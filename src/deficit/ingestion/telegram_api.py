"""Collector of PUBLIC Telegram channel history through the official API (Telethon).

Writes the SAME per-channel Parquet as ``telegram_public`` (``data/raw/telegram/<channel>.parquet``, same columns,
dedupe by (channel, post_id)), so ``deficit extract`` / ``region-restoration`` work unchanged.

Safety / politeness rules (enforced in code and by tests/test_telegram_api.py):
  * read-only: only ``get_entity`` (public username lookup) and ``iter_messages``; never joins, sends, lists dialogs
    or touches private chats; only channels listed in config/sources.yaml are accepted;
  * one process, one client; batches of 100 messages with >= 2 s pause (``wait_time``); ``FloodWaitError`` is
    honoured by sleeping the required time (+ margin);
  * credentials come from the environment (TELEGRAM_API_ID / TELEGRAM_API_HASH, via .env) and are never logged;
    the login session lives in data/telegram_session/ (git- and docker-ignored, never baked into the image).

Login is interactive and done once by the user: ``deficit tg-login``.
"""
from __future__ import annotations

import getpass
import logging
import os
import time
from pathlib import Path

import pandas as pd

from .telegram_public import (COLUMNS, load_store, merge_store, quality_report, save_store,  # noqa: F401
                              store_path, to_frame)

log = logging.getLogger("deficit")

BATCH = 100               # messages per request
BATCH_PAUSE = 2.0         # seconds between batches (Telethon wait_time); never lower than 1 s
CHANNEL_PAUSE = 10        # seconds between channels
FLOOD_MARGIN = 5          # extra seconds on top of a FloodWait
MAX_FLOOD_RETRIES = 20
CHECKPOINT_BATCHES = 20
SESSION_NAME = "ua_deficit"


class TelegramApiError(RuntimeError):
    pass


def session_dir(data_root: Path) -> Path:
    return Path(data_root) / "telegram_session"


def credentials() -> tuple[int, str]:
    """(api_id, api_hash) from the environment; values are never printed."""
    api_id, api_hash = os.environ.get("TELEGRAM_API_ID", "").strip(), os.environ.get("TELEGRAM_API_HASH", "").strip()
    if not api_id or not api_hash:
        raise TelegramApiError("TELEGRAM_API_ID / TELEGRAM_API_HASH are not set (put them in .env)")
    try:
        return int(api_id), api_hash
    except ValueError:
        raise TelegramApiError("TELEGRAM_API_ID must be an integer") from None


def allowed_channels(sources_yaml: Path) -> set[str]:
    import yaml
    src = yaml.safe_load(Path(sources_yaml).read_text(encoding="utf-8"))
    return {c["name"] for c in src["telegram"]}


def make_client(data_root: Path):
    from telethon import TelegramClient
    api_id, api_hash = credentials()
    d = session_dir(data_root)
    d.mkdir(parents=True, exist_ok=True)
    return TelegramClient(str(d / SESSION_NAME), api_id, api_hash)


# --------------------------------------------------------------------------------------------- login
def login(data_root: Path, input_fn=input, getpass_fn=getpass.getpass) -> None:
    """One-time interactive login: phone number, code from the Telegram app, optional 2FA password."""
    client = make_client(data_root)
    client.start(phone=lambda: input_fn("Phone number (international format, +380...): ").strip(),
                 code_callback=lambda: input_fn("Code from the Telegram app: ").strip(),
                 password=lambda: getpass_fn("2FA password (input hidden): "))
    log.info("login OK; session stored in %s (keep it private, never commit it)", session_dir(data_root))
    client.disconnect()


# --------------------------------------------------------------------------------------------- conversion
def _forward_name(m) -> str | None:
    f = getattr(m, "fwd_from", None)
    if not f:
        return None
    name = getattr(f, "from_name", None)
    if name:
        return str(name)
    fid = getattr(f, "from_id", None)
    if fid is not None:
        for attr in ("channel_id", "user_id", "chat_id"):
            v = getattr(fid, attr, None)
            if v is not None:
                return f"{attr.split('_')[0]}:{v}"
    return "forwarded"


def message_to_post(m, channel: str) -> dict | None:
    """Telethon Message -> post dict (schema of telegram_public). Service messages / no date -> None."""
    dt = getattr(m, "date", None)
    if dt is None or getattr(m, "action", None) is not None:
        return None
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    rep = getattr(getattr(m, "reply_to", None), "reply_to_msg_id", None)
    views = getattr(m, "views", None)
    return {"channel": channel, "post_id": int(m.id), "datetime": ts,
            "text": getattr(m, "message", None) or "",
            "views": int(views) if views is not None else None,
            "has_media": getattr(m, "media", None) is not None,
            "forwarded_from": _forward_name(m),
            "reply_to": int(rep) if rep is not None else None,
            "url": f"https://t.me/{channel}/{int(m.id)}"}


# --------------------------------------------------------------------------------------------- collection
def _flood_error():
    from telethon.errors import FloodWaitError
    return FloodWaitError


def _iter_batches(client, entity, channel: str, offset_id: int, sleep, flood_exc):
    """Yield posts newest -> oldest starting below `offset_id` (0 = newest). Resumes after FloodWait."""
    retries = 0
    while True:
        try:
            for m in client.iter_messages(entity, limit=None, offset_id=offset_id, wait_time=BATCH_PAUSE):
                offset_id = int(m.id)      # resume point if a flood wait interrupts
                p = message_to_post(m, channel)
                if p:
                    yield p
            return
        except flood_exc as e:
            retries += 1
            if retries > MAX_FLOOD_RETRIES:
                raise
            wait = int(getattr(e, "seconds", 60)) + FLOOD_MARGIN
            log.warning("%s: FloodWait, sleeping %ds (retry %d)", channel, wait, retries)
            sleep(wait)


def _walk(client, entity, channel: str, start_before: int, since: pd.Timestamp, stop_at_id: int | None,
          path: Path, store: pd.DataFrame, sleep, flood_exc) -> tuple[pd.DataFrame, int]:
    buf, n_new, n_seen = [], 0, 0
    for p in _iter_batches(client, entity, channel, start_before, sleep, flood_exc):
        n_seen += 1
        if p["datetime"] < since or (stop_at_id is not None and p["post_id"] <= stop_at_id):
            break
        buf.append(p)
        n_new += 1
        if len(buf) >= BATCH * CHECKPOINT_BATCHES:
            store = merge_store(store, to_frame(buf))
            save_store(store, path)
            buf = []
            log.info("%s: +%d posts, oldest %s", channel, n_new, p["datetime"].date())
    if buf:
        store = merge_store(store, to_frame(buf))
        save_store(store, path)
    return store, n_new


def collect_channel_api(client, channel: str, data_root: Path, since: str = "2022-10-01", update: bool = False,
                        sleep=time.sleep, flood_exc=None) -> pd.DataFrame:
    """Forward catch-up from the stored max id, then (unless `update`) backward fill from the stored min id to
    `since`. Saves checkpoints; deduplicates by (channel, post_id)."""
    flood_exc = flood_exc or _flood_error()
    since_ts = pd.Timestamp(since, tz="UTC")
    path = store_path(data_root, channel)
    store = load_store(path)
    entity = client.get_entity(channel)          # public username lookup only
    total = 0
    if len(store):
        max_id = int(store["post_id"].max())
        store, n = _walk(client, entity, channel, 0, since_ts, max_id, path, store, sleep, flood_exc)
        total += n
        log.info("%s: forward catch-up +%d posts (stored max id was %d)", channel, n, max_id)
    if not update:
        if len(store):
            min_id = int(store["post_id"].min())
            if store["datetime"].min() > since_ts and min_id > 1:
                store, n = _walk(client, entity, channel, min_id, since_ts, None, path, store, sleep, flood_exc)
                total += n
        else:
            store, n = _walk(client, entity, channel, 0, since_ts, None, path, store, sleep, flood_exc)
            total += n
    if len(store) == 0:
        log.warning("%s: no posts collected", channel)
    else:
        log.info("%s: %d posts stored (%s .. %s), +%d this run", channel, len(store),
                 store["datetime"].min().date(), store["datetime"].max().date(), total)
    return store


def channel_message_count(client, channel: str) -> int:
    """Total messages in a channel via one cheap call (limit=0 returns only the total). Needs the session."""
    return int(client.get_messages(channel, limit=0).total)


def run_collect_api(channels: list[str], data_root: Path, reports_dir: Path, sources_yaml: Path,
                    since: str, update: bool, client=None, sleep=time.sleep) -> None:
    allowed = allowed_channels(sources_yaml)
    bad = [c for c in channels if c not in allowed]
    if bad:
        raise TelegramApiError(f"channels not listed in config/sources.yaml (refusing): {bad}")
    own = client is None
    if own:
        client = make_client(data_root)
        client.connect()
        if not client.is_user_authorized():
            client.disconnect()
            raise TelegramApiError("no valid session: run `deficit tg-login` first")
    try:
        for i, ch in enumerate(channels):
            if i:
                log.info("pausing %ds between channels", CHANNEL_PAUSE)
                sleep(CHANNEL_PAUSE)
            try:
                collect_channel_api(client, ch, data_root, since=since, update=update, sleep=sleep)
            except Exception as e:   # one bad channel (private/renamed) must not stop the others
                if e.__class__.__name__ == "FloodWaitError":
                    raise
                log.error("%s skipped: %s: %s", ch, e.__class__.__name__, e)
    finally:
        if own:
            client.disconnect()
    quality_report(data_root, reports_dir)
