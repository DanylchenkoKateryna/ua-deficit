"""Polite scraper of public Telegram channel web previews (https://t.me/s/<channel>).

No login and no Telegram API keys: only the public HTML preview, 20 posts per page, paginated backwards
with ``?before=<post_id>``.  Output: one Parquet per channel, ``data/raw/telegram/<channel>.parquet``,
deduplicated by (channel, post_id).  Parsing uses only the standard library (html.parser + re).

Columns: channel, post_id, datetime (UTC), text, views, has_media, forwarded_from, reply_to, url, matched_query, window_strike.

Modes (see ``collect``):
  * backfill (default): catch up on posts newer than the stored max id, then walk backwards from the
    stored min id (or from the newest post when nothing is stored) until ``since`` is reached.
  * update: only the forward catch-up (cheap, meant for frequent scheduled runs).
  * window (``collect_windows``): only posts from [strike day - 1, strike day + window_days] (UTC) for the strikes of
    ``data/external/attacks.csv``; the post id of a date is found by interpolation + a few probe requests, pages are then read
    forwards with ``?after=<id>``.  Posts get column ``window_strike`` (strike dates whose window contains the post, joined by
    ``|``); finished windows are listed in ``_windows.csv`` (the observation periods the oblast-level episodes are censored at).
  * query (``collect_channel_query``): uses the preview's built-in search ``?q=<term>`` (same ``before=`` pagination, 20 hits per
    page, newest first) so that high-volume channels (regional military administrations, ~200 posts/day, mostly unrelated)
    need only the matching posts.  Hits are stored in the same per-channel Parquet with column ``matched_query`` (the terms that
    found the post, joined by ``|``; empty for posts collected in the normal modes).  The search matches whole words
    (lemma-aware, e.g. "відключення" = "відключень", but not the stem "відключен"), so terms must be full word forms.

This module only collects raw posts; text extraction / any use as model input is a later step and must
respect the no-look-ahead rules (README §5.1) - ``datetime`` is the post publication time.
"""
from __future__ import annotations

import html
import logging
import math
import os
import random
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("deficit")

BASE_URL = "https://t.me/s/{channel}"
USER_AGENT = ("ua-deficit-forecast-thesis/0.1 (academic research on power-system outages; "
              "public t.me/s previews only; contact: kateryna.danylchenko@asgnederland.nl)")
TERM_COLS = ("matched_query", "window_strike")    # '|'-joined sets, unioned when a post is scraped again
COLUMNS = ["channel", "post_id", "datetime", "text", "views", "has_media", "forwarded_from", "reply_to", "url",
           *TERM_COLS]
MIN_DELAY, MAX_DELAY = 3.0, 5.0     # seconds between requests (mean 4 s = at most 900/hour, capped below)
MAX_REQUESTS_PER_HOUR = 800         # hard cap, enforced with a sliding window (retries count)
CHANNEL_PAUSE = 60                  # seconds between channels (used by the CLI)
MAX_429 = 3                         # HTTP 429 answers tolerated per process; the third one stops everything
CHECKPOINT_PAGES = 25
MEDIA_MARKERS = ("tgme_widget_message_photo_wrap", "tgme_widget_message_video_player",
                 "tgme_widget_message_roundvideo", "tgme_widget_message_voice",
                 "tgme_widget_message_document_wrap", "tgme_widget_message_sticker",
                 "tgme_widget_message_poll", "tgme_widget_message_location")


class ChannelUnavailable(RuntimeError):
    """The preview page is not public (redirects to the channel page / login) or does not exist."""


class TooManyRequests(RuntimeError):
    """HTTP 429 was answered MAX_429 times: the whole collection must stop (never retried by the callers)."""


# --------------------------------------------------------------------------------------------- parsing
class _TextExtractor(HTMLParser):
    """HTML fragment -> plain text; <br> and block ends become line breaks, tags are dropped."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("p", "blockquote", "pre", "li"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(fragment: str) -> str:
    p = _TextExtractor()
    p.feed(fragment)
    p.close()
    text = "".join(p.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_views(s: str | None) -> int | None:
    """'42.2K' -> 42200, '1.3M' -> 1300000, '987' -> 987."""
    if not s:
        return None
    s = s.strip().replace(",", ".").replace("\xa0", "").replace(" ", "")
    m = re.fullmatch(r"([\d.]+)([KkMm]?)", s)
    if not m:
        return None
    mult = {"": 1, "k": 1_000, "m": 1_000_000}[m.group(2).lower()]
    try:
        return int(round(float(m.group(1)) * mult))
    except ValueError:
        return None


def _inner_div(chunk: str, class_marker: str) -> str | None:
    """Inner HTML of the first <div> whose class attribute contains `class_marker` (no nested divs
    expected inside message text; a depth counter keeps it safe if there are)."""
    m = re.search(r'<div class="[^"]*%s[^"]*"[^>]*>' % re.escape(class_marker), chunk)
    if not m:
        return None
    pos, depth = m.end(), 1
    for t in re.finditer(r"<(/?)div\b[^>]*>", chunk[pos:]):
        depth += -1 if t.group(1) else 1
        if depth == 0:
            return chunk[pos:pos + t.start()]
    return chunk[pos:]


def parse_page(page_html: str, channel: str | None = None) -> list[dict]:
    """Parse one t.me/s page into post dicts (oldest first as on the page). Service messages without an id
    or datetime are skipped."""
    chunks = re.split(r'<div class="tgme_widget_message_wrap', page_html)[1:]
    posts: list[dict] = []
    for ch in chunks:
        m = re.search(r'data-post="([^"/]+)/(\d+)"', ch)
        if not m:
            continue
        ch_name, post_id = m.group(1), int(m.group(2))
        dt = re.search(r'<time[^>]*\bdatetime="([^"]+)"', ch)
        if not dt:
            continue
        # message text (not the reply preview, which uses js-message_reply_text)
        txt_html = _inner_div(ch, "js-message_text")
        text = html_to_text(txt_html) if txt_html else ""
        views_m = re.search(r'class="tgme_widget_message_views">([^<]*)<', ch)
        fwd = re.search(r'tgme_widget_message_forwarded_from[^"]*">(.*?)</div>', ch, re.S)
        fwd_name = None
        if fwd:
            fwd_name = html_to_text(fwd.group(1)).replace("Forwarded from", "").strip(" \n()") or None
            fwd_name = re.sub(r"\s*\n\s*", " ", fwd_name) if fwd_name else None
        reply = re.search(r'<a class="tgme_widget_message_reply[^"]*"[^>]*href="[^"]*?/(\d+)"', ch)
        if not reply:
            reply = re.search(r'<a[^>]*href="[^"]*?/(\d+)"[^>]*class="tgme_widget_message_reply', ch)
        ts = pd.Timestamp(dt.group(1))
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        posts.append({
            "channel": channel or ch_name,
            "post_id": post_id,
            "datetime": ts,
            "text": text,
            "views": parse_views(html.unescape(views_m.group(1))) if views_m else None,
            "has_media": any(k in ch for k in MEDIA_MARKERS),
            "forwarded_from": fwd_name,
            "reply_to": int(reply.group(1)) if reply else None,
            "url": f"https://t.me/{ch_name}/{post_id}",
        })
    return posts


# --------------------------------------------------------------------------------------------- storage
def raw_dir(data_root: Path) -> Path:
    return Path(data_root) / "raw" / "telegram"


def store_path(data_root: Path, channel: str) -> Path:
    return raw_dir(data_root) / f"{channel}.parquet"


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "channel": pd.Series(dtype="object"), "post_id": pd.Series(dtype="int64"),
        "datetime": pd.Series(dtype="datetime64[ns, UTC]"), "text": pd.Series(dtype="object"),
        "views": pd.Series(dtype="Int64"), "has_media": pd.Series(dtype="bool"),
        "forwarded_from": pd.Series(dtype="object"), "reply_to": pd.Series(dtype="Int64"),
        "url": pd.Series(dtype="object"), "matched_query": pd.Series(dtype="object"),
        "window_strike": pd.Series(dtype="object")})


def to_frame(posts: list[dict]) -> pd.DataFrame:
    if not posts:
        return _empty()
    df = pd.DataFrame(posts, columns=COLUMNS)   # missing keys (e.g. matched_query) become None
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["views"] = df["views"].astype("Int64")
    df["reply_to"] = df["reply_to"].astype("Int64")
    df["has_media"] = df["has_media"].astype(bool)
    df["post_id"] = df["post_id"].astype("int64")
    return df


def load_store(path: Path) -> pd.DataFrame:
    """Stored posts; files written before the query / window modes get empty ``matched_query`` / ``window_strike``."""
    if not path.exists():
        return _empty()
    df = pd.read_parquet(path)
    for c in TERM_COLS:
        if c not in df.columns:
            df[c] = None
    return df


def _union_terms(values: pd.Series) -> str | None:
    terms = sorted({t for v in values if isinstance(v, str) for t in v.split("|") if t})
    return "|".join(terms) or None


def merge_store(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union, dedupe by (channel, post_id); the newer scrape wins (fresher view counts); ``matched_query`` and
    ``window_strike`` are the unions of the values of all scrapes of the post."""
    parts = [d for d in (old, new) if len(d)]
    if not parts:
        return _empty()
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0].copy()
    for c in TERM_COLS:
        if c not in df.columns:
            df[c] = None
    dup = df.duplicated(["channel", "post_id"], keep=False)
    if dup.any():
        terms = {c: df.loc[dup].groupby(["channel", "post_id"])[c].agg(_union_terms) for c in TERM_COLS}
        df = df.drop_duplicates(["channel", "post_id"], keep="last").reset_index(drop=True)
        key = pd.MultiIndex.from_frame(df[["channel", "post_id"]])
        for c in TERM_COLS:
            df[c] = df[c].astype(object)
            hit = key.isin(terms[c].index)
            df.loc[hit, c] = terms[c].reindex(key[hit]).to_numpy()
    return df.sort_values("post_id").reset_index(drop=True)


def save_store(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------- HTTP
class Fetcher:
    """requests wrapper: descriptive UA, jittered delay between requests, retry/backoff, 429 handling."""

    def __init__(self, min_delay: float = MIN_DELAY, max_delay: float = MAX_DELAY, max_retries: int = 6,
                 session: requests.Session | None = None, sleep=time.sleep,
                 max_per_hour: int = MAX_REQUESTS_PER_HOUR) -> None:
        self.min_delay, self.max_delay, self.max_retries = min_delay, max_delay, max_retries
        self.max_per_hour = max_per_hour
        self._stamps: list[float] = []
        self._slow = 1.0   # multiplier on the delay, raised after every HTTP 429
        self.n429 = 0      # HTTP 429 answers so far (process lifetime); MAX_429 of them stop everything
        self.n_requests = 0
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "uk,en;q=0.8"})
        self._sleep = sleep
        self._last = 0.0

    def _throttle(self) -> None:
        wait = random.uniform(self.min_delay, self.max_delay) * self._slow - (time.monotonic() - self._last)
        if wait > 0:
            self._sleep(wait)
        now = time.monotonic()
        self._stamps = [t for t in self._stamps if now - t < 3600]
        if len(self._stamps) >= self.max_per_hour:
            pause = 3600 - (now - self._stamps[0]) + 1
            log.info("hourly request cap (%d) reached; pausing %.0fs", self.max_per_hour, pause)
            self._sleep(pause)
            now = time.monotonic()
            self._stamps = [t for t in self._stamps if now - t < 3600]
        self._stamps.append(now)
        self._last = now
        self.n_requests += 1

    def get_page(self, channel: str, before: int | None = None, query: str | None = None,
                 after: int | None = None) -> str:
        url = BASE_URL.format(channel=channel)
        params = {k: v for k, v in (("q", query), ("before", before), ("after", after)) if v is not None and v != ""}
        params = params or None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self.s.get(url, params=params, timeout=30, allow_redirects=False)
            except requests.RequestException as e:
                backoff = min(300, 5 * 2 ** attempt)
                log.warning("%s: network error %s; retry in %ds", channel, e.__class__.__name__, backoff)
                self._sleep(backoff)
                continue
            if r.status_code == 200:
                return r.text
            if r.status_code in (301, 302, 303, 307, 308):
                raise ChannelUnavailable(f"{channel}: redirect to {r.headers.get('Location')} "
                                         "(preview disabled or channel does not exist)")
            if r.status_code == 429:
                try:
                    wait = int(r.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 0
                self.n429 += 1
                if self.n429 >= MAX_429:
                    raise TooManyRequests(f"{channel}: HTTP 429 answered {self.n429} times; stopping the collection")
                wait = max(wait, 60 * 2 ** (self.n429 - 1))   # growing backoff: 60 s, 120 s, ...
                self._slow = min(self._slow * 1.5, 8.0)   # continue slower afterwards
                log.warning("%s: HTTP 429 (#%d); stopping requests for %ds", channel, self.n429, wait)
                self._sleep(wait)
                continue
            if r.status_code in (404, 400):
                raise ChannelUnavailable(f"{channel}: HTTP {r.status_code}")
            backoff = min(300, 5 * 2 ** attempt)
            log.warning("%s: HTTP %s; retry in %ds", channel, r.status_code, backoff)
            self._sleep(backoff)
        raise RuntimeError(f"{channel}: giving up after {self.max_retries} retries (before={before})")


# --------------------------------------------------------------------------------------------- collect
def _walk_back(fetcher: Fetcher, channel: str, start_before: int | None, since: pd.Timestamp,
               stop_at_id: int | None, path: Path, store: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Paginate backwards from `start_before` (None = newest page) until posts older than `since`, ids
    <= `stop_at_id` (already stored) or the beginning of the channel. Checkpoints to disk."""
    before, pages, buf, n_new = start_before, 0, [], 0
    while True:
        posts = parse_page(fetcher.get_page(channel, before), channel)
        pages += 1
        if not posts:
            break
        min_id = min(p["post_id"] for p in posts)
        keep = [p for p in posts if p["datetime"] >= since and (stop_at_id is None or p["post_id"] > stop_at_id)]
        buf += keep
        n_new += len(keep)
        oldest = min(p["datetime"] for p in posts)
        if pages % 10 == 0:
            log.info("%s: page %d, oldest %s (id %d), +%d posts", channel, pages, oldest.date(), min_id, n_new)
        if pages % CHECKPOINT_PAGES == 0 and buf:
            store = merge_store(store, to_frame(buf))
            save_store(store, path)
            buf = []
        if oldest < since or (stop_at_id is not None and min_id <= stop_at_id) or min_id <= 1:
            break
        before = min_id
    if buf:
        store = merge_store(store, to_frame(buf))
        save_store(store, path)
    return store, n_new


def collect_channel(channel: str, data_root: Path, since: str = "2022-02-01", update: bool = False,
                    fetcher: Fetcher | None = None) -> pd.DataFrame:
    fetcher = fetcher or Fetcher()
    since_ts = pd.Timestamp(since, tz="UTC")
    path = store_path(data_root, channel)
    store = load_store(path)
    total = 0
    # 1) forward catch-up (also the very first page when nothing is stored)
    if len(store):
        max_id = int(store["post_id"].max())
        store, n = _walk_back(fetcher, channel, None, since_ts, max_id, path, store)
        total += n
        log.info("%s: forward catch-up +%d posts (stored max id was %d)", channel, n, max_id)
    # 2) backward fill to `since`
    if not update:
        if len(store):
            min_id = int(store["post_id"].min())
            oldest = store["datetime"].min()
            if oldest > since_ts and min_id > 1:
                store, n = _walk_back(fetcher, channel, min_id, since_ts, None, path, store)
                total += n
        else:
            store, n = _walk_back(fetcher, channel, None, since_ts, None, path, store)
            total += n
    if len(store) == 0:
        log.warning("%s: no posts collected", channel)
    else:
        log.info("%s: %d posts stored (%s .. %s), +%d this run", channel, len(store),
                 store["datetime"].min().date(), store["datetime"].max().date(), total)
    return store


def _walk_query(fetcher: Fetcher, channel: str, term: str, since: pd.Timestamp) -> tuple[list[dict], int]:
    """Search `term` in the channel preview, newest hit first, paginating with ``before=<lowest id of the page>``
    until a page is empty, hits are older than `since` or the ids stop decreasing. Returns (posts, pages)."""
    before, pages, out = None, 0, []
    while True:
        posts = parse_page(fetcher.get_page(channel, before, query=term), channel)
        pages += 1
        if not posts:
            break
        min_id = min(p["post_id"] for p in posts)
        out += [dict(p, matched_query=term) for p in posts if p["datetime"] >= since]
        if min(p["datetime"] for p in posts) < since or min_id <= 1 or (before is not None and min_id >= before):
            break
        before = min_id
    return out, pages


def collect_channel_query(channel: str, terms: list[str], data_root: Path, since: str = "2022-10-01",
                          fetcher: Fetcher | None = None) -> pd.DataFrame:
    """Query mode: per term, all search hits since `since`; merged into the channel's Parquet (dedupe by
    channel+post_id, existing posts are kept, ``matched_query`` accumulates the terms). Saved after every term."""
    fetcher = fetcher or Fetcher()
    since_ts = pd.Timestamp(since, tz="UTC")
    path = store_path(data_root, channel)
    store = load_store(path)
    n_before = len(store)
    for term in terms:
        hits, pages = _walk_query(fetcher, channel, term, since_ts)
        if hits:
            store = merge_store(store, to_frame(hits))
            save_store(store, path)
        log.info("%s: query %r: %d hits in %d requests (oldest %s)", channel, term, len(hits), pages,
                 min(h["datetime"] for h in hits).date() if hits else "-")
    log.info("%s: %d posts stored, +%d new this run", channel, len(store), len(store) - n_before)
    return store


# --------------------------------------------------------------------------------------------- window mode
PAGE = 20                  # posts per t.me/s page
PROBES_PER_WINDOW = 6      # hard limit of probe requests to find the post id of a window start
PLAN_PROBES = 3            # assumed mean number of probes per window in the cost estimate
WINDOWS_FILE = "_windows.csv"
WINDOW_COLS = ["channel", "window_start", "window_end", "strikes", "first_id", "last_id", "n_posts", "requests", "probes",
               "end_of_channel", "collected_at"]


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class IdClock:
    """Known (post id, time) points of a channel; estimates the post id at a time by linear interpolation (ids and times
    both increase), extrapolating with the slope of the nearest segment."""

    def __init__(self, default_ids_per_day: float = 40.0) -> None:
        self._pts: dict[int, float] = {}
        self.default_rate = default_ids_per_day / 86400.0
        self._cache = None

    def __len__(self) -> int:
        return len(self._pts)

    def add(self, posts) -> None:
        """`posts`: dicts with post_id / datetime. Only the first and last post of a long list are kept."""
        ps = list(posts)
        if len(ps) > 4:
            ps = [ps[0], ps[-1]]
        for p in ps:
            self._pts[int(p["post_id"])] = _utc(p["datetime"]).timestamp()
        self._cache = None

    def add_frame(self, df: pd.DataFrame) -> None:
        if len(df):
            self._pts.update(zip(df["post_id"].astype(int), df["datetime"].map(lambda t: t.timestamp())))
            self._cache = None

    def _arr(self):
        if self._cache is None:
            ids = np.array(sorted(self._pts), dtype=float)
            ts = np.array([self._pts[int(i)] for i in ids])
            order = np.argsort(ts, kind="stable")
            self._cache = (ts[order], ids[order])
        return self._cache

    @property
    def max_id(self) -> int:
        return max(self._pts) if self._pts else 0

    def estimate(self, ts) -> float:
        t = _utc(ts).timestamp()
        xs, ys = self._arr()
        if len(xs) == 0:
            raise ValueError("no known points")
        if len(xs) == 1:
            return max(float(ys[0] + (t - xs[0]) * self.default_rate), 0.0)
        if xs[0] <= t <= xs[-1]:
            return float(np.interp(t, xs, ys))
        a, b = (0, 1) if t < xs[0] else (len(xs) - 2, len(xs) - 1)
        dt = xs[b] - xs[a]
        slope = (ys[b] - ys[a]) / dt if dt > 0 else self.default_rate
        if slope <= 0:
            slope = self.default_rate
        ref = a if t < xs[0] else b
        return max(float(ys[ref] + (t - xs[ref]) * slope), 0.0)

    def bracket(self, ts):
        """(lo, hi): lo = known (id, t) with the largest time <= ts, hi = the one with the smallest time > ts (None if none)."""
        t = _utc(ts).timestamp()
        xs, ys = self._arr()
        k = int(np.searchsorted(xs, t, side="right"))
        lo = (int(ys[k - 1]), float(xs[k - 1])) if k > 0 else None
        hi = (int(ys[k]), float(xs[k])) if k < len(xs) else None
        return lo, hi


class _Counted:
    """Counts the requests of any fetcher (per-window accounting, also for test fakes)."""

    def __init__(self, inner) -> None:
        self.inner, self.n = inner, 0

    def get_page(self, *a, **k):
        self.n += 1
        return self.inner.get_page(*a, **k)


def _fetch_posts(fetcher, channel: str, **kw) -> list[dict]:
    return parse_page(fetcher.get_page(channel, **kw), channel)


def locate_start(fetcher, channel: str, clock: IdClock, ts, max_probes: int = PROBES_PER_WINDOW,
                 page: int = PAGE) -> tuple[int, list[dict] | None, int]:
    """Find where to start reading forwards for time `ts`.  Returns (after_id, prefetched, probes): reading ``?after=after_id``
    yields the first posts at/before `ts`; `prefetched` is the already fetched page when a probe page started at or before
    `ts` (then reading continues after its last id).  Each probe is one request, at most `max_probes`.  The estimate
    interpolates between the nearest known points (store, earlier probes and windows) and aims half a page before `ts`;
    when the probes run out, reading starts from the nearest known id below `ts` (extra pages, never a missed post)."""
    t = _utc(ts)
    probes = 0
    tried: set[int] = set()
    ceil_id: int | None = None          # an `after` id that returned nothing: the channel ends below it
    side, streak = None, 0              # which bound the last probes moved; two in a row -> bisect instead of interpolate
    while True:
        if len(clock) == 0:             # nothing known: the newest page anchors the clock
            posts = _fetch_posts(fetcher, channel)
            probes += 1
            if not posts:
                return 0, None, probes
            clock.add(posts)
            continue
        lo, hi = clock.bracket(t)
        lo_id = lo[0] if lo else 0
        hi_id = hi[0] if hi else ceil_id
        if hi_id is not None and hi_id - lo_id <= 2 * page:     # cheap to read forwards from lo
            return lo_id, None, probes
        if probes >= max_probes:
            return lo_id, None, probes
        g = max(int(clock.estimate(t)) - page // 2, lo_id)
        if hi_id is not None:
            g = max(min(g, hi_id - page), lo_id)
        if g in tried or streak >= 2:   # no progress / one-sided convergence: bisect the bracket
            g = max((lo_id + (hi_id if hi_id is not None else clock.max_id)) // 2 - page // 2, lo_id)
        tried.add(g)
        posts = [p for p in _fetch_posts(fetcher, channel, after=g) if p["post_id"] > g]
        probes += 1
        if not posts:                   # g is at/after the newest post
            if g <= lo_id:
                return lo_id, None, probes
            ceil_id = g if ceil_id is None else min(ceil_id, g)
            continue
        clock.add(posts)
        if posts[0]["datetime"] <= t <= posts[-1]["datetime"]:
            return g, posts, probes
        moved = "lo" if posts[-1]["datetime"] < t else "hi"
        streak = streak + 1 if moved == side else 1
        side = moved


def read_window(fetcher, channel: str, start_after: int, prefetched: list[dict] | None, start: pd.Timestamp,
                end: pd.Timestamp, clock: IdClock | None = None, checkpoint=None) -> tuple[list[dict], int, bool]:
    """Read forwards (``?after=``) from `start_after` until a page reaches `end`; returns (posts in [start, end), page
    requests, reached the end of the channel).  `checkpoint(new_posts)` is called every CHECKPOINT_PAGES requests."""
    cur, buf, n_req, posts, saved = start_after, [], 0, prefetched, 0
    while True:
        if posts is None:
            posts = [p for p in _fetch_posts(fetcher, channel, after=cur) if p["post_id"] > cur]
            n_req += 1
        if not posts:
            return buf, n_req, True
        if clock is not None:
            clock.add(posts)
        buf += [p for p in posts if start <= p["datetime"] < end]
        mx = max(p["post_id"] for p in posts)
        if mx <= cur:
            return buf, n_req, False
        cur = mx
        if max(p["datetime"] for p in posts) >= end:
            return buf, n_req, False
        posts = None
        if checkpoint and n_req and n_req % CHECKPOINT_PAGES == 0 and len(buf) > saved:
            checkpoint(buf[saved:])
            saved = len(buf)


def strike_windows(dates, window_days: int = 7, pre_days: int = 1) -> list[dict]:
    """[strike day - pre_days 00:00, strike day + window_days + 1 day 00:00) UTC per strike date (both end days included)."""
    out = []
    for d in sorted({pd.Timestamp(x).strftime("%Y-%m-%d") for x in dates}):
        day = _utc(d)
        out.append({"strike": d, "start": day - pd.Timedelta(days=pre_days),
                    "end": day + pd.Timedelta(days=window_days + 1)})
    return out


def merge_windows(wins: list[dict]) -> list[dict]:
    """Union of overlapping / touching windows (chained strikes); a merged window keeps its strikes with their own spans."""
    out: list[dict] = []
    for w in sorted(wins, key=lambda x: x["start"]):
        if out and w["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], w["end"])
            out[-1]["strikes"].append(w)
        else:
            out.append({"start": w["start"], "end": w["end"], "strikes": [w]})
    return out


def tag_window_strike(dt: pd.Timestamp, strikes: list[dict]) -> str | None:
    return "|".join(s["strike"] for s in strikes if s["start"] <= dt < s["end"]) or None


def estimate_windows(clock: IdClock, merged: list[dict], page: int = PAGE, probes: int = PLAN_PROBES) -> dict:
    """Upper-bound request count of the merged windows: ids in the window / page size (ids >= posts, gaps are deleted
    posts) + `probes` per window."""
    rows = []
    top = clock.max_id
    for w in merged:
        a, b = clock.estimate(w["start"]), min(clock.estimate(w["end"]), max(top, 1))
        ids = max(b - a, page)
        rows.append({"start": w["start"], "end": w["end"], "strikes": len(w["strikes"]), "ids": ids,
                     "requests": math.ceil(ids / page) + probes})
    return {"windows": rows, "ids": sum(r["ids"] for r in rows), "requests": sum(r["requests"] for r in rows)}


def request_rate(min_delay: float = MIN_DELAY, max_delay: float = MAX_DELAY, cap: int = MAX_REQUESTS_PER_HOUR) -> float:
    """Effective requests per hour: the slower of the jittered delay (mean) and the hourly cap."""
    return min(float(cap), 3600.0 / ((min_delay + max_delay) / 2))


def select_strikes(attacks: pd.DataFrame, clocks: dict[str, IdClock], since: str, window_days: int = 7,
                   pre_days: int = 1, share: float = 1.0, max_hours: float = 6.0,
                   until: dict[str, str] | None = None) -> dict:
    """Which strikes to collect.  Outcome-independent rule: every strike on/after `since` when the estimated total time
    fits `max_hours`; otherwise strikes in descending `weapons_through` (ties: earlier first) until the budget is met
    (the largest k whose estimate fits), and never more than ceil(share * n)."""
    until = until or {}
    a = attacks.copy()
    a["date"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")
    a = a[a["date"] >= pd.Timestamp(since).strftime("%Y-%m-%d")].drop_duplicates("date")
    a["_w"] = pd.to_numeric(a["weapons_through"], errors="coerce").fillna(-1) if "weapons_through" in a else -1.0
    order = a.sort_values(["_w", "date"], ascending=[False, True])
    rate = request_rate()
    pause_h = max(len(clocks) - 1, 0) * CHANNEL_PAUSE / 3600.0

    def cost(dates) -> dict:
        per, tot = {}, 0
        for ch, clk in clocks.items():
            ds = [d for d in dates if d <= until.get(ch, "9999")]
            est = (estimate_windows(clk, merge_windows(strike_windows(ds, window_days, pre_days))) if ds
                   else {"windows": [], "ids": 0, "requests": 0})
            per[ch] = est
            tot += est["requests"]
        return {"per_channel": per, "requests": tot, "hours": tot / rate + pause_h}

    all_dates = list(order["date"])
    k_share = max(1, math.ceil(share * len(all_dates)))
    full = cost(all_dates[:k_share])
    if full["hours"] <= max_hours:
        k, res = k_share, full
        rule = "all strikes" if k_share == len(all_dates) else f"top {share:.0%} of strikes by weapons_through"
    else:
        k, res = 0, cost([])
        for kk in range(1, k_share + 1):
            c = cost(all_dates[:kk])
            if c["hours"] > max_hours:
                break
            k, res = kk, c
        rule = (f"strikes in descending weapons_through (ties: earlier date first) while the estimated total time stays within "
                f"{max_hours:g} h: top {k} of {len(all_dates)}")
    chosen = sorted(all_dates[:k])
    return {"selected": chosen, "excluded": sorted(set(all_dates) - set(chosen)), "cost": res, "rule": rule,
            "full_cost": cost(all_dates), "n_strikes": len(all_dates), "weapons": dict(zip(order["date"], order["_w"])),
            "rate_per_hour": rate}


def write_plan(plan: dict, path: Path, window_days: int, pre_days: int, max_hours: float, since: str,
               until: dict[str, str] | None = None) -> None:
    c = plan["cost"]
    md = ["# Telegram window collection: plan", "",
          f"Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC by `deficit collect --windows-from-attacks`. The estimate "
          "makes no request: it uses the (post id, time) points already stored per channel (keyword-search samples) and "
          "linear interpolation between them.", "",
          "## Selection rule (does not look at any outage outcome)", "",
          f"- Candidates: every row of `data/external/attacks.csv` dated {since} or later ({plan['n_strikes']} strikes).",
          f"- Window of a strike: [strike day - {pre_days} d, strike day + {window_days} d], both days included, UTC; overlapping "
          "windows (chained strikes) are merged.",
          f"- Rule applied: **{plan['rule']}**. Strikes are ranked by `weapons_through` "
          "(weapons that reached their targets, an attack-size column of attacks.csv), never by observed impact or "
          "restoration times.",
          f"- Selected {len(plan['selected'])} of {plan['n_strikes']} strikes; time budget {max_hours:g} h."]
    if until:
        md.append("- Channel-specific end of the window period: " + ", ".join(f"{k} until {v}" for k, v in until.items()) + ".")
    md += ["", "## Request budget (hard limits in `telegram_public.py`)", "",
           f"- One collector process, never parallel; {MIN_DELAY:g}-{MAX_DELAY:g} s random delay between requests, at most "
           f"{MAX_REQUESTS_PER_HOUR} requests per hour (sliding window, retries included), {CHANNEL_PAUSE} s pause between "
           f"channels, `Retry-After` honoured with growing backoff, descriptive User-Agent, everything stops at the "
           f"{MAX_429}rd HTTP 429.",
           f"- Effective rate: min({MAX_REQUESTS_PER_HOUR}/h cap, 3600 / mean delay = "
           f"{3600 / ((MIN_DELAY + MAX_DELAY) / 2):.0f}/h) = **{plan['rate_per_hour']:.0f} requests/hour** "
           f"(one request per {3600 / plan['rate_per_hour']:.1f} s).", "",
           "## Estimate", "",
           f"Requests per window = ids in the window / {PAGE} (an upper bound: deleted posts leave id gaps) + {PLAN_PROBES} "
           f"probes (hard limit {PROBES_PER_WINDOW} per window).", "",
           "| channel | windows | est. ids | est. requests | est. hours |", "|---|---|---|---|---|"]
    for ch, e in c["per_channel"].items():
        md.append(f"| {ch} | {len(e['windows'])} | {e['ids']:.0f} | {e['requests']} | "
                  f"{e['requests'] / plan['rate_per_hour']:.2f} |")
    md += [f"| **total** | | | {c['requests']} | **{c['hours']:.2f}** (incl. channel pauses) |", "",
           f"With all {plan['n_strikes']} strikes the estimate would be {plan['full_cost']['requests']} requests "
           f"({plan['full_cost']['hours']:.2f} h).", "", "## Strikes", "",
           "| date | weapons_through | selected |", "|---|---|---|"]
    sel = set(plan["selected"])
    for d in sorted(plan["weapons"]):
        w = plan["weapons"][d]
        md.append(f"| {d} | {'' if w < 0 else int(w)} | {'yes' if d in sel else 'no'} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def load_windows_log(data_root: Path) -> pd.DataFrame:
    """Finished windows (`_windows.csv`): channel, window_start / window_end (UTC, end exclusive), strikes, counts."""
    p = raw_dir(data_root) / WINDOWS_FILE
    if not p.exists():
        return pd.DataFrame(columns=WINDOW_COLS)
    df = pd.read_csv(p)
    for c in ("window_start", "window_end"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def _log_window(data_root: Path, row: dict) -> None:
    p = raw_dir(data_root) / WINDOWS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    df = load_windows_log(data_root)
    new = pd.DataFrame([row], columns=WINDOW_COLS)
    for c in ("window_start", "window_end"):
        new[c] = pd.to_datetime(new[c], utc=True)
    df = pd.concat([d for d in (df, new) if len(d)], ignore_index=True)
    df = df.drop_duplicates(["channel", "window_start", "window_end"], keep="last").sort_values(["channel", "window_start"])
    tmp = p.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, p)


def collect_channel_windows(channel: str, dates: list[str], data_root: Path, window_days: int = 7, pre_days: int = 1,
                            fetcher: Fetcher | None = None) -> dict:
    """Window mode for one channel: the posts of the (merged) windows of the strike `dates`, with ``window_strike`` set,
    merged into the channel's Parquet (existing posts kept).  Windows already listed in ``_windows.csv`` are skipped
    (resume).  Saved after every window and every CHECKPOINT_PAGES requests."""
    fetcher = _Counted(fetcher or Fetcher())
    path = store_path(data_root, channel)
    store = load_store(path)
    clock = IdClock()
    clock.add_frame(store)
    merged = merge_windows(strike_windows(dates, window_days, pre_days))
    log_df = load_windows_log(data_root)
    done = {(r.channel, r.window_start, r.window_end) for r in log_df.itertuples()}
    stats = {"channel": channel, "windows": 0, "skipped": 0, "posts": 0, "requests": 0, "probes": 0}

    def save(posts: list[dict]) -> None:
        nonlocal store
        if posts:
            store = merge_store(store, to_frame(posts))
            save_store(store, path)

    for i, w in enumerate(merged):
        if (channel, w["start"], w["end"]) in done:
            stats["skipped"] += 1
            continue
        n0 = fetcher.n
        saved = [0]

        def tagged(buf, w=w):
            return [dict(p, window_strike=tag_window_strike(p["datetime"], w["strikes"])) for p in buf]

        def ckpt(buf, tagged=tagged, saved=saved):
            save(tagged(buf))
            saved[0] += len(buf)

        after, pre, probes = locate_start(fetcher, channel, clock, w["start"])
        posts, n_pages, at_end = read_window(fetcher, channel, after, pre, w["start"], w["end"], clock, ckpt)
        save(tagged(posts[saved[0]:]))
        ids = [p["post_id"] for p in posts]
        _log_window(data_root, {"channel": channel, "window_start": w["start"], "window_end": w["end"],
                                "strikes": "|".join(s["strike"] for s in w["strikes"]),
                                "first_id": min(ids) if ids else None, "last_id": max(ids) if ids else None,
                                "n_posts": len(posts), "requests": fetcher.n - n0, "probes": probes,
                                "end_of_channel": at_end, "collected_at": pd.Timestamp.now("UTC").isoformat()})
        stats["windows"] += 1
        stats["posts"] += len(posts)
        stats["probes"] += probes
        log.info("%s: window %d/%d %s .. %s (%d strikes): %d posts, %d probes + %d pages; %d requests this channel",
                 channel, i + 1, len(merged), w["start"].date(), w["end"].date(), len(w["strikes"]), len(posts), probes,
                 n_pages, fetcher.n)
    stats["requests"] = fetcher.n
    return stats


def append_actuals(plan_path: Path, stats: list[dict], n_requests: int, hours: float, n429: int, code: int) -> None:
    """Append the measured outcome of a window run to the plan report."""
    md = ["", "## Actual run", "",
          f"Finished {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC ({'stopped by repeated HTTP 429' if code else 'completed'}): "
          f"{n_requests} requests in {hours:.2f} h = **{n_requests / max(hours, 1e-9):.0f} requests/hour** effective, "
          f"{n429} x HTTP 429.", "",
          "| channel | windows | skipped (already done) | posts in windows | requests (incl. probes) | probes |",
          "|---|---|---|---|---|---|"]
    md += [f"| {s['channel']} | {s['windows']} | {s['skipped']} | {s['posts']} | {s['requests']} | {s['probes']} |" for s in stats]
    with open(plan_path, "a", encoding="utf-8") as f:
        f.write(chr(10).join(md) + chr(10))


# --------------------------------------------------------------------------------------------- quality
def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def quality_report(data_root: Path, reports_dir: Path, channels: list[str] | None = None,
                   gap_days: int = 7) -> pd.DataFrame:
    """Write data/raw/telegram/_manifest.csv and reports/telegram_quality.md from the stored Parquet files."""
    d = raw_dir(data_root)
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    rows, monthly, gaps_all = [], {}, []
    for f in files:
        if channels and f.stem not in channels:
            continue
        df = pd.read_parquet(f)
        if df.empty:
            continue
        dup = int(df.duplicated(["channel", "post_id"]).sum())
        df = df.sort_values("datetime")
        gap = df["datetime"].diff()
        big = df.loc[gap > pd.Timedelta(days=gap_days)]
        for i in big.index:
            prev = df.loc[:i].iloc[-2]["datetime"]
            gaps_all.append((f.stem, prev, df.loc[i, "datetime"], (df.loc[i, "datetime"] - prev).days))
        has_text = df["text"].fillna("").str.strip().ne("")
        rows.append({
            "channel": f.stem, "posts": len(df), "first_date": df["datetime"].min().date(),
            "last_date": df["datetime"].max().date(), "first_id": int(df["post_id"].min()),
            "last_id": int(df["post_id"].max()),
            "gaps_gt_%dd" % gap_days: len(big), "max_gap_days": int(gap.max().days) if len(df) > 1 else 0,
            "duplicates": dup, "share_with_text": round(float(has_text.mean()), 4),
            "share_with_media": round(float(df["has_media"].mean()), 4),
            "median_views": float(df["views"].dropna().median()) if df["views"].notna().any() else None,
        })
        monthly[f.stem] = df.groupby(df["datetime"].dt.strftime("%Y-%m")).size()
    man = pd.DataFrame(rows)
    d.mkdir(parents=True, exist_ok=True)
    man.to_csv(d / "_manifest.csv", index=False)
    reports_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Telegram collection quality", "",
          f"Generated {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M} UTC from `data/raw/telegram/*.parquet`. "
          "Source: public `t.me/s/<channel>` previews (no login). Posts deleted by the channel before "
          "collection, and any posts the preview hides, cannot be recovered; id gaps are normal.", "",
          "## Summary", "", _md_table(man) if len(man) else "_no data_", ""]
    md += ["## Gaps longer than %d days" % gap_days, ""]
    if gaps_all:
        g = pd.DataFrame(gaps_all, columns=["channel", "from", "to", "days"])
        g["from"], g["to"] = g["from"].dt.strftime("%Y-%m-%d"), g["to"].dt.strftime("%Y-%m-%d")
        md += [_md_table(g), ""]
    else:
        md += ["None.", ""]
    md += ["## Posts per month", ""]
    if monthly:
        mt = pd.DataFrame(monthly).fillna(0).astype(int).sort_index()
        mt.index.name = "month"
        md += [_md_table(mt.reset_index()), ""]
    (reports_dir / "telegram_quality.md").write_text("\n".join(md), encoding="utf-8")
    log.info("wrote %s and %s", d / "_manifest.csv", reports_dir / "telegram_quality.md")
    return man
