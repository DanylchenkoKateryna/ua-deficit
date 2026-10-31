"""Telegram API collector with a fake client: no network, no secrets, read-only guarantees."""
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from deficit.ingestion import telegram_api as ta
from deficit.ingestion import telegram_public as tp

FORBIDDEN = ("join", "send_message", "send_file", "get_dialogs", "iter_dialogs", "ImportChatInvite",
             "JoinChannel", "forward_messages", "delete_messages", "edit_message")


class FloodWait(Exception):
    def __init__(self, seconds):
        self.seconds = seconds


def msg(i, day, text="x", **kw):
    return SimpleNamespace(id=i, date=datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(days=day), message=text,
                           views=kw.get("views", 10), media=kw.get("media"), fwd_from=kw.get("fwd"),
                           reply_to=kw.get("reply"), action=kw.get("action"))


class FakeClient:
    """Channel with ids 1..N (day = id); records calls; raises on anything outside the read-only API."""

    def __init__(self, n=50, flood_at=None):
        self.n, self.flood_at, self.calls, self.waits = n, flood_at, [], []
        self.flooded = False

    def __getattr__(self, name):
        raise AssertionError(f"forbidden client call: {name}")

    def get_entity(self, ch):
        self.calls.append(("get_entity", ch))
        return ch

    def iter_messages(self, entity, limit=None, offset_id=0, wait_time=None, **kw):
        self.calls.append(("iter_messages", entity, offset_id, wait_time))
        assert wait_time is not None and wait_time >= 2
        start = self.n if not offset_id else offset_id - 1
        for i in range(start, 0, -1):
            if self.flood_at == i and not self.flooded:
                self.flooded = True
                raise FloodWait(7)
            yield msg(i, i)


def run(client, tmp_path, **kw):
    sleeps = []
    out = ta.collect_channel_api(client, "chan", tmp_path, sleep=sleeps.append, flood_exc=FloodWait, **kw)
    return out, sleeps


def test_schema_matches_public(tmp_path):
    out, _ = run(FakeClient(5), tmp_path, since="2000-01-01")
    assert list(out.columns) == tp.COLUMNS
    assert out["datetime"].dt.tz is not None and out["url"].iloc[0] == "https://t.me/chan/1"
    assert out.dtypes["post_id"] == "int64"


def test_since_and_resume_and_dedupe(tmp_path):
    out, _ = run(FakeClient(50), tmp_path, since="2023-01-31")       # day = id -> ids >= 30
    assert out["post_id"].min() == 30 and out["post_id"].max() == 50
    out, _ = run(FakeClient(60), tmp_path, since="2023-01-21")       # forward +10, backward to 20
    assert out["post_id"].min() == 20 and out["post_id"].max() == 60
    assert not out.duplicated(["channel", "post_id"]).any() and len(out) == 41
    out, _ = run(FakeClient(70), tmp_path, since="2023-01-01", update=True)   # update: forward only
    assert out["post_id"].min() == 20 and out["post_id"].max() == 70


def test_floodwait_sleeps_and_resumes(tmp_path):
    c = FakeClient(30, flood_at=15)
    out, sleeps = run(c, tmp_path, since="2000-01-01")
    assert sleeps == [7 + ta.FLOOD_MARGIN]
    assert sorted(out["post_id"]) == list(range(1, 31))


def test_service_messages_skipped():
    assert ta.message_to_post(msg(1, 1, action=object()), "c") is None
    p = ta.message_to_post(msg(2, 1, media=object(), reply=SimpleNamespace(reply_to_msg_id=1),
                               fwd=SimpleNamespace(from_name="Src")), "c")
    assert p["has_media"] and p["reply_to"] == 1 and p["forwarded_from"] == "Src"


def test_rejects_unlisted_channel(tmp_path):
    y = tmp_path / "s.yaml"
    y.write_text("telegram:\n  - {name: chan}\n", encoding="utf-8")
    with pytest.raises(ta.TelegramApiError):
        ta.run_collect_api(["my_private_chat"], tmp_path, tmp_path, y, "2022-10-01", False, client=FakeClient())
    ta.run_collect_api(["chan"], tmp_path, tmp_path, y, "2000-01-01", False, client=FakeClient(3), sleep=lambda s: None)
    assert (tmp_path / "raw" / "telegram" / "chan.parquet").exists()


def test_only_read_calls_used(tmp_path):
    c = FakeClient(5)
    run(c, tmp_path, since="2000-01-01")
    assert {x[0] for x in c.calls} <= {"get_entity", "iter_messages"}


def test_module_source_has_no_write_or_dialog_calls():
    src = Path(ta.__file__).read_text(encoding="utf-8")
    for name in FORBIDDEN:
        assert not re.search(r"\b%s\b" % name, src), name


def test_credentials_missing_and_not_leaked(monkeypatch):
    monkeypatch.delenv("TELEGRAM_API_ID", raising=False)
    monkeypatch.delenv("TELEGRAM_API_HASH", raising=False)
    with pytest.raises(ta.TelegramApiError):
        ta.credentials()
    monkeypatch.setenv("TELEGRAM_API_ID", "123")
    monkeypatch.setenv("TELEGRAM_API_HASH", "sekrit")
    assert ta.credentials() == (123, "sekrit")


def test_session_dir_under_data():
    assert ta.session_dir(Path("data")) == Path("data") / "telegram_session"
