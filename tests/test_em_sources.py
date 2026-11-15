"""Energy Map source collector: URL cleaning, planning, robots, resume, blocking. No network (fake fetcher)."""
import gzip

import pandas as pd
import requests

from deficit.ingestion import em_sources as em


def test_clean_links():
    cell = 'https://a.ua/x, "https://b.ua/y?z=1" https://c.ua/z.'
    assert em.clean_links(cell) == ["https://a.ua/x", "https://b.ua/y?z=1", "https://c.ua/z"]
    assert em.clean_links(float("nan")) == []


def _setup(tmp_path):
    emd, tgd, out = tmp_path / "em", tmp_path / "tg", tmp_path / "out"
    emd.mkdir()
    tgd.mkdir()
    rows = ["https://ok.ua/1", "https://ok.ua/1", "https://www.facebook.com/p/1", "https://t.me/Chan/5",
            "https://t.me/Chan/6", "https://nope.ua/a", "https://ok.ua/2"]
    pd.DataFrame({"link": rows}).to_csv(emd / "x_oblasti_osr.csv", index=False)
    pd.DataFrame({"link": ["https://ok.ua/3"]}).to_csv(emd / "x_oblasti_minenergo.csv", index=False)
    pd.DataFrame({"post_id": [5]}).to_parquet(tgd / "chan.parquet")
    return emd, tgd, out


class Fake:
    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, url):
        self.calls.append(url)
        if url.endswith("/robots.txt"):
            return (200, b"User-agent: *\nDisallow: /a\n") if "nope.ua" in url else (404, b"")
        a = self.answers.get(url, (200, b"<html>" + url.encode() + b"</html>"))
        if isinstance(a, Exception):
            raise a
        return a


def test_plan(tmp_path):
    emd, tgd, out = _setup(tmp_path)
    rep = em.plan_report(emd, tgd, out)
    assert rep["unique_urls"] == 7
    assert rep["per_domain"] == {"ok.ua": 3, "t.me": 1, "nope.ua": 1}
    assert rep["skipped"][("login_wall", "www.facebook.com")] == 1
    assert rep["skipped"][("in_telegram_parquet", "t.me")] == 1


def test_collect_and_resume(tmp_path):
    emd, tgd, out = _setup(tmp_path)
    f, sleeps = Fake(), []
    st = em.collect(emd, tgd, out, fetcher=f, sleep=sleeps.append)
    assert "https://t.me/s/Chan/6" in f.calls and "https://t.me/s/Chan/5" not in f.calls
    assert not any("facebook" in c for c in f.calls) and "https://nope.ua/a" not in f.calls
    assert st[("skipped:robots_disallow", "nope.ua")] == 1
    assert sum(v for (k, _), v in st.items() if k == "ok") == 4
    assert all(3.0 <= s <= 5.0 for s in sleeps)
    man = pd.read_csv(out / "_manifest.csv", dtype=str, keep_default_na=False)
    assert list(man.columns) == em.MANIFEST_COLS
    ok = man[man.status == "200"].iloc[0]
    assert b"<html>" in gzip.decompress((out / "html" / f"{ok.sha1}.html.gz").read_bytes())
    f2 = Fake()
    em.collect(emd, tgd, out, fetcher=f2, sleep=lambda s: None)
    assert f2.calls == []


def test_retries_and_errors(tmp_path):
    emd, tgd, out = _setup(tmp_path)
    ans = {"https://ok.ua/1": (503, b""), "https://ok.ua/2": (403, b""), "https://ok.ua/3": requests.Timeout()}
    f = Fake(ans)
    em.collect(emd, tgd, out, fetcher=f, sleep=lambda s: None)
    assert f.calls.count("https://ok.ua/1") == 1 + em.MAX_RETRIES
    man = pd.read_csv(out / "_manifest.csv", dtype=str, keep_default_na=False).set_index("url")
    assert man.loc["https://ok.ua/2", "status"] == "403"
    assert man.loc["https://ok.ua/3", "reason"] == "Timeout"


def test_three_blocked_drops_domain(tmp_path):
    emd, tgd, out = tmp_path / "em", tmp_path / "tg", tmp_path / "out"
    emd.mkdir()
    tgd.mkdir()
    urls = [f"https://b.ua/{i}" for i in range(6)]
    pd.DataFrame({"link": urls}).to_csv(emd / "x_oblasti_osr.csv", index=False)
    f = Fake({u: (429, b"") for u in urls})
    em.collect(emd, tgd, out, fetcher=f, sleep=lambda s: None)
    assert len([c for c in f.calls if "/robots" not in c]) == 3
    man = pd.read_csv(out / "_manifest.csv", dtype=str, keep_default_na=False)
    assert (man.reason == "domain_blocked").sum() == 3


def test_bot_challenge_not_stored_and_blocks(tmp_path):
    emd, tgd, out = tmp_path / "em", tmp_path / "tg", tmp_path / "out"
    emd.mkdir()
    tgd.mkdir()
    urls = [f"https://c.ua/{i}" for i in range(5)]
    pd.DataFrame({"link": urls}).to_csv(emd / "x_oblasti_osr.csv", index=False)
    body = b'<html><script src="/_Incapsula_Resource?x=1"></script></html>'
    f = Fake({u: (200, body) for u in urls})
    em.collect(emd, tgd, out, fetcher=f, sleep=lambda s: None)
    assert len([c for c in f.calls if "/robots" not in c]) == 3
    assert not list((out / "html").glob("*.gz"))
    man = pd.read_csv(out / "_manifest.csv", dtype=str, keep_default_na=False)
    assert (man.reason == "bot_challenge").sum() == 3
