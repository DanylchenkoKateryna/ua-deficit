"""Telegram collector: parsing, storage and pagination logic. No network (fixture HTML + fake fetcher)."""
from pathlib import Path

import pandas as pd
import pytest

from deficit.ingestion import telegram_public as tp

FIX = Path(__file__).parent / "fixtures" / "telegram_page.html"


@pytest.fixture(scope="module")
def posts():
    return tp.parse_page(FIX.read_text(encoding="utf-8"), "testchan")


def test_parse_ids_and_urls(posts):
    assert [p["post_id"] for p in posts] == [101, 102, 104, 105]
    assert posts[0]["url"] == "https://t.me/testchan/101"
    assert all(p["channel"] == "testchan" for p in posts)


def test_parse_text_keeps_line_breaks(posts):
    t = posts[0]["text"]
    assert t.startswith("УВАГА!\n\nВнаслідок атаки без світла 1 200 абонентів & ще\nдругий рядок")
    assert "<" not in t


def test_parse_datetime_utc(posts):
    assert posts[0]["datetime"] == pd.Timestamp("2026-10-13T16:27:08", tz="UTC")
    assert posts[2]["datetime"] == pd.Timestamp("2026-10-13T23:30:00", tz="UTC")   # +03:00 converted


def test_parse_views(posts):
    assert [p["views"] for p in posts] == [42200, 987, 1_300_000, None]
    assert tp.parse_views("1,5K") == 1500 and tp.parse_views(None) is None


def test_parse_media_forward_reply(posts):
    assert [p["has_media"] for p in posts] == [False, True, False, True]
    assert posts[1]["forwarded_from"] == "Рівненська АЕС" and posts[0]["forwarded_from"] is None
    assert posts[2]["reply_to"] == 101 and posts[0]["reply_to"] is None
    assert posts[2]["text"] == "Відповідь: світло повернули"     # not the quoted reply preview
    assert posts[3]["text"] == ""                                  # media-only post


def test_merge_dedupes_and_keeps_newer(posts):
    a = tp.to_frame(posts)
    b = tp.to_frame([dict(posts[0], views=99999)])
    m = tp.merge_store(a, b)
    assert len(m) == 4 and m.loc[m.post_id == 101, "views"].iloc[0] == 99999
    assert m["post_id"].is_monotonic_increasing


class FakeFetcher:
    """Serves a synthetic channel with ids 1..N, one post per hour, 20 per page like t.me/s."""

    def __init__(self, n=100):
        self.n, self.calls = n, []

    def get_page(self, channel, before=None):
        self.calls.append(before)
        hi = self.n if before is None else before - 1
        out = ""
        for i in range(max(1, hi - 19), hi + 1):
            ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i)
            out += ('<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" '
                    f'data-post="{channel}/{i}"><div class="tgme_widget_message_text js-message_text">p{i}</div>'
                    f'<a class="tgme_widget_message_date"><time datetime="{ts.isoformat()}"></time></a></div></div>')
        return out


def test_backfill_then_update(tmp_path):
    f = FakeFetcher(100)
    df = tp.collect_channel("c", tmp_path, since="2026-01-01", fetcher=f)
    assert len(df) == 100 and df["post_id"].is_unique and df["post_id"].min() == 1
    assert tp.store_path(tmp_path, "c").exists()
    f.n = 107                                   # 7 new posts; update must not walk back through old ones
    f.calls.clear()
    df = tp.collect_channel("c", tmp_path, since="2026-01-01", update=True, fetcher=f)
    assert len(df) == 107 and f.calls == [None]


def test_since_stops_pagination(tmp_path):
    f = FakeFetcher(1000)
    since = "2026-01-02"
    df = tp.collect_channel("c", tmp_path, since=since, fetcher=f)
    assert df["datetime"].min() >= pd.Timestamp(since, tz="UTC")
    assert len(f.calls) < 60


def test_resume_extends_backwards(tmp_path):
    f = FakeFetcher(100)
    tp.collect_channel("c", tmp_path, since="2026-01-04", fetcher=f)       # ids >= 72
    df = tp.collect_channel("c", tmp_path, since="2026-01-01", fetcher=f)
    assert df["post_id"].min() == 1 and len(df) == 100


def test_quality_report(tmp_path):
    tp.collect_channel("c", tmp_path, since="2026-01-01", fetcher=FakeFetcher(50))
    man = tp.quality_report(tmp_path, tmp_path / "reports")
    assert man.loc[0, "posts"] == 50 and man.loc[0, "duplicates"] == 0 and man.loc[0, "share_with_text"] == 1.0
    assert (tmp_path / "raw" / "telegram" / "_manifest.csv").exists()
    assert "Posts per month" in (tmp_path / "reports" / "telegram_quality.md").read_text(encoding="utf-8")


def test_fetcher_rate_cap_and_429_slowdown():
    slept = []

    class R:
        def __init__(self, code): self.status_code, self.headers, self.text = code, {"Retry-After": "7"}, "ok"

    class S:
        headers = {}
        def __init__(self): self.codes = [429, 200, 200, 200]
        def get(self, *a, **k): return R(self.codes.pop(0))

    f = tp.Fetcher(session=S(), sleep=slept.append, max_per_hour=2)
    assert f.get_page("c") == "ok" and f._slow > 1            # 429 -> waited, slowed down, then ok
    assert any(x >= 60 for x in slept)
    f.get_page("c"); f.get_page("c")
    assert any(x > 3000 for x in slept)                       # hourly cap forced a long pause
    assert tp.MIN_DELAY == 3 and tp.MAX_DELAY == 5 and tp.MAX_REQUESTS_PER_HOUR == 800 and tp.CHANNEL_PAUSE == 60


# ------------------------------------------------------------------------------------------ query mode
SEARCH_FIX = Path(__file__).parent / "fixtures" / "telegram_search_page.html"


def test_parse_search_page_fixture():
    ps = tp.parse_page(SEARCH_FIX.read_text(encoding="utf-8"), "kyivoda")
    assert [p["post_id"] for p in ps] == [49797, 51234]
    assert ps[1]["datetime"] == pd.Timestamp("2026-06-30T08:04:22", tz="UTC")
    assert "світло повернули" in ps[1]["text"]


def test_fetcher_sends_query_and_before():
    seen = []

    class R:
        status_code, headers, text = 200, {}, "ok"

    class S:
        headers = {}
        def get(self, url, params=None, **k):
            seen.append((url, params))
            return R()

    f = tp.Fetcher(session=S(), sleep=lambda s: None)
    f.get_page("c", query="світло")
    f.get_page("c", before=51282, query="світло")
    f.get_page("c")
    assert seen == [("https://t.me/s/c", {"q": "світло"}),
                    ("https://t.me/s/c", {"q": "світло", "before": 51282}),
                    ("https://t.me/s/c", None)]


class FakeSearchFetcher:
    """Hits (ids, one per 3 h) per term, 20 per page newest first like t.me/s/<ch>?q=; `before` = lowest id of the previous page."""

    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def get_page(self, channel, before=None, query=None):
        self.calls.append((query, before))
        ids = sorted((i for i in self.hits.get(query, []) if before is None or i < before), reverse=True)[:20]
        out = ""
        for i in sorted(ids):
            ts = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=3 * i)
            out += ('<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" '
                    f'data-post="{channel}/{i}"><div class="tgme_widget_message_text js-message_text">p{i}</div>'
                    f'<a class="tgme_widget_message_date"><time datetime="{ts.isoformat()}"></time></a></div></div>')
        return out


def test_query_collect_paginates_dedupes_and_marks_terms(tmp_path):
    # pre-existing partial file (normal mode) must be kept; post 30 is also found by both terms
    tp.collect_channel("c", tmp_path, since="2026-01-01", fetcher=FakeFetcher(40), update=False)
    n_old = len(tp.load_store(tp.store_path(tmp_path, "c")))
    f = FakeSearchFetcher({"світло": list(range(100, 150, 2)) + [30], "графік": [30, 120, 200]})
    df = tp.collect_channel_query("c", ["світло", "графік"], tmp_path, since="2026-01-01", fetcher=f)
    assert df["post_id"].is_unique and len(df) == n_old + 25 + 1          # 25 new (incl. 120) + 200; post 30 already stored
    mq = df.set_index("post_id")["matched_query"]
    assert mq[30] == "графік|світло" and mq[120] == "графік|світло" and mq[102] == "світло" and mq[200] == "графік"
    assert pd.isna(mq[5])                                                  # untouched old post keeps no term
    # 26 hits -> pages of 20 and 6, then an empty page ends the walk (a short page is not trusted as the last)
    assert [c for c in f.calls if c[0] == "світло"] == [("світло", None), ("світло", 110), ("світло", 30)]
    assert len(f.calls) == 5


def test_query_collect_stops_at_since(tmp_path):
    hits = {"t": list(range(1, 400))}                     # 3 h apart: id 8 = 2026-01-02 00:00
    f = FakeSearchFetcher(hits)
    df = tp.collect_channel_query("c", ["t"], tmp_path, since="2026-01-02", fetcher=f)
    assert df["datetime"].min() >= pd.Timestamp("2026-01-02", tz="UTC") and len(f.calls) < 25


def test_merge_store_old_file_without_matched_query(tmp_path, posts):
    old = tp.to_frame(posts).drop(columns="matched_query")
    path = tmp_path / "x.parquet"
    old.to_parquet(path)
    loaded = tp.load_store(path)
    assert "matched_query" in loaded and loaded["matched_query"].isna().all()
    m = tp.merge_store(loaded, tp.to_frame([dict(posts[0], matched_query="світло")]))
    assert len(m) == 4 and m.set_index("post_id").loc[101, "matched_query"] == "світло"


# ------------------------------------------------------------------------------------------ window mode
AFTER_FIX = Path(__file__).parent / "fixtures" / "telegram_after_page.html"
T0 = pd.Timestamp("2026-01-01", tz="UTC")


def _post_html(channel, i, ts):
    return ('<div class="tgme_widget_message_wrap js-widget_message_wrap"><div class="tgme_widget_message" '
            f'data-post="{channel}/{i}"><div class="tgme_widget_message_text js-message_text">p{i}</div>'
            f'<a class="tgme_widget_message_date"><time datetime="{ts.isoformat()}"></time></a></div></div>')


class FakeWindowFetcher:
    """Channel with ids 1..n whose posting rate changes (2 ids/h for the first 30 days, 6 ids/h afterwards, an id gap every
    7th id), served like t.me/s: `after=g` -> the next 20 existing ids ascending, `before=b` -> the 20 before, none -> newest."""

    def __init__(self, n=4000, gap=7):
        self.n, self.gap = n, gap
        self.ids = [i for i in range(1, n + 1) if i % gap]
        self.calls = []

    @staticmethod
    def time_of(i):
        return T0 + pd.Timedelta(hours=i / 2.0 if i <= 1440 else 720 + (i - 1440) / 6.0)

    def get_page(self, channel, before=None, query=None, after=None):
        self.calls.append(("after" if after is not None else "before" if before else "newest", after or before))
        if after is not None:
            sel = [i for i in self.ids if i > after][:20]
        elif before:
            sel = [i for i in self.ids if i < before][-20:]
        else:
            sel = self.ids[-20:]
        return "".join(_post_html(channel, i, self.time_of(i)) for i in sel)


def test_parse_after_page_fixture():
    ps = tp.parse_page(AFTER_FIX.read_text(encoding="utf-8"), "kyivoda")
    assert [p["post_id"] for p in ps] == [50001, 50002, 50005]            # ascending, id gaps allowed
    assert ps[2]["datetime"] == pd.Timestamp("2026-06-08T07:36:30", tz="UTC") and ps[0]["views"] == 1200


def test_fetcher_sends_after_and_third_429_stops_everything():
    seen, slept = [], []

    class R:
        def __init__(self, code): self.status_code, self.headers, self.text = code, {}, "ok"

    class S:
        headers = {}
        def __init__(self): self.codes = [200, 429, 429, 200]
        def get(self, url, params=None, **k):
            seen.append(params)
            return R(self.codes.pop(0))

    f = tp.Fetcher(session=S(), sleep=slept.append)
    f.get_page("c", after=50000)
    assert seen[0] == {"after": 50000}
    assert f.get_page("c") == "ok"                    # two 429 answers, backoff 60 s then 120 s, then ok
    assert f.n429 == 2 and [x for x in slept if x >= 60] == [60, 120]
    f.s.codes = [429]
    with pytest.raises(tp.TooManyRequests):           # the third 429 stops everything, without waiting
        f.get_page("c")


def test_strike_windows_and_merge():
    w = tp.strike_windows(["2026-03-10", "2026-03-12", "2026-04-20"], window_days=7, pre_days=1)
    assert w[0]["start"] == pd.Timestamp("2026-03-09", tz="UTC") and w[0]["end"] == pd.Timestamp("2026-03-18", tz="UTC")
    m = tp.merge_windows(w)
    assert len(m) == 2 and [s["strike"] for s in m[0]["strikes"]] == ["2026-03-10", "2026-03-12"]
    assert m[0]["end"] == pd.Timestamp("2026-03-20", tz="UTC")                       # 03-12 + 7 d, end day included
    assert len(tp.merge_windows(tp.strike_windows(["2026-03-10", "2026-03-19"]))) == 1   # 03-19 window starts at the end of the first
    assert tp.tag_window_strike(pd.Timestamp("2026-03-13T12:00", tz="UTC"), m[0]["strikes"]) == "2026-03-10|2026-03-12"
    assert tp.tag_window_strike(pd.Timestamp("2026-03-19T12:00", tz="UTC"), m[0]["strikes"]) == "2026-03-12"
    assert tp.tag_window_strike(pd.Timestamp("2026-03-25", tz="UTC"), m[0]["strikes"]) is None


def test_idclock_interpolates_and_brackets():
    c = tp.IdClock()
    c.add([{"post_id": 100, "datetime": T0}, {"post_id": 300, "datetime": T0 + pd.Timedelta(days=2)}])
    assert c.estimate(T0 + pd.Timedelta(days=1)) == pytest.approx(200)
    assert c.estimate(T0 + pd.Timedelta(days=3)) == pytest.approx(400)               # extrapolates with the last slope
    lo, hi = c.bracket(T0 + pd.Timedelta(days=1))
    assert (lo[0], hi[0]) == (100, 300)
    assert c.bracket(T0 - pd.Timedelta(days=1)) == (None, (100, T0.timestamp()))


def test_locate_and_read_window_few_probes():
    f = FakeWindowFetcher()
    clock = tp.IdClock()
    clock.add([{"post_id": 1, "datetime": f.time_of(1)}, {"post_id": 4000, "datetime": f.time_of(4000)}])   # only the ends known
    start, end = T0 + pd.Timedelta(days=40), T0 + pd.Timedelta(days=43)
    after, pre, probes = tp.locate_start(f, "c", clock, start)
    assert probes <= tp.PROBES_PER_WINDOW and len(f.calls) == probes
    posts, n_req, at_end = tp.read_window(f, "c", after, pre, start, end, clock)
    truth = [i for i in f.ids if start <= f.time_of(i) < end]
    assert [p["post_id"] for p in posts] == truth and not at_end
    assert n_req <= len(truth) // 20 + 3                       # pages of the window plus at most the boundary pages
    # a second window nearby needs no more probes: the first one added points to the clock
    _, _, pr2 = tp.locate_start(f, "c", clock, T0 + pd.Timedelta(days=46))
    assert pr2 <= probes


def test_locate_far_from_known_points_is_bounded():
    f = FakeWindowFetcher()
    clock = tp.IdClock()
    # only two nearby, late points are known (like a keyword sample that starts months after the window): the estimate
    # extrapolates backwards with a wrong slope; probes + bisection must still land close, never read from id 0 for long
    clock.add([{"post_id": 3000, "datetime": f.time_of(3000)}, {"post_id": 3100, "datetime": f.time_of(3100)}])
    start, end = T0 + pd.Timedelta(days=12), T0 + pd.Timedelta(days=14)
    after, pre, probes = tp.locate_start(f, "c", clock, start)
    posts, n_req, _ = tp.read_window(f, "c", after, pre, start, end, clock)
    truth = [i for i in f.ids if start <= f.time_of(i) < end]
    assert [p["post_id"] for p in posts] == truth
    assert probes <= tp.PROBES_PER_WINDOW and probes + n_req <= len(truth) // 20 + 2 * tp.PROBES_PER_WINDOW


def test_locate_before_channel_start_and_after_last_post():
    f = FakeWindowFetcher(n=300)
    clock = tp.IdClock()
    clock.add([{"post_id": 50, "datetime": f.time_of(50)}, {"post_id": 250, "datetime": f.time_of(250)}])
    s0 = T0 - pd.Timedelta(days=5)                                                    # before the first post
    after, pre, _ = tp.locate_start(f, "c", clock, s0)
    posts, _, _ = tp.read_window(f, "c", after, pre, s0, T0 + pd.Timedelta(days=1), clock)
    assert [p["post_id"] for p in posts] == [i for i in f.ids if f.time_of(i) < T0 + pd.Timedelta(days=1)]
    late = f.time_of(300) + pd.Timedelta(days=1)                                      # after the newest post
    after, pre, _ = tp.locate_start(f, "c", clock, late)
    posts, _, at_end = tp.read_window(f, "c", after, pre, late, late + pd.Timedelta(days=3), clock)
    assert posts == [] and at_end


def test_collect_windows_end_to_end(tmp_path):
    f = FakeWindowFetcher()
    # stored keyword-search sample: sparse posts (kept; the ones inside a window get tagged when scraped again)
    sample = [{"channel": "c", "post_id": i, "datetime": f.time_of(i), "text": "old", "views": 1, "has_media": False,
               "forwarded_from": None, "reply_to": None, "url": f"https://t.me/c/{i}", "matched_query": "світло"}
              for i in (10, 500, 2000, 3500, 3999)]
    tp.save_store(tp.to_frame(sample), tp.store_path(tmp_path, "c"))
    d1, d2, d3 = [(T0 + pd.Timedelta(days=x)).strftime("%Y-%m-%d") for x in (20, 22, 40)]
    st = tp.collect_channel_windows("c", [d1, d2, d3], tmp_path, window_days=3, pre_days=1, fetcher=f)
    assert st["windows"] == 2                                   # d1 and d2 are chained into one window
    df = tp.load_store(tp.store_path(tmp_path, "c"))
    assert df["post_id"].is_unique and {10, 500, 2000, 3500, 3999} <= set(df["post_id"])
    w = df[df["window_strike"].notna()]
    ts = lambda d, k: pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=k)
    assert w["datetime"].min() >= ts(d1, -1) and w["datetime"].max() < ts(d3, 4)
    both = w[(w["datetime"] >= ts(d2, -1)) & (w["datetime"] < ts(d1, 4))]
    assert len(both) and (both["window_strike"] == f"{d1}|{d2}").all()
    truth = {i for i in f.ids if ts(d1, -1) <= f.time_of(i) < ts(d2, 4)} | {i for i in f.ids if ts(d3, -1) <= f.time_of(i) < ts(d3, 4)}
    assert truth == set(w["post_id"])                           # exactly the posts of the windows, no more, no fewer
    log = tp.load_windows_log(tmp_path)
    assert len(log) == 2 and log["n_posts"].sum() == len(truth)
    # resume: finished windows are skipped, no request is made
    f.calls.clear()
    st2 = tp.collect_channel_windows("c", [d1, d2, d3], tmp_path, window_days=3, pre_days=1, fetcher=f)
    assert st2["skipped"] == 2 and st2["windows"] == 0 and f.calls == []


def test_select_strikes_budget_rule_ignores_outcomes():
    clk = tp.IdClock()
    clk.add([{"post_id": 1, "datetime": T0}, {"post_id": 100_000, "datetime": T0 + pd.Timedelta(days=500)}])   # 200 ids/day
    dates = [(T0 + pd.Timedelta(days=20 * k)).strftime("%Y-%m-%d") for k in range(1, 21)]
    att = pd.DataFrame({"date": dates, "weapons_through": [(k * 37) % 91 for k in range(20)]})
    ok = tp.select_strikes(att, {"c": clk}, since=dates[0], max_hours=100)
    assert ok["selected"] == dates and ok["rule"] == "all strikes"
    tight = tp.select_strikes(att, {"c": clk}, since=dates[0], max_hours=1.0)
    k = len(tight["selected"])
    assert 0 < k < 20 and tight["cost"]["hours"] <= 1.0
    ranked = att.sort_values(["weapons_through", "date"], ascending=[False, True])["date"].tolist()
    assert tight["selected"] == sorted(ranked[:k])               # top-k by weapons_through, nothing else
    capped = tp.select_strikes(att, {"c": clk}, since=dates[0], max_hours=100, share=0.25)
    assert len(capped["selected"]) == 5
    until = tp.select_strikes(att, {"c": clk}, since=dates[0], max_hours=100, until={"c": dates[4]})
    assert len(until["cost"]["per_channel"]["c"]["windows"]) <= 5


def test_merge_store_unions_window_strike():
    row = {"channel": "c", "post_id": 1, "datetime": T0, "text": "x", "has_media": False, "forwarded_from": None,
           "reply_to": None, "url": "u"}
    a = tp.to_frame([dict(row, views=1, window_strike="2026-01-01")])
    b = tp.to_frame([dict(row, views=2, window_strike="2026-01-02", matched_query="q")])
    m = tp.merge_store(a, b)
    assert len(m) == 1 and m.loc[0, "window_strike"] == "2026-01-01|2026-01-02" and m.loc[0, "matched_query"] == "q"


def _sample_store(tmp_path, f):
    s = [{"channel": "c", "post_id": i, "datetime": f.time_of(i), "text": "old", "views": 1, "has_media": False,
          "forwarded_from": None, "reply_to": None, "url": f"https://t.me/c/{i}", "matched_query": "q"} for i in (10, 500, 2000, 3500, 3999)]
    tp.save_store(tp.to_frame(s), tp.store_path(tmp_path, "c"))


def test_uncovered_intervals():
    t = lambda d: T0 + pd.Timedelta(days=d)
    assert tp.uncovered(t(0), t(10), []) == [(t(0), t(10))]
    assert tp.uncovered(t(0), t(10), [(t(-2), t(3)), (t(2), t(5)), (t(7), t(8))]) == [(t(5), t(7)), (t(8), t(10))]
    assert tp.uncovered(t(0), t(10), [(t(-1), t(10))]) == []


def test_gap_fill_skips_covered_and_fetches_only_the_rest(tmp_path):
    f = FakeWindowFetcher()
    _sample_store(tmp_path, f)
    d = (T0 + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    ts = lambda k: pd.Timestamp(d, tz="UTC") + pd.Timedelta(days=k)
    tp.collect_channel_windows("c", [d], tmp_path, window_days=3, pre_days=1, fetcher=f)       # [d-1, d+4)
    # fully covered (narrower) window: skipped, no request
    f.calls.clear()
    st = tp.collect_channel_windows("c", [d], tmp_path, window_days=2, pre_days=1, fetcher=f)
    assert st["skipped"] == 1 and st["windows"] == 0 and f.calls == []
    # wider window: only [d+4, d+7) is requested, the full window is logged
    st = tp.collect_channel_windows("c", [d], tmp_path, window_days=6, pre_days=1, fetcher=f)
    assert st["windows"] == 1 and st["skipped"] == 0
    fresh = FakeWindowFetcher()
    other = tmp_path / "fresh"
    _sample_store(other, fresh)
    st_full = tp.collect_channel_windows("c", [d], other, window_days=6, pre_days=1, fetcher=fresh)
    assert 0 < st["requests"] < st_full["requests"]
    new_calls = [c for c in f.calls if c[0] == "after"]
    assert all(c[1] >= min(i for i in f.ids if f.time_of(i) >= ts(4)) - 2 * tp.PAGE for c in new_calls)    # nothing read from the covered part
    df = tp.load_store(tp.store_path(tmp_path, "c"))
    w = df[df["window_strike"].notna()]
    truth = {i for i in f.ids if ts(-1) <= f.time_of(i) < ts(7)}
    assert set(w["post_id"]) == truth and set(w["post_id"]) == set(tp.load_store(tp.store_path(other, "c")).query("window_strike == window_strike")["post_id"])
    log = tp.load_windows_log(tmp_path)
    assert list(log["window_start"]) == [ts(-1)] * 2 and sorted(log["window_end"]) == [ts(4), ts(7)]
    # the plan counts only the uncovered part (here: nothing)
    clk = tp.IdClock()
    clk.add_frame(tp.load_store(tp.store_path(tmp_path, "c")))
    m = tp.merge_windows(tp.strike_windows([d], 6, 1))
    cov = tp.covered_intervals(log, "c")
    e = tp.estimate_windows(clk, m, covered=cov)
    assert e["requests"] == 0 and e["todo"] == 0 and e["requests_full"] > 0
    assert tp.estimate_windows(clk, m)["requests"] == e["requests_full"]
