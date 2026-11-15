"""Oblast-level mapping of Telegram posts and (strike, oblast) episodes: fixed example texts, no network."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deficit.ingestion import telegram_extract as te
from deficit.ingestion import telegram_regional as tr
from deficit.ingestion.telegram_extract import EpisodeParams

CH = {"sumyoblenergo": "Sumy"}
STRIKE = pd.DataFrame({"strike_at": [pd.Timestamp("2024-03-22 06:00", tz="UTC")], "major": [True], "description": ["x"]})
RAW = [
    ("ukrenergo", "2024-03-22 05:30", "Вночі ворог здійснив масовану атаку на енергооб'єкти. Пошкоджено підстанцію у Харківській "
                                      "області та ТЕС у Сумській області. Без світла залишаються 500 тисяч споживачів."),
    ("sumyoblenergo", "2024-03-22 06:30", "Через обстріл знеструмлено 12 тисяч споживачів."),
    ("ukrenergo", "2024-03-22 15:00", "Завтра, 23 березня, у Сумській області з 06:00 до 22:00 вимушено будуть застосовуватись "
                                      "графіки погодинних відключень."),
    ("sumyoblenergo", "2024-03-22 20:00", "Енергетики повернули світло всім споживачам."),
    ("ukrenergo", "2024-03-23 09:00", "Енергетики повернули світло для 58 тис. родин Харківщини. Без світла залишаються 20 тисяч "
                                      "споживачів Полтавщини."),
    ("ukrenergo", "2024-03-24 09:00", "На Харківщині внаслідок обстрілів знеструмлено 3 тисячі споживачів. Аварійні відключення "
                                      "діють у Полтавській області."),
    ("ukrenergo", "2024-03-27 15:00", "УВАГА! 28 БЕРЕЗНЯ ВІДКЛЮЧЕННЯ НЕ ПРОГНОЗУЮТЬСЯ"),
    ("sumyoblenergo", "2024-03-29 15:00", "Завтра графіки відключень не застосовуватимуться."),
]


def T(s):
    return pd.Timestamp(s, tz="UTC")


def _raw(rows) -> pd.DataFrame:
    return pd.DataFrame([{"channel": c, "post_id": i, "datetime": T(t), "text": x, "url": f"https://t.me/{c}/{i}"}
                         for i, (c, t, x) in enumerate(rows, 1)])


@pytest.fixture(scope="module")
def built():
    raw = _raw(RAW)
    posts = te.add_derived(te.extract_frame(raw))
    facts = tr.region_facts(posts, raw, CH)
    nat = posts[(posts.channel == "ukrenergo") & posts.schedules_lifted][["datetime", "channel"]]
    eps = tr.build_region_episodes(facts, STRIKE, CH, posts.groupby("channel").datetime.min().to_dict(), nat, EpisodeParams())
    return posts, facts, eps


def fact(facts, ch, pid, ob):
    r = facts[(facts.channel == ch) & (facts.post_id == pid) & (facts.oblast == ob)]
    assert len(r) == 1, f"expected one fact for {ch}/{pid}/{ob}, got {len(r)}"
    return r.iloc[0]


def test_own_channel_post_belongs_to_its_oblast(built):
    _, facts, _ = built
    f = fact(facts, "sumyoblenergo", 2, "Sumy")
    assert f["source"] == "own" and f["consumers_without_power"] == 12_000 and f["attack"]
    assert fact(facts, "sumyoblenergo", 4, "Sumy")["restoration_power"] == "full"
    assert fact(facts, "sumyoblenergo", 8, "Sumy")["lifted"]


def test_national_post_is_split_per_oblast_and_counts_need_a_single_oblast(built):
    _, facts, _ = built
    # one sentence names two oblasts: attack evidence for both, but the count cannot be attributed
    for ob in ("Kharkiv", "Sumy"):
        f = fact(facts, "ukrenergo", 1, ob)
        assert f["source"] == "text" and f["attack"] and f["multi_region"] and pd.isna(f["consumers_without_power"])
    # a digest line per oblast: restored for one, still without power for the other
    k, p = fact(facts, "ukrenergo", 5, "Kharkiv"), fact(facts, "ukrenergo", 5, "Poltava")
    assert k["restoration_power"] == "partial" and k["consumers_restored"] == 58_000 and pd.isna(k["consumers_without_power"])
    assert p["consumers_without_power"] == 20_000 and pd.isna(p["restoration_power"])
    assert fact(facts, "ukrenergo", 6, "Poltava")["emergency"]
    assert fact(facts, "ukrenergo", 6, "Kharkiv")["consumers_without_power"] == 3_000


def test_schedules_attributed_only_to_named_oblast(built):
    _, facts, _ = built
    assert fact(facts, "ukrenergo", 3, "Sumy")["schedules"]
    assert facts[(facts.post_id == 3) & (facts.oblast != "Sumy")].empty


def test_spans_point_into_the_raw_text(built):
    _, facts, _ = built
    raw = _raw(RAW).set_index("post_id")["text"]
    for r in facts.itertuples():
        for field, items in json.loads(r.spans).items():
            for it in items:
                assert raw[r.post_id][it["start"]:it["end"]] == it["text"], (r.channel, r.post_id, field)


def test_episode_oblast_with_own_channel(built):
    _, _, eps = built
    assert len(eps) == len(tr.OBLASTS)
    e = eps[eps.oblast == "Sumy"].iloc[0]
    assert e["coverage"] == "own_channel" and e["outage"] and e["max_consumers_without_power"] == 12_000
    assert e["first_impact_at"] == T("2024-03-22 06:30") and e["first_post_at"] == T("2024-03-22 05:30")
    assert e["restored_at"] == T("2024-03-22 20:00") and e["lifted_at"] == T("2024-03-29 15:00")
    assert e["resolution_kind"] == "restrictions_lifted" and e["resolution_source"] == "oblast" and e["status"] == "observed"
    assert e["hours_to_resolution"] == pytest.approx((T("2024-03-29 15:00") - T("2024-03-22 06:00")).total_seconds() / 3600)
    assert not e["censored"]
    ev = {x["role"]: x for x in json.loads(e["spans_used"])}
    assert ev["max_consumers"]["url"] == "https://t.me/sumyoblenergo/2" and "12 тисяч" in ev["max_consumers"]["text"]


def test_episode_national_lift_is_flagged_and_text_only_coverage(built):
    _, _, eps = built
    p = eps[eps.oblast == "Poltava"].iloc[0]
    assert p["coverage"] == "text_only" and p["resolution_source"] == "national_lift" and p["status"] == "observed"
    k = eps[eps.oblast == "Kharkiv"].iloc[0]          # outage reported, no restriction, no 'full' restoration -> censored
    assert k["status"] == "censored" and k["censored"] and np.isnan(k["hours_to_resolution"])


def test_oblast_without_any_post_has_no_impact(built):
    _, _, eps = built
    e = eps[eps.oblast == "Lviv"].iloc[0]
    assert not e["outage"] and e["status"] == "no_impact_reported" and e["own_posts_in_window"] == 0


def test_only_posts_inside_the_window_are_used():
    raw = _raw([("sumyoblenergo", "2024-03-21 23:00", "Знеструмлено 5 тисяч споживачів через обстріл."),
                ("sumyoblenergo", "2024-05-30 10:00", "Знеструмлено 7 тисяч споживачів через обстріл.")])
    posts = te.add_derived(te.extract_frame(raw))
    facts = tr.region_facts(posts, raw, CH)
    eps = tr.build_region_episodes(facts, STRIKE, CH, posts.groupby("channel").datetime.min().to_dict(), None, EpisodeParams())
    assert not eps[eps.oblast == "Sumy"].iloc[0]["outage"]


def test_channel_oblast_loader(tmp_path: Path):
    y = tmp_path / "s.yaml"
    y.write_text("telegram:\n  - {name: a, oblast: Sumy}\n  - {name: b}\n", encoding="utf-8")
    assert tr.load_channel_oblasts(y) == {"a": "Sumy"}
    y.write_text("telegram:\n  - {name: a, oblast: Atlantis}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        tr.load_channel_oblasts(y)


def test_score_region_gold_counts_sets_per_oblast(built):
    _, facts, _ = built
    gold = pd.DataFrame({"channel": ["ukrenergo", "sumyoblenergo"], "post_id": [5, 2], "stratum": ["x", "x"],
                         "attack": ["", "Sumy"], "consumers": ["Poltava:20000", "Sumy:12000"], "schedules": ["", ""],
                         "emergency": ["", ""], "lifted": ["", ""], "restored_full": ["", ""], "restored_any": ["Kharkiv", ""]})
    sc = tr.score_region_gold(gold, facts).set_index("field")
    assert sc.loc["consumers", "tp"] == 2 and sc.loc["consumers", "fp"] == 0 and sc.loc["consumers", "fn"] == 0
    assert sc.loc["restored_any", "tp"] == 1 and sc.loc["attack", "tp"] == 1


def test_run_extract_writes_regional_files(tmp_path: Path):
    r = _raw(RAW)

    def setup(root: Path):
        (root / "raw" / "telegram").mkdir(parents=True)
        (root / "external").mkdir()
        for ch, g in r.groupby("channel"):
            g.to_parquet(root / "raw" / "telegram" / f"{ch}.parquet")
        pd.DataFrame({"date": ["2024-03-22"], "time_utc": ["06:00"], "description": ["x"], "major": [True]}).to_csv(
            root / "external" / "attacks.csv", index=False)

    setup(tmp_path / "a")
    out = te.run_extract(tmp_path / "a", prm=EpisodeParams(), channel_oblast=CH)
    assert {"region_posts", "region_episodes"} <= set(out)
    proc = tmp_path / "a" / "processed"
    assert (proc / "telegram_region_posts.parquet").exists()
    e = pd.read_parquet(proc / "telegram_region_episodes.parquet")
    assert len(e) == len(tr.OBLASTS) and e[e.oblast == "Sumy"].iloc[0]["outage"]
    # without a channel -> oblast mapping the regional step is skipped
    setup(tmp_path / "b")
    assert "region_episodes" not in te.run_extract(tmp_path / "b", prm=EpisodeParams())


def test_regional_channel_reposting_a_national_digest_is_not_about_its_own_oblast():
    raw = _raw([("sumyoblenergo", "2024-03-22 09:00", "Енергетики повернули світло всім споживачам Київщини."),
                ("sumyoblenergo", "2024-03-22 10:00", "Енергетики повернули світло всім споживачам.")])
    posts = te.add_derived(te.extract_frame(raw))
    facts = tr.region_facts(posts, raw, CH)
    assert facts[(facts.post_id == 1) & (facts.oblast == "Sumy")].empty
    k = facts[(facts.post_id == 1) & (facts.oblast == "Kyiv")]
    assert len(k) == 1 and k.iloc[0]["source"] == "text" and k.iloc[0]["restoration_power"] == "full"
    own = facts[(facts.post_id == 2) & (facts.oblast == "Sumy")]
    assert len(own) == 1 and own.iloc[0]["source"] == "own"          # no other oblast named: the channel's own oblast


def test_episode_stops_at_the_next_strike_day_and_keeps_the_30_day_variant():
    strikes = pd.DataFrame({"strike_at": [T("2024-03-22 06:00"), T("2024-03-25 06:00")], "major": [True, True], "description": ["x", "y"]})
    raw = _raw([("sumyoblenergo", "2024-03-22 08:00", "Через обстріл знеструмлено 12 тисяч споживачів."),
                ("sumyoblenergo", "2024-03-22 15:00", "Завтра, 23 березня, у Сумській області з 06:00 до 22:00 вимушено будуть "
                                                      "застосовуватись графіки погодинних відключень."),
                ("sumyoblenergo", "2024-03-28 12:00", "Завтра графіки відключень не застосовуватимуться.")])
    posts = te.add_derived(te.extract_frame(raw))
    facts = tr.region_facts(posts, raw, CH)
    first = posts.groupby("channel").datetime.min().to_dict()
    e = tr.build_region_episodes(facts, strikes, CH, first, None, EpisodeParams())
    a = e[(e.oblast == "Sumy") & (e.strike_at == strikes.strike_at[0])].iloc[0]
    assert a["status"] == "censored" and a["censored"] and a["window_end_reason"] == "next_strike"
    assert a["censor_at"] == T("2024-03-25 00:00") and a["censor_hours"] == pytest.approx(66.0)
    assert a["status_30d"] == "observed"
    assert a["hours_to_resolution_30d"] == pytest.approx((T("2024-03-28 12:00") - T("2024-03-22 06:00")).total_seconds() / 3600)
    assert a["resolution_at_30d"] == T("2024-03-28 12:00") and pd.isna(a["resolution_at"])
    # the later strike keeps the lifting post
    b = e[(e.oblast == "Sumy") & (e.strike_at == strikes.strike_at[1])].iloc[0]
    assert b["window_end_reason"] == "horizon" and b["status"] == "no_impact_reported"   # the lifting post alone is no impact


def test_region_gold_file_is_well_formed():
    g = pd.read_csv(Path(__file__).parent / "fixtures" / "telegram_region_gold.csv", dtype=str, keep_default_na=False)
    assert len(g) >= 60 and not g.duplicated(["channel", "post_id"]).any()
    for col, _ in tr.GOLD_FIELDS:
        for v in g[col]:
            for item in tr._gold_items(v):
                assert item.split(":")[0] in tr.OBLASTS, (col, item)


def test_window_channel_censors_at_the_end_of_the_window():
    """A window-collected channel is observed only inside the strike's window: later posts (keyword-search leftovers) are ignored,
    the episode is right-censored at the window end; a strike without window gets no own-channel coverage from it."""
    ch = {"wchan": "Sumy"}
    strikes = pd.DataFrame({"strike_at": [T("2024-03-22 06:00"), T("2024-05-10 06:00")], "major": [True, True],
                            "description": ["x", "y"]})
    raw = _raw([("wchan", "2024-03-22 08:00", "Через обстріл знеструмлено 12 тисяч споживачів."),
                ("wchan", "2024-03-22 15:00", "Завтра, 23 березня, у Сумській області з 06:00 до 22:00 вимушено будуть "
                                              "застосовуватись графіки погодинних відключень."),
                ("wchan", "2024-03-28 12:00", "Завтра графіки відключень не застосовуватимуться."),      # after the window
                ("wchan", "2024-05-10 08:00", "Через обстріл знеструмлено 5 тисяч споживачів.")])    # strike without window
    posts = te.add_derived(te.extract_frame(raw))
    facts = tr.region_facts(posts, raw, ch)
    first = posts.groupby("channel").datetime.min().to_dict()
    win = pd.DataFrame({"window_start": [T("2024-03-21")], "window_end": [T("2024-03-26")], "strikes": ["2024-03-22"]})
    info = {"wchan": {"continuous_from": None, "windows": win}}
    e = tr.build_region_episodes(facts, strikes, ch, first, None, EpisodeParams(), window_info=info)
    a = e[(e.oblast == "Sumy") & (e.strike_at == strikes.strike_at[0])].iloc[0]
    assert a["coverage"] == "own_channel" and a["status"] == "censored" and a["window_end_reason"] == "collection_window"
    assert a["censor_at"] == T("2024-03-26") and a["censor_hours"] == pytest.approx(90.0)
    assert a["status_30d"] == "censored" and a["censor_hours_30d"] == pytest.approx(90.0)     # the 03-28 lifting post is not seen
    b = e[(e.oblast == "Sumy") & (e.strike_at == strikes.strike_at[1])].iloc[0]
    assert b["coverage"] == "text_only" and b["status"] == "no_impact_reported"             # its posts are outside any window
    # without window information the same data behave as before (30-day window, post seen)
    e0 = tr.build_region_episodes(facts, strikes, ch, first, None, EpisodeParams())
    assert e0[(e0.oblast == "Sumy") & (e0.strike_at == strikes.strike_at[0])].iloc[0]["status_30d"] == "observed"
    # a channel in its continuous period (continuous_from) is an ordinary channel again
    info2 = {"wchan": {"continuous_from": T("2024-05-01"), "windows": win}}
    e2 = tr.build_region_episodes(facts, strikes, ch, first, None, EpisodeParams(), window_info=info2)
    b2 = e2[(e2.oblast == "Sumy") & (e2.strike_at == strikes.strike_at[1])].iloc[0]
    assert b2["coverage"] == "own_channel" and b2["outage"]


def test_load_window_info(tmp_path):
    y = tmp_path / "s.yaml"
    y.write_text("telegram:\n  - {name: a, oblast: Sumy, window_collected: true, continuous_from: '2024-11-01'}\n"
                 "  - {name: b, oblast: Kyiv}\n", encoding="utf-8")
    raw = tmp_path / "raw" / "telegram"
    raw.mkdir(parents=True)
    pd.DataFrame({"channel": ["a"], "window_start": ["2024-03-21T00:00:00+00:00"], "window_end": ["2024-03-26T00:00:00+00:00"],
                  "strikes": ["2024-03-22"]}).to_csv(raw / "_windows.csv", index=False)
    w = tr.load_window_info(y, tmp_path)
    assert list(w) == ["a"] and w["a"]["continuous_from"] == T("2024-11-01")
    assert tr._strike_window(w["a"], T("2024-03-22")) == (T("2024-03-21"), T("2024-03-26")) and tr._strike_window(w["a"], T("2024-04-01")) is None


def test_strike_window_is_the_union_of_overlapping_logged_windows():
    # an older short window and a later merged window (more strikes) both hold 2024-03-22
    win = pd.DataFrame({"window_start": [T("2024-03-21"), T("2024-03-21")], "window_end": [T("2024-03-30"), T("2024-04-06")],
                        "strikes": ["2024-03-22", "2024-03-22|2024-03-29"]})
    info = {"continuous_from": None, "windows": win}
    assert tr._strike_window(info, T("2024-03-22")) == (T("2024-03-21"), T("2024-04-06"))
    assert tr._strike_window(info, T("2024-03-29")) == (T("2024-03-21"), T("2024-04-06"))


# ---------------------------------------------------------------- regional gold v2 (dev / holdout halves)
GOLD_V2 = ("dev", "holdout")


def _gold_v2(half: str) -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "fixtures" / f"telegram_region_gold_v2_{half}.csv", dtype=str, keep_default_na=False)


@pytest.mark.parametrize("half", GOLD_V2)
def test_region_gold_v2_file_is_well_formed(half):
    g = _gold_v2(half)
    assert len(g) >= 100 and not g.duplicated(["channel", "post_id"]).any()
    assert list(g.columns) == ["channel", "post_id", "stratum"] + [c for c, _ in tr.GOLD_FIELDS]
    assert {"kw", "other", "restore_kw"} <= set(g["stratum"])
    for col, _ in tr.GOLD_FIELDS:
        for v in g[col]:
            for item in tr._gold_items(v):
                assert item.split(":")[0] in tr.OBLASTS, (half, col, item)
    old = pd.read_csv(Path(__file__).parent / "fixtures" / "telegram_region_gold.csv", dtype=str, keep_default_na=False)
    assert not set(zip(g.channel, g.post_id)) & set(zip(old.channel, old.post_id))


def test_region_gold_v2_halves_are_disjoint_and_every_field_has_positives():
    d, h = _gold_v2("dev"), _gold_v2("holdout")
    assert not set(zip(d.channel, d.post_id)) & set(zip(h.channel, h.post_id))
    for half in (d, h):
        for col, _ in tr.GOLD_FIELDS:
            assert (half[col] != "").any(), col


def test_scorer_counts_emergency_outages_as_schedules_of_any_kind():
    facts = pd.DataFrame([{"channel": "c", "post_id": 1, "oblast": "Sumy", "attack": False, "consumers_without_power": np.nan,
                           "schedules": False, "emergency": True, "lifted": False, "restoration_power": None}])
    gold = pd.DataFrame({"channel": ["c"], "post_id": [1], "stratum": ["x"], "attack": [""], "consumers": [""], "schedules": ["Sumy"],
                         "emergency": ["Sumy"], "lifted": [""], "restored_full": [""], "restored_any": [""]})
    assert tr.score_region_gold(gold, facts).set_index("field").loc["schedules", "tp"] == 1
    assert tr.score_region_gold(gold, facts, emergency_is_schedule=False).set_index("field").loc["schedules", "fn"] == 1


def test_headline_oblast_governs_the_whole_post_and_followup_sentence_is_carried():
    nl = chr(10)
    raw = _raw([("ukrenergo", "2024-03-22 06:00", nl.join(["Одеська область: ДТЕК повертає світло після нічної атаки",
                                                          "Було знеструмлено 1063 родини.",
                                                          "Станом на 17:00 всі оселі знову зі світлом."])),
                ("poltavskaoda", "2024-03-22 07:00", "Уночі ворог завдав ударів по об'єктах енергетики на Одещині. "
                                                    "На ранок без світла залишаються понад 90 тисяч споживачів.")])
    posts = te.add_derived(te.extract_frame(raw))
    facts = tr.region_facts(posts, raw, {"poltavskaoda": "Poltava"})
    a = fact(facts, "ukrenergo", 1, "Odesa")
    assert a["restoration_power"] == "full" and a["consumers_without_power"] == 1063
    assert fact(facts, "poltavskaoda", 2, "Odesa")["consumers_without_power"] == 90_000
