"""Guards for the most important rule of the project: no information from the future."""
import numpy as np
import pandas as pd

from deficit.features.build import build_features, feature_columns
from deficit.ingestion.synthetic import generate
from deficit.models.base import Forecaster
from deficit.preprocessing.master import build_master
from deficit.training.walkforward import DriftDetector, DriftParams, fit_adaptive


def _master(cfg):
    generate(cfg)
    return build_master(cfg)


def test_features_do_not_see_the_future(cfg):
    master = _master(cfg)
    f1 = build_features(master, cfg)
    h = cfg.features.horizon_hours
    t = f1.index[len(f1) // 2]
    # scramble everything at and after the issue time (t - h) except exogenous columns
    m2 = master.copy()
    future = m2.index > t - pd.Timedelta(hours=h)
    for c in ("deficit", "actual_load_MW", "net_import_MW", "capacity_lost_MW", "attack_flag", "weapons_through", "weapons_fired", "dam_price_ratio"):
        m2.loc[future, c] = 1e6
    f2 = build_features(m2, cfg)
    cols = [c for c in feature_columns(f1)]
    pd.testing.assert_series_equal(f1.loc[t, cols], f2.loc[t, cols], check_names=False)


class Spy(Forecaster):
    """Records what it was trained on and asked to predict."""
    log: list = []

    def fit(self, frame, train_mask):
        self.train_max = frame.index[np.asarray(train_mask)].max()
        return self

    def predict(self, frame, mask):
        Spy.log.append((self.train_max, frame.index[np.asarray(mask)].min()))
        return np.zeros(int(np.asarray(mask).sum()))


def test_walk_forward_timing(cfg):
    master = _master(cfg)
    frame = build_features(master, cfg)
    days = pd.date_range(frame.index[0].normalize(), frame.index[-1], freq="D")
    signal = pd.DataFrame({"cp_prob": 0.0, "map_run_length": 0}, index=days)
    signal.iloc[300, 0] = 0.99  # one alarm
    Spy.log = []
    start = cfg.train_end_ts + pd.Timedelta(hours=1)
    res = fit_adaptive(Spy, frame, cfg, signal, start)
    h = pd.Timedelta(hours=cfg.features.horizon_hours)
    assert Spy.log
    for train_max, first_pred in Spy.log:
        assert first_pred - train_max > h, "forecast issued before its training data was observed"
    assert "alarm" in set(res.refits["reason"])
    assert res.forecasts["yhat"].notna().mean() > 0.99


def test_regime_mode_trains_on_the_new_regime_only(cfg):
    master = _master(cfg)
    frame = build_features(master, cfg)
    days = pd.date_range(frame.index[0].normalize(), frame.index[-1], freq="D")
    signal = pd.DataFrame({"cp_prob": 0.0, "map_run_length": 0}, index=days)
    tau_i = 330
    signal.iloc[tau_i, 0] = 0.99
    tau = days[tau_i]
    start = cfg.train_end_ts + pd.Timedelta(hours=1)
    regime = fit_adaptive(Spy, frame, cfg, signal, start, mode="regime").refits
    literal = fit_adaptive(Spy, frame, cfg, signal, start, mode="tau_w").refits
    late = lambda r: r[r["refit_at"] >= tau + pd.Timedelta(days=cfg.adaptive.min_train_days + 2)]  # noqa: E731
    assert len(late(regime)) > 0
    assert (late(regime)["train_start"] >= tau).all() or (late(regime)["train_start"] >= late(regime)["refit_at"] - pd.Timedelta(days=cfg.adaptive.window_days)).all()
    assert (late(regime)["train_start"] >= tau).all()                # nothing from before the break
    assert (late(literal)["train_start"] < tau).any()                # the literal formula keeps the old regime


# ---------------------------------------------------------------- attack- and drift-triggered refits
def _setup(cfg):
    frame = build_features(_master(cfg), cfg)
    days = pd.date_range(frame.index[0].normalize(), frame.index[-1], freq="D")
    signal = pd.DataFrame({"cp_prob": 0.0, "map_run_length": 0}, index=days)
    return frame, signal, cfg.train_end_ts + pd.Timedelta(hours=1)


def test_attack_refit_starts_the_day_after_the_attack(cfg):
    frame, signal, start = _setup(cfg)
    h = pd.Timedelta(hours=cfg.features.horizon_hours)
    attacks = [start + pd.Timedelta(days=d, hours=hr) for d, hr in ((10, 6), (40, 23), (41, 1), (80, 0))]
    Spy.log = []
    res = fit_adaptive(Spy, frame, cfg, signal, start, use_alarms=False, attack_times=attacks,
                       trigger_window_days=14)
    r = res.refits[res.refits["reason"] == "attack"]
    assert len(r) >= 3
    for T in r["refit_at"]:
        # usable from the start of the day after the attack, never earlier
        assert any(a.floor("D") + pd.Timedelta(days=1) == T for a in attacks)
        assert all(a < T for a in attacks if a.floor("D") + pd.Timedelta(days=1) == T)
    for T, row in zip(res.refits["refit_at"], res.refits.itertuples()):
        if row.reason == "attack":
            assert row.train_start >= T - pd.Timedelta(days=14)   # recent window
    for train_max, first_pred in Spy.log:
        assert first_pred - train_max > h

    # attacks at or after X cannot influence any refit up to X
    X = attacks[2]
    res2 = fit_adaptive(Spy, frame, cfg, signal, start, use_alarms=False,
                        attack_times=[a for a in attacks if a < X], trigger_window_days=14)
    a, b = (d[d["refit_at"] <= X].reset_index(drop=True) for d in (res.refits, res2.refits))
    pd.testing.assert_frame_equal(a, b)


def test_drift_detector_cusum():
    rng = np.random.default_rng(0)
    d = DriftDetector(DriftParams(k=0.5, h=3.0, min_days=14))
    assert not any(d.update(1.0 + 0.1 * rng.standard_normal()) for _ in range(120))   # stationary: silent
    fired = [d.update(3.0) for _ in range(5)]                                           # error triples
    assert any(fired) and not fired[0]                                                  # noticed within days, not at once
    d.reset()
    assert d.s == 0.0
    assert not DriftDetector(DriftParams(min_days=14)).update(100.0)                    # no reference yet


def test_drift_trigger_uses_only_errors_of_the_past(cfg):
    frame, signal, start = _setup(cfg)
    p = DriftParams(k=0.5, h=2.0, ref_days=30, min_days=14)   # one exploding day is enough to fire
    Spy.log = []
    base = fit_adaptive(Spy, frame, cfg, signal, start, use_alarms=False, drift=p, trigger_window_days=14)
    X = start + pd.Timedelta(days=60, hours=7)      # from here on the targets explode
    f2 = frame.copy()
    f2.loc[f2.index >= X, "y"] = f2.loc[f2.index >= X, "y"] * 50 + 1e4
    Spy.log = []
    pert = fit_adaptive(Spy, f2, cfg, signal, start, use_alarms=False, drift=p, trigger_window_days=14)
    # 1. nothing that happens at or after X (here: the targets) changes a decision taken up to X
    a, b = (d[d["refit_at"] <= X].reset_index(drop=True) for d in (base.refits, pert.refits))
    pd.testing.assert_frame_equal(a, b)
    # 2. the jump is noticed - but only once the erroneous hours are over, i.e. after X
    drift = pert.refits[pert.refits["reason"] == "drift"]
    assert (drift["refit_at"] > X).any()
    assert drift["refit_at"].min() > X
    # 3. a drift refit trains on the short recent window
    d0 = drift.iloc[0]
    assert d0["train_start"] >= d0["refit_at"] - pd.Timedelta(days=14)
    h = pd.Timedelta(hours=cfg.features.horizon_hours)
    for train_max, first_pred in Spy.log:
        assert first_pred - train_max > h


def test_recency_weights_use_only_the_training_window():
    from deficit.models.xgb_model import XGBForecaster
    idx = pd.date_range("2022-01-01", periods=24 * 60, freq="h", tz="UTC")
    m = XGBForecaster({}, halflife_days=10)
    w = m._weights(idx)
    assert w[-1] == 1.0 and abs(w[-1 - 24 * 10] - 0.5) < 1e-9 and (np.diff(w) > 0).all()
    # the weights of a row depend only on the newest TRAINING row, never on later data
    np.testing.assert_allclose(m._weights(idx[:24 * 30]), m._weights(idx[:24 * 30 + 1])[:-1] * 2 ** (1 / 24 / 10), rtol=1e-9)
    assert XGBForecaster({})._weights(idx) is None


# ---------------------------------------------------------------- telegram features (posts before the issue time only)
def _tg_posts(n=900, seed=3, start="2022-03-01"):
    """Synthetic posts with every column the telegram features and episodes read."""
    rng = np.random.default_rng(seed)
    t = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 120 * 86400, n)), unit="s")
    regions = ["Kyiv", "Lviv", "Odesa", "Sumy", "Kharkiv"]
    p = pd.DataFrame({"datetime": t, "channel": rng.choice(["ukrenergo", "dtek_ua"], n)})
    p["post_id"] = np.arange(n)
    p["is_attack_report"] = rng.random(n) < 0.08
    p["outage_schedules_announced"] = rng.random(n) < 0.15
    p["emergency_outages"] = rng.random(n) < 0.05
    p["schedules_lifted"] = rng.random(n) < 0.05
    p["consumers_without_power"] = np.where(rng.random(n) < 0.05, rng.integers(1000, 10**6, n), np.nan)
    p["damaged_objects"] = [list(rng.choice(["generation", "substation", "transmission"], rng.integers(0, 3), replace=False)) for _ in range(n)]
    p["regions_hit"] = [list(rng.choice(regions, rng.integers(0, 4), replace=False)) for _ in range(n)]
    p["restoration_power"] = np.where(rng.random(n) < 0.04, "full", None)
    p["nationwide"] = False
    p["schedule_queues_max"] = np.nan
    for c in ("settlements_without_power", "deficit_mw", "import_mw", "limit_mw", "capacity_lost_mw"):
        p[c] = np.nan
    p["impact_post"] = p["outage_schedules_announced"] | p["emergency_outages"] | p["consumers_without_power"].notna()
    return p


def _loud(p, cut):
    """Rewrite every post at or after `cut` so that it screams, and add posts exactly at `cut` and later."""
    q = p.copy()
    late = q["datetime"] >= cut
    n_late = int(late.sum())
    q.loc[late, ["emergency_outages", "outage_schedules_announced", "schedules_lifted", "impact_post"]] = True
    q.loc[late, "consumers_without_power"] = 9e9
    q.loc[late, "damaged_objects"] = pd.Series([["generation", "substation"]] * n_late, index=q.index[late])
    q.loc[late, "regions_hit"] = pd.Series([list("abcdefg")] * n_late, index=q.index[late])
    extra = q.iloc[:3].copy()
    extra["datetime"] = [cut, cut + pd.Timedelta(days=3), cut + pd.Timedelta(days=9)]
    extra[["emergency_outages", "outage_schedules_announced", "is_attack_report", "impact_post"]] = True
    extra["consumers_without_power"] = 9e9
    extra["damaged_objects"] = pd.Series([["generation"]] * 3, index=extra.index)
    extra["regions_hit"] = pd.Series([list("abcdefg")] * 3, index=extra.index)
    return pd.concat([q, extra], ignore_index=True).sort_values("datetime").reset_index(drop=True)


def test_telegram_features_ignore_posts_published_after_the_issue_time():
    from deficit.features.telegram import REGIME, TG_COLS, issue_features, regime_state
    posts = _tg_posts()
    idx = pd.date_range("2022-02-20", "2022-06-20", freq="D", tz="UTC")
    d0 = pd.Timestamp("2022-04-15", tz="UTC")
    issue = d0 + pd.Timedelta(days=1)                                   # end of day d0
    base = issue_features(posts, idx)
    pert = issue_features(_loud(posts, issue), idx)                      # everything at/after the issue time is rewritten
    assert base.loc[d0, "tg_sched_days30"] > 0                           # the test sees real signal
    pd.testing.assert_frame_equal(base.loc[:d0], pert.loc[:d0])          # nothing up to the issue time changes
    assert not base.loc[d0 + pd.Timedelta(days=1):].equals(pert.loc[d0 + pd.Timedelta(days=1):])   # but the change is visible later
    assert regime_state(posts, issue) == regime_state(_loud(posts, issue), issue)
    assert set(regime_state(posts, issue)) == set(REGIME)
    # a post one second before the issue time IS used, a post exactly at the issue time is not
    inside = posts.iloc[:1].copy()
    inside["datetime"] = [issue - pd.Timedelta(seconds=1)]
    inside[["emergency_outages", "outage_schedules_announced"]] = True
    assert issue_features(pd.concat([posts, inside]), idx).loc[d0, "tg_emerg_d0"] == 1.0
    exactly = inside.copy()
    exactly["datetime"] = [issue]
    pd.testing.assert_series_equal(issue_features(pd.concat([posts, exactly]), idx).loc[d0, TG_COLS], base.loc[d0, TG_COLS])
    # the features of an issue day do not depend on which later days are requested
    pd.testing.assert_frame_equal(issue_features(posts, idx[idx <= d0]), base.loc[:d0])


def test_post_event_telegram_sets_do_not_see_posts_after_the_issue_day():
    from deficit.config import PostEvent
    from deficit.evaluation.post_event import direct_forecast, direct_frame
    from deficit.features.telegram import issue_features
    from test_post_event import XCFG, _toy_daily
    d = _toy_daily(n=420)
    posts = _tg_posts(n=2500, start="2022-01-10")
    H, thr, i0 = 30, 0.5, 300
    pe = PostEvent(direct_n_estimators=40, direct_min_rows=50)
    issue = d.index[i0] + pd.Timedelta(days=1)
    loud = _loud(posts, issue)
    out = {}
    for name, p in (("base", posts), ("loud", loud)):
        fr = direct_frame(d, H, thr, pe, issue_features(p, d.index))
        out[name] = {s: direct_forecast(d, i0, H, thr, pe, XCFG, 0, fr, s) for s in ("g", "tension_g")}
    assert np.isfinite(list(out["base"]["g"].values())).all()
    assert out["base"] == out["loud"]
    # one day later the same posts DO matter (the check is not vacuous)
    fr_a = direct_frame(d, H, thr, pe, issue_features(posts, d.index))
    fr_b = direct_frame(d, H, thr, pe, issue_features(loud, d.index))
    assert not np.allclose(fr_a.loc[i0 + 1:, "tg_sched_days7"], fr_b.loc[i0 + 1:, "tg_sched_days7"])
    assert fr_a.loc[:i0, "tg_sched_days7"].equals(fr_b.loc[:i0, "tg_sched_days7"])


def test_restoration_forecast_uses_only_posts_and_labels_known_at_the_issue_time():
    from deficit.evaluation.restoration import (
        BASELINES, forecast_one, issue_time, km_median, known_labels, label_as_of, strike_features)
    from deficit.ingestion.telegram_extract import EpisodeParams
    posts = _tg_posts(n=1500, seed=5, start="2022-03-01")
    strikes = pd.DataFrame({"strike_at": pd.to_datetime(["2022-03-12 06:00", "2022-03-20 06:00", "2022-03-29 06:00", "2022-04-08 06:00",
                                                         "2022-04-15 06:00", "2022-04-26 06:00", "2022-05-06 06:00", "2022-05-17 06:00"], utc=True),
                            "weapons_through": [5, 10, 3, 20, 8, 12, 6, 9]})
    prm, i = EpisodeParams(), 6
    nat = lambda p: p[p["channel"] == "ukrenergo"]  # noqa: E731
    cut = issue_time(strikes["strike_at"].iloc[i])
    loud = _loud(posts, cut)
    ch = dict(feature_channels=["ukrenergo"], regime_lookback=60, all_channels=["dtek_ua", "ukrenergo"])
    F1, F2 = strike_features(posts, strikes, **ch), strike_features(loud, strikes, **ch)
    pd.testing.assert_frame_equal(F1.iloc[: i + 1], F2.iloc[: i + 1])                  # features of the strikes up to i: unchanged
    t1, e1 = known_labels(posts, nat(posts), strikes, i, prm)
    t2, e2 = known_labels(loud, nat(loud), strikes, i, prm)
    np.testing.assert_array_equal(t1, t2)                                              # labels as known at the issue time: unchanged
    np.testing.assert_array_equal(e1, e2)
    assert e1.sum() >= 1, "the toy data needs at least one observed label"
    f1, f2 = forecast_one(F1, i, t1, e1, [1.0]), forecast_one(F2, i, t2, e2, [1.0])
    assert f1 == f2 and np.isfinite(list(f1.values())).all()
    assert set(BASELINES) <= set(f1)
    # a label known at an early instant is censored at the time watched; with all posts it may become an observed duration
    s0, s1 = strikes["strike_at"].iloc[0], strikes["strike_at"].iloc[1]
    early = label_as_of(posts, nat(posts), s0, s1, prm, issue_time(s0) - pd.Timedelta(microseconds=1))
    late = label_as_of(posts, nat(posts), s0, s1, prm, posts["datetime"].max())
    assert early[2] <= 1.0 and late[2] > early[2]
    assert km_median(np.array([1.0, 2.0, 3.0, 30.0]), np.array([True, True, False, False])) <= 30.0


def test_region_restoration_forecast_ignores_posts_published_after_the_issue_time():
    """Oblast-level forecast of strike i: rewriting every fact at or after its issue time changes neither its features, its
    cohort, the training labels nor any prediction (the final label of strike i itself may change)."""
    from test_region_restoration import OBL, _episodes, _facts, _loud, _strikes
    from deficit.evaluation import region_restoration as rr
    strikes = _strikes(16)
    facts = _facts(strikes, seed=3)
    i = 10
    cut = rr.issue_time(strikes["strike_at"].iloc[i])
    loud = _loud(facts, cut)
    methods = rr.all_methods([10.0], extra=True)           # incl. IPCW ridge and boosted AFT trees
    checked = 0
    for variant in rr.VARIANTS:
        out = {}
        for name, f in (("base", facts), ("loud", loud)):
            tab = rr.build_table(_episodes(f, strikes), f, strikes, None, pd.Timestamp("2022-05-31", tz="UTC"), variant)
            out[name] = rr.run_walk_forward(tab[tab["idx"] <= i], [10.0], min_obs=5, min_oblast=2, extra=True)
        a, b = (out[k][out[k]["idx"] == i].reset_index(drop=True) for k in ("base", "loud"))
        assert len(a) > 0 and a["evaluated"].all()
        cols = ["oblast", "issue", "n_train", "n_train_observed"] + rr.STRIKE_F + rr.SEVERITY_F + rr.ACTIVITY_F + rr.HIST_F + methods
        pd.testing.assert_frame_equal(a[cols], b[cols])
        # the earlier strikes' final labels are also unchanged for the next-strike window (they end before the issue time)
        if variant == "next_strike":
            e0, e1 = (out[k][out[k]["idx"] < i].reset_index(drop=True) for k in ("base", "loud"))
            pd.testing.assert_series_equal(e0["hours"], e1["hours"])
        checked += len(a)
    assert checked > 0 and set(OBL) >= set(a["oblast"])


def test_region_history_features_ignore_outcomes_and_posts_after_the_issue_time():
    """The oblast history of strike i (previous duration, prior count, days since, regime flag) is unchanged when the
    resolution posts / outcomes of the previous episodes after the issue time of strike i, or any later post, are rewritten."""
    from test_region_restoration import _episodes, _facts, _loud, _strikes
    from deficit.evaluation import region_restoration as rr
    strikes = _strikes(16)
    facts = _facts(strikes, seed=5)
    i = 9
    cut = rr.issue_time(strikes["strike_at"].iloc[i])
    vend = pd.Timestamp("2022-05-31", tz="UTC")
    variants = {"loud": _loud(facts, cut),
                "no_lift_after": facts[~((facts["datetime"] >= cut) & facts["lifted"])],
                "extra_later": pd.concat([facts, facts[facts["datetime"] >= cut].assign(datetime=lambda d: d["datetime"] + pd.Timedelta(days=1))],
                                         ignore_index=True).sort_values("datetime").reset_index(drop=True)}
    for variant in rr.VARIANTS:
        base = rr.build_table(_episodes(facts, strikes), facts, strikes, None, vend, variant)
        a = base[base["idx"] == i].reset_index(drop=True)
        assert len(a) > 0
        for name, f in variants.items():
            t = rr.build_table(_episodes(f, strikes), f, strikes, None, vend, variant)
            b = t[t["idx"] == i].reset_index(drop=True)
            pd.testing.assert_frame_equal(a[["oblast"] + rr.HIST_F], b[["oblast"] + rr.HIST_F], obj=f"{variant}/{name}")


def test_region_binary_task_ignores_posts_published_after_the_issue_time():
    """Long-outage probabilities of strike i (logistic regression, base rates) are unchanged when every fact at or after its issue
    time is rewritten; the binary training labels are those known at the issue time."""
    from test_region_restoration import _episodes, _facts, _loud, _strikes
    from deficit.evaluation import region_binary as rb, region_restoration as rr
    strikes = _strikes(16)
    facts = _facts(strikes, seed=7)
    i = 10
    cut = rr.issue_time(strikes["strike_at"].iloc[i])
    loud = _loud(facts, cut)
    for variant in rr.VARIANTS:
        out = {}
        for name, f in (("base", facts), ("loud", loud)):
            tab = rr.build_table(_episodes(f, strikes), f, strikes, None, pd.Timestamp("2022-05-31", tz="UTC"), variant)
            dev = tab[tab["idx"] <= i]
            out[name] = rb.binary_walk_forward(dev, [i], [10.0], (24.0, 72.0), 2)
        for T in (24.0, 72.0):
            a, b = out["base"][T], out["loud"][T]
            pc = [c for c in a.columns if c != "y"]
            assert len(a) > 0
            pd.testing.assert_frame_equal(a[pc], b[pc])



def test_region_early_features_ignore_posts_at_or_after_the_early_cut():
    """The early-impact features of strike i (K hours after strike_at) do not change when extra impact posts are added at or after
    the cut-off (the cohort may change for oblasts whose first impact post comes later, so rows are matched by oblast)."""
    from test_region_restoration import OBL, _episodes, _fact, _facts, _strikes
    from deficit.evaluation import region_restoration as rr
    strikes = _strikes(16)
    facts = _facts(strikes, seed=7)
    i, vend = 9, pd.Timestamp("2022-05-31", tz="UTC")
    st = strikes["strike_at"].iloc[i]
    checked = 0
    for K in (3.0, 6.0):
        cut = st + pd.Timedelta(hours=K)
        late = pd.DataFrame([_fact(ob, cut + pd.Timedelta(minutes=m), post_id=20_000 + 10 * j + m, attack=True, schedules=True, emergency=True,
                                   consumers_without_power=9e9, damaged_objects=["generation"]) for j, ob in enumerate(OBL) for m in (0, 1, 120)])
        late["datetime"] = pd.to_datetime(late["datetime"], utc=True)
        pert = pd.concat([facts, late], ignore_index=True).sort_values("datetime").reset_index(drop=True)
        for variant in rr.VARIANTS:
            a = rr.build_table(_episodes(facts, strikes), facts, strikes, None, vend, variant, K)
            b = rr.build_table(_episodes(pert, strikes), pert, strikes, None, vend, variant, K)
            a, b = (x[x["idx"] == i].set_index("oblast")[rr.EARLY_F] for x in (a, b))
            common = a.index.intersection(b.index)
            assert len(common) > 0
            pd.testing.assert_frame_equal(a.loc[common], b.loc[common], obj=f"K={K}/{variant}")
            checked += len(common)
    assert checked > 0


def test_duration_dev_features_exclude_whole_day_counts_and_use_only_labels_resolved_before_the_strike_day():
    from deficit.evaluation import duration_dev as dd
    from deficit.ingestion.telegram_extract import EpisodeParams
    prm = EpisodeParams()
    # 1) whole-day counts of derived strikes are not features, and cannot leak in through the strike list
    cols = {c for s in dd.SETS.values() for c in s}
    assert not cols & set(dd.FORBIDDEN) and {"n_channels", "n_attack_posts"} <= set(dd.FORBIDDEN)
    posts = _tg_posts(n=1500, seed=5, start="2022-03-01")
    strikes = pd.DataFrame({"strike_at": pd.to_datetime(["2022-03-12 06:00", "2022-03-20 06:00", "2022-03-29 06:00", "2022-04-08 06:00",
                                                         "2022-04-15 06:00", "2022-04-26 06:00", "2022-05-06 06:00", "2022-05-17 06:00"], utc=True),
                            "weapons_through": [5, np.nan, 3, np.nan, 8, 12, np.nan, 9]})
    F = dd.duration_features(strikes, dd.LabelBook(posts, strikes, prm), None)
    assert set(F.columns) == cols and list(F["is_af"]) == [1, 0, 1, 0, 1, 1, 0, 1]
    wild = strikes.assign(n_channels=[9, 1, 7, 2, 5, 3, 8, 4], n_attack_posts=[99, 1, 50, 2, 7, 1, 30, 4], derived=[False, True] * 4)
    pd.testing.assert_frame_equal(F, dd.duration_features(wild, dd.LabelBook(posts, wild, prm), None))
    # 2) posts of the strike day (and later) never enter the features of that strike; the labels of the walk-forward stop at the issue time
    i = 5
    day = strikes["strike_at"].iloc[i].floor("D")
    loud = _loud(posts, day)
    b1, b2 = dd.LabelBook(posts, strikes, prm), dd.LabelBook(loud, strikes, prm)
    F2 = dd.duration_features(strikes, b2, None)
    pd.testing.assert_frame_equal(F.iloc[: i + 1], F2.iloc[: i + 1])
    assert not F.iloc[i + 1:].equals(F2.iloc[i + 1:]), "later strikes do see them (the check is not vacuous)"
    cut = dd.issue_time(strikes["strike_at"].iloc[i]) - pd.Timedelta(microseconds=1)
    for x, y in zip(b1.known(i, cut), dd.LabelBook(_loud(posts, cut + pd.Timedelta(microseconds=1)), strikes, prm).known(i, cut)):
        np.testing.assert_array_equal(x, y)
    # 3) a strike counts as resolved only if its lifting post is older than the strike day: same-day lifting is not used
    p = _tg_posts(n=4, seed=1, start="2022-03-01")
    p["channel"], p["datetime"] = "ukrenergo", pd.to_datetime(["2022-03-01 10:00", "2022-03-04 12:00", "2022-03-08 09:00", "2022-03-09 09:00"], utc=True)
    for c in ("is_attack_report", "outage_schedules_announced", "emergency_outages", "schedules_lifted", "impact_post"):
        p[c] = False
    p["consumers_without_power"], p["restoration_power"] = np.nan, None
    p.loc[0, ["outage_schedules_announced", "impact_post"]] = True       # regime announced
    p.loc[1, "schedules_lifted"] = True                                   # lifted 2022-03-04 12:00 = 3.25 d after the strike at 06:00
    st = pd.DataFrame({"strike_at": pd.to_datetime(["2022-03-01 06:00", "2022-03-04 15:00", "2022-03-09 06:00"], utc=True), "weapons_through": [4, 6, 2]})
    bk = dd.LabelBook(p, st, prm)
    assert bk.label(0, pd.Timestamp("2022-03-04 11:59", tz="UTC"))[1] is False and bk.label(0, pd.Timestamp("2022-03-04 12:00", tz="UTC"))[1] is True
    G = dd.duration_features(st, bk, None)
    assert G["prev_obs"].iloc[1] == 0.0 and np.isnan(G["hist_logmed5"].iloc[1]) and G["n_unresolved"].iloc[1] == 1.0   # lifted later on the strike day
    assert G["prev_obs"].iloc[2] == 0.0 and np.isclose(G["hist_logmed5"].iloc[2], np.log(3.25)) and G["n_unresolved"].iloc[2] == 1.0
    assert np.isclose(G["prev_log_t"].iloc[1], np.log(2.75)) and np.isclose(G["prev_log_t"].iloc[2], np.log(4.375))   # censored: time watched at the strike-day start


def _strike_toy(n=420, seed=3):
    """Daily toy table + extended strike list (major and derived rows) for the post-event dev tests."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    day = np.sort(rng.choice(np.arange(5, n - 5), 60, replace=False))
    derived = rng.random(len(day)) < 0.7
    sk = pd.DataFrame({"day": idx[day], "major": ~derived, "derived": derived})
    sk["datetime"] = sk["day"] + pd.to_timedelta(rng.integers(0, 24, len(sk)), unit="h")
    sk["weapons_through"] = np.where(derived, np.nan, rng.integers(1, 40, len(sk)).astype(float))
    y = np.zeros(n)
    hit = np.zeros(n)
    hit[day] = 1
    for i in range(1, n):
        y[i] = 0.9 * y[i - 1] + 0.6 * hit[i] * rng.uniform(0.5, 1.5) + 0.05 * rng.random()
    d = pd.DataFrame({"y": y, "temp": rng.normal(5, 8, n), "attack": hit, "through": hit}, index=idx)
    return d, sk


def test_post_event_extended_events_use_no_whole_day_counts_and_nothing_after_the_issue_day(tmp_path):
    from deficit.config import PostEvent
    from deficit.evaluation import post_event_dev as pdv
    from deficit.evaluation.post_event import direct_frame

    d, sk = _strike_toy()
    # 1) the whole-day counts of the derived rows (n_channels, n_attack_posts) never reach the loader or the features
    raw = pd.DataFrame({"date": sk["datetime"].dt.strftime("%Y-%m-%d"), "time_utc": sk["datetime"].dt.strftime("%H:%M"), "major": sk["major"],
                        "derived": sk["derived"], "weapons_through": sk["weapons_through"].fillna(7.0), "n_channels": 1, "n_attack_posts": 1})
    raw.to_csv(tmp_path / "a.csv", index=False)
    raw.assign(n_channels=9, n_attack_posts=99).to_csv(tmp_path / "b.csv", index=False)
    la, lb = pdv.load_strike_list(tmp_path / "a.csv"), pdv.load_strike_list(tmp_path / "b.csv")
    assert not {"n_channels", "n_attack_posts"} & set(la.columns)
    pd.testing.assert_frame_equal(la, lb)
    assert la.loc[la["derived"], "weapons_through"].isna().all()           # strength of a derived strike is unknown, not 0 (even if the file had a number)
    # 2) event selection: a later strike never changes an earlier decision, major strikes are always kept, the gap is respected
    ev = pdv.select_events(sk, 7)
    assert set(sk.loc[sk["major"], "day"]) <= set(ev["day"])
    dd = ev.loc[ev["derived"], "day"]
    assert all(min(abs((x - k).days) for k in ev["day"] if k < x) >= 7 for x in dd if (ev["day"] < x).any())
    cut = sk["datetime"].iloc[30]
    early = pdv.select_events(sk[sk["datetime"] <= cut], 7)
    pd.testing.assert_frame_equal(early, ev[ev["day"] <= early["day"].max()].reset_index(drop=True))
    # 3) features of the extended frame at an issue day use days <= d0 only (future y, strikes and counts scrambled)
    pe = PostEvent(direct_n_estimators=30, direct_min_rows=50, dev_min_event_rows=10)
    H, thr, i0 = 30, 0.5, 300
    d0 = pdv.ext_daily(d, sk[sk["day"] <= d.index[i0]])
    fa = pdv.ext_frame(pdv.ext_daily(d, sk), H, thr, pe)
    rng = np.random.default_rng(5)
    sk2 = sk.copy()
    late = sk2["day"] > d.index[i0]
    sk2.loc[late, "major"] = rng.random(late.sum()) < 0.5
    sk2.loc[late, "derived"] = ~sk2.loc[late, "major"]
    sk2.loc[late, "weapons_through"] = np.where(sk2.loc[late, "derived"], np.nan, 99.0)
    d2 = d.copy()
    d2.iloc[i0 + 1:, d2.columns.get_loc("y")] = rng.permutation(d2["y"].to_numpy()[i0 + 1:]) * 3 + 1
    fb = pdv.ext_frame(pdv.ext_daily(d2, sk2), H, thr, pe)
    cols = fa.attrs["feature_sets"]["base"] + ["m28", "max28"]
    pd.testing.assert_series_equal(fa.loc[i0, cols], fb.loc[i0, cols])
    assert {"derived7", "derived30"} <= set(fa.attrs["feature_sets"]["base"])
    # a window with strikes but none of them major has unknown (NaN) strength and a derived-strike indicator
    only_derived = sk[sk["derived"]].iloc[[3]]
    f1 = pdv.ext_frame(pdv.ext_daily(d, only_derived), H, thr, pe)
    j = int(d.index.get_loc(only_derived["day"].iloc[0])) + 2
    assert np.isnan(f1.loc[j, "through7"]) and f1.loc[j, "derived7"] == 1 and f1.loc[j + 8, "through7"] == 0
    # 4) forecasts of every variant at i0 do not see the days after i0
    major_idx = np.array(sorted(d.index.get_loc(x) for x in sk.loc[sk["major"], "day"]))
    ev_idx = np.array(sorted(d.index.get_loc(x) for x in sk["day"].unique()))

    def forecasts(dx, frame_o, skx):
        fe = pdv.ext_frame(pdv.ext_daily(dx, skx), H, thr, pe)
        e_idx = np.array(sorted(dx.index.get_loc(x) for x in skx["day"].unique()))
        return pdv.forecast_variants(frame_o, fe, e_idx[e_idx <= i0], i0, H, pe, {"subsample": 0.8, "colsample_bytree": 0.8}, 0,
                                     major_idx[major_idx <= i0])

    base = forecasts(d, direct_frame(d, H, thr, pe), sk)
    other = forecasts(d2, direct_frame(d2, H, thr, pe), sk2)
    assert np.isfinite(base["ev"]["depth"]) and np.isfinite(base["ext"]["depth"])
    for v in base:
        for k in base[v]:
            assert base[v][k] == other[v][k] or (np.isnan(base[v][k]) and np.isnan(other[v][k])), (v, k)
    assert ev_idx.max() > i0                                                  # the scrambled strikes are in the future of the issue day

def _dmg_posts(rows):
    """Post table for the damage features: rows = (datetime, tags, regions, queue_next, capacity_lost_mw)."""
    from deficit.ingestion.damage_features import POST_COLS, TAGS
    recs = []
    for t, tags, regs, q, cap in rows:
        r = {"datetime": pd.Timestamp(t, tz="UTC"), "is_attack_report": True, "regions_hit": list(regs), "capacity_lost_mw": cap, "queue_next": q}
        r.update({k: (k in tags) for k in TAGS})
        recs.append(r)
    return pd.DataFrame(recs)[POST_COLS]


def test_damage_features_use_only_posts_of_the_strike_day_before_the_issue_time_and_older_history():
    from deficit.ingestion.damage_features import DMG_F, damage_features
    strikes = pd.DataFrame({"strike_at": pd.to_datetime(["2024-03-10 03:00", "2024-03-20 22:00", "2024-03-21 05:00"], utc=True)})
    base = [("2024-03-05 12:00", {"tpp"}, ["Kyiv"], np.nan, np.nan),                       # history (15 days before strike 2)
            ("2024-03-10 00:00", {"tpp", "sub_hv"}, ["Kharkiv", "Sumy"], np.nan, np.nan),   # first instant of strike day 1: counts
            ("2024-03-10 23:59", {"distribution"}, ["Sumy"], 3.0, 9000.0),                 # last minute before the issue time: counts
            ("2024-03-20 10:00", {"line_hv"}, ["Odesa"], np.nan, np.nan)]
    F0 = damage_features(strikes, _dmg_posts(base))
    assert list(F0.columns) == DMG_F
    r = F0.iloc[0]
    assert r["dmg_tpp"] == 1 and r["dmg_sub_hv"] == 1 and r["dmg_distr"] == 1 and r["dmg_line_hv"] == 0 and r["dmg_n_oblasts"] == 2
    assert r["dmg_queue_next"] == 3 and np.isclose(r["dmg_cap_lost"], np.log1p(9000))
    assert r["dmg_repeat"] == 1                                                              # generation (tpp) seen 5 days earlier and again today
    assert F0["dmg_repeat"].iloc[1] == 1 and F0["dmg_gen"].iloc[1] == 0                      # strike 2: line_hv now, transmission (sub_hv) on 03-10
    # posts at or after the issue time change nothing for that strike (but later strikes may see them)
    loud = base + [("2024-03-11 00:00", {"npp", "hpp", "tpp", "distribution"}, ["Lviv", "Kyiv", "Odesa"], 6.0, 27000.0),
                   ("2024-03-12 10:00", {"npp"}, ["Lviv"], 6.0, 27000.0),
                   ("2024-03-21 00:00", {"npp", "hpp"}, ["Lviv"], 6.0, 27000.0)]
    F1 = damage_features(strikes, _dmg_posts(loud))
    pd.testing.assert_frame_equal(F0.iloc[:2], F1.iloc[:2])
    assert not F0.iloc[2:].equals(F1.iloc[2:]), "the third strike sees the posts of its own day: the check is not vacuous"
    # a post published before the strike day never counts as same-day damage
    assert damage_features(strikes, _dmg_posts(base[:1] + base[3:]))["dmg_tpp"].iloc[0] == 0
    one = strikes.iloc[[1]]
    same_day = [("2024-03-20 10:00", {"line_hv"}, [], np.nan, np.nan), ("2024-03-20 23:00", {"line_hv"}, [], np.nan, np.nan)]
    assert damage_features(one, _dmg_posts(same_day))["dmg_repeat"].iloc[0] == 0               # same-day posts are not "earlier"
    assert damage_features(one, _dmg_posts([("2024-03-05 10:00", {"line_hv"}, [], np.nan, np.nan)] + same_day))["dmg_repeat"].iloc[0] == 1
    assert damage_features(one, _dmg_posts([("2024-02-10 10:00", {"line_hv"}, [], np.nan, np.nan)] + same_day))["dmg_repeat"].iloc[0] == 0   # > 30 days
    s2 = pd.DataFrame({"strike_at": pd.to_datetime(["2024-03-21 05:00"], utc=True)})
    assert damage_features(s2, _dmg_posts(base + [("2024-03-21 11:00", {"line_hv"}, [], np.nan, np.nan)]))["dmg_repeat"].iloc[0] == 1
    assert damage_features(s2, _dmg_posts(base))["dmg_repeat"].iloc[0] == 0


def test_chain_episode_features_ignore_later_strikes_and_posts_of_the_chain():
    from deficit.evaluation import duration_dev as dd
    from deficit.ingestion.telegram_extract import EpisodeParams
    prm = EpisodeParams()
    posts = _tg_posts(n=1500, seed=5, start="2022-03-01")
    day = ["2022-03-12", "2022-03-14", "2022-03-15", "2022-03-27", "2022-03-29", "2022-04-12", "2022-04-13", "2022-04-25", "2022-05-08", "2022-05-09"]
    strikes = pd.DataFrame({"strike_at": pd.to_datetime([d + " 06:00" for d in day], utc=True), "weapons_through": 5.0})
    mask = dd.episode_start_mask(strikes)
    assert list(mask) == [True, False, False, True, False, True, False, True, True, False]            # gap <= 2 days merges
    mk = lambda rows: _dmg_posts(rows)  # noqa: E731
    dp = mk([("2022-03-12 10:00", {"tpp"}, ["Kyiv"], np.nan, np.nan), ("2022-03-27 10:00", {"sub_hv"}, ["Lviv"], np.nan, np.nan),
             ("2022-04-12 10:00", {"distribution"}, ["Sumy"], np.nan, np.nan)])
    # later chain members and their posts (03-14, 03-15, 03-29, 04-13) must not move the episode features of their first strike
    dp_loud = pd.concat([dp, mk([("2022-03-14 10:00", {"npp", "hpp"}, ["Lviv", "Odesa"], 6.0, 27000.0), ("2022-03-15 10:00", {"npp"}, ["Lviv"], 6.0, 27000.0),
                                ("2022-03-29 10:00", {"hpp"}, ["Lviv", "Odesa"], 6.0, 27000.0), ("2022-04-13 10:00", {"tpp"}, ["Lviv"], 6.0, 27000.0)])], ignore_index=True)
    out, strike_lvl = [], []
    for d in (dp, dp_loud):
        F = dd.duration_features(strikes, dd.LabelBook(posts, strikes, prm), None, d)
        strike_lvl.append(F)
        ep = strikes[mask].reset_index(drop=True)
        out.append(dd.episode_features(F, mask, ep, dd.LabelBook(posts, ep, prm)))
    a, b = out
    # episode 0 starts 03-12 and its chain has posts on 03-14 / 03-15: unchanged. Episode 1 (03-27, chain 03-29) likewise.
    pd.testing.assert_frame_equal(a.iloc[:2], b.iloc[:2])
    assert a["dmg_tpp"].iloc[0] == 1 and a["dmg_sub_hv"].iloc[1] == 1 and a["dmg_hpp"].iloc[1] == 0
    assert not strike_lvl[0].equals(strike_lvl[1]), "the later strikes of the chain do see the extra posts at strike level: the check is not vacuous"
    assert set(dd.DMG_F) <= set(a.columns)


def test_damage_tags_name_new_damage_only():
    from deficit.ingestion.damage_features import tag_text
    assert tag_text("Вночі російські ракети пошкодили ТЕС та ГЕС у двох областях.")["tpp"]
    t = tag_text("Вночі російські ракети пошкодили ТЕС та ГЕС у двох областях.")
    assert t["tpp"] and t["hpp"] and not t["sub_hv"]
    assert tag_text("Внаслідок атаки пошкоджено підстанцію 330 кВ.", "ukrenergo")["sub_hv"]
    assert tag_text("Через обстріли є нові пошкодження в мережах обленерго в Харківській області.")["distribution"]
    assert not any(tag_text("Після ремонту в мережу включено блок на ТЕС. Ремонтні бригади відновлюють пошкоджені лінії.").values())
    assert not any(tag_text("Нагадаємо, головна причина обмежень - наслідки ракетних ударів по ТЕС і ГЕС.").values())
    assert not any(tag_text("Через негоду пошкоджено підстанцію та лінію 110 кВ.").values())


def test_energy_map_test_walk_forward_ignores_labels_that_are_not_yet_known_at_the_issue_time():
    """`energy_map_test.walk_forward_chosen`: the forecast of strike i uses training labels as known just before its issue time.
    A restoration of an EARLIER strike that is recorded only after the issue time of strike i, and the final label of strike i
    itself, change none of its predictions; a restoration known before the issue time does (the check is not vacuous)."""
    from test_region_restoration import _episodes, _facts, _strikes
    from deficit.evaluation import energy_map_test as emt
    from deficit.evaluation import region_restoration as rr
    strikes = _strikes(16)
    facts = _facts(strikes, seed=3)
    tab = rr.build_table(_episodes(facts, strikes), facts, strikes, None, pd.Timestamp("2022-05-31", tz="UTC"), "30d")
    ids = sorted(tab["idx"].unique())
    i, j = ids[10], ids[9]
    issue_i = tab.loc[tab["idx"] == i, "issue"].iloc[0]
    model = "ridge_strike_oblast_a100"

    def run(res_at_shift, hours_after, own_hours=None):
        t = tab.copy()
        m = t["idx"] == j
        t.loc[m, "res_at"] = issue_i + pd.Timedelta(days=res_at_shift)
        t.loc[m, "res_hours"] = hours_after
        t.loc[m, "observed"] = True
        t.loc[m, "hours"] = hours_after
        if own_hours is not None:
            k = t["idx"] == i
            t.loc[k, "res_at"] = t.loc[k, "issue"] + pd.Timedelta(days=2)
            t.loc[k, "res_hours"] = own_hours
            t.loc[k, "hours"] = own_hours
            t.loc[k, "observed"] = True
        wf = emt.walk_forward_chosen(t[t["idx"] <= i], model, min_obs=5, min_oblast=2)
        out = wf[wf["idx"] == i].reset_index(drop=True)
        assert len(out) > 0 and out["evaluated"].all()
        return out[["oblast", "n_train", "n_train_observed", model, "oblast_median", "global_median"]]

    after_a, after_b = run(+3, 30.0), run(+9, 900.0)               # recorded after the issue time of strike i in both cases
    pd.testing.assert_frame_equal(after_a, after_b)
    pd.testing.assert_frame_equal(after_a, run(+3, 30.0, own_hours=1234.0))      # the label of the forecast strike itself
    before_a, before_b = run(-1, 30.0), run(-1, 900.0)             # recorded before the issue time (the day before)
    assert not before_a[[model, "oblast_median", "global_median"]].equals(before_b[[model, "oblast_median", "global_median"]])


def _em_improve_inputs():
    """12 strikes of one oblast every 3 days, restoration 40 h after each strike; blackout/restoration records on the day after every strike."""
    from deficit.evaluation.restoration import issue_time
    st = pd.date_range("2025-03-01 03:00", periods=12, freq="3D", tz="UTC")
    tab = pd.DataFrame({"strike_at": st, "idx": range(12), "oblast": "Sumy"})
    tab["issue"] = [issue_time(s) for s in st]
    tab["res_at"] = tab["strike_at"] + pd.Timedelta(hours=40)
    tab["res_hours"] = [30.0 + 7 * k for k in range(12)]
    days = [s.tz_convert(None).floor("D") for s in st]
    rec = pd.DataFrame({"date": [d + pd.Timedelta(days=1) for d in days], "oblast": "Sumy", "is_period": False, "has_blackout": True,
                        "has_restored": False, "blackout_consumers": 1000.0})
    return tab, rec, days


def test_energy_map_improve_features_use_only_what_is_known_at_the_issue_time():
    """`energy_map_improve.add_em_features`: records dated on or after the strike day and durations not yet known at the issue time
    (restoration after the issue time, the row's own label, later episodes) change none of a row's em_* features; records dated
    before the strike day and durations known before the issue time do (the check is not vacuous)."""
    from deficit.evaluation import energy_map_improve as ei
    tab, rec, days = _em_improve_inputs()
    i = 8
    D = days[i]
    base = ei.add_em_features(tab, rec).loc[i, ei.EM_F]

    def feats(t=tab, r=rec):
        return ei.add_em_features(t, r).loc[i, ei.EM_F]

    extra = pd.DataFrame({"date": [D, D + pd.Timedelta(days=1), D + pd.Timedelta(days=5)], "oblast": "Sumy", "is_period": False,
                          "has_blackout": True, "has_restored": True, "blackout_consumers": 9.9e6})
    pd.testing.assert_series_equal(base, feats(r=pd.concat([rec, extra], ignore_index=True)))       # records of the strike day or later
    late = rec[rec["date"] < D]
    pd.testing.assert_series_equal(base, feats(r=late))                                              # only records before the strike day matter
    early = pd.DataFrame({"date": [D - pd.Timedelta(days=2)], "oblast": "Sumy", "is_period": False, "has_blackout": True,
                          "has_restored": True, "blackout_consumers": 5e6})
    assert not base.equals(feats(r=pd.concat([rec, early], ignore_index=True)))                      # a record before the strike day changes them
    t2 = tab.copy()                                                                                  # own label and later episodes
    t2.loc[t2["idx"] >= i, "res_hours"] = 5000.0
    t2.loc[t2["idx"] >= i, "res_at"] = t2.loc[t2["idx"] >= i, "strike_at"] + pd.Timedelta(hours=10)
    pd.testing.assert_series_equal(base, feats(t=t2))
    t3 = tab.copy()                                                                                  # restoration of episode i-1 after the issue time of i
    t3.loc[t3["idx"] == i - 1, "res_at"] = tab.loc[i, "issue"] + pd.Timedelta(days=1)
    t4 = t3.copy()
    t4.loc[t4["idx"] == i - 1, "res_hours"] = 4000.0
    pd.testing.assert_series_equal(feats(t=t3), feats(t=t4))
    assert not feats(t=t3).equals(base)                                                              # ... but it is not counted as known
    t5 = tab.copy()
    t5.loc[t5["idx"] == i - 1, "res_hours"] = 4000.0                                                 # known before the issue time: changes the features
    assert not feats(t=t5).equals(base)


def test_alert_weather_features_use_only_the_strike_day_and_before():
    """`energy_map_alerts_weather`: alerts starting at or after the issue time, the part of a running alert after it, and weather of the days after the
    strike day change none of a row's al_* / w_* features; alerts / weather of the strike day and before do (the check is not vacuous)."""
    from deficit.evaluation import energy_map_alerts_weather as aw
    from deficit.ingestion.alerts import AlertIndex

    def T(s):
        return pd.Timestamp(s, tz="UTC")

    iss = T("2025-12-03")                                                    # strike day Dec 2
    tab = pd.DataFrame({"oblast": ["Kharkiv"], "issue": [iss]})
    base_iv = pd.DataFrame({"oblast": "Kharkiv", "start": [T("2025-12-02 10:00"), T("2025-11-30 22:00")], "end": [T("2025-12-02 12:00"), T("2025-12-01 01:00")]})
    days = pd.date_range("2025-11-25", "2025-12-10")
    wx = pd.DataFrame({"oblast": "Kharkiv", "date": days, "temperature_2m_min": np.linspace(-10, 5, len(days)), "wind_gusts_10m_max": 20.0, "snowfall_sum": 1.0})
    base_a = aw.alert_features(tab, AlertIndex(base_iv)).iloc[0]
    base_w = aw.weather_features(tab, wx).iloc[0]
    later = pd.concat([base_iv, pd.DataFrame({"oblast": "Kharkiv", "start": [iss, T("2025-12-03 08:00")], "end": [T("2025-12-03 02:00"), T("2025-12-03 09:00")]})], ignore_index=True)
    pd.testing.assert_series_equal(base_a, aw.alert_features(tab, AlertIndex(later)).iloc[0])                    # alerts at or after the issue time
    running = base_iv.copy()
    running.loc[0, "end"] = T("2025-12-09")                                                                    # the same alert lasting days after the issue time
    running = pd.concat([running, pd.DataFrame({"oblast": "Kharkiv", "start": [T("2025-12-02 23:00")], "end": [T("2025-12-05")]})], ignore_index=True)
    r = aw.alert_features(tab, AlertIndex(running)).iloc[0]
    assert r["al_h_day"] == 14.0 and r["al_n_day"] == 1.0                                                      # 10:00 .. 24:00 (14 h), the 23:00 alert merges into it, clipped at the issue time
    assert aw.alert_features(tab, AlertIndex(base_iv)).iloc[0]["al_h_day"] == 2.0
    pd.testing.assert_series_equal(base_a, aw.alert_features(tab, AlertIndex(base_iv[base_iv["start"] < iss])).iloc[0])
    earlier = pd.concat([base_iv, pd.DataFrame({"oblast": "Kharkiv", "start": [T("2025-12-01 12:00")], "end": [T("2025-12-01 15:00")]})], ignore_index=True)
    assert aw.alert_features(tab, AlertIndex(earlier)).iloc[0]["al_h_prev"] != base_a["al_h_prev"]            # an earlier alert does change them
    w2 = wx.copy()
    after = w2["date"] > pd.Timestamp("2025-12-02")
    w2.loc[after, ["temperature_2m_min", "wind_gusts_10m_max", "snowfall_sum"]] = [-40.0, 200.0, 50.0]         # weather of the days after the strike day
    pd.testing.assert_series_equal(base_w, aw.weather_features(tab, w2).iloc[0])
    w3 = wx.copy()
    w3.loc[w3["date"] == pd.Timestamp("2025-12-01"), "snowfall_sum"] = 9.0                                    # a day inside the 3-day window
    assert aw.weather_features(tab, w3).iloc[0]["w_snow_3d"] != base_w["w_snow_3d"]
    w4 = wx.copy()
    w4.loc[w4["date"] == pd.Timestamp("2025-11-29"), "temperature_2m_min"] = -40.0                            # older than 3 days: no effect
    pd.testing.assert_series_equal(base_w, aw.weather_features(tab, w4).iloc[0])
