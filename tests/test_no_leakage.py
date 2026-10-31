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
    methods = rr.REFERENCES + [d[0] for d in rr.designs([10.0])]
    checked = 0
    for variant in rr.VARIANTS:
        out = {}
        for name, f in (("base", facts), ("loud", loud)):
            tab = rr.build_table(_episodes(f, strikes), f, strikes, None, pd.Timestamp("2022-05-31", tz="UTC"), variant)
            out[name] = rr.run_walk_forward(tab[tab["idx"] <= i], [10.0], min_obs=5, min_oblast=2)
        a, b = (out[k][out[k]["idx"] == i].reset_index(drop=True) for k in ("base", "loud"))
        assert len(a) > 0 and a["evaluated"].all()
        cols = ["oblast", "issue", "n_train", "n_train_observed"] + rr.STRIKE_F + rr.SEVERITY_F + rr.ACTIVITY_F + methods
        pd.testing.assert_frame_equal(a[cols], b[cols])
        # the earlier strikes' final labels are also unchanged for the next-strike window (they end before the issue time)
        if variant == "next_strike":
            e0, e1 = (out[k][out[k]["idx"] < i].reset_index(drop=True) for k in ("base", "loud"))
            pd.testing.assert_series_equal(e0["hours"], e1["hours"])
        checked += len(a)
    assert checked > 0 and set(OBL) >= set(a["oblast"])
