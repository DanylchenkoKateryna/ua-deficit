"""Rules of the post-attack forecast: duration from flags, paired tests against persistence."""
import numpy as np
import pandas as pd

from deficit.evaluation.post_event import duration_from_flags, paired_tests


def test_duration_from_flags_needs_three_recovered_days_in_a_row():
    assert duration_from_flags(np.array([0, 0, 1, 1, 0, 1, 1, 1, 0])) == 5
    assert duration_from_flags(np.array([1, 1, 1, 0])) == 0
    assert duration_from_flags(np.array([0, 1, 1, 0, 1, 1])) == 6          # never recovers: all days count


def test_paired_tests_compare_each_method_with_persistence():
    rng = np.random.default_rng(0)
    n = 30
    actual = rng.uniform(1, 2, n)
    t = pd.DataFrame({"issue_day": pd.date_range("2024-01-01", periods=n).astype(str), "actual_depth": actual,
                      "actual_duration_days": 5, "persistence_depth": actual + 0.5, "persistence_duration_days": 5,
                      "persistence_traj_MAE": 1.0,
                      "xgb_rolling_depth": actual + 0.1 * rng.uniform(0.5, 1.5, n), "xgb_rolling_duration_days": 5,
                      "xgb_rolling_traj_MAE": 0.9})
    res = paired_tests(t).set_index(["method", "metric"])
    assert res.loc[("xgb_rolling", "depth"), "p_value"] < 0.001                    # clearly closer to the truth
    assert res.loc[("xgb_rolling", "depth"), "share_better"] == 1.0
    assert ("xgb_rolling", "duration") not in res.index                            # identical errors: nothing to test


# ---------------------------------------------------------------- direct targets and the leakage rule
from deficit.config import PostEvent  # noqa: E402
from deficit.evaluation.post_event import (  # noqa: E402
    DirectModel, depth_duration, direct_features, direct_forecast, direct_frame, direct_targets, known_rows, summarize)

XCFG = {"subsample": 0.8, "colsample_bytree": 0.8}


def _toy_daily(n=420, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    attack = (rng.random(n) < 0.06).astype(float)
    y = np.zeros(n)
    for i in range(1, n):                                   # strikes push the level up, it decays afterwards
        y[i] = 0.9 * y[i - 1] + 0.6 * attack[i] * rng.uniform(0.5, 1.5) + 0.05 * rng.random()
    return pd.DataFrame({"y": y, "temp": rng.normal(5, 8, n), "attack": attack, "through": attack * rng.integers(1, 50, n),
                         "dam": rng.uniform(0.3, 1.0, n)}, index=idx)


def test_direct_targets_use_the_window_after_the_issue_day():
    y = np.array([0.0, 5, 6, 4, 1, 1, 1, 9, 0, 0, 0, 0.0])
    T = direct_targets(y, H=5, thr=2.0, min_share=0.7)
    # issue day 0 sees y[1..5] = 5 6 4 1 1: never three calm days in a row inside the window -> censored at 5 days
    assert T.loc[0, "depth"] == 6 and T.loc[0, "duration"] == 5 and T.loc[0, "censored"] == 1 and T.loc[0, "ndays"] == 3
    # issue day 1 sees y[2..6] = 6 4 1 1 1: recovered after two days
    assert T.loc[1, "depth"] == 6 and T.loc[1, "duration"] == 2 and T.loc[1, "censored"] == 0
    assert T.loc[2, "depth"] == depth_duration(y[3:8], 2.0)[0] == 9
    assert T["end_i"].tolist() == [i + 5 for i in range(len(y))]
    assert T["depth"].iloc[len(y) - 5:].isna().all()          # the last H issue days have no complete window


def test_known_rows_require_the_whole_window_to_be_observed():
    d = _toy_daily()
    pe = PostEvent()
    frame = direct_frame(d, 30, 0.5, pe)
    for i0 in (100, 250, 400):
        k = known_rows(frame, i0)
        assert len(k) > 0
        assert (k["end_i"] <= i0).all() and k.index.max() == i0 - 30    # the newest usable example was issued H days ago


def test_direct_forecast_does_not_see_the_days_after_the_issue_day():
    d = _toy_daily()
    H, thr, i0 = 30, 0.5, 300
    pe = PostEvent(direct_n_estimators=40, direct_min_rows=50)
    base = direct_forecast(d, i0, H, thr, pe, XCFG, 0)
    base_t = direct_forecast(d, i0, H, thr, pe, XCFG, 0, feature_set="tension")
    assert np.isfinite(list(base.values()) + list(base_t.values())).all()

    # (a) only the first day after d0 is changed: every example whose window reaches it (issued after d0 - H) must be excluded
    d1 = d.copy()
    d1.iloc[i0 + 1, d1.columns.get_loc("y")] += 1000.0
    f_ok, f_bad = direct_frame(d, H, thr, pe), direct_frame(d1, H, thr, pe)
    assert not np.allclose(known_rows(f_ok, i0 + 1)["depth"], known_rows(f_bad, i0 + 1)["depth"])   # the change IS visible one day later
    assert direct_forecast(d1, i0, H, thr, pe, XCFG, 0) == base
    assert direct_forecast(d1, i0, H, thr, pe, XCFG, 0, feature_set="tension") == base_t

    # (b) everything after d0 is scrambled (all columns, including attacks, weapons, temperature and the day-ahead price)
    rng = np.random.default_rng(7)
    d2 = d.copy()
    for c in d2.columns:
        d2.iloc[i0 + 1:, d2.columns.get_loc(c)] = rng.permutation(d2[c].to_numpy()[i0 + 1:]) * 3.0 + 1.0
    assert direct_forecast(d2, i0, H, thr, pe, XCFG, 0) == base
    assert direct_forecast(d2, i0, H, thr, pe, XCFG, 0, feature_set="tension") == base_t

    # (c) features of day d0 use days <= d0 only
    f_a, f_b = direct_features(d, thr).iloc[i0], direct_features(d2, thr).iloc[i0]
    pd.testing.assert_series_equal(f_a, f_b)


def test_direct_model_predicts_integer_censoring_aware_durations():
    d = _toy_daily()
    H, thr, i0 = 30, 0.5, 350
    pe = PostEvent(direct_n_estimators=40)
    frame = direct_frame(d, H, thr, pe)
    sets = frame.attrs["feature_sets"]
    assert "dam7" in sets["tension"] and "dam7" not in sets["base"]          # tension features use the day-ahead price when it is there
    assert "dam7" not in direct_frame(d.drop(columns="dam"), H, thr, pe).attrs["feature_sets"]["tension"]
    m = DirectModel(known_rows(frame, i0), sets["tension"], H, pe, XCFG, 0)
    p = m.predict(frame.loc[[i0, i0 - 1]])
    assert ((p["duration"] >= 0) & (p["duration"] <= H) & (p["duration"] == p["duration"].round())).all()
    assert ((p["aft_duration"] >= 0) & (p["aft_duration"] <= H)).all() and ((p["ndays"] >= 0) & (p["ndays"] <= H)).all()
    assert np.isfinite(p[["ridge_depth", "depth"]].to_numpy()).all()


def test_summary_is_reported_per_split():
    t = pd.DataFrame({"trigger": ["attack"] * 4, "split": ["validation", "validation", "test", "test"],
                      "actual_depth": [1.0, 1, 1, 1], "persistence_depth": [0.0, 0, 0.5, 0.5], "actual_duration_days": [5, 5, 5, 5],
                      "persistence_duration_days": [0, 0, 5, 5], "actual_ndays": [4.0] * 4, "persistence_ndays": [0.0, 0, 4, 4]})
    s = summarize(t).set_index(["split", "trigger", "method"])
    assert s.loc[("validation", "attack", "persistence"), "depth_MAE"] == 1.0
    assert s.loc[("test", "attack", "persistence"), "depth_MAE"] == 0.5
    assert s.loc[("test", "attack", "persistence"), "duration_MAE_days"] == 0.0
    assert s.loc[("all", "attack", "persistence"), "deficit_days_MAE"] == 2.0
