"""Oblast-level restoration forecast: cohort, labels as known at the issue time, models, tests, no look-ahead."""
import numpy as np
import pandas as pd
import pytest

from deficit.evaluation import region_restoration as rr
from deficit.ingestion import telegram_regional as tr
from deficit.ingestion.telegram_extract import EpisodeParams

OBL = ["Sumy", "Lviv", "Kyiv"]
CH = {"c_sumy": "Sumy", "c_lviv": "Lviv"}


def T(s):
    return pd.Timestamp(s, tz="UTC")


def _fact(ob, when, **kw):
    base = {"channel": "c_" + ob.lower(), "post_id": 0, "datetime": when, "url": None, "oblast": ob, "source": "own", "attack": False,
            "consumers_without_power": np.nan, "schedules": False, "emergency": False, "lifted": False, "restoration_power": None,
            "consumers_restored": np.nan, "multi_region": False, "damaged_objects": [], "spans": "{}"}
    base.update(kw)
    return base


def _strikes(n=14):
    t = pd.date_range("2022-03-01 06:00", periods=n, freq="10D", tz="UTC")
    return pd.DataFrame({"strike_at": t, "major": True, "description": "x", "weapons_through": np.arange(n) * 3 % 17 + 2})


def _facts(strikes, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for st in strikes["strike_at"]:
        day = st.floor("D")
        for ob in OBL:
            if rng.random() < 0.15:
                continue
            pid += 1
            rows.append(_fact(ob, day + pd.Timedelta(hours=10), post_id=pid, attack=True, schedules=True,
                              consumers_without_power=float(rng.integers(1_000, 90_000)), damaged_objects=["substation"]))
            if rng.random() < 0.75:                                     # a lifting post some days later (before the next strike)
                pid += 1
                rows.append(_fact(ob, day + pd.Timedelta(days=int(rng.integers(1, 8)), hours=int(rng.integers(0, 20))), post_id=pid, lifted=True))
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def _loud(facts: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Everything at or after `cut` is replaced by loud posts (huge outages, restoration, lifting)."""
    keep = facts[facts["datetime"] < cut]
    extra = [_fact(ob, cut + pd.Timedelta(hours=h), post_id=10_000 + k, attack=True, schedules=True, emergency=True, lifted=True,
                   consumers_without_power=9e9, restoration_power="full", damaged_objects=["generation", "substation"])
             for k, (ob, h) in enumerate((o, h) for o in OBL for h in (0, 3, 30, 200))]
    return pd.concat([keep, pd.DataFrame(extra)], ignore_index=True).sort_values("datetime").reset_index(drop=True)


def _episodes(facts, strikes):
    first = {c: facts["datetime"].min() - pd.Timedelta(days=1) for c in CH}
    return tr.build_region_episodes(facts, strikes, CH, first, None, EpisodeParams())


def _table(facts, strikes, variant):
    eps = _episodes(facts, strikes)
    return rr.build_table(eps, facts, strikes, None, T("2022-05-31"), variant)


@pytest.fixture(scope="module")
def toy():
    strikes = _strikes()
    facts = _facts(strikes)
    return strikes, facts


def test_cohort_needs_an_impact_post_before_the_issue_time_and_no_resolution_yet(toy):
    strikes, facts = toy
    for v in rr.VARIANTS:
        tab = _table(facts, strikes, v)
        assert len(tab) > 10
        assert (tab["res_at"].isna() | (tab["res_at"] >= tab["issue"])).all()      # not resolved before the issue time
        assert (tab["observed"] == tab["res_at"].notna()).all()
        assert (tab[["hours"]] >= rr.MIN_H).all().all()
        assert set(tab["split"]) <= {"validation", "test"} and (tab.loc[tab["split"] == "validation", "strike_at"] <= T("2022-05-31 23:00")).all()


def test_labels_as_of_hide_resolutions_after_the_instant(toy):
    strikes, facts = toy
    tab = _table(facts, strikes, "30d")
    obs = tab[tab["observed"]].iloc[0]
    before = obs["res_at"] - pd.Timedelta(minutes=1)
    r = rr.labels_as_of(tab[tab.index == obs.name], before).iloc[0]
    assert not r["observed"] and r["hours"] == pytest.approx(max((before - obs["strike_at"]).total_seconds() / 3600, rr.MIN_H))
    r = rr.labels_as_of(tab[tab.index == obs.name], obs["res_at"]).iloc[0]
    assert r["observed"] and r["hours"] == pytest.approx(obs["res_hours"])


def test_the_next_strike_window_labels_are_final_before_the_next_issue_time(toy):
    """With the next-strike window an earlier episode never needs a post later than the day of the next strike."""
    strikes, facts = toy
    tab = _table(facts, strikes, "next_strike")
    nxt = dict(zip(strikes.index, strikes["strike_at"].shift(-1).dt.floor("D")))
    for r in tab.itertuples():
        end = nxt[r.idx]
        if pd.notna(end):
            assert r.res_at is pd.NaT or r.res_at <= end


def test_aft_recovers_a_censored_lognormal_law_better_than_the_observed_only_ridge():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    y = 3.0 + 0.8 * x + 0.5 * rng.normal(size=n)
    t = np.exp(y)
    c = np.exp(3.2 + 0.3 * rng.normal(size=n))                    # censoring times
    ev = t <= c
    obs = np.minimum(t, c)
    X = x[:, None]
    pred = rr.fit_aft(X, obs, ev, 0.01, ["log_through"])(X)
    # median of the true law is exp(3 + 0.8 x): the censoring-aware fit recovers it, the observed-only fit is biased low
    ridge = rr.fit_ridge(X, obs, ev, 0.01, ["log_through"])(X)
    truth = np.exp(3.0 + 0.8 * x)
    assert np.mean(np.abs(np.log(pred) - np.log(truth))) < np.mean(np.abs(np.log(ridge) - np.log(truth)))
    assert np.corrcoef(np.log(pred), x)[0, 1] > 0.95


def test_aft_intercept_only_recovers_the_median_of_a_heavily_censored_lognormal():
    """No covariates, 55 % censored: the AFT median is the median of the latent law (no intercept / scale bug); observed-only is biased low."""
    rng = np.random.default_rng(7)
    n = 800
    t = np.exp(3.0 + 1.0 * rng.normal(size=n))
    c = np.exp(2.9 + 0.2 * rng.normal(size=n))
    ev, obs = t <= c, np.minimum(t, c)
    X = np.zeros((n, 1))
    aft = rr.fit_aft(X, obs, ev, 1e-3, ["log_through"])(X[:1])[0]
    ridge = rr.fit_ridge(X, obs, ev, 1e-3, ["log_through"])(X[:1])[0]
    assert 0.35 < (~ev).mean() < 0.8
    assert abs(np.log(aft) - 3.0) < 0.2 and abs(np.log(aft) - 3.0) < abs(np.log(ridge) - 3.0)


def test_censoring_km_and_ipcw_weights():
    t = np.array([2.0, 3.0, 3.0, 5.0, 8.0])
    ev = np.array([True, False, True, True, False])
    G = rr.censor_survival(t, ev)
    # censor times 3 (at risk 4, 1 censored) and 8 (at risk 1): G(<=3)=1, G(5)=3/4, G(8)=3/4, G(9)=0
    np.testing.assert_allclose(G([1.0, 3.0, 3.5, 5.0, 8.0, 9.0]), [1.0, 1.0, 0.75, 0.75, 0.75, 0.0])
    assert rr.censor_survival(t, np.ones(5, bool))([1, 100]).tolist() == [1.0, 1.0]       # nothing censored -> weight 1


def test_ipcw_ridge_equals_ridge_without_censoring_and_upweights_long_observed_episodes():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.normal(size=(n, 1))
    t = np.exp(3 + 0.5 * x[:, 0] + 0.3 * rng.normal(size=n))
    all_ev = np.ones(n, bool)
    np.testing.assert_allclose(rr.fit_ipcw_ridge(x, t, all_ev, 1.0, ["log_through"])(x), rr.fit_ridge(x, t, all_ev, 1.0, ["log_through"])(x))
    # administrative censoring at 25 h drops the long episodes; IPCW (G only known up to 25 h, floored) lifts the fit above observed-only
    c = np.full(n, 25.0)
    ev, obs = t <= c, np.minimum(t, c)
    w = rr.fit_ipcw_ridge(x, obs, ev, 1.0, ["log_through"])(x)
    r = rr.fit_ridge(x, obs, ev, 1.0, ["log_through"])(x)
    assert np.all(np.isfinite(w)) and np.mean(np.log(w)) >= np.mean(np.log(r)) - 1e-9


def test_xgb_aft_recovers_a_censored_lognormal_law_better_than_the_observed_only_ridge():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    t = np.exp(3.0 + 0.8 * x + 0.5 * rng.normal(size=n))
    c = np.exp(3.2 + 0.3 * rng.normal(size=n))
    ev, obs = t <= c, np.minimum(t, c)
    X = x[:, None]
    p = rr.fit_xgb_aft(X, obs, ev, 2, 60, ["log_through"], sigma=0.5)(X)
    ridge = rr.fit_ridge(X, obs, ev, 1.0, ["log_through"])(X)
    truth = 3.0 + 0.8 * x
    assert np.corrcoef(np.log(p), x)[0, 1] > 0.9
    assert np.mean(np.abs(np.log(p) - truth)) < np.mean(np.abs(np.log(ridge) - truth))
    # deterministic and bounded
    np.testing.assert_allclose(p, rr.fit_xgb_aft(X, obs, ev, 2, 60, ["log_through"], sigma=0.5)(X))
    assert (p >= rr.MIN_H).all() and (p <= rr.HORIZON_H).all()


def test_walk_forward_uses_only_earlier_strikes_and_all_methods_are_finite(toy):
    strikes, facts = toy
    for v in rr.VARIANTS:
        tab = rr.run_walk_forward(_table(facts, strikes, v), [10.0], min_obs=5, min_oblast=2)
        ev = tab[tab["evaluated"]]
        assert len(ev) > 5 and (ev["n_train"] > 0).all()
        methods = rr.REFERENCES + [d[0] for d in rr.designs([10.0])]
        assert np.isfinite(ev[methods].to_numpy(float)).all()
        assert tab.loc[~tab["evaluated"], methods].isna().all().all()
        first = ev["idx"].min()
        assert tab[tab["idx"] < first]["evaluated"].eq(False).all()


def test_scores_choice_and_tests_run(toy):
    strikes, facts = toy
    tab = rr.run_walk_forward(_table(facts, strikes, "30d"), [10.0], min_obs=5, min_oblast=2)
    methods = rr.REFERENCES + [d[0] for d in rr.designs([10.0])]
    sc = rr.score_table(tab, methods)
    assert {"validation", "test"} >= set(sc["split"]) and {"MAE_h", "MAE_per_strike_h", "C_within_strike"} <= set(sc.columns)
    ch = rr.choose_on_validation(sc)
    assert ch["baseline"] in rr.BASELINES and "_a" in ch["overall"]
    t = rr.paired_tests(tab, [(ch["overall"], ch["baseline"])], n_boot=50)
    assert {"p_episodes", "p_per_strike", "boot_ci95_lo_h"} <= set(t.columns)


def test_issue_features_ignore_facts_at_or_after_the_issue_time(toy):
    strikes, facts = toy
    st = strikes["strike_at"].iloc[7]
    issue = rr.issue_time(st)
    base = rr.issue_features(facts[facts["oblast"] == "Sumy"], facts, st, issue)
    pert = rr.issue_features(_loud(facts, issue)[lambda d: d["oblast"] == "Sumy"], _loud(facts, issue), st, issue)
    assert base == pert
    assert base["sev_posts"] >= 0 and set(rr.SEVERITY_F + rr.ACTIVITY_F) == set(base)
    # a post one second before the issue time counts, one exactly at the issue time does not
    inside = pd.concat([facts, pd.DataFrame([_fact("Sumy", issue - pd.Timedelta(seconds=1), post_id=99_001, attack=True,
                                                    consumers_without_power=5e5, emergency=True)])], ignore_index=True)
    f = rr.issue_features(inside[inside["oblast"] == "Sumy"], inside, st, issue)
    assert f["sev_emergency"] == 1.0 and f["sev_log_consumers"] == pytest.approx(np.log1p(5e5))
    at = pd.concat([facts, pd.DataFrame([_fact("Sumy", issue, post_id=99_002, attack=True, consumers_without_power=5e5, emergency=True)])],
                   ignore_index=True)
    assert rr.issue_features(at[at["oblast"] == "Sumy"], at, st, issue) == base


def test_strike_table_temperature_uses_only_the_24h_before_the_issue_time(toy):
    strikes, _ = toy
    idx = pd.date_range("2022-02-01", "2022-08-01", freq="h", tz="UTC")
    w = pd.DataFrame({"temperature_C": np.arange(len(idx), dtype=float)}, index=idx)
    base = rr.strike_table(strikes, w)
    w2 = w.copy()
    k = 4
    w2.loc[w2.index >= base.loc[k, "issue"], "temperature_C"] = 1e6
    again = rr.strike_table(strikes, w2)
    assert base.loc[k, "temp_24h"] == again.loc[k, "temp_24h"] < 1e5
    assert again.loc[k + 1, "temp_24h"] != base.loc[k + 1, "temp_24h"]      # the change is visible one strike later


def _hist_row(tab, ob, k):
    g = tab[tab["oblast"] == ob].sort_values("idx").reset_index(drop=True)
    return g, g.iloc[k]


def test_hist_features_definition_and_missing_indicator(toy):
    strikes, facts = toy
    for v in rr.VARIANTS:
        tab = _table(facts, strikes, v)
        assert set(rr.HIST_F) <= set(tab.columns)
        assert "strike_oblast_hist" in rr.feature_sets() and "ridge_strike_oblast_hist_a10" in [d[0] for d in rr.designs([10.0])]
        assert "ridge_strike_oblast_a10" in [d[0] for d in rr.designs([10.0])]           # existing designs are kept
        for ob in OBL:
            g = tab[tab["oblast"] == ob].sort_values("idx").reset_index(drop=True)
            if g.empty:
                continue
            first = g.iloc[0]
            assert first["hist_has_prev"] == 0 and first["hist_n_prior"] == 0 and first["hist_prev_observed"] == 0
            assert np.isnan(first["hist_prev_log_hours"]) and np.isnan(first["hist_days_since_prev"])
            for k in range(1, len(g)):
                r, q = g.iloc[k], g.iloc[k - 1]
                assert r["hist_has_prev"] == 1 and r["hist_n_prior"] == k
                assert r["hist_days_since_prev"] == pytest.approx(min((r["strike_at"] - q["strike_at"]).days + 0.0, 60.0), abs=1.0)
                as_of = r["issue"] - pd.Timedelta(microseconds=1)
                lab = rr.labels_as_of(g.iloc[[k - 1]], as_of).iloc[0]
                assert bool(r["hist_prev_observed"]) == bool(lab["observed"])
                assert r["hist_prev_log_hours"] == pytest.approx(np.log(lab["hours"]))


def test_hist_prev_open_episode_uses_elapsed_time_not_the_final_label():
    """30-day window: the previous episode of the oblast is resolved only AFTER the next issue time -> elapsed time, flag 0."""
    strikes = _strikes(4)
    s0, s1 = strikes["strike_at"].iloc[0], strikes["strike_at"].iloc[1]
    d0, d1 = s0.floor("D"), s1.floor("D")
    rows = [_fact("Sumy", d0 + pd.Timedelta(hours=10), post_id=1, attack=True, schedules=True, consumers_without_power=5e4),
            _fact("Sumy", d1 + pd.Timedelta(hours=10), post_id=2, attack=True, schedules=True, consumers_without_power=5e4),
            _fact("Sumy", d1 + pd.Timedelta(days=3), post_id=3, lifted=True)]       # lifts the 1st episode after strike 2's issue time
    f = pd.DataFrame(rows)
    f["datetime"] = pd.to_datetime(f["datetime"], utc=True)
    tab = _table(f, strikes, "30d")
    r = tab[(tab["oblast"] == "Sumy") & (tab["idx"] == 1)].iloc[0]
    assert r["hist_prev_observed"] == 0
    assert r["hist_prev_log_hours"] == pytest.approx(np.log((r["issue"] - pd.Timedelta(microseconds=1) - s0).total_seconds() / 3600))
    assert r["hist_under_restr"] == 1.0                                              # schedules of strike 1 not lifted before day 2


def test_hist_under_restr_needs_a_restriction_not_lifted_before_the_strike_day():
    strikes = _strikes(3)
    s1 = strikes["strike_at"].iloc[1]
    d0, d1 = strikes["strike_at"].iloc[0].floor("D"), s1.floor("D")
    base = [_fact("Sumy", d0 + pd.Timedelta(hours=10), post_id=1, attack=True, schedules=True, consumers_without_power=5e4),
            _fact("Sumy", d1 + pd.Timedelta(hours=10), post_id=3, attack=True, schedules=True, consumers_without_power=5e4)]
    lift = _fact("Sumy", d0 + pd.Timedelta(days=2), post_id=2, lifted=True)
    late = _fact("Sumy", d1 + pd.Timedelta(days=5), post_id=4, lifted=True)          # after the strike day: must not matter
    for extra, want in (([late], 1.0), ([lift, late], 0.0)):
        f = pd.DataFrame(base + extra)
        f["datetime"] = pd.to_datetime(f["datetime"], utc=True)
        tab = _table(f, strikes, "30d")
        assert tab[(tab["oblast"] == "Sumy") & (tab["idx"] == 1)].iloc[0]["hist_under_restr"] == want


def test_prep_fills_hist_missing_values_from_training_only():
    cols = ["hist_has_prev", "hist_prev_log_hours"]
    Xtr = np.array([[1.0, 2.0], [1.0, 4.0], [0.0, np.nan], [1.0, 6.0]])
    pr = rr._Prep(Xtr, cols)
    z = pr(np.array([[0.0, np.nan]]))
    assert z[0, 0] == 0.0                                                         # the indicator stays 0/1
    assert z[0, 1] == pytest.approx((4.0 - 4.0) / pr.sd[1])                         # NaN -> train median (4), centred
    z2 = rr._Prep(np.vstack([Xtr, [[1.0, 1000.0]]]), cols)(np.array([[0.0, np.nan]]))
    assert z2[0, 1] != z[0, 1]                                                     # fill depends on the data it was fitted on only


# ---------------------------------------------------------------------------------- dev mode (test blind)
def _dev(tab, k=3):
    from deficit.evaluation import region_dev as rd
    return rd.dev_cv(tab, [10.0], min_obs=5, min_oblast=2, k=k)


def test_dev_cv_never_touches_test_rows(toy):
    strikes, facts = toy
    for v in rr.VARIANTS:
        tab = _table(facts, strikes, v)
        assert (tab["split"] == "test").any() and (tab["split"] == "validation").any()
        junk = tab.copy()
        m = junk["split"] == "test"
        num = [c for c in junk.columns if pd.api.types.is_float_dtype(junk[c])]
        junk.loc[m, num] = np.nan
        junk.loc[m, "hours"] = 1e9
        junk.loc[m, "observed"] = ~junk.loc[m, "observed"]
        a, b = _dev(tab), _dev(junk)
        pd.testing.assert_frame_equal(a["table"], b["table"])
        pd.testing.assert_frame_equal(a["fold_table"], b["fold_table"])
        assert a["selection"] == b["selection"]
        assert set(a["table"].columns).isdisjoint({"split"}) and a["info"]["last_scored"] <= T("2022-05-31 23:00")


def test_dev_folds_are_consecutive_and_selection_follows_the_one_se_rule(toy):
    from deficit.evaluation import region_dev as rd
    strikes, facts = toy
    res = _dev(_table(facts, strikes, "30d"), k=3)
    folds = res["info"]["folds"]
    flat = [i for f in folds for i in f]
    assert flat == sorted(flat) and 1 <= len(folds) <= 3
    t, s = res["table"], res["selection"]
    sel = t.set_index("method").loc[s["selected"], "MAE_cv_h"]
    assert sel <= s["threshold"] + 1e-9 and s["selected"] not in rd.REFERENCE_ONLY
    assert rd.complexity_key("global_median") < rd.complexity_key("ridge_strike_a10") < rd.complexity_key("ridge_strike_oblast_a10") \
        < rd.complexity_key("ridge_strike_oblast_hist_a10") < rd.complexity_key("ridge_full_a10")


def test_one_se_prefers_the_simpler_design_inside_the_band():
    from deficit.evaluation import region_dev as rd
    t = pd.DataFrame({"method": ["ridge_full_a1", "ridge_strike_a10", "global_median", "km_global"],
                      "MAE_cv_h": [10.0, 11.0, 14.0, 5.0], "MAE_cv_se_h": [2.0, 2.0, 2.0, 1.0]})
    assert rd.select_one_se(t)["selected"] == "ridge_strike_a10"          # km_global is reference-only; baseline is outside 12 h
    t.loc[2, "MAE_cv_h"] = 11.9
    assert rd.select_one_se(t)["selected"] == "global_median"


def test_extra_designs_walk_forward_and_ensembles(toy):
    from deficit.evaluation import region_dev as rd
    strikes, facts = toy
    for v in rr.VARIANTS:
        tab = _table(facts, strikes, v)
        wf = rr.run_walk_forward(tab, [10.0], min_obs=5, min_oblast=2, extra=True)
        ev = wf[wf["evaluated"]]
        methods = rr.all_methods([10.0], extra=True)
        assert len(methods) > len(rr.all_methods([10.0])) and np.isfinite(ev[methods].to_numpy(float)).all()
        base = rr.run_walk_forward(tab, [10.0], min_obs=5, min_oblast=2)                       # original designs are unchanged by `extra`
        cols = rr.all_methods([10.0])
        pd.testing.assert_frame_equal(base[cols], wf[cols])
        res = _dev(tab)
        t = res["table"].set_index("method")
        assert {"ens_fixed", "ens_best"} <= set(t.index) and "C_all_pairs" in t.columns
        mem = res["info"]["ensemble_members"]["ens_best"]
        assert mem[0] == "geo_mean" and mem[1].startswith("ridge_") and mem[2].split("_")[0] in ("ipcw", "xgbaft", "aft")
    assert rd.complexity_key("ridge_strike_oblast_hist_a10") < rd.complexity_key("ipcw_strike_a10") < rd.complexity_key("xgbaft_full_d1n25")         < rd.complexity_key("xgbaft_full_d2n60") < rd.complexity_key("ens_fixed") < rd.complexity_key("ens_best")
    assert rd.complexity_key("aft_full_a10") < rd.complexity_key("ipcw_strike_a10")


# ---------------------------------------------------------------------------------- dev binary long-outage task
def test_binary_labels_censoring_before_and_after_T():
    from deficit.evaluation import region_binary as rb
    h = np.array([10.0, 72.0, 100.0, 50.0, 72.0, 90.0])
    ob = np.array([True, True, True, False, False, False])
    y = rb.binary_labels(h, ob, 72.0)
    # observed 10 -> short; observed exactly T -> short; observed 100 -> long; censored at 50 (< T) -> unknown;
    # censored exactly at T -> unknown; censored at 90 (> T) -> known long
    assert y[0] == 0 and y[1] == 0 and y[2] == 1 and np.isnan(y[3]) and np.isnan(y[4]) and y[5] == 1


def test_binary_training_labels_are_those_known_at_the_issue_time():
    from deficit.evaluation import region_binary as rb
    s0 = T("2022-03-01 06:00")
    tab = pd.DataFrame({"strike_at": [s0] * 4,
                        "res_at": [s0 + pd.Timedelta(hours=10), s0 + pd.Timedelta(hours=100), pd.NaT, pd.NaT],
                        "res_hours": [10.0, 100.0, np.nan, np.nan], "end_hours": [200.0, 200.0, 200.0, 40.0]})
    as_of = s0 + pd.Timedelta(hours=80)
    tr_ = rr.labels_as_of(tab, as_of)
    y = rb.binary_labels(tr_["hours"], tr_["observed"], 72.0)
    # row 1 is resolved at 100 h, i.e. AFTER as_of: it is censored at 80 h (> T, known long), not an observed 100 h; row 3 ends at 40 h < T: unknown
    assert list(tr_["observed"]) == [True, False, False, False] and list(tr_["hours"]) == [10.0, 80.0, 80.0, 40.0]
    assert y[0] == 0 and y[1] == 1 and y[2] == 1 and np.isnan(y[3])
    early = rb.binary_labels(*(lambda r: (r["hours"], r["observed"]))(rr.labels_as_of(tab, s0 + pd.Timedelta(hours=50))), 72.0)
    assert early[0] == 0 and np.isnan(early[1]) and np.isnan(early[2]) and np.isnan(early[3])      # nothing is known long before T


def test_binary_walk_forward_ignores_later_strikes_and_labels(toy):
    from deficit.evaluation import region_binary as rb
    strikes, facts = toy
    tab = _table(facts, strikes, "30d")
    dev = tab[tab["split"] == "validation"].copy()
    ids = sorted(dev["idx"].unique())[6:]
    a = rb.binary_walk_forward(dev, ids[:4], [10.0], (24.0, 72.0), 2)
    junk = dev.copy()
    later = junk["idx"] > ids[3]
    junk.loc[later, "hours"] = 1e6
    junk.loc[later, "observed"] = True
    b = rb.binary_walk_forward(junk, ids[:4], [10.0], (24.0, 72.0), 2)
    for T in a:
        pc = [c for c in a[T].columns if c != "y"]
        pd.testing.assert_frame_equal(a[T][pc], b[T][pc])
        assert np.isfinite(a[T][pc].to_numpy(float)).all()


def test_auc_and_brier_helpers():
    from deficit.evaluation import region_binary as rb
    y = np.array([0, 0, 1, 1, 1])
    assert rb.auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0
    assert rb.auc(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == 0.0
    assert rb.auc(y, np.ones(5)) == 0.5
    assert np.isnan(rb.auc(np.ones(5), np.arange(5)))
    s = np.array([0.2, 0.1, 0.4, 0.4, 0.9])                                     # one tie across classes: pairs (0.4 vs 0.1/0.2 ...)
    pairs = [(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]]
    assert rb.auc(y, s) == pytest.approx(np.mean(pairs))
    assert rb.brier(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    assert rb.brier(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.25)


def test_block_bootstrap_matches_direct_computation_and_ranking_and_selection(toy):
    from deficit.evaluation import region_binary as rb
    from deficit.evaluation import region_dev as rd
    rng = np.random.default_rng(0)
    n = 60
    y = (rng.random(n) < 0.4).astype(float)
    s = y + rng.normal(0, 1, n)
    g = np.repeat(np.arange(12), 5)
    A = rb._auc_blocks(y, s, g, 12)
    D = np.outer(np.bincount(g, weights=y, minlength=12), np.bincount(g, weights=1 - y, minlength=12))
    assert A.sum() / D.sum() == pytest.approx(rb.auc(y, s))
    cnt = rng.multinomial(12, np.full(12, 1 / 12), size=5).astype(np.float32)
    got = rb._ratio_boot(A, D.astype(np.float32), cnt)
    for c, v in zip(cnt, got):
        rows = np.concatenate([np.flatnonzero(g == k) for k in range(12) for _ in range(int(c[k]))])
        assert v == pytest.approx(rb.auc(y[rows], s[rows]), abs=1e-5)
    strikes, facts = toy
    res = rd.dev_cv(_table(facts, strikes, "30d"), [10.0], 5, 2, 3, thresholds=(24.0, 72.0), n_boot=200)
    for T, b in res["binary"].items():
        t = b["table"].set_index("method")
        assert t.loc["bin_const", "BSS_fold_mean"] == pytest.approx(0.0)
        assert b["info"]["n_labelled"] == b["info"]["n_long"] + b["info"]["n_short"] and b["info"]["n_unknown"] >= 0
        s_ = b["selection"]
        assert s_["selected"] in t.index and t.loc[s_["selected"], "BSS_fold_mean"] >= s_["threshold"] - 1e-9
        assert rb.bin_complexity_key("bin_const") < rb.bin_complexity_key("bin_oblast") < rb.bin_complexity_key("logit_strike_a10") \
            < rb.bin_complexity_key("logit_strike_oblast_a10") < rb.bin_complexity_key("logit_full_a10")
    rk = res["ranking"].set_index("method")
    assert (rk["C_within_lo"] <= rk["C_within_hi"]).all()
    assert rk.loc["global_median", "C_within"] == pytest.approx(0.5)      # constant inside a strike; C_all of it mixes strikes (time drift)
    ev_c = res["table"].set_index("method")["C_all_pairs"]
    assert rk.loc["oblast_median", "C_all"] == pytest.approx(ev_c["oblast_median"], abs=1e-6)
    assert res["table"].set_index("method")["C_cv"].notna().any()


def test_one_se_binary_prefers_simpler_within_band():
    from deficit.evaluation import region_binary as rb
    t = pd.DataFrame({"method": ["bin_const", "logit_full_a1", "logit_strike_a10"], "BSS_fold_mean": [0.0, 0.10, 0.07],
                      "BSS_fold_se": [0.0, 0.05, 0.05]})
    assert rb.select_binary(t)["selected"] == "logit_strike_a10"
    t.loc[1, "BSS_fold_mean"] = 0.13
    assert rb.select_binary(t)["selected"] == "logit_full_a1"          # threshold 0.13 - 0.05 = 0.08 > 0.07
    t.loc[1, "BSS_fold_mean"] = 0.04
    t.loc[2, "BSS_fold_mean"] = 0.04
    assert rb.select_binary(t)["selected"] == "bin_const"


def test_early_features_use_only_posts_between_the_day_start_and_the_cut(toy):
    day = T("2022-06-01")
    st = day + pd.Timedelta(hours=6)
    issue = rr.issue_time(st)
    f = pd.DataFrame([_fact("Sumy", day - pd.Timedelta(minutes=1), post_id=1, attack=True, consumers_without_power=9e9),      # before the day
                      _fact("Sumy", day + pd.Timedelta(hours=1), post_id=2, attack=True, consumers_without_power=1000.0, damaged_objects=["substation"]),
                      _fact("Sumy", day + pd.Timedelta(hours=8), post_id=3, emergency=True, consumers_without_power=5e5, damaged_objects=["generation"]),
                      _fact("Sumy", day + pd.Timedelta(hours=9), post_id=4, attack=True, consumers_without_power=9e9, damaged_objects=["x"]),   # at the cut (K=3)
                      _fact("Sumy", day + pd.Timedelta(hours=20), post_id=5, schedules=True)])
    f["datetime"] = pd.to_datetime(f["datetime"], utc=True)
    a = rr.early_features(f, st, issue, 3.0, True)
    assert a == {"early_log_consumers": pytest.approx(np.log1p(5e5)), "early_attack_posts": 1, "early_impact_posts": 2, "early_object_kinds": 2,
                 "early_emergency_schedule": 1.0, "early_any_impact": 1.0, "early_first_impact_h": 1.0, "early_own_channel": 1.0}
    b = rr.early_features(f, st, issue, 6.0, False)                                            # K = 6 reads the post at 09:00 too
    assert b["early_attack_posts"] == 2 and b["early_object_kinds"] == 3 and b["early_log_consumers"] == pytest.approx(np.log1p(9e9))
    assert b["early_own_channel"] == 0.0
    none = rr.early_features(f.iloc[:0], st, issue, 3.0, False)                                  # nothing: window length as delay, flag 0
    assert none["early_any_impact"] == 0.0 and none["early_first_impact_h"] == 9.0 and none["early_log_consumers"] == 0.0
    late = rr.early_features(f, day + pd.Timedelta(hours=22), issue, 6.0, True)                  # the cut is clipped at the issue time
    assert late["early_first_impact_h"] == 1.0 and late["early_impact_posts"] == 4


def test_early_columns_in_the_table_and_designs_are_finite(toy):
    strikes, facts = toy
    eps = _episodes(facts, strikes)
    t3 = rr.build_table(eps, facts, strikes, None, T("2022-05-31"), "30d", 3.0)
    t6 = rr.build_table(eps, facts, strikes, None, T("2022-05-31"), "30d", 6.0)
    assert t3[rr.EARLY_F].notna().all().all() and (t3["early_own_channel"] == (t3["coverage"] == "own_channel")).all()
    assert (t6["early_impact_posts"] >= t3["early_impact_posts"]).all()          # a longer window can only add posts
    assert set(rr.feature_sets()["strike_oblast_hist_early"]) == set(rr.STRIKE_F + rr.dummies() + rr.HIST_F + rr.EARLY_F)
    wf = rr.run_walk_forward(t6, [10.0], min_obs=5, min_oblast=2, extra=True)
    names = [m for m in rr.all_methods([10.0], True) if "early" in m]
    assert len(names) >= 8 and np.isfinite(wf.loc[wf["evaluated"], names].to_numpy(float)).all()
