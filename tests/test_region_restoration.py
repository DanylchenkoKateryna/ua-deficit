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
