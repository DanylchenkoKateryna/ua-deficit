"""Final test of the Energy Map restoration labels (FROZEN protocol: docs/preregistration_energy_map.md).

Step A (`select_for_test`, dev only): dev CV of the cohort 'all oblasts', window 30 d; model = the simplest regression design whose mean
C within strike (5 folds) is within one SE of the best (SE over folds of the best design).  Written to
reports/<scenario>/energy_map_chosen_for_test.md BEFORE any test metric exists.  No test-period strike enters the dev tables.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion import energy_map as em
from . import energy_map_labels as eml
from . import region_dev as rd
from . import region_restoration as rr

log = logging.getLogger(__name__)

CHOSEN_FILE = "energy_map_chosen_for_test.md"
RULE = ("Protocol rule: candidates are the regression variants of the development mode (ridge, ipcw, aft, xgbaft, ensembles; baselines are not "
        "selected); criterion is the mean within-strike C over 5 consecutive folds, cohort 'all oblasts', 30-day window; among the variants no "
        "worse than the best by more than one SE (SE over the folds of the best), the simplest is taken (`complexity_key` in `region_dev.py`).")


def _read_inputs(cfg: Config):
    from ..ingestion.telegram_extract import load_strikes
    root = cfg._abs(cfg.paths.data_dir)
    pdir = cfg.telegram_processed_dir
    strikes = load_strikes(cfg.strike_list_path)
    facts = pd.read_parquet(pdir / "telegram_region_posts.parquet")
    wpath = root / "raw" / "weather.parquet"
    weather = pd.read_parquet(wpath) if wpath.exists() else None
    eps = eml.episodes_for_cfg(cfg, strikes, "energy_map")
    return strikes, facts, weather, eps


def select_one_se_c(table: pd.DataFrame) -> dict:
    """Max mean C_within (C_cv) among the regression designs, one-SE rule over folds, simplest by `complexity_key`."""
    base = set(rd.SELECTABLE_BASELINES) | set(rd.REFERENCE_ONLY)
    cand = table[~table["method"].isin(base) & table["C_cv"].notna()]
    best = cand.loc[cand["C_cv"].idxmax()]
    se = float(best["C_cv_se"]) if pd.notna(best["C_cv_se"]) else 0.0
    thr = float(best["C_cv"]) - se
    within = cand[cand["C_cv"] >= thr - 1e-12]
    pick = min(within.itertuples(), key=lambda r: rd.complexity_key(r.method, r.MAE_cv_h))
    return {"best": str(best["method"]), "best_c": float(best["C_cv"]), "best_se": se, "threshold": thr, "n_within": len(within),
            "within": [str(m) for m in within["method"]], "selected": str(pick.method), "selected_c": float(pick.C_cv),
            "selected_se": float(pick.C_cv_se)}


def select_for_test(cfg: Config, k: int | None = None) -> dict:
    tc = cfg.telegram
    k = int(k or tc.region_cv_folds)
    strikes, facts, weather, eps = _read_inputs(cfg)
    vend = pd.Timestamp(tc.region_validation_end, tz="UTC")
    cut = vend + pd.Timedelta(hours=23)
    sdev = strikes[pd.to_datetime(strikes["strike_at"], utc=True) <= cut].reset_index(drop=True)      # test-period strikes never enter
    eps = eps[pd.to_datetime(eps["strike_at"], utc=True) <= cut]
    tab = rr.build_table(eps, facts, sdev, weather, vend, "30d", tc.region_early_hours)
    tab = tab[tab["split"] == "validation"]
    res = eml.dev_cv_for_subset(tab, tc, k)
    if res.get("skipped"):
        raise RuntimeError(f"dev CV skipped: {res}")
    t = res["table"]
    sel = select_one_se_c(t)
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    keep = sorted(set(sel["within"]) | {sel["best"], sel["selected"], "oblast_median", "global_median"},
                  key=lambda m: -float(t.set_index("method").loc[m, "C_cv"]))
    ti = t.set_index("method")
    L = ["# Energy Map: model chosen in the development mode for the final test", "",
         "This file is written before any test code is run; no test metrics exist at the time of writing. "
         "Protocol: `docs/preregistration_energy_map.md` (frozen).", "", RULE, "",
         f"Development mode: cohort 'all oblasts', 30-day window, K = {k} folds; {res['info']['n_dev_episodes']} pairs "
         f"({res['info']['n_dev_observed']} observed); {res['info']['n_strikes_scored']} of {res['info']['n_strikes_total']} strikes scored.", "",
         f"- Best by mean C: `{sel['best']}`, C = {sel['best_c']:.4f}, SE = {sel['best_se']:.4f}.",
         f"- 1-SE threshold: C >= {sel['threshold']:.4f}; variants within the threshold: {sel['n_within']}.",
         f"- **Model chosen for the test: `{sel['selected']}`**, development C = {sel['selected_c']:.4f} (SE over folds {sel['selected_se']:.4f}).",
         "- Test forecasts: only this model + `oblast_median` + `global_median`.", "",
         "Variants within 1 SE (and baselines for reference), by decreasing C:", "",
         "| design | C_cv | SE | MAE_cv h |", "|---|---|---|---|"]
    for m in keep:
        x = ti.loc[m]
        L.append(f"| {m} | {x.C_cv:.4f} | {x.C_cv_se:.4f} | {x.MAE_cv_h:.1f} |")
    (d / CHOSEN_FILE).write_text("\r\n".join(L), encoding="utf-8")
    t.to_csv(d / "energy_map_dev_all_oblasts_30d.csv", index=False)
    log.info("selected %s (best %s)", sel["selected"], sel["best"])
    return sel


# --------------------------------------------------------------------------------------------------
# Step B/C: the test (ONE model chosen in `select_for_test` + oblast_median + global_median)
# --------------------------------------------------------------------------------------------------
BASE_METHODS = ("oblast_median", "global_median")


def chosen_model(path: Path) -> str:
    """The model name written by `select_for_test` (the line 'Model chosen for the test: `name`'); reports written by
    older versions in Ukrainian are read too."""
    m = re.search(r"(?:Model chosen for the test|\u0412\u0438\u0431\u0440\u0430\u043d\u0430 \u043c\u043e\u0434\u0435\u043b\u044c "
                  r"\u0434\u043b\u044f \u0442\u0435\u0441\u0442\u0443): `([^`]+)`", Path(path).read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"no chosen model in {path}")
    return m.group(1)


def forecast_chosen(train: pd.DataFrame, test: pd.DataFrame, model: str, min_oblast: int) -> pd.DataFrame:
    """Predictions of `model` (ridge_/aft_<set>_a<alpha>) and the two baselines for `test` from `train` (labels as known at the issue time).
    Same computation as `region_restoration.forecast_strike` for these methods."""
    m = re.fullmatch(r"(ridge|aft)_(.+)_a([0-9.eE+-]+)", model)
    if not m or m.group(2) not in rr.feature_sets():
        raise ValueError(f"unsupported model {model!r}: the test path handles ridge/aft designs")
    fam, fs, alpha = m.group(1), m.group(2), float(m.group(3))
    t, ev = train["hours"].to_numpy(float), train["observed"].to_numpy(bool)
    glob = float(np.median(t[ev]))
    obs_by = train[train["observed"]].groupby("oblast")["hours"].agg(["median", "size"])
    out = pd.DataFrame(index=test.index)
    out["global_median"] = glob
    out["oblast_median"] = [float(obs_by.loc[o, "median"]) if o in obs_by.index and obs_by.loc[o, "size"] >= min_oblast else glob
                            for o in test["oblast"]]
    cols = rr.feature_sets()[fs]
    fit = rr.fit_aft if fam == "aft" else rr.fit_ridge
    out[model] = fit(train[cols].to_numpy(float), t, ev, alpha, cols)(test[cols].to_numpy(float))
    return out


def walk_forward_chosen(tab: pd.DataFrame, model: str, min_obs: int, min_oblast: int) -> pd.DataFrame:
    """`region_restoration.run_walk_forward` restricted to `model` + the baselines: the forecast of strike i is fitted on the episodes of
    strikes j < i with their labels as known just before the issue time of strike i (`labels_as_of`)."""
    methods = [model, *BASE_METHODS]
    pred = pd.DataFrame(np.nan, index=tab.index, columns=methods)
    tab = tab.copy()
    tab["evaluated"] = False
    tab["n_train"] = 0
    tab["n_train_observed"] = 0
    for i in sorted(tab["idx"].unique()):
        test = tab[tab["idx"] == i]
        train = rr.labels_as_of(tab[tab["idx"] < i], test["issue"].iloc[0] - pd.Timedelta(microseconds=1))
        tab.loc[test.index, "n_train"] = len(train)
        tab.loc[test.index, "n_train_observed"] = int(train["observed"].sum())
        if int(train["observed"].sum()) < min_obs:
            continue
        pred.loc[test.index] = forecast_chosen(train, test, model, min_oblast).reindex(columns=methods)
        tab.loc[test.index, "evaluated"] = True
    return pd.concat([tab, pred], axis=1)


def c_parts(pred: np.ndarray, t: np.ndarray, event: np.ndarray, group: np.ndarray) -> pd.DataFrame:
    """Per group (strike): numerator and denominator of Harrell's C within the group (same definition as `harrell_c_within`:
    a observed, t[a] < t[b]; a tie in the prediction counts 0.5)."""
    rows = []
    for g in np.unique(group):
        k = np.flatnonzero(group == g)
        p, tt, e = pred[k], t[k], event[k]
        comp = e[:, None] & (tt[:, None] < tt[None, :])
        num = np.where(p[:, None] < p[None, :], 1.0, np.where(p[:, None] == p[None, :], 0.5, 0.0))
        rows.append((g, float((num * comp).sum()), float(comp.sum())))
    return pd.DataFrame(rows, columns=["idx", "num", "den"]).set_index("idx")


def paired_c_bootstrap(ev: pd.DataFrame, model: str, base: str, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap over strikes of C_within(model) - C_within(base): the strikes are resampled ONCE per replicate and both C are
    computed on the same resample (pair counts add up over the drawn strikes with multiplicity). Also the CI of C(model) and
    the one-sided p of C(model) > 0.5 from the same resamples (p = (1 + #(C* <= 0.5)) / (B + 1))."""
    t, ob, g = ev["hours"].to_numpy(float), ev["observed"].to_numpy(bool), ev["idx"].to_numpy()
    a, b = c_parts(ev[model].to_numpy(float), t, ob, g), c_parts(ev[base].to_numpy(float), t, ob, g)
    S = len(a)
    rng = np.random.default_rng(seed)
    cnt = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(n_boot)]).astype(float)
    den = cnt @ a["den"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ca, cb = cnt @ a["num"].to_numpy() / den, cnt @ b["num"].to_numpy() / den
    ok = den > 0
    d = (ca - cb)[ok]
    pt_a, pt_b = a["num"].sum() / a["den"].sum(), b["num"].sum() / b["den"].sum()
    return {"n_strikes": S, "n_boot": int(ok.sum()), "n_pairs": int(a["den"].sum()), "C_model": float(pt_a), "C_base": float(pt_b),
            "diff": float(pt_a - pt_b), "diff_lo": float(np.percentile(d, 2.5)), "diff_hi": float(np.percentile(d, 97.5)),
            "p_diff_le_0": float((1 + np.sum(d <= 0)) / (len(d) + 1)),
            "C_model_lo": float(np.percentile(ca[ok], 2.5)), "C_model_hi": float(np.percentile(ca[ok], 97.5)),
            "p_C_model_gt_half": float((1 + np.sum(ca[ok] <= 0.5)) / (ok.sum() + 1)),
            "C_base_lo": float(np.percentile(cb[ok], 2.5)), "C_base_hi": float(np.percentile(cb[ok], 97.5))}


def mae_vs_global(ev: pd.DataFrame, model: str) -> dict:
    """MAE on observed episodes of the model, oblast_median and global_median; Wilcoxon of per-strike mean |error| differences
    model - global_median."""
    o = ev[ev["observed"]]
    err = {m: (o[m] - o["hours"]).abs() for m in (model, *BASE_METHODS)}
    ps = (err[model] - err["global_median"]).groupby(o["idx"]).mean()
    return {"n_observed": len(o), "n_strikes_obs": len(ps), **{f"MAE_{m}": float(e.mean()) for m, e in err.items()},
            "mean_diff_per_strike_h": float(ps.mean()), "share_better_strikes": float((ps < 0).mean()),
            "p_wilcoxon": rd._wilcoxon_p(ps.to_numpy(), min_n=2)}


def _tab_test(cfg: Config) -> pd.DataFrame:
    tc = cfg.telegram
    strikes, facts, weather, eps = _read_inputs(cfg)
    end = pd.Timestamp(tc.energy_map_test_end, tz="UTC") + pd.Timedelta(days=1)         # strikes of the last day included
    strikes = strikes[pd.to_datetime(strikes["strike_at"], utc=True) < end].reset_index(drop=True)
    eps = eps[pd.to_datetime(eps["strike_at"], utc=True) < end]
    vend = pd.Timestamp(tc.region_validation_end, tz="UTC")
    return rr.build_table(eps, facts, strikes, weather, vend, "30d", tc.region_early_hours)


def _score(tab: pd.DataFrame, model: str, tc, n_boot: int, seed: int) -> dict:
    wf = walk_forward_chosen(tab, model, int(tc.region_min_obs), int(tc.region_min_oblast))
    ev = wf[(wf["split"] == "test") & wf["evaluated"]]
    is_test = wf["split"] == "test"
    return {"ev": ev, "primary": paired_c_bootstrap(ev, model, "oblast_median", n_boot, seed),
            "vs_global": paired_c_bootstrap(ev, model, "global_median", n_boot, seed), "mae": mae_vs_global(ev, model),
            "counts": {"test_pairs": int(is_test.sum()), "test_observed": int((is_test & wf["observed"]).sum()),
                       "test_strikes": int(wf.loc[is_test, "idx"].nunique()), "scored_pairs": len(ev),
                       "scored_observed": int(ev["observed"].sum()), "scored_strikes": int(ev["idx"].nunique())}}


def run_energy_map_test(cfg: Config, n_boot: int = 2000, seed: int = 42) -> dict:
    """The final test of docs/preregistration_energy_map.md; run it ONCE after `select_for_test`."""
    tc = cfg.telegram
    d = cfg.reports_dir
    model = chosen_model(d / CHOSEN_FILE)
    tab = _tab_test(cfg)
    res = {"model": model, "min_obs": int(tc.region_min_obs), "all": _score(tab, model, tc, n_boot, seed),
           "rear": _score(tab[~tab["oblast"].isin(em.FRONTLINE)], model, tc, n_boot, seed)}
    res["all"]["ev"].to_csv(d / "energy_map_test_predictions.csv", index=False)
    write_test_report(d / "energy_map_test_results.md", res, n_boot, seed)
    return res


def _ci(x, lo, hi, nd=3):
    return f"{x:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def write_test_report(path, res: dict, n_boot: int, seed: int) -> None:
    m = res["model"]
    A, R = res["all"], res["rear"]
    p, c = A["primary"], A["counts"]
    ok = p["diff_lo"] > 0
    L = ["# Energy Map: final test (protocol `docs/preregistration_energy_map.md`)", "",
         f"Model chosen in development (`energy_map_chosen_for_test.md`): `{m}`. Comparison: `oblast_median`, `global_median`. "
         f"Training: walk-forward, labels as known at forecast time. Bootstrap over strikes: {n_boot} resamples, seed {seed}, "
         "paired (strikes are resampled once, both C on the same resample). The test was run once.", "",
         "## Main metric (cohort: all oblasts)", "",
         f"- Within-strike C, `{m}`: {_ci(p['C_model'], p['C_model_lo'], p['C_model_hi'])}",
         f"- Within-strike C, `oblast_median`: {_ci(p['C_base'], p['C_base_lo'], p['C_base_hi'])}",
         f"- Difference C(model) - C(`oblast_median`): {_ci(p['diff'], p['diff_lo'], p['diff_hi'], 4)} (95% paired bootstrap interval); "
         f"one-sided p (difference <= 0) = {p['p_diff_le_0']:.4f}",
         f"- **Success criterion (lower bound of the interval > 0): {'YES' if ok else 'NO'}** (lower bound {p['diff_lo']:.4f})", "",
         "## C against 0.5 (all oblasts)", "",
         "| method | within-strike C [95% CI] | p (C > 0.5) |", "|---|---|---|",
         f"| {m} | {_ci(p['C_model'], p['C_model_lo'], p['C_model_hi'])} | {p['p_C_model_gt_half']:.4f} |",
         f"| oblast_median | {_ci(p['C_base'], p['C_base_lo'], p['C_base_hi'])} | |",
         "| global_median | 0.500 (constant within a strike) | |", ""]
    g = A["vs_global"]
    L += [f"Difference C(model) - C(`global_median`): {_ci(g['diff'], g['diff_lo'], g['diff_hi'], 4)}.", ""]
    rp, rc = R["primary"], R["counts"]
    L += ["## Additional metrics (no conclusions about success)", "",
          "### 1. Rear: without Donetsk, Luhansk, Zaporizhzhia, Kherson", "",
          "Test table without these oblasts; training (including `oblast_median`) is also without them, as in the development mode.", "",
          f"- C `{m}`: {_ci(rp['C_model'], rp['C_model_lo'], rp['C_model_hi'])}; p (C > 0.5) = {rp['p_C_model_gt_half']:.4f}",
          f"- C `oblast_median`: {_ci(rp['C_base'], rp['C_base_lo'], rp['C_base_hi'])}",
          f"- Difference: {_ci(rp['diff'], rp['diff_lo'], rp['diff_hi'], 4)}; p (difference <= 0) = {rp['p_diff_le_0']:.4f}",
          f"- Strikes {rp['n_strikes']}, pairs scored {rc['scored_pairs']} ({rc['scored_observed']} observed), "
          f"comparable within-strike pairs {rp['n_pairs']}.", "",
          "### 2. MAE on observed episodes against `global_median`", "",
          "| cohort | observed | MAE model, h | MAE oblast_median, h | MAE global_median, h | "
          "mean difference over strikes (model - global), h | share of better strikes | Wilcoxon p (over strikes) |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in (("all oblasts", A), ("rear", R)):
        x = r["mae"]
        L.append(f"| {name} | {x['n_observed']} | {x['MAE_' + m]:.1f} | {x['MAE_oblast_median']:.1f} | {x['MAE_global_median']:.1f} | "
                 f"{x['mean_diff_per_strike_h']:.1f} | {x['share_better_strikes']:.2f} | {x['p_wilcoxon']:.4f} |")
    L += ["", "### 3. C against 0.5", "", "See the table above (all oblasts) and item 1 (rear).", "",
          "## Counts", "",
          f"- Test pairs (strike x oblast) in the table: {c['test_pairs']} ({c['test_observed']} observed) in {c['test_strikes']} strikes.",
          f"- Scored by the model (needs >= {res['min_obs']} observed past episodes): {c['scored_pairs']} pairs, "
          f"{c['scored_observed']} of them observed, in {c['scored_strikes']} strikes; comparable within-strike pairs (denominator of C): {p['n_pairs']}.",
          "- Forecasts and labels: `energy_map_test_predictions.csv`.", "",
          "## Limitations (from the protocol)", "",
          "- The label is the first restoration report, not full restoration; the records are events, not a continuous series.",
          "- Episodes of the same strike are dependent; intervals are computed over strikes. A large part of C comes from differences "
          "between oblasts, so the criterion is an advantage over `oblast_median`.", ""]
    Path(path).write_text("\r\n".join(L), encoding="utf-8")
