"""Development mode of `deficit region-restoration --dev`: rolling-origin CV over the train+validation strikes, TEST BLIND.

Only strikes up to `telegram.region_validation_end` (the existing validation/test boundary) are used. Test-period strikes are
removed before the features are built, and `dev_cv` drops every row whose split is not "validation" again, so no test-period
row can influence (or appear in) any dev output. No test metric / prediction is computed, printed or written.

Rolling-origin (expanding window) CV
  * The cohort episodes of the dev period are ordered by strike time. The walk-forward of the normal mode is run on them: the
    forecast of strike i is fitted on the episodes of strikes j < i only, with their labels AS KNOWN at the issue time of strike i
    (`labels_as_of`), exactly as in the normal mode. Strikes that cannot be forecast (fewer than `region_min_obs` observed past
    episodes) are not scored.
  * The scored strikes are split into K consecutive blocks (folds, `np.array_split` in time order). Every fold is therefore
    trained on all earlier strikes (including the earlier strikes of the same fold, which are earlier in time) = expanding window.
  * Per design and fold: MAE on observed episodes (pooled episodes of the fold), censoring-aware lower-bound MAE (all episodes)
    and Harrell's C within strikes. Reported: mean over folds, standard error over folds (std ddof=1 / sqrt(K)).
  * Paired comparison with `global_median`: per-strike mean of (|err design| - |err global_median|) on observed episodes, pooled
    over folds; Wilcoxon signed-rank on these per-strike means.

Selection: one-standard-error rule. Candidates = baselines and ridge/AFT designs (not the reference-only km_global and
const_30d). threshold = best CV MAE + SE of the best design. Among candidates with CV MAE <= threshold the SIMPLEST is chosen.
Complexity order (low -> high): baselines (global_median < oblast_median < geo_mean) < strike < strike_oblast <
strike_oblast_hist < other feature sets (full, full_no_oblast); inside a feature set ridge < aft, then the larger penalty alpha
(stronger shrinkage = simpler), then the lower CV MAE. The censoring-aware designs come after all of these: ipcw_* (IPCW-weighted
ridge) < xgbaft_* (boosted AFT trees; fewer rounds / shallower = simpler) < the ensembles (ens_*).

Early-impact designs (*_strike_oblast_early, *_strike_oblast_hist_early; ridge, aft, ipcw, xgbaft): oblast-level features from the posts
of the first K hours (`rr.early_features`). K is chosen on this dev CV only: the dev run repeats the CV for every K in
`telegram.region_early_candidates` (result["early"][K]); the main tables use `telegram.region_early_hours`. They rank after strike_oblast_hist
in the complexity order (new information = more complex).

Extra designs (`rr.extra_designs`, dev only): IPCW-weighted ridge and XGBoost `survival:aft`; ensembles = mean of the log-predictions
of fixed members, added after the walk-forward from the stored (causal) predictions: `ens_fixed` = geo_mean + ridge_strike_oblast_hist
+ ipcw_strike_oblast_hist (alpha = the largest in the grid); `ens_best` = geo_mean + the ridge design + the censoring-aware design
(ipcw/xgbaft/aft) with the lowest CV MAE (members picked on the same dev CV, so its CV MAE is mildly optimistic).
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.telegram_extract import load_strikes
from . import region_binary as rb
from . import region_restoration as rr

log = logging.getLogger(__name__)

SELECTABLE_BASELINES = ["global_median", "oblast_median", "geo_mean"]
SET_ORDER = ["strike", "strike_oblast", "strike_oblast_hist", "strike_oblast_early", "strike_oblast_hist_early"]               # everything else ranks after these
REFERENCE_ONLY = ["km_global", "const_30d"]
ENSEMBLES = ["ens_fixed", "ens_best"]


def complexity_key(name: str, mae: float = 0.0) -> tuple:
    """Sort key of the complexity order (smaller = simpler); see the module docstring."""
    if name in SELECTABLE_BASELINES:
        return (0, SELECTABLE_BASELINES.index(name), 0, 0.0, mae)
    if name.startswith("ens_"):
        return (9, ENSEMBLES.index(name) if name in ENSEMBLES else len(ENSEMBLES), 0, 0.0, mae)
    x = re.fullmatch(r"xgbaft_(.+)_d(\d+)n(\d+)", name)
    if x:
        return (8, SET_ORDER.index(x.group(1)) if x.group(1) in SET_ORDER else len(SET_ORDER), int(x.group(2)) * 1000 + int(x.group(3)), 0.0, mae)
    m = re.fullmatch(r"(ipcw|ridge|aft)_(.+)_a([0-9.eE+-]+)", name)
    if not m:
        return (10, 0, 0, 0.0, mae)
    fam, fs, alpha = m.group(1), m.group(2), float(m.group(3))
    if fam == "ipcw":
        return (7, SET_ORDER.index(fs) if fs in SET_ORDER else len(SET_ORDER), 0, -alpha, mae)
    tier = 1 + (SET_ORDER.index(fs) if fs in SET_ORDER else len(SET_ORDER))
    return (tier, ["ridge", "aft"].index(fam), 0 if fam == "ridge" else 1, -alpha, mae)


def make_folds(strike_ids: list, k: int) -> list[list]:
    """K consecutive blocks (time order) of the scored strikes; K is capped by the number of strikes."""
    ids = sorted(strike_ids)
    k = max(1, min(int(k), len(ids)))
    return [list(map(int, b)) for b in np.array_split(np.array(ids), k) if len(b)]


def _fold_metrics(s: pd.DataFrame, m: str) -> tuple[float, float, float]:
    ob = s["observed"].to_numpy(bool)
    err = (s[m] - s["hours"]).abs()
    mae = float(err[ob].mean()) if ob.any() else np.nan
    lb = float(rr.loss_lb(s[m], s["hours"], s["observed"]).mean())
    c = rr.harrell_c_within(s[m].to_numpy(), s["hours"].to_numpy(), ob, s["idx"].to_numpy())
    return mae, lb, c


def _wilcoxon_p(x: np.ndarray, min_n: int = 8) -> float:
    from scipy.stats import wilcoxon
    return float(wilcoxon(x).pvalue) if len(x) >= min_n and (x != 0).any() else np.nan


def _log_mean(*cols: pd.Series) -> pd.Series:
    return np.exp(np.mean([np.log(np.clip(c.to_numpy(float), rr.MIN_H, None)) for c in cols], axis=0))


def _fold_table(ev: pd.DataFrame, folds: list[list], methods: list[str]) -> pd.DataFrame:
    rows = []
    for fi, ids in enumerate(folds):
        s = ev[ev["idx"].isin(ids)]
        if not s["observed"].any():
            continue
        for m in methods:
            mae, lb, c = _fold_metrics(s, m)
            rows.append({"fold": fi, "method": m, "n_strikes": len(ids), "n_episodes": len(s), "n_observed": int(s["observed"].sum()),
                         "MAE_h": mae, "MAE_lower_bound_h": lb, "C_within_strike": c})
    return pd.DataFrame(rows)


def _cv_row(fr: pd.DataFrame, obs: pd.DataFrame, ev: pd.DataFrame, m: str, base: str) -> dict:
    f = fr[fr["method"] == m]
    n = len(f)

    def se(c):
        return f[c].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    row = {"method": m, "n_folds": n, "MAE_cv_h": f["MAE_h"].mean(), "MAE_cv_se_h": se("MAE_h"),
           "LB_cv_h": f["MAE_lower_bound_h"].mean(), "LB_cv_se_h": se("MAE_lower_bound_h"),
           "C_cv": f["C_within_strike"].mean(), "C_cv_se": se("C_within_strike"),
           # Harrell's C over ALL comparable pairs of the scored dev episodes (also across strikes); cheap at this size
           "C_all_pairs": rr.harrell_c_within(ev[m].to_numpy(), ev["hours"].to_numpy(), ev["observed"].to_numpy(bool), np.zeros(len(ev), int))}
    if m != base:
        d = (obs[m] - obs["hours"]).abs() - (obs[base] - obs["hours"]).abs()
        ps = d.groupby(obs["idx"]).mean()
        row.update({"mean_diff_vs_base_h": float(ps.mean()), "share_better_strikes": float((ps < 0).mean()), "n_strikes_paired": len(ps),
                    "p_per_strike_vs_base": _wilcoxon_p(ps.to_numpy())})
    return row


def add_ensembles(ev: pd.DataFrame, table: pd.DataFrame, alphas: list[float]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Ensemble columns (mean of log-predictions of fixed members) from the stored causal predictions; see the module docstring."""
    ev = ev.copy()
    t = table.set_index("method")["MAE_cv_h"]
    a = f"{max(alphas):g}"
    members = {"ens_fixed": ["geo_mean", f"ridge_strike_oblast_hist_a{a}", f"ipcw_strike_oblast_hist_a{a}"]}
    ridge = t[t.index.str.startswith("ridge_")].idxmin()
    cens = t[t.index.str.match(r"(ipcw|xgbaft|aft)_")].idxmin()
    members["ens_best"] = ["geo_mean", ridge, cens]
    for name, mem in members.items():
        if all(m in ev for m in mem):
            ev[name] = _log_mean(*[ev[m] for m in mem])
    return ev, {k: v for k, v in members.items() if k in ev}


def dev_cv(tab: pd.DataFrame, alphas: list[float], min_obs: int, min_oblast: int, k: int = 5, base: str = "global_median",
           extra: bool = True, thresholds: tuple = (), n_boot: int = 1000) -> dict:
    """Rolling-origin CV on the validation-split rows of `tab` (a `build_table` result). Rows of any other split are dropped first.
    `extra`: also the censoring-aware designs and the ensembles. `thresholds`: if given, also the binary long-outage task
    (`region_binary`) for each T and the bootstrap ranking test of the regression designs."""
    dev = tab[tab["split"] == "validation"].copy()
    if dev.empty:
        raise ValueError("no dev (validation) episodes")
    wf = rr.run_walk_forward(dev, alphas, min_obs, min_oblast, extra=extra)
    ev = wf[wf["evaluated"]]
    methods = rr.all_methods(alphas, extra)
    folds = make_folds(list(ev["idx"].unique()), k)
    obs = ev[ev["observed"]]
    fr = _fold_table(ev, folds, methods)
    table = pd.DataFrame([_cv_row(fr, obs, ev, m, base) for m in methods])
    members: dict[str, list[str]] = {}
    if extra:
        ev, members = add_ensembles(ev, table, alphas)
        obs = ev[ev["observed"]]
        ens = list(members)
        fr = pd.concat([fr, _fold_table(ev, folds, ens)], ignore_index=True)
        table = pd.concat([table, pd.DataFrame([_cv_row(fr, obs, ev, m, base) for m in ens])], ignore_index=True)
    table = table.sort_values("MAE_cv_h").reset_index(drop=True)
    sel = select_one_se(table)
    info = {"n_dev_episodes": len(dev), "n_dev_observed": int(dev["observed"].sum()), "n_strikes_scored": int(ev["idx"].nunique()),
            "n_strikes_total": int(dev["idx"].nunique()), "folds": folds, "min_obs": min_obs, "ensemble_members": members,
            "fold_sizes": [{"fold": i, "strikes": len(ids), "episodes": int(ev["idx"].isin(ids).sum()),
                            "observed": int((ev["idx"].isin(ids) & ev["observed"]).sum())} for i, ids in enumerate(folds)],
            "first_scored": ev["strike_at"].min(), "last_scored": ev["strike_at"].max(), "base": base}
    out = {"table": table, "fold_table": fr, "selection": sel, "info": info}
    if thresholds:
        score_methods = [m for m in list(methods) + list(members) if m not in ("km_global", "const_30d")]
        bw = rb.binary_walk_forward(dev, ev["idx"].unique(), alphas, thresholds, min_oblast)
        out["binary"] = {float(T): rb.binary_scores(bw[T], ev, folds, score_methods, alphas, n_boot=n_boot) for T in thresholds}
        out["ranking"] = rb.ranking_bootstrap(ev, score_methods, n_boot=max(n_boot, 1000))
    return out


def select_one_se(table: pd.DataFrame) -> dict:
    """One-standard-error rule among the selectable designs (see the module docstring)."""
    cand = table[~table["method"].isin(REFERENCE_ONLY) & table["MAE_cv_h"].notna()]
    best = cand.loc[cand["MAE_cv_h"].idxmin()]
    se = float(best["MAE_cv_se_h"]) if pd.notna(best["MAE_cv_se_h"]) else 0.0
    thr = float(best["MAE_cv_h"]) + se
    within = cand[cand["MAE_cv_h"] <= thr + 1e-12]
    pick = min(within.itertuples(), key=lambda r: complexity_key(r.method, r.MAE_cv_h))
    return {"best": str(best["method"]), "best_mae": float(best["MAE_cv_h"]), "best_se": se, "threshold": thr,
            "n_within": len(within), "selected": str(pick.method), "selected_mae": float(pick.MAE_cv_h)}


# --------------------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------------------
def _fmt(x, nd=1):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def _md_table(t: pd.DataFrame, base: str) -> list[str]:
    head = ["design", "MAE_cv h", "SE", "LB_cv h", "C_within", "C_all", "diff vs " + base + " h", "better strikes", "p per strike"]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in t.itertuples():
        out.append(f"| {r.method} | {_fmt(r.MAE_cv_h)} | {_fmt(r.MAE_cv_se_h)} | {_fmt(r.LB_cv_h)} | {_fmt(r.C_cv, 3)} | {_fmt(r.C_all_pairs, 3)} | "
                   f"{_fmt(getattr(r, 'mean_diff_vs_base_h', np.nan))} | {_fmt(getattr(r, 'share_better_strikes', np.nan), 2)} | "
                   f"{_fmt(getattr(r, 'p_per_strike_vs_base', np.nan), 3)} |")
    return out


def _binary_section(r: dict, primary: float, top: int = 8) -> list[str]:
    L = []
    for T, b in r["binary"].items():
        i, t, s = b["info"], b["table"], b["selection"]
        tag = " (PRIMARY, fixed in advance)" if T == primary else ""
        L += [f"#### Long-outage task, T = {T:g} h{tag}", "",
              f"Labelled scored episodes {i['n_labelled']} in {i['n_strikes']} strikes: long (> T h) {i['n_long']} ({100 * i['long_share']:.1f}%), "
              f"short {i['n_short']}; unknown (censored before T, excluded) {i['n_unknown']}.", ""]
        if not s or i["n_short"] == 0 or i["n_long"] == 0:
            L += ["Degenerate threshold: only one class among the scored episodes, no metric is defined.", ""]
            continue
        L += [f"Best mean Brier skill: `{s['best']}` {s['best_bss']:.3f} (SE {s['best_se']:.3f}); 1-SE threshold {s['threshold']:.3f}; {s['n_within']} designs within. "
              f"**Selected (simplest within 1 SE): `{s['selected']}`** (skill {s['selected_bss']:.3f}).", ""]
        prob = t[t["BSS_fold_mean"].notna()].sort_values("BSS_fold_mean", ascending=False)
        score = t[t["BSS_fold_mean"].isna()].sort_values("AUC_within", ascending=False)
        keep = pd.concat([prob.head(top), score.head(top), t[t["method"].isin(rb.BIN_BASELINES)]]).drop_duplicates("method")
        head = ["design", "AUC pooled", "AUC CI95 (strikes)", "p AUC>0.5", "AUC within strikes [CI95]", "p within>0.5", "AUC folds", "Brier", "Brier skill (folds)", "SE", "share better strikes", "p Brier vs base rate"]
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for x in keep.itertuples():
            ci = f"[{_fmt(x.AUC_ci_lo, 3)}, {_fmt(x.AUC_ci_hi, 3)}]" if pd.notna(getattr(x, "AUC_ci_lo", np.nan)) else ""
            L.append(f"| {x.method} | {_fmt(x.AUC_pooled, 3)} | {ci} | {_fmt(getattr(x, 'p_AUC_gt_half', np.nan), 3)} | "
                     f"{_fmt(getattr(x, 'AUC_within', np.nan), 3)} [{_fmt(getattr(x, 'AUC_within_lo', np.nan), 3)}, {_fmt(getattr(x, 'AUC_within_hi', np.nan), 3)}] | "
                     f"{_fmt(getattr(x, 'p_AUC_within_gt_half', np.nan), 3)} | "
                     f"{_fmt(x.AUC_fold_mean, 3)} +- {_fmt(x.AUC_fold_se, 3)} | {_fmt(getattr(x, 'Brier', np.nan), 3)} | "
                     f"{_fmt(x.BSS_fold_mean, 3)} | {_fmt(x.BSS_fold_se, 3)} | {_fmt(getattr(x, 'share_better_strikes', np.nan), 2)} | "
                     f"{_fmt(getattr(x, 'p_brier_vs_base', np.nan), 3)} |")
        L.append("")
    return L


def _ranking_section(rk: pd.DataFrame, top: int = 10) -> list[str]:
    head = ["design", "C_within [95% CI]", "p (C_within>0.5)", "C_all [95% CI]", "p (C_all>0.5)"]
    L = [f"#### Ranking skill of the regression designs (bootstrap over {int(rk['n_strikes'].iloc[0])} strikes)", "",
         f"Top {top} by C_within of {len(rk)} designs (p-values are NOT corrected for the choice among {len(rk)} designs), plus the baselines.", "",
         "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    keep = pd.concat([rk.head(top), rk[rk["method"].isin(SELECTABLE_BASELINES)]]).drop_duplicates("method")
    for x in keep.itertuples():
        L.append(f"| {x.method} | {x.C_within:.3f} [{x.C_within_lo:.3f}, {x.C_within_hi:.3f}] | {x.p_C_within_gt_half:.3f} | "
                 f"{x.C_all:.3f} [{x.C_all_lo:.3f}, {x.C_all_hi:.3f}] | {x.p_C_all_gt_half:.3f} |")
    return L + [""]


def write_dev_report(path, res: dict[str, dict], k: int, top: int = 10, primary: float = 72.0) -> None:
    L = ["# Oblast restoration forecast: DEV report (rolling-origin CV on train+validation, test period blind)", "",
         f"CV design: expanding window over the validation-period strikes (up to `telegram.region_validation_end`), K = {k} consecutive "
         "blocks of scored strikes; every strike is forecast from the earlier strikes only, with their labels as known at its issue time. "
         "MAE on observed episodes: mean over folds, SE over folds. LB = censoring-aware lower-bound MAE over all episodes; C = Harrell's C "
         "within strikes (0.5 = no ranking skill). Paired p: Wilcoxon on per-strike mean |error| differences vs the baseline. "
         "Selection: one-standard-error rule, complexity order baselines < strike < strike_oblast < *_hist < other sets "
         "(ridge < aft, larger alpha simpler). No test-period number is computed or shown.", ""]
    for v, r in res.items():
        i, s, t = r["info"], r["selection"], r["table"]
        L += [f"## Window `{v}`", "",
              f"Dev episodes {i['n_dev_episodes']} ({i['n_dev_observed']} observed); strikes scored {i['n_strikes_scored']} of {i['n_strikes_total']} "
              f"({i['first_scored']:%Y-%m-%d} .. {i['last_scored']:%Y-%m-%d}).", "",
              "Folds: " + "; ".join(f"#{f['fold']}: {f['strikes']} strikes / {f['episodes']} episodes / {f['observed']} observed" for f in i["fold_sizes"]), "",
              f"Best CV MAE: `{s['best']}` {s['best_mae']:.1f} h (SE {s['best_se']:.1f}); 1-SE threshold {s['threshold']:.1f} h; "
              f"{s['n_within']} designs within it. **Selected (simplest within 1 SE): `{s['selected']}`** ({s['selected_mae']:.1f} h).", "",
              "Ensemble members: " + "; ".join(f"`{k_}` = {' + '.join(v)}" for k_, v in i["ensemble_members"].items()), "",
              f"Top {top} by CV MAE, plus baselines and references:", ""]
        keep = list(t["method"].head(top)) + [m for m in SELECTABLE_BASELINES + REFERENCE_ONLY if m not in set(t["method"].head(top))]
        L += _md_table(t[t["method"].isin(keep)], i["base"]) + [""]
        fam = t.assign(f=t["method"].str.extract(r"^(ens|xgbaft|ipcw|aft|ridge)")[0]).dropna(subset=["f"])
        best_f = fam.loc[fam.groupby("f")["MAE_cv_h"].idxmin()].sort_values("MAE_cv_h")
        L += ["Best design per family (CV MAE): " + "; ".join(f"`{x.method}` {x.MAE_cv_h:.1f} h / C {x.C_cv:.3f}" for x in best_f.itertuples()), ""]
        if "binary" in r:
            L += [f"### Binary task: will the restoration take longer than T hours? (primary T = {primary:g} h)", "",
                  "Label as known at issue time / final dev label: long = observed duration > T or still unresolved at an elapsed time > T; short = "
                  "observed <= T; censored before T = unknown, excluded. Learners: L2 logistic regression on the same feature sets, base rates "
                  "(`bin_const` training base rate, `bin_oblast` oblast base rate) and the regression designs as scores (predicted hours; AUC only). "
                  "AUC pooled over the scored episodes (also across strikes, so it is confounded by the drift of the base rate over time: `bin_const` is the reference, "
                  "not 0.5; AUC within strikes uses pairs of the same strike only); CI = bootstrap over strikes; p AUC>0.5 one-sided bootstrap. "
                  "Brier skill = 1 - Brier / Brier(bin_const), mean and SE over folds; p = Wilcoxon over per-strike Brier differences vs `bin_const`. "
                  "Selection: best mean Brier skill, one-SE rule toward the simpler design (bin_const < bin_oblast < logit_strike < _oblast < _hist < "
                  "other sets, larger alpha simpler).", ""]
            L += _binary_section(r, primary)
            L += _ranking_section(r["ranking"])
        er = r.get("early", {})
        if er:
            L += [f"Early-impact features: dev CV for each cut-off K (hours after the strike time; main tables use K = {r['early_main']:g}). "
                  "C within strikes, mean over folds (SE); MAE_cv h (SE).", "",
                  "| K h | design | C_within (SE) | MAE_cv h (SE) |", "|---|---|---|---|"]
            for h, q in sorted(er.items()):
                tt = q["table"].set_index("method")
                sel = [m for m in tt.index if re.search(r"strike_oblast(_hist)?_early", m)]
                best = tt.loc[sel].sort_values("C_cv", ascending=False).head(4).index.tolist()
                for m in best + [x for x in ("oblast_median", "geo_mean") if x in tt.index]:
                    x = tt.loc[m]
                    L.append(f"| {h:g} | {m} | {x.C_cv:.3f} ({x.C_cv_se:.3f}) | {x.MAE_cv_h:.1f} ({x.MAE_cv_se_h:.1f}) |")
            L.append("")
        w = r.get("warmup", {})
        if w:
            L += [f"Warm-up sensitivity (`region_dev_min_obs` = observed past episodes needed before the first forecast; dev value {i['min_obs']}, normal mode 30):", "",
                  "| min_obs | strikes scored | first scored | global_median MAE | geo_mean MAE | best design (MAE, SE) | selected (1 SE) |",
                  "|---|---|---|---|---|---|---|"]
            for mo, q in sorted({i["min_obs"]: r, **w}.items()):
                tt = q["table"].set_index("method")
                sl = q["selection"]
                L.append(f"| {mo} | {q['info']['n_strikes_scored']} of {q['info']['n_strikes_total']} | {q['info']['first_scored']:%Y-%m-%d} | "
                         f"{tt.loc['global_median', 'MAE_cv_h']:.1f} | {tt.loc['geo_mean', 'MAE_cv_h']:.1f} | "
                         f"`{sl['best']}` ({sl['best_mae']:.1f}, {sl['best_se']:.1f}) | `{sl['selected']}` |")
            L.append("")
    from pathlib import Path
    Path(path).write_text("\n".join(L), encoding="utf-8")


def run_region_dev(cfg: Config, k: int | None = None, warmups: tuple[int, ...] = (30,)) -> dict[str, dict]:
    tc = cfg.telegram
    k = int(k or tc.region_cv_folds)
    root = cfg._abs(cfg.paths.data_dir)
    pdir = cfg.telegram_processed_dir
    facts = pd.read_parquet(pdir / "telegram_region_posts.parquet")
    strikes = load_strikes(cfg.strike_list_path)
    from .energy_map_labels import episodes_for_cfg                          # label source switch (telegram | energy_map, dev only)
    eps = episodes_for_cfg(cfg, strikes)
    tag = "telegram_region_dev" if tc.region_label_source == "telegram" else f"{tc.region_label_source}_region_dev"
    wpath = root / "raw" / "weather.parquet"
    weather = pd.read_parquet(wpath) if wpath.exists() else None
    vend = pd.Timestamp(tc.region_validation_end, tz="UTC")
    cut = vend + pd.Timedelta(hours=23)                                    # same boundary as build_table's split
    st = pd.to_datetime(strikes["strike_at"], utc=True)
    strikes = strikes[st <= cut].reset_index(drop=True)                    # test-period strikes never enter the dev tables
    eps = eps[pd.to_datetime(eps["strike_at"], utc=True) <= cut]
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    res = {}
    mo_dev = int(tc.region_dev_min_obs)                                     # dev warm-up; the normal mode keeps region_min_obs
    for v in rr.VARIANTS:
        tab = rr.build_table(eps, facts, strikes, weather, vend, v, tc.region_early_hours)
        if tc.region_exclude_oblasts:
            tab = tab[~tab["oblast"].isin(tc.region_exclude_oblasts)]
        res[v] = dev_cv(tab, tc.region_alphas, mo_dev, tc.region_min_oblast, k, thresholds=tuple(tc.region_long_hours))
        res[v]["warmup"] = {mo: dev_cv(tab, tc.region_alphas, mo, tc.region_min_oblast, k) for mo in warmups if mo != mo_dev}
        res[v]["early"] = {float(h): dev_cv(rr.build_table(eps, facts, strikes, weather, vend, v, h), tc.region_alphas, mo_dev, tc.region_min_oblast, k)
                           for h in tc.region_early_candidates}
        for h, q in res[v]["early"].items():
            q["table"].to_csv(d / f"{tag}_{v}_early{h:g}h.csv", index=False)
        for T, b in res[v]["binary"].items():
            b["table"].to_csv(d / f"{tag}_binary_{v}_T{T:g}.csv", index=False)
        res[v]["ranking"].to_csv(d / f"{tag}_ranking_{v}.csv", index=False)
        res[v]["table"].to_csv(d / f"{tag}_{v}.csv", index=False)
        log.info("[dev %s] selected %s (best %s)", v, res[v]["selection"]["selected"], res[v]["selection"]["best"])
    for v in res:
        res[v]["early_main"] = float(tc.region_early_hours)
    write_dev_report(d / f"{tag}.md", res, k, primary=float(tc.region_long_primary))
    return res
