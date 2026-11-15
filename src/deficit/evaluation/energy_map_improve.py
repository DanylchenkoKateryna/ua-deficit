"""Improvement study of the Energy Map restoration forecast (DEV ONLY): rolling-origin CV over ALL strikes up to `energy_map_test_end`.

Status of the data: the frozen test of `docs/preregistration_energy_map.md` (strikes 2025-01-01 .. 2026-06-26) is SPENT.  From now on every
strike up to 2026-06-26 is development data; the next blind test (strikes from 2026-06-27, a fresher Energy Map file) is pre-registered
separately (`docs/preregistration_energy_map_v2_draft.md`) and nothing in this module touches that file.

Design (the same label, cohort and walk-forward as the old test: `energy_map_labels`, `region_restoration.build_table`,
`labels_as_of`; warm-up `region_dev_min_obs`).  New, all computed at the issue time (end of the strike day) only:
  EM history   previous Energy Map durations of the oblast as KNOWN at the issue time (observed iff the restoration day ended before the
               issue time): median of the last 3 / 10, recency-weighted mean (half-life 5 episodes), count; and the oblast's blackout /
               restoration RECORDS dated STRICTLY BEFORE the strike day (days since the last one, blackout days in the last 7 / 30 days,
               log consumers of the last record, number of oblasts with a blackout record in the last 7 days).  Records dated on the strike
               day are never used (day resolution: they may be published after the issue time).
  geography    front-line flag, a 4-level proximity tier and a Russia/Belarus border flag (fixed from the map before any result),
  context      Air Force strike (vs derived from attack-report posts), seasonality sin/cos of the day of year (plus `heating`, `log_through`).
Models: ridge / AFT / IPCW ridge / boosted AFT trees on feature sets built on `strike_oblast` (the model of the old test is
`ridge_so_a100`), a boosted pairwise RANKING model (xgboost `rank:pairwise`, grouped by strike; ordering only, no hours), blends of
a model with `oblast_median` in log space (weight chosen in CV) and ensembles of the top designs.
Selection: 8 expanding-window folds of the scored strikes; mean C within strike (SE over folds); one-SE rule, simplest within one SE
(`complexity`).  With ~50 designs on the same CV the winner's CV numbers are optimistic (selection effect); the blind test decides.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion import energy_map as em
from ..ingestion.telegram_regional import OBLASTS
from . import energy_map_labels as eml
from . import energy_map_test as emt
from . import region_dev as rd
from . import region_restoration as rr

log = logging.getLogger(__name__)

FRONT = em.FRONTLINE
NEAR = ("Kharkiv", "Sumy", "Dnipropetrovsk", "Mykolaiv", "Chernihiv")
MID = ("Odesa", "Poltava", "Kirovohrad", "Kyiv", "Kyiv_city", "Cherkasy", "Zhytomyr")
BORDER = ("Chernihiv", "Sumy", "Kharkiv", "Luhansk", "Donetsk", "Kyiv", "Zhytomyr", "Rivne", "Volyn")      # border with Russia / Belarus

EM_F = ["em_prev_med3", "em_prev_med10", "em_prev_ewm", "em_n_obs", "em_days_since_blackout", "em_days_since_restored", "em_bl_7d",
        "em_bl_30d", "em_last_log_cons", "em_nat_7d"]
GEO_F = ["geo_front", "geo_tier", "geo_border"]
CTX_F = ["is_af", "sin_doy", "cos_doy"]
rr.CONTINUOUS.update(set(EM_F) | {"geo_tier", "sin_doy", "cos_doy"})      # standardised like the other continuous features

SO = rr.STRIKE_F + rr.dummies()
SETS = {"so": SO, "so_geo": SO + GEO_F + CTX_F, "so_em": SO + EM_F, "so_emh": SO + rr.HIST_F + EM_F,
        "so_all": SO + rr.HIST_F + EM_F + GEO_F + CTX_F}
SET_RANK = {s: i for i, s in enumerate(SETS)}
ALPHAS = (1.0, 10.0, 100.0)
XGB_GRID = ((1, 60), (2, 25), (2, 60))
RANK_GRID = ((1, 60), (2, 60))
EM_START = "2025-01-01"                    # the (spent) test period: used only to report the period-restricted CV
BLEND_W = (0.25, 0.5, 0.75)
REFERENCES = ["global_median", "oblast_median", "oblast_last10_median"]
OLD_MODEL = "ridge_so_a100"                # = `ridge_strike_oblast_a100` of the old test


# --------------------------------------------------------------------------------------------------
# features at the issue time
# --------------------------------------------------------------------------------------------------
def geo_features(oblast: pd.Series) -> pd.DataFrame:
    tier = oblast.map(lambda o: 3 if o in FRONT else 2 if o in NEAR else 1 if o in MID else 0)
    return pd.DataFrame({"geo_front": oblast.isin(FRONT).astype(float), "geo_tier": tier.astype(float),
                         "geo_border": oblast.isin(BORDER).astype(float)}, index=oblast.index)


def context_features(strike_at: pd.Series, is_af: pd.Series) -> pd.DataFrame:
    doy = pd.to_datetime(strike_at, utc=True).dt.dayofyear.to_numpy(float)
    ang = 2 * np.pi * doy / 365.25
    return pd.DataFrame({"is_af": is_af.astype(float).to_numpy(), "sin_doy": np.sin(ang), "cos_doy": np.cos(ang)}, index=strike_at.index)


def _hist_features(out: pd.DataFrame) -> dict[str, np.ndarray]:
    """Previous Energy Map durations of the oblast as known at `issue - 1 us` (restoration day ended before the issue time)."""
    cols = {c: np.full(len(out), np.nan) for c in ("em_prev_med3", "em_prev_med10", "em_prev_ewm", "em_n_obs")}
    cols["em_n_obs"][:] = 0.0
    for _, g in out.sort_values(["strike_at", "idx"]).groupby("oblast"):
        ix = g.index.to_numpy()
        ra = pd.to_datetime(g["res_at"], utc=True).dt.tz_convert(None).to_numpy()
        lh = np.log(np.clip(g["res_hours"].to_numpy(float), rr.MIN_H, None))
        iss = (pd.to_datetime(g["issue"], utc=True).dt.tz_convert(None) - pd.Timedelta(microseconds=1)).to_numpy()
        for k in range(len(ix)):
            known = np.flatnonzero(~np.isnat(ra[:k]) & (ra[:k] <= iss[k]))
            if not len(known):
                continue
            v = lh[known]
            w = 0.5 ** (np.arange(len(v) - 1, -1, -1) / 5.0)
            cols["em_prev_med3"][ix[k]] = float(np.median(v[-3:]))
            cols["em_prev_med10"][ix[k]] = float(np.median(v[-10:]))
            cols["em_prev_ewm"][ix[k]] = float((w * v).sum() / w.sum())
            cols["em_n_obs"][ix[k]] = float(len(v))
    return cols


def _record_features(out: pd.DataFrame, rec: pd.DataFrame) -> dict[str, np.ndarray]:
    """Energy Map records of the oblast dated STRICTLY BEFORE the strike day (point records only)."""
    r = rec[~rec["is_period"]]
    bl, rs, cons = {}, {}, {}
    for o in OBLASTS:
        b = r[(r["oblast"] == o) & r["has_blackout"]]
        bl[o] = np.sort(b["date"].unique()).astype("datetime64[D]")
        cons[o] = b.groupby("date")["blackout_consumers"].max()
        rs[o] = np.sort(r.loc[(r["oblast"] == o) & r["has_restored"], "date"].unique()).astype("datetime64[D]")
    n = len(out)
    cols = {c: np.zeros(n) for c in ("em_days_since_blackout", "em_days_since_restored", "em_bl_7d", "em_bl_30d", "em_last_log_cons", "em_nat_7d")}
    day = pd.to_datetime(out["strike_at"], utc=True).dt.tz_convert(None).dt.floor("D").to_numpy().astype("datetime64[D]")
    one = np.timedelta64(1, "D")
    for i, (o, D) in enumerate(zip(out["oblast"], day)):
        b, s = bl[o], rs[o]
        p = np.searchsorted(b, D, "left")
        cols["em_days_since_blackout"][i] = min(float((D - b[p - 1]) / one), 60.0) if p else 60.0
        cols["em_bl_7d"][i] = p - np.searchsorted(b, D - 7 * one, "left")
        cols["em_bl_30d"][i] = p - np.searchsorted(b, D - 30 * one, "left")
        if p:
            c = cons[o].get(pd.Timestamp(b[p - 1]), np.nan)
            cols["em_last_log_cons"][i] = float(np.log1p(c)) if pd.notna(c) else 0.0
        q = np.searchsorted(s, D, "left")
        cols["em_days_since_restored"][i] = min(float((D - s[q - 1]) / one), 60.0) if q else 60.0
        cols["em_nat_7d"][i] = sum(int(np.searchsorted(bl[x], D, "left") - np.searchsorted(bl[x], D - 7 * one, "left") > 0) for x in OBLASTS)
    return cols


def add_em_features(tab: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """`tab` = `region_restoration.build_table` result with energy_map episodes (needs res_at, res_hours, issue, strike_at, oblast, idx)."""
    out = tab.copy().reset_index(drop=True)
    for c, v in {**_hist_features(out), **_record_features(out, rec)}.items():
        out[c] = v
    return out


def build_dev_table(cfg: Config) -> pd.DataFrame:
    """All strikes up to `energy_map_test_end`, features as in the old test + the new ones; split = 'validation' for every row."""
    tc = cfg.telegram
    strikes, facts, weather, eps = emt._read_inputs(cfg)
    end = pd.Timestamp(tc.energy_map_test_end, tz="UTC") + pd.Timedelta(days=1)
    strikes = strikes[pd.to_datetime(strikes["strike_at"], utc=True) < end].reset_index(drop=True)
    eps = eps[pd.to_datetime(eps["strike_at"], utc=True) < end]
    tab = rr.build_table(eps, facts, strikes, weather, pd.Timestamp(tc.energy_map_test_end, tz="UTC"), "30d", tc.region_early_hours)
    rec = em.load_energy_map(cfg._abs(cfg.paths.data_dir) / "raw" / "energy_map")
    tab = add_em_features(tab, rec)
    typ = eml.strike_types(cfg.strike_list_path)
    tab = tab.merge(typ, on="strike_at", how="left")
    tab = pd.concat([tab, geo_features(tab["oblast"]), context_features(tab["strike_at"], tab["type"].eq("air_force"))], axis=1)
    tab["split"] = "validation"
    return tab


# --------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------
def fit_xgb_rank(X: np.ndarray, t: np.ndarray, event: np.ndarray, group: np.ndarray, depth: int, rounds: int, cols: list[str],
                 eta: float = 0.1, reg_lambda: float = 10.0, seed: int = 42):
    """Pairwise ranking trees (`rank:pairwise`) on the OBSERVED episodes, one query group per strike; the label is the dense rank of the
    duration inside the strike (longer = higher).  Returns a SCORE (higher = longer), not hours.  Groups with one episode carry no pair."""
    import xgboost as xgb

    pr = rr._Prep(X, cols)
    m = np.asarray(event, bool)
    Z, y, g = pr(X)[m], np.log(np.clip(t[m], rr.MIN_H, None)), np.asarray(group)[m]
    order = np.argsort(g, kind="stable")
    Z, y, g = Z[order], y[order], g[order]
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= 2
    Z, y, g = Z[keep], y[keep], g[keep]
    if len(g) < 4:
        return lambda Xn: np.zeros(len(Xn))
    rel = pd.Series(y).groupby(g).rank(method="dense").astype(int).to_numpy() - 1
    _, cnt = np.unique(g, return_counts=True)
    d = xgb.DMatrix(Z, label=rel)
    d.set_group(cnt)
    params = {"objective": "rank:pairwise", "max_depth": depth, "learning_rate": eta, "reg_lambda": reg_lambda, "min_child_weight": 0.0,
              "tree_method": "hist", "nthread": 1, "seed": seed, "verbosity": 0}
    bst = xgb.train(params, d, rounds)
    return lambda Xn: bst.predict(xgb.DMatrix(pr(Xn)))


def design_list() -> list[tuple[str, str, str, object]]:
    out = [(f"ridge_{s}_a{a:g}", "ridge", s, a) for s in SETS for a in ALPHAS]
    out += [(f"aft_{s}_a{a:g}", "aft", s, a) for s in ("so", "so_em", "so_all") for a in ALPHAS]
    out += [(f"ipcw_{s}_a{a:g}", "ipcw", s, a) for s in ("so", "so_all") for a in (10.0, 100.0)]
    out += [(f"xgbaft_{s}_d{d}n{n}", "xgbaft", s, (d, n)) for s in ("so", "so_geo", "so_em", "so_all") for d, n in XGB_GRID]
    out += [(f"rank_{s}_d{d}n{n}", "rank", s, (d, n)) for s in ("so_em", "so_all") for d, n in RANK_GRID]
    return out


def complexity(name: str, mae: float = 0.0) -> tuple:
    """Smaller = simpler: references < ridge < aft < ipcw < xgbaft < rank < blends/ensembles; inside a family the feature set (in the order
    of SETS), then the larger penalty (stronger shrinkage = simpler) / the smaller forest."""
    if name in REFERENCES:
        return (0, REFERENCES.index(name), 0, 0.0, mae)
    if name.startswith(("blend", "ens_")):
        base = re.sub(r"^blend\d+_", "", name)
        return (10 + complexity(base)[0] if not name.startswith("ens_") else 20, 0, 0, 0.0, mae)
    m = re.fullmatch(r"(ridge|aft|ipcw)_(.+)_a([0-9.eE+-]+)", name)
    if m:
        fam = ["ridge", "aft", "ipcw"].index(m.group(1))
        return (1 + fam, SET_RANK[m.group(2)], 0, -float(m.group(3)), mae)
    m = re.fullmatch(r"(xgbaft|rank)_(.+)_d(\d+)n(\d+)", name)
    if m:
        return (4 if m.group(1) == "xgbaft" else 5, SET_RANK[m.group(2)], int(m.group(3)) * 1000 + int(m.group(4)), 0.0, mae)
    return (30, 0, 0, 0.0, mae)


def forecast_all(train: pd.DataFrame, test: pd.DataFrame, designs: list, min_oblast: int) -> pd.DataFrame:
    t, ev = train["hours"].to_numpy(float), train["observed"].to_numpy(bool)
    glob = float(np.median(t[ev]))
    obs = train[train["observed"]].sort_values("strike_at")
    by = obs.groupby("oblast")["hours"].agg(["median", "size"])
    last10 = obs.groupby("oblast")["hours"].apply(lambda x: float(np.median(x.tail(10))))
    out = pd.DataFrame(index=test.index)
    out["global_median"] = glob
    out["oblast_median"] = [float(by.loc[o, "median"]) if o in by.index and by.loc[o, "size"] >= min_oblast else glob for o in test["oblast"]]
    out["oblast_last10_median"] = [float(last10[o]) if o in by.index and by.loc[o, "size"] >= min_oblast else glob for o in test["oblast"]]
    grp = train["idx"].to_numpy()
    for name, fam, s, p in designs:
        cols = SETS[s] if isinstance(s, str) else list(s)      # a design may carry its own column list (energy_map_alerts_weather)
        Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
        if fam == "ridge":
            out[name] = rr.fit_ridge(Xtr, t, ev, p, cols)(Xte)
        elif fam == "aft":
            out[name] = rr.fit_aft(Xtr, t, ev, p, cols)(Xte)
        elif fam == "ipcw":
            out[name] = rr.fit_ipcw_ridge(Xtr, t, ev, p, cols)(Xte)
        elif fam == "xgbaft":
            out[name] = rr.fit_xgb_aft(Xtr, t, ev, p[0], p[1], cols)(Xte)
        else:
            out[name] = fit_xgb_rank(Xtr, t, ev, grp, p[0], p[1], cols)(Xte)
    return out


def _one_strike(i: int, train: pd.DataFrame, test: pd.DataFrame, designs: list, min_oblast: int) -> tuple[int, pd.DataFrame]:
    return i, forecast_all(train, test, designs, min_oblast)


def walk_forward(tab: pd.DataFrame, designs: list, min_obs: int, min_oblast: int, n_jobs: int = 6) -> pd.DataFrame:
    """Forecast of the pairs of strike i from the pairs of strikes j < i with labels as known just before the issue time of strike i."""
    import os
    from joblib import Parallel, delayed

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"                      # one BLAS thread per worker process

    tab = tab.copy()
    tab["evaluated"] = False
    jobs = []
    for i in sorted(tab["idx"].unique()):
        test = tab[tab["idx"] == i]
        train = rr.labels_as_of(tab[tab["idx"] < i], test["issue"].iloc[0] - pd.Timedelta(microseconds=1))
        if int(train["observed"].sum()) < min_obs:
            continue
        jobs.append((i, train, test))
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one_strike)(i, tr, te, designs, min_oblast) for i, tr, te in jobs)
    pred = pd.concat([p for _, p in res])
    tab.loc[pred.index, "evaluated"] = True
    return pd.concat([tab, pred.reindex(tab.index)], axis=1)


# --------------------------------------------------------------------------------------------------
# scoring: per-fold C within strike and MAE, paired differences, 1-SE selection
# --------------------------------------------------------------------------------------------------
def _log_blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return np.exp(w * np.log(np.clip(a, rr.MIN_H, None)) + (1 - w) * np.log(np.clip(b, rr.MIN_H, None)))


class Scorer:
    """Caches per-strike C parts (numerator / denominator) and per-row absolute errors of every prediction column of `ev`."""

    def __init__(self, ev: pd.DataFrame):
        self.ev = ev
        self.t, self.ob, self.g = ev["hours"].to_numpy(float), ev["observed"].to_numpy(bool), ev["idx"].to_numpy()
        self.parts: dict[str, pd.DataFrame] = {}
        self.err: dict[str, np.ndarray] = {}

    def prep(self, cols):
        for m in cols:
            if m in self.parts:
                continue
            p = self.ev[m].to_numpy(float)
            self.parts[m] = emt.c_parts(p, self.t, self.ob, self.g)
            self.err[m] = np.abs(p - self.t) if not m.startswith("rank_") else np.full(len(p), np.nan)

    def fold_stats(self, m: str, folds: list[list]) -> tuple[np.ndarray, np.ndarray]:
        pa = self.parts[m]
        c, mae = [], []
        for ids in folds:
            q = pa.loc[pa.index.intersection(ids)]
            c.append(q["num"].sum() / q["den"].sum() if q["den"].sum() > 0 else np.nan)
            sel = np.isin(self.g, ids) & self.ob
            mae.append(float(np.nanmean(self.err[m][sel])) if sel.any() else np.nan)
        return np.array(c), np.array(mae)


def _se(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def cv_table(sc: Scorer, folds: list[list], methods: list[str], base: str = "oblast_median", glob: str = "global_median") -> pd.DataFrame:
    sc.prep(set(methods) | {base, glob})
    fs = {m: sc.fold_stats(m, folds) for m in set(methods) | {base, glob}}
    rows = []
    for m in methods:
        c, a = fs[m]
        r = {"method": m, "n_folds": len(folds), "C_cv": float(np.nanmean(c)), "C_cv_se": _se(c), "MAE_cv_h": float(np.nanmean(a)) if not np.isnan(a).all() else np.nan,
             "MAE_cv_se_h": _se(a)}
        for tag, b in (("om", base), ("gm", glob)):
            dc, da = c - fs[b][0], a - fs[b][1]
            r.update({f"dC_{tag}": float(np.nanmean(dc)), f"dC_{tag}_se": _se(dc), f"dC_{tag}_better": int((dc > 0).sum()),
                      f"dMAE_{tag}_h": float(np.nanmean(da)) if not np.isnan(da).all() else np.nan, f"dMAE_{tag}_se": _se(da),
                      f"dMAE_{tag}_better": int((da < 0).sum())})
        rows.append(r)
    return pd.DataFrame(rows)


def select_one_se(tab: pd.DataFrame, what: str = "C") -> dict:
    """One-SE rule on mean C (higher better) or mean MAE (lower better) among the designs (references excluded), simplest by `complexity`."""
    col, se = ("C_cv", "C_cv_se") if what == "C" else ("MAE_cv_h", "MAE_cv_se_h")
    cand = tab[~tab["method"].isin(REFERENCES) & tab[col].notna()]
    best = cand.loc[cand[col].idxmax() if what == "C" else cand[col].idxmin()]
    s = float(best[se]) if pd.notna(best[se]) else 0.0
    within = cand[cand[col] >= best[col] - s - 1e-12] if what == "C" else cand[cand[col] <= best[col] + s + 1e-12]
    mae_col = "MAE_cv_h"
    pick = min(within.itertuples(), key=lambda r: complexity(r.method, 0.0 if pd.isna(getattr(r, mae_col)) else getattr(r, mae_col)))
    return {"what": what, "best": str(best["method"]), "best_val": float(best[col]), "best_se": s, "n_within": len(within),
            "selected": str(pick.method), "selected_val": float(getattr(pick, col)), "selected_se": float(getattr(pick, se)),
            "within": [str(m) for m in within["method"]]}


def paired_mae_bootstrap(ev: pd.DataFrame, model: str, base: str, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap over strikes of pooled MAE(model) - MAE(base) on the observed episodes."""
    o = ev[ev["observed"]]
    g = o.groupby("idx")
    a = g.apply(lambda x: float((x[model] - x["hours"]).abs().sum()))
    b = g.apply(lambda x: float((x[base] - x["hours"]).abs().sum()))
    n = g.size().to_numpy(float)
    S = len(a)
    rng = np.random.default_rng(seed)
    cnt = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(n_boot)]).astype(float)
    d = (cnt @ a.to_numpy() - cnt @ b.to_numpy()) / (cnt @ n)
    return {"diff": float((a.sum() - b.sum()) / n.sum()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def add_blends(ev: pd.DataFrame, anchors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    ev = ev.copy()
    names = []
    for m in anchors:
        if m.startswith("rank_"):
            continue
        for w in BLEND_W:
            n = f"blend{int(w * 100)}_{m}"
            ev[n] = _log_blend(ev[m].to_numpy(float), ev["oblast_median"].to_numpy(float), w)
            names.append(n)
    return ev, names


def add_ensembles(ev: pd.DataFrame, tab: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    ev = ev.copy()
    t = tab[~tab["method"].isin(REFERENCES) & ~tab["method"].str.startswith(("rank_", "blend"))].copy()
    members = {"ens_top3_C": list(t.sort_values("C_cv", ascending=False)["method"].head(3)),
               "ens_top3_MAE": list(t.sort_values("MAE_cv_h")["method"].head(3))}
    for n, mem in members.items():
        ev[n] = np.exp(np.mean([np.log(np.clip(ev[m].to_numpy(float), rr.MIN_H, None)) for m in mem], axis=0))
    return ev, members


def run_improve(cfg: Config, k: int | None = None, n_boot: int = 2000, seed: int = 42, n_jobs: int = 6) -> dict:
    tc = cfg.telegram
    k = int(k or 8)
    tab = build_dev_table(cfg)
    designs = design_list()
    log.info("dev table: %d pairs (%d observed) in %d strikes; %d designs", len(tab), int(tab["observed"].sum()), tab["idx"].nunique(), len(designs))
    wf = walk_forward(tab, designs, int(tc.region_dev_min_obs), int(tc.region_min_oblast), n_jobs)
    ev = wf[wf["evaluated"]].copy()
    names = [d[0] for d in designs]
    folds = rd.make_folds(list(ev["idx"].unique()), k)
    sc = Scorer(ev)
    t0 = cv_table(sc, folds, REFERENCES + names)
    top = [m for m in t0.sort_values("C_cv", ascending=False)["method"] if m not in REFERENCES][:3] + \
          [m for m in t0.sort_values("MAE_cv_h")["method"] if m not in REFERENCES][:3] + [OLD_MODEL]
    anchors = list(dict.fromkeys(top))
    ev, blends = add_blends(ev, anchors)
    ev, members = add_ensembles(ev, t0)
    sc = Scorer(ev)
    allm = REFERENCES + names + blends + list(members)
    tab_cv = cv_table(sc, folds, allm)
    sel_c, sel_m = select_one_se(tab_cv, "C"), select_one_se(tab_cv, "MAE")
    key = list(dict.fromkeys([sel_c["selected"], sel_c["best"], sel_m["selected"], sel_m["best"], OLD_MODEL] + list(tab_cv.sort_values("C_cv", ascending=False)["method"].head(8))))
    key = [m for m in key if m not in REFERENCES]
    boot = {m: emt.paired_c_bootstrap(ev, m, "oblast_median", n_boot, seed) for m in key}
    maeb = {m: paired_mae_bootstrap(ev, m, "oblast_median", n_boot, seed) for m in key if not m.startswith("rank_")}
    boot_gm = {m: emt.paired_c_bootstrap(ev, m, "global_median", n_boot, seed) for m in key}
    # period-restricted CV: scored strikes of the (spent) old test period and of the earlier period, 4 consecutive blocks each
    st = ev.groupby("idx")["strike_at"].first()
    late = list(st.index[st >= pd.Timestamp(EM_START, tz="UTC")])
    early = list(st.index[st < pd.Timestamp(EM_START, tz="UTC")])
    per = {"2025-01..2026-06-26 (the old test period, now dev)": rd.make_folds(late, 4), "2022..2024": rd.make_folds(early, 4)}
    tab_per = {n: cv_table(sc, fo, allm) for n, fo in per.items()}
    out = {"table": tab_cv, "folds": folds, "per": tab_per, "per_folds": per, "sel_c": sel_c, "sel_mae": sel_m, "boot": boot, "boot_gm": boot_gm,
           "mae_boot": maeb, "key": key, "members": members, "anchors": anchors, "ev": ev,
           "info": {"pairs": len(tab), "observed": int(tab["observed"].sum()), "strikes": int(tab["idx"].nunique()), "scored": int(ev["idx"].nunique()),
                    "first": ev["strike_at"].min(), "last": ev["strike_at"].max(), "k": k, "n_boot": n_boot, "min_obs": int(tc.region_dev_min_obs),
                    "fold_sizes": [(len(ids), int(ev["idx"].isin(ids).sum()), int((ev["idx"].isin(ids) & ev["observed"]).sum())) for ids in folds]}}
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    tab_cv.to_csv(d / "energy_map_improve_dev_cv.csv", index=False)
    for n, t in tab_per.items():
        t.to_csv(d / f"energy_map_improve_dev_period_{n.split(' ')[0].replace('..', '_').replace('-', '')}.csv", index=False)
    ev.drop(columns=[c for c in ev.columns if c.startswith("sev_") or c.startswith("early_")]).to_csv(d / "energy_map_improve_dev_predictions.csv", index=False)
    write_report(d / "energy_map_improve_dev.md", out)
    return out


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
def _f(x, nd=3):
    return "" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _row(m: str, r) -> str:
    return (f"| {m} | {_f(r['C_cv'])} ({_f(r['C_cv_se'])}) | {_f(r['dC_om'], 3)} ({_f(r['dC_om_se'])}), {int(r['dC_om_better'])}/{int(r['n_folds'])} | "
            f"{_f(r['dC_gm'], 3)} ({_f(r['dC_gm_se'])}) | {_f(r['MAE_cv_h'], 1)} ({_f(r['MAE_cv_se_h'], 1)}) | "
            f"{_f(r['dMAE_om_h'], 1)} ({_f(r['dMAE_om_se'], 1)}), {int(r['dMAE_om_better'])}/{int(r['n_folds'])} | "
            f"{_f(r['dMAE_gm_h'], 1)} ({_f(r['dMAE_gm_se'], 1)}) |")


HEAD = ["| design | C_cv (SE) | dC vs oblast_median (SE), folds better | dC vs global_median (SE) | MAE_cv h (SE) | dMAE vs oblast_median h (SE), folds better | dMAE vs global_median h (SE) |",
        "|---|---|---|---|---|---|---|"]


def _tbl(t: pd.DataFrame, methods: list[str]) -> list[str]:
    x = t.set_index("method")
    return HEAD + [_row(m, x.loc[m]) for m in methods]


def write_report(path, o: dict) -> None:
    t, i, sc_, sm = o["table"], o["info"], o["sel_c"], o["sel_mae"]
    cand = t[~t["method"].isin(REFERENCES)]
    L = ["# Energy Map restoration forecast: improvement study (DEV; rolling-origin CV over ALL strikes up to 2026-06-26)", "",
         "The frozen test of `docs/preregistration_energy_map.md` (strikes 2025-01-01 .. 2026-06-26) is spent, so these strikes are now development "
         "data. Label, cohort and walk-forward are exactly those of the old test (first Energy Map restoration report, 30 d window, labels as known at "
         "the issue time, warm-up %d observed episodes). The next blind test (strikes from 2026-06-27, a fresher Energy Map file) has not been looked at; "
         "its draft protocol is `docs/preregistration_energy_map_v2_draft.md`." % i["min_obs"], "",
         f"Dev pairs {i['pairs']} ({i['observed']} observed) in {i['strikes']} strikes; scored {i['scored']} strikes ({i['first']:%Y-%m-%d} .. {i['last']:%Y-%m-%d}). "
         f"{i['k']} expanding-window folds (strikes / pairs / observed): " + "; ".join(f"{a} / {b} / {c}" for a, b, c in i["fold_sizes"]) + ".", "",
         "Metrics: C = Harrell's C within strike (pairs of one strike, ties 0.5; constant forecast = 0.5), pooled inside a fold; MAE on observed episodes, pooled "
         "inside a fold. Cells: mean over folds (SE over folds). Paired columns: per-fold difference to the reference (positive dC = better, negative dMAE = "
         "better), mean (SE) and number of folds where the design is better. `rank_*` models give an order only (no hours, MAE blank).", "",
         "Features (all at the issue time = end of the strike day): `em_*` = Energy Map history of the oblast (known durations of earlier episodes: last-3 / last-10 "
         "median, recency-weighted mean, count; blackout/restoration records dated STRICTLY BEFORE the strike day: days since last, blackout days in the last 7 / 30 days, "
         "consumers of the last record, oblasts with a blackout record in the last 7 days); `geo_*` = front-line flag, 4-level proximity tier, border flag; "
         "`is_af`, `sin_doy`, `cos_doy` (context); `hist_*` = previous-episode features of `region_restoration`. Sets: so = strike + oblast dummies (the old "
         "model `ridge_so_a100`), so_geo = so + geo + context, so_em = so + em, so_emh = so + hist + em, so_all = everything.", "",
         "## 1. Selection (one-SE rule, simplest within one SE)", "",
         f"- **C**: best `{sc_['best']}` C = {sc_['best_val']:.4f} (SE {sc_['best_se']:.4f}); {sc_['n_within']} designs within one SE; "
         f"**selected `{sc_['selected']}`** C = {sc_['selected_val']:.4f} (SE {sc_['selected_se']:.4f}).",
         f"- **MAE** (separate design for hours): best `{sm['best']}` {sm['best_val']:.1f} h (SE {sm['best_se']:.1f}); {sm['n_within']} within one SE; "
         f"**selected `{sm['selected']}`** {sm['selected_val']:.1f} h (SE {sm['selected_se']:.1f}).",
         "- With ~%d designs on one CV the best numbers are optimistic (selection effect); blends/ensembles use members and weights picked on the same CV." % len(cand), ""]
    key = o["key"]
    show = list(dict.fromkeys(list(t[t["method"].isin(cand["method"])].sort_values("C_cv", ascending=False)["method"].head(12)) + key))
    L += ["## 2. CV table (8 folds), by C: top 12 plus the selected / best / old designs; references at the bottom", ""]
    L += _tbl(t, show + REFERENCES) + [""]
    L += ["## 3. Pooled bootstrap over strikes (paired with oblast_median; all scored dev strikes, 2000 resamples, seed 42)", "",
          "| design | C pooled | C oblast_median | dC [95% CI] | dC vs global_median [95% CI] | MAE pooled h | dMAE vs oblast_median h [95% CI] |", "|---|---|---|---|---|---|---|"]
    for m in key:
        b, g = o["boot"][m], o["boot_gm"][m]
        mb = o["mae_boot"].get(m)
        mae = o["ev"].loc[o["ev"]["observed"]]
        mp = float((mae[m] - mae["hours"]).abs().mean()) if not m.startswith("rank_") else np.nan
        L.append(f"| {m} | {b['C_model']:.3f} | {b['C_base']:.3f} | {b['diff']:+.4f} [{b['diff_lo']:+.4f}, {b['diff_hi']:+.4f}] | "
                 f"{g['diff']:+.4f} [{g['diff_lo']:+.4f}, {g['diff_hi']:+.4f}] | {_f(mp, 1)} | " +
                 (f"{mb['diff']:+.1f} [{mb['lo']:+.1f}, {mb['hi']:+.1f}]" if mb else "") + " |")
    L += ["", "## 4. Period-restricted CV (4 consecutive blocks of the scored strikes of the period; the models are still the walk-forward ones)", "",
          "**The period 2025-01..2026-06-26 was the frozen test of the previous protocol and is now development data; numbers on it are NOT a blind result.**", ""]
    for n, tp in o["per"].items():
        fo = o["per_folds"][n]
        L += [f"### {n}", "", f"{len(sum(fo, []))} scored strikes in {len(fo)} blocks of " + ", ".join(str(len(x)) for x in fo) + " strikes.", ""]
        L += _tbl(tp, [m for m in key if m in set(tp["method"])] + REFERENCES) + [""]
    L += ["## 5. What helps: best C per family and feature set (8-fold CV)", "",
          "| family_set | best design | C_cv (SE) | dC vs oblast_median | MAE_cv h |", "|---|---|---|---|---|"]
    x = cand[~cand["method"].str.startswith(("blend", "ens_"))].copy()
    x["fs"] = x["method"].str.replace(r"_(a[0-9.]+|d\d+n\d+)$", "", regex=True)
    for _, r in x.loc[x.groupby("fs")["C_cv"].idxmax()].sort_values("C_cv", ascending=False).iterrows():
        L.append(f"| {r['fs']} | {r['method']} | {_f(r['C_cv'])} ({_f(r['C_cv_se'])}) | {_f(r['dC_om'])} | {_f(r['MAE_cv_h'], 1)} |")
    L += ["", "## 6. Blends with oblast_median (log space; weight on the model) and ensembles", "",
          f"Anchors (top 3 by C, top 3 by MAE, old model): {', '.join(o['anchors'])}. Ensemble members: " +
          "; ".join(f"`{k_}` = {' + '.join(v)}" for k_, v in o["members"].items()) + ".", ""]
    bl = cand[cand["method"].str.startswith(("blend", "ens_"))].sort_values("MAE_cv_h")
    L += _tbl(t, list(bl["method"].head(12))) + [""]
    L += ["## 7. Caveats", "",
          "- The label is the first restoration report (maybe partial), event records, day resolution; frontline records are dominated by shelling.",
          "- Features from records use only records dated before the strike day; the cohort itself (blackout reported on D or D+1) is label-side, so all numbers are conditional on it.",
          "- ~%d designs on one CV; the winner is optimistic. Pairs of one strike are dependent: bootstrap is over strikes, folds are blocks of strikes." % len(cand), ""]
    Path(path).write_text("\r\n".join(L), encoding="utf-8")
