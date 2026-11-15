"""Dev-only binary task: "after a strike, will this oblast's restoration take LONGER than T hours?" (risk of a long outage).

Thresholds T in {24, 72, 168} h (`telegram.region_long_hours`); PRIMARY T = 72 h (`telegram.region_long_primary`), fixed before the
test period was looked at. Test strikes never enter (the caller passes dev rows only).

Label of an episode at a given instant (`binary_labels`):
  1  (long)     the duration is > T: observed with hours > T, or still unresolved (censored) at an elapsed time > T (known long)
  0  (short)    observed with hours <= T
  NaN (unknown) censored at a time <= T (it may still turn out either way) -> excluded from training and scoring
The TRAINING labels of strike i are taken as known at the issue time of strike i (`rr.labels_as_of`, strikes j < i only). The evaluated
label is the final label (`hours`/`observed` of the table).

Learners (same walk-forward as the regression designs, same feature sets; imputation / scaling statistics from the training rows only)
  bin_const        training base rate (share of long among the labelled training rows)
  bin_oblast       base rate of the oblast (labelled training rows of the oblast; global base rate if fewer than `min_oblast`)
  logit_<set>_a<alpha>   L2 logistic regression (sklearn C = 1/alpha); constant base rate if the training rows have < 3 of either class
  every regression design (and ensemble): the predicted hours used as a score (AUC only; it is not a probability)

Metrics (per T): ROC AUC (pooled over all scored episodes, which also compares episodes of different strikes, so a per-strike constant
such as the base rate scores != 0.5 when the base rate drifts over time; therefore also AUC within strikes = pos/neg pairs of the same strike; mean +- SE over folds; bootstrap CI over strikes), Brier score and
Brier skill = 1 - Brier / Brier(bin_const) per fold (mean +- SE over folds), per-strike paired Brier difference vs bin_const (Wilcoxon).
Selection: best mean Brier skill, one-standard-error rule toward the simpler design.

`ranking_bootstrap`: bootstrap over strikes of Harrell's C (within strikes and over all comparable pairs) of the regression designs.
Both use the exact block trick: the statistic is a ratio of quadratic forms in the per-strike resample counts, so every method is
resampled in the same replicates at negligible cost.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import region_restoration as rr

MIN_CLASS = 3                                  # logistic regression needs at least this many training rows of each class
BIN_BASE = "bin_const"
BIN_BASELINES = ["bin_const", "bin_oblast"]
SET_ORDER = ["strike", "strike_oblast", "strike_oblast_hist"]


def binary_labels(hours, observed, T: float) -> np.ndarray:
    """1 = long (> T, observed or censored beyond T), 0 = observed <= T, NaN = censored at <= T (unknown)."""
    h, ob = np.asarray(hours, float), np.asarray(observed, bool)
    y = np.full(len(h), np.nan)
    y[h > T] = 1.0
    y[ob & (h <= T)] = 0.0
    return y


def logit_names(alphas: list[float]) -> list[str]:
    return [f"logit_{s}_a{a:g}" for s in rr.feature_sets() for a in alphas]


def bin_methods(alphas: list[float]) -> list[str]:
    return BIN_BASELINES + logit_names(alphas)


def fit_logit(X: np.ndarray, y: np.ndarray, alpha: float, cols: list[str]):
    """L2 logistic regression on the labelled rows; returns P(long). Constant base rate if a class is nearly absent."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, int)
    if min(int(y.sum()), int((1 - y).sum())) < MIN_CLASS:
        p = float(y.mean()) if len(y) else 0.5
        return lambda Xn: np.full(len(Xn), p)
    pr = rr._Prep(X, cols)
    m = LogisticRegression(C=1.0 / alpha, max_iter=2000).fit(pr(X), y)
    return lambda Xn: m.predict_proba(pr(Xn))[:, 1]


def predict_strike(train: pd.DataFrame, test: pd.DataFrame, T: float, alphas: list[float], min_oblast: int) -> pd.DataFrame:
    """Probabilities of every binary learner for the rows `test` from `train` (labels as known at the issue time)."""
    y = binary_labels(train["hours"], train["observed"], T)
    k = np.isfinite(y)
    base = float(y[k].mean()) if k.any() else 0.5
    out = pd.DataFrame(index=test.index)
    out["bin_const"] = base
    g = pd.DataFrame({"oblast": train["oblast"].to_numpy()[k], "y": y[k]}).groupby("oblast")["y"].agg(["mean", "size"])
    out["bin_oblast"] = [float(g.loc[o, "mean"]) if o in g.index and g.loc[o, "size"] >= min_oblast else base for o in test["oblast"]]
    for s, cols in rr.feature_sets().items():
        Xtr, Xte = train[cols].to_numpy(float)[k], test[cols].to_numpy(float)
        for a in alphas:
            out[f"logit_{s}_a{a:g}"] = fit_logit(Xtr, y[k], a, cols)(Xte) if k.any() else base
    return out


def binary_walk_forward(dev: pd.DataFrame, scored_idx, alphas: list[float], thresholds, min_oblast: int) -> dict[float, pd.DataFrame]:
    """For every scored strike i: train on the rows of strikes j < i with labels as known at the issue time of i. Returns per T a frame
    indexed like `dev` rows of the scored strikes with the binary learners' probabilities and the FINAL binary label `y`."""
    out = {T: [] for T in thresholds}
    for i in sorted(scored_idx):
        test = dev[dev["idx"] == i]
        train = rr.labels_as_of(dev[dev["idx"] < i], test["issue"].iloc[0] - pd.Timedelta(microseconds=1))
        for T in thresholds:
            p = predict_strike(train, test, T, alphas, min_oblast)
            p["y"] = binary_labels(test["hours"], test["observed"], T)
            out[T].append(p)
    return {T: pd.concat(v) for T, v in out.items()}


# --------------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------------
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC (ties 0.5); NaN if one class is absent."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(np.asarray(s, float))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2)) if len(y) else np.nan


def _se(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def _wilcoxon_p(x: np.ndarray, min_n: int = 8) -> float:
    from scipy.stats import wilcoxon
    return float(wilcoxon(x).pvalue) if len(x) >= min_n and (x != 0).any() else np.nan


def _auc_blocks(y: np.ndarray, s: np.ndarray, g: np.ndarray, S: int):
    """Per-strike block matrices of the AUC numerator: A[a, b] = concordance of positives of strike a vs negatives of strike b."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    conc = (s[pos][:, None] > s[neg][None, :]).astype(np.float32) + 0.5 * (s[pos][:, None] == s[neg][None, :])
    Pp, Pn = np.zeros((len(pos), S), np.float32), np.zeros((len(neg), S), np.float32)
    Pp[np.arange(len(pos)), g[pos]] = 1.0
    Pn[np.arange(len(neg)), g[neg]] = 1.0
    return Pp.T @ conc @ Pn


def _ratio_boot(A: np.ndarray, D: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    num = np.einsum("bs,bs->b", cnt @ A, cnt)
    den = np.einsum("bs,bs->b", cnt @ D, cnt)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def binary_scores(bw: pd.DataFrame, ev: pd.DataFrame, folds: list[list], score_methods: list[str], alphas: list[float],
                  n_boot: int = 1000, seed: int = 42) -> dict:
    """All binary metrics for one T. `bw`: binary_walk_forward output (probabilities + y), `ev`: scored rows with the regression
    predictions (same index)."""
    d = bw.join(ev[["idx"] + [m for m in score_methods if m not in bw]])
    d = d[d["y"].notna()].copy()
    prob = bin_methods(alphas)
    methods = prob + [m for m in score_methods if m not in prob]
    y = d["y"].to_numpy(float)
    info = {"n_labelled": len(d), "n_long": int(y.sum()), "n_short": int((1 - y).sum()), "n_unknown": int(len(bw) - len(d)),
            "long_share": float(y.mean()) if len(d) else np.nan, "n_strikes": int(d["idx"].nunique())}
    fold_rows = []
    for fi, ids in enumerate(folds):
        f = d[d["idx"].isin(ids)]
        yf = f["y"].to_numpy(float)
        bc = brier(yf, f[BIN_BASE])
        for m in methods:
            r = {"fold": fi, "method": m, "n": len(f), "n_long": int(yf.sum()), "AUC": auc(yf, f[m].to_numpy(float))}
            if m in prob:
                b = brier(yf, f[m])
                r.update({"Brier": b, "BSS": 1 - b / bc if bc > 0 else np.nan})
            fold_rows.append(r)
    ft = pd.DataFrame(fold_rows)
    # bootstrap over strikes (AUC)
    ids, g = np.unique(d["idx"].to_numpy(), return_inverse=True)
    S = len(ids)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(np.float32) if S else np.zeros((n_boot, 0), np.float32)
    npos = np.bincount(g, weights=y, minlength=S).astype(np.float32)
    nneg = np.bincount(g, weights=1 - y, minlength=S).astype(np.float32)
    D = np.outer(npos, nneg)
    rows = []
    d_c = ((d[BIN_BASE] - d["y"]) ** 2)
    for m in methods:
        s = d[m].to_numpy(float)
        fm = ft[ft["method"] == m]
        row = {"method": m, "AUC_pooled": auc(y, s), "AUC_fold_mean": fm["AUC"].mean(), "AUC_fold_se": _se(fm["AUC"].to_numpy(float))}
        if S > 1 and 0 < y.sum() < len(y):
            A = _auc_blocks(y, s, g, S)
            bs = _ratio_boot(A, D, cnt)
            dw, Dw = np.diag(A), np.diag(D)
            if Dw.sum() > 0:                         # AUC within strikes: positive vs negative pairs of the same strike only
                bwn = (cnt @ dw) / np.where(cnt @ Dw > 0, cnt @ Dw, np.nan)
                row.update({"AUC_within": float(dw.sum() / Dw.sum()), "AUC_within_lo": float(np.nanpercentile(bwn, 2.5)),
                            "AUC_within_hi": float(np.nanpercentile(bwn, 97.5)),
                            "p_AUC_within_gt_half": float((1 + np.nansum(bwn <= 0.5)) / (np.isfinite(bwn).sum() + 1))})
            row.update({"AUC_ci_lo": float(np.nanpercentile(bs, 2.5)), "AUC_ci_hi": float(np.nanpercentile(bs, 97.5)),
                        "p_AUC_gt_half": float((1 + np.nansum(bs <= 0.5)) / (np.isfinite(bs).sum() + 1))})
        if m in prob:
            row.update({"Brier": brier(y, s), "BSS_fold_mean": fm["BSS"].mean(), "BSS_fold_se": _se(fm["BSS"].to_numpy(float)),
                        "BSS_pooled": 1 - brier(y, s) / brier(y, d[BIN_BASE]) if brier(y, d[BIN_BASE]) > 0 else np.nan})
            if m != BIN_BASE:
                ps = (((d[m] - d["y"]) ** 2) - d_c).groupby(d["idx"]).mean()
                row.update({"brier_diff_vs_base": float(ps.mean()), "share_better_strikes": float((ps < 0).mean()),
                            "p_brier_vs_base": _wilcoxon_p(ps.to_numpy())})
        rows.append(row)
    table = pd.DataFrame(rows)
    return {"table": table, "fold_table": ft, "info": info, "selection": select_binary(table)}


def bin_complexity_key(name: str, bss: float = 0.0) -> tuple:
    if name in BIN_BASELINES:
        return (0, BIN_BASELINES.index(name), 0.0, -bss)
    m = re.fullmatch(r"logit_(.+)_a([0-9.eE+-]+)", name)
    if not m:
        return (9, 0, 0.0, -bss)
    fs = m.group(1)
    return (1 + (SET_ORDER.index(fs) if fs in SET_ORDER else len(SET_ORDER)), 0, -float(m.group(2)), -bss)


def select_binary(table: pd.DataFrame) -> dict:
    """Best mean Brier skill over folds; one-SE rule toward the simpler design (bin_const < bin_oblast < logit_strike < _oblast < _hist <
    other sets; larger alpha simpler; then higher skill)."""
    cand = table[table["BSS_fold_mean"].notna()]
    if cand.empty:
        return {}
    best = cand.loc[cand["BSS_fold_mean"].idxmax()]
    se = float(best["BSS_fold_se"]) if pd.notna(best["BSS_fold_se"]) else 0.0
    thr = float(best["BSS_fold_mean"]) - se
    within = cand[cand["BSS_fold_mean"] >= thr - 1e-12]
    pick = min(within.itertuples(), key=lambda r: bin_complexity_key(r.method, r.BSS_fold_mean))
    return {"best": str(best["method"]), "best_bss": float(best["BSS_fold_mean"]), "best_se": se, "threshold": thr,
            "n_within": len(within), "selected": str(pick.method), "selected_bss": float(pick.BSS_fold_mean)}


# --------------------------------------------------------------------------------------------------
# ranking test of the regression designs
# --------------------------------------------------------------------------------------------------
def _c_blocks(pred: np.ndarray, M: np.ndarray, starts: np.ndarray) -> np.ndarray:
    conc = M * ((pred[:, None] < pred[None, :]).astype(np.float32) + 0.5 * (pred[:, None] == pred[None, :]))
    return np.add.reduceat(np.add.reduceat(conc, starts, axis=0), starts, axis=1)


def ranking_bootstrap(ev: pd.DataFrame, methods: list[str], n_boot: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Bootstrap over strikes of Harrell's C: C_within (comparable pairs inside a strike) and C_all (all comparable pairs) with 95% CI
    and the one-sided bootstrap p-value of C > 0.5. Comparable: i observed and hours_i < hours_j (final labels of the scored episodes)."""
    e = ev.sort_values("idx", kind="stable")
    t, ob, g = e["hours"].to_numpy(float), e["observed"].to_numpy(bool), e["idx"].to_numpy()
    ids, inv = np.unique(g, return_inverse=True)
    S = len(ids)
    starts = np.r_[0, np.flatnonzero(np.diff(inv)) + 1]
    M = (ob[:, None] & (t[:, None] < t[None, :])).astype(np.float32)
    Dm = np.add.reduceat(np.add.reduceat(M, starts, axis=0), starts, axis=1)
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(np.float32)
    rows = []
    for m in methods:
        A = _c_blocks(e[m].to_numpy(float), M, starts)
        cw = lambda c: (c @ np.diag(A)) / (c @ np.diag(Dm))                  # noqa: E731  (within-strike C of the resample)
        pt_w, pt_a = float(np.trace(A) / np.trace(Dm)), float(A.sum() / Dm.sum())
        bw, ba = cw(cnt), _ratio_boot(A, Dm, cnt)
        rows.append({"method": m, "n_strikes": S, "C_within": pt_w,
                     "C_within_lo": float(np.nanpercentile(bw, 2.5)), "C_within_hi": float(np.nanpercentile(bw, 97.5)),
                     "p_C_within_gt_half": float((1 + np.nansum(bw <= 0.5)) / (np.isfinite(bw).sum() + 1)),
                     "C_all": pt_a, "C_all_lo": float(np.nanpercentile(ba, 2.5)), "C_all_hi": float(np.nanpercentile(ba, 97.5)),
                     "p_C_all_gt_half": float((1 + np.nansum(ba <= 0.5)) / (np.isfinite(ba).sum() + 1))})
    return pd.DataFrame(rows).sort_values("C_within", ascending=False).reset_index(drop=True)
