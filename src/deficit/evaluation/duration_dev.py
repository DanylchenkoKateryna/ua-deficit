"""Development mode of `deficit restoration --dev`: national restriction-regime duration, rolling-origin CV, TEST BLIND.

Target     `days_to_resolution` of the national episode (`telegram_episodes.parquet`, scope ALL; see `restoration.py`): strike ->
           first Ukrenergo post saying that restrictions are not applied any more (else "all consumers have power"); right-censored
           when no such post appears within the 30-day window. Run on the extended strike list (`config/telegram_extended.yaml`:
           attacks.csv + days with attack reports in the continuous channels). Strikes in a chain share one lifting post, so the
           strikes are NOT independent; p-values and CIs below are optimistic (the bootstrap resamples blocks of consecutive strikes).

Scope      Only strikes up to `telegram.region_validation_end` (test start 2025-01-01). Test-period strikes are removed before anything
           is built; posts are cut at validation end + 30 days (only to complete the 30-day window of the last dev strikes). No test
           strike is scored, printed or written.

Timing     The forecast of strike i is issued at the END of the strike day (`restoration.issue_time`).
  * Training labels: strikes j < i with their label AS KNOWN at the issue time (observed if the lifting post is older than the
    issue time, else right-censored at the time watched). No later strike and no later post enters.
  * Features of strike i use only: the strike list up to strike i (weapons_through of Air Force strikes, gap to the previous
    strike, strikes in the last 30 days, heating season), weather (observed weather = perfect weather forecast, stated assumption)
    and HISTORY features computed from posts published BEFORE 00:00 UTC of the strike day (before any post of the strike day,
    so nothing from the same day beyond what the strike list gives). `n_channels` / `n_attack_posts` of derived strikes are
    whole-day counts (they use the posts of the rest of the day), so they are NOT features (`FORBIDDEN`, guarded by a test).
  * The target (ex-post label) is only used for scoring.
  * Damage features (`dmg_*`, ingestion/damage_features.py): object classes (TPP/CHP, HPP, NPP, transmission substation, HV line, distribution)
    named in attack-report posts of Ukrenergo / DTEK published from 00:00 UTC of the strike day up to the issue time (end of the strike day;
    strictly `< issue`), number of oblasts hit, repeat of an object type seen in the 30 days before the strike day, queue count announced for
    "tomorrow", capacity lost (MW). No later post is read (tests/test_no_leakage.py).
  * Episode level (second report `telegram_restoration_dev_episodes.md`): strikes with a calendar-day gap <= 2 days to the previous strike
    form one chain episode; the episode is forecast at the issue time of its first strike, trained on earlier episodes only.

Predictors  (days)
  median_obs    median of the observed durations known at the issue time              (baseline)
  last_episode  duration of the most recent strike whose label is known to be observed (persistence)
  prev_strike   label of the immediately previous strike if known to be observed, else `last_episode` (persistence)
  km_global     Kaplan-Meier median of all labels known so far (censoring aware, capped at 30 days)
  ridge_<set>_a<alpha>   ridge on log(duration), observed strikes only
  aft_<set>_a<alpha>     log-normal AFT with right censoring, L2 penalty
  ipcw_<set>_a<alpha>    ridge weighted by 1/G(t) (inverse probability of censoring weights)
  xgbaft_<set>_d<D>n<N>  boosted trees, objective survival:aft
  const_30               always 30 days (reference only: shows that the lower-bound metric alone can be gamed)
Feature sets: strike (log_through, is_af, gap_prev, n_strikes_30, heating, temp_24h), hist (previous label, observed flag, median of
the last 5 known durations, number of unresolved strikes), strike_hist (both); dmg (damage features), strike_dmg, strike_hist_dmg.

CV         The scored strikes (>= `telegram.duration_dev_min_obs` observed known labels) are split into K consecutive blocks. Per
           design and fold: MAE on observed strikes (days), censoring-aware lower-bound MAE over all strikes, Harrell's C over strikes.
           Reported: mean over folds and SE over folds (std ddof=1 / sqrt(K)). Paired: per-strike |error| difference vs `median_obs`
           on observed strikes, Wilcoxon, block-bootstrap CI (blocks of `BLOCK` consecutive strikes).
Selection  One-standard-error rule on the CV MAE (observed strikes): threshold = best MAE + its SE; among the designs within it the
           simplest (complexity order: median_obs < last_episode < prev_strike < km_global < ridge < aft by feature set < ipcw < xgbaft;
           larger penalty = simpler). `const_30` is not selectable.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.damage_features import DMG_F, load_damage_posts, damage_features
from ..ingestion.telegram_extract import EpisodeParams, add_derived, load_strikes
from . import region_dev as rd
from . import region_restoration as rr
from .restoration import HORIZON, MIN_DAYS, issue_time, km_median, label_as_of

log = logging.getLogger(__name__)

STRIKE_F = ["log_through", "is_af", "gap_prev", "n_strikes_30", "heating", "temp_24h"]
HIST_F = ["prev_log_t", "prev_obs", "hist_logmed5", "n_unresolved"]
SETS = {"strike": STRIKE_F, "hist": HIST_F, "strike_hist": STRIKE_F + HIST_F,
        "dmg": DMG_F, "strike_dmg": STRIKE_F + DMG_F, "strike_hist_dmg": STRIKE_F + HIST_F + DMG_F}
SET_ORDER = list(SETS)
BASELINES = ["median_obs", "last_episode", "prev_strike", "km_global"]
REFERENCE_ONLY = ["const_30"]
# whole-day counts of derived strikes (posts of the rest of the day) and other columns that must never be features
FORBIDDEN = ["n_channels", "n_attack_posts", "n_attack_reports", "n_posts", "derived", "near_major", "description"]
XGB_SETS = ("strike_hist", "strike_hist_dmg")
XGB_GRID = [(1, 25), (1, 60), (2, 25), (2, 60)]
BLOCK = 10
rr.CONTINUOUS.update({"prev_log_t", "hist_logmed5", "n_unresolved", "dmg_n_types", "dmg_n_oblasts", "dmg_repeat", "dmg_queue_next", "dmg_cap_lost"})     # standardised by rr._Prep (names are unique to this module)


# ---------------------------------------------------------------------------------------- labels at any instant
class LabelBook:
    """Labels of the strikes as they were known at an arbitrary instant, fast: a strike whose 30-day window has ended is final."""

    def __init__(self, posts: pd.DataFrame, strikes: pd.DataFrame, prm: EpisodeParams):
        self.prm = prm
        self.p = posts.sort_values("datetime").reset_index(drop=True)
        self.nat = self.p[self.p["channel"].isin(prm.national_channels)].reset_index(drop=True)
        self.tp = self.p["datetime"].to_numpy(dtype="datetime64[ns]")
        self.tn = self.nat["datetime"].to_numpy(dtype="datetime64[ns]")
        self.at = [pd.Timestamp(s) for s in strikes["strike_at"]]
        self.start = [s.floor("D") for s in self.at]
        self.end = np.array([(s + pd.Timedelta(days=prm.horizon_days)).to_datetime64() for s in self.start])
        self._final: dict[int, tuple[float, bool, float]] = {}

    def _slice(self, j: int):
        lo, hi = self.start[j].to_datetime64(), self.end[j]
        a, b = np.searchsorted(self.tp, lo, "left"), np.searchsorted(self.tp, hi, "right")
        c, d = np.searchsorted(self.tn, lo, "left"), np.searchsorted(self.tn, hi, "right")
        return self.p.iloc[a:b], self.nat.iloc[c:d]

    def label(self, j: int, as_of) -> tuple[float, bool, float]:
        """(days, observed, censor_days) of strike j from posts published at or before `as_of`."""
        as_of = pd.Timestamp(as_of)
        if as_of.to_datetime64() >= self.end[j] and j in self._final:
            return self._final[j]
        p, nat = self._slice(j)
        out = label_as_of(p, nat, self.at[j], pd.NaT, self.prm, as_of)
        if as_of.to_datetime64() >= self.end[j]:
            self._final[j] = out
        return out

    def final(self, j: int) -> tuple[float, bool, float]:
        """Ex-post label (full window)."""
        return self.label(j, pd.Timestamp(self.end[j]))

    def known(self, i: int, as_of) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(days, observed, window still open at as_of) of the strikes j < i as known at `as_of`."""
        a = pd.Timestamp(as_of).to_datetime64()
        labs = [self.label(j, as_of) for j in range(i)]
        t = np.array([x[0] for x in labs], float)
        ev = np.array([x[1] for x in labs], bool)
        return t, ev, self.end[:i] > a


# ---------------------------------------------------------------------------------------- features
def episode_start_mask(strikes: pd.DataFrame, gap_days: int = 2) -> np.ndarray:
    """True for the first strike of a chain episode: strikes whose calendar-day gap to the previous strike is <= `gap_days` are merged.
    Whether strike i starts an episode depends on the previous strike only (known at its issue time)."""
    d = pd.to_datetime(strikes["strike_at"], utc=True).dt.floor("D").reset_index(drop=True)
    gap = d.diff().dt.days
    return (gap.isna() | (gap > gap_days)).to_numpy()


def duration_features(strikes: pd.DataFrame, book: LabelBook, weather: pd.DataFrame | None,
                      dposts: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per strike (same order), every value known before the forecast of that strike is issued.
    Strike-list columns other than weapons_through / strike_at are never read (`FORBIDDEN` columns cannot leak in).
    History features come from labels as known at 00:00 UTC of the strike day (before any post of that day).
    Damage features (`dmg_*`) come from `dposts` (posts of the day of the strike, up to its issue time; without `dposts` they are 0)."""
    base = rr.strike_table(strikes[["strike_at"] + [c for c in ("weapons_through",) if c in strikes]], weather)
    out = base[["log_through", "gap_prev", "n_strikes_30", "heating", "temp_24h"]].copy()
    through = strikes["weapons_through"] if "weapons_through" in strikes else pd.Series(np.nan, index=strikes.index)
    out["is_af"] = through.notna().astype(float).to_numpy()
    dmg = damage_features(strikes, dposts) if dposts is not None else pd.DataFrame(0.0, index=strikes.index, columns=DMG_F)
    out = pd.concat([out, dmg.reset_index(drop=True)], axis=1)
    return pd.concat([out, history_features(strikes, book)], axis=1)[STRIKE_F + HIST_F + DMG_F]


def history_features(strikes: pd.DataFrame, book: LabelBook) -> pd.DataFrame:
    rows = []
    for i in range(len(strikes)):
        day0 = book.start[i] - pd.Timedelta(microseconds=1)
        t, ev, still_open = book.known(i, day0)
        r = {"prev_log_t": np.nan, "prev_obs": np.nan, "hist_logmed5": np.nan, "n_unresolved": 0.0}
        if i:
            r["prev_log_t"] = float(np.log(max(t[-1], MIN_DAYS)))
            r["prev_obs"] = float(ev[-1])
            r["n_unresolved"] = float((~ev & still_open).sum())
            if ev.any():
                r["hist_logmed5"] = float(np.median(np.log(np.clip(t[ev][-5:], MIN_DAYS, None))))
        rows.append(r)
    return pd.DataFrame(rows)


def episode_features(F: pd.DataFrame, mask: np.ndarray, ep_strikes: pd.DataFrame, ep_book: LabelBook) -> pd.DataFrame:
    """Features of the chain episodes: strike-list and damage features of the FIRST strike (window = its own day, up to its own issue time;
    later strikes of the chain are not read), history features recomputed over the earlier episodes only."""
    F2 = F.loc[mask].reset_index(drop=True).copy()
    H = history_features(ep_strikes, ep_book)
    F2[HIST_F] = H[HIST_F].to_numpy()
    return F2


# ---------------------------------------------------------------------------------------- models
def designs(alphas: list[float]) -> list[tuple[str, str, str, object]]:
    d = [(f"{m}_{s}_a{a:g}", m, s, a) for m in ("ridge", "aft", "ipcw") for s in SETS for a in alphas]
    d += [(f"xgbaft_{s}_d{dp}n{n}", "xgbaft", s, (dp, n)) for s in XGB_SETS for dp, n in XGB_GRID]
    return d


def all_methods(alphas: list[float]) -> list[str]:
    return BASELINES + REFERENCE_ONLY + [d[0] for d in designs(alphas)]


def _fit_predict(model: str, par, X: np.ndarray, t: np.ndarray, ev: np.ndarray, Xn: np.ndarray, cols: list[str]) -> float:
    th = t * 24.0                                                       # region_restoration models work in hours
    if model == "ridge":
        f = rr.fit_ridge(X, th, ev, par, cols)
    elif model == "aft":
        f = rr.fit_aft(X, th, ev, par, cols)
    elif model == "ipcw":
        f = rr.fit_ipcw_ridge(X, th, ev, par, cols)
    else:
        f = rr.fit_xgb_aft(X, th, ev, par[0], par[1], cols)
    return float(np.clip(f(Xn)[0] / 24.0, MIN_DAYS, HORIZON))


def forecast_one(F: pd.DataFrame, i: int, t: np.ndarray, ev: np.ndarray, alphas: list[float]) -> dict[str, float]:
    """All predictors for strike i from the labels (t, ev) of the strikes before it (as known at the issue time)."""
    last = float(t[np.flatnonzero(ev)[-1]])
    out = {"median_obs": float(np.median(t[ev])), "last_episode": last,
           "prev_strike": float(t[-1]) if ev[-1] else last, "km_global": km_median(t, ev, HORIZON), "const_30": HORIZON}
    for name, model, s, par in designs(alphas):
        cols = SETS[s]
        out[name] = _fit_predict(model, par, F.loc[: i - 1, cols].to_numpy(float), t, ev, F.loc[[i], cols].to_numpy(float), cols)
    return out


def walk_forward(strikes: pd.DataFrame, book: LabelBook, F: pd.DataFrame, alphas: list[float], min_obs: int) -> pd.DataFrame:
    rows = []
    for i in range(len(strikes)):
        t, ev, _ = book.known(i, issue_time(strikes["strike_at"].iloc[i]) - pd.Timedelta(microseconds=1)) if i else (np.array([]), np.array([], bool), None)
        days, observed, censor = book.final(i)
        row = {"idx": i, "strike_at": strikes["strike_at"].iloc[i], "days": days, "observed": observed, "censor_days": censor,
               "is_af": bool(F["is_af"].iloc[i]), "n_train": int(len(t)), "n_train_observed": int(ev.sum()), "evaluated": bool(ev.sum() >= min_obs)}
        if row["evaluated"]:
            row.update(forecast_one(F, i, t, ev, alphas))
        rows.append(row)
        if i % 50 == 0:
            log.info("duration dev: strike %d/%d (%s)", i, len(strikes), pd.Timestamp(row["strike_at"]).date())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- metrics
def harrell_c(pred: np.ndarray, t: np.ndarray, ev: np.ndarray) -> float:
    """Harrell's C over strikes: pair (a, b) comparable if a is observed and t[a] < t[b]; ties in the prediction count 0.5."""
    pred, t, ev = np.asarray(pred, float), np.asarray(t, float), np.asarray(ev, bool)
    comp = ev[:, None] & (t[:, None] < t[None, :])
    if not comp.any():
        return np.nan
    conc = (pred[:, None] < pred[None, :]) + 0.5 * (pred[:, None] == pred[None, :])
    return float(conc[comp].sum() / comp.sum())


def loss_lb(pred: pd.Series, s: pd.DataFrame) -> pd.Series:
    """|error| on observed strikes, max(0, censor time - prediction) on censored ones (a lower bound of the true error), days."""
    return pd.Series(np.where(s["observed"], (pred - s["days"]).abs(), np.maximum(s["censor_days"] - pred, 0.0)), index=pred.index)


def _fold_rows(ev: pd.DataFrame, folds: list[list], methods: list[str]) -> pd.DataFrame:
    rows = []
    for fi, ids in enumerate(folds):
        s = ev[ev["idx"].isin(ids)]
        ob = s["observed"].to_numpy(bool)
        if not ob.any():
            continue
        for m in methods:
            rows.append({"fold": fi, "method": m, "n_strikes": len(s), "n_observed": int(ob.sum()),
                         "MAE_d": float((s[m] - s["days"]).abs()[ob].mean()), "LB_d": float(loss_lb(s[m], s).mean()),
                         "C": harrell_c(s[m].to_numpy(), s["days"].to_numpy(), ob)})
    return pd.DataFrame(rows)


def block_bootstrap(d: np.ndarray, block: int = BLOCK, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    """95% CI of the mean of a time-ordered series, resampling blocks of consecutive items (moving blocks)."""
    rng = np.random.default_rng(seed)
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    b = max(1, min(block, n))
    starts = np.arange(n - b + 1)
    k = int(np.ceil(n / b))
    means = [np.concatenate([d[s:s + b] for s in rng.choice(starts, size=k)])[:n].mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def cv_table(ev: pd.DataFrame, folds: list[list], methods: list[str], base: str = "median_obs") -> tuple[pd.DataFrame, pd.DataFrame]:
    fr = _fold_rows(ev, folds, methods)
    obs = ev[ev["observed"]]
    rows = []
    for m in methods:
        f = fr[fr["method"] == m]
        n = len(f)
        se = lambda c: f[c].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan  # noqa: E731
        row = {"method": m, "n_folds": n, "MAE_cv_d": f["MAE_d"].mean(), "MAE_cv_se_d": se("MAE_d"), "LB_cv_d": f["LB_d"].mean(),
               "LB_cv_se_d": se("LB_d"), "C_cv": f["C"].mean(), "C_cv_se": se("C"),
               "C_pooled": harrell_c(ev[m].to_numpy(), ev["days"].to_numpy(), ev["observed"].to_numpy(bool))}
        if m != base:
            d = ((obs[m] - obs["days"]).abs() - (obs[base] - obs["days"]).abs()).to_numpy()
            lb = (loss_lb(ev[m], ev) - loss_lb(ev[base], ev)).to_numpy()
            lo, hi = block_bootstrap(d)
            row.update({"diff_vs_base_d": float(d.mean()), "diff_ci_lo": lo, "diff_ci_hi": hi, "share_better": float((d < 0).mean()),
                        "n_paired": len(d), "p_wilcoxon": rd._wilcoxon_p(d), "diff_lb_vs_base_d": float(lb.mean()),
                        "p_wilcoxon_lb": rd._wilcoxon_p(lb)})
        rows.append(row)
    return pd.DataFrame(rows), fr


def complexity_key(name: str, mae: float = 0.0) -> tuple:
    """Smaller = simpler (see the module docstring)."""
    if name in BASELINES:
        return (0, BASELINES.index(name), 0, 0.0, mae)
    x = re.fullmatch(r"xgbaft_(.+)_d(\d+)n(\d+)", name)
    if x:
        return (8, SET_ORDER.index(x.group(1)), int(x.group(2)) * 1000 + int(x.group(3)), 0.0, mae)
    m = re.fullmatch(r"(ipcw|ridge|aft)_(.+)_a([0-9.eE+-]+)", name)
    if not m:
        return (10, 0, 0, 0.0, mae)
    fam, fs, alpha = m.group(1), m.group(2), float(m.group(3))
    if fam == "ipcw":
        return (7, SET_ORDER.index(fs), 0, -alpha, mae)
    return (1 + SET_ORDER.index(fs), ["ridge", "aft"].index(fam), 0, -alpha, mae)


def select_one_se(table: pd.DataFrame) -> dict:
    cand = table[~table["method"].isin(REFERENCE_ONLY) & table["MAE_cv_d"].notna()]
    best = cand.loc[cand["MAE_cv_d"].idxmin()]
    se = float(best["MAE_cv_se_d"]) if pd.notna(best["MAE_cv_se_d"]) else 0.0
    thr = float(best["MAE_cv_d"]) + se
    within = cand[cand["MAE_cv_d"] <= thr + 1e-12]
    pick = min(within.itertuples(), key=lambda r: complexity_key(r.method, r.MAE_cv_d))
    return {"best": str(best["method"]), "best_mae": float(best["MAE_cv_d"]), "best_se": se, "threshold": thr,
            "n_within": len(within), "selected": str(pick.method), "selected_mae": float(pick.MAE_cv_d)}


def dev_cv(wf: pd.DataFrame, alphas: list[float], k: int = 5, base: str = "median_obs") -> dict:
    ev = wf[wf["evaluated"]].reset_index(drop=True)
    if ev.empty:
        raise ValueError("no scored dev strikes")
    methods = all_methods(alphas)
    folds = rd.make_folds(list(ev["idx"]), k)
    table, fr = cv_table(ev, folds, methods, base)
    table = table.sort_values("MAE_cv_d").reset_index(drop=True)
    sel = select_one_se(table)
    groups = []
    for name, g in (("Air Force strikes", ev[ev["is_af"]]), ("derived strikes", ev[~ev["is_af"]])):
        if g["observed"].any():
            groups.append({"group": name, "n": len(g), "n_observed": int(g["observed"].sum()),
                           **{m: float((g.loc[g["observed"], m] - g.loc[g["observed"], "days"]).abs().mean()) for m in (base, sel["selected"], sel["best"])}})
    info = {"n_strikes": len(wf), "n_scored": len(ev), "n_observed_all": int(wf["observed"].sum()), "n_observed_scored": int(ev["observed"].sum()),
            "folds": folds, "first_scored": ev["strike_at"].min(), "last_scored": ev["strike_at"].max(), "base": base,
            "fold_sizes": [{"fold": i, "strikes": len(ids), "observed": int((ev["idx"].isin(ids) & ev["observed"]).sum())} for i, ids in enumerate(folds)],
            "groups": groups}
    return {"table": table, "fold_table": fr, "selection": sel, "info": info}


# ---------------------------------------------------------------------------------------- report
def write_dev_report(path, res: dict, k: int, min_obs: int, top: int = 10, header: str = "") -> None:
    t, s, i = res["table"], res["selection"], res["info"]
    f = rd._fmt
    base = i["base"]
    b = t.set_index("method")
    L = ["# National restriction-regime duration: DEV report (rolling-origin CV on strikes up to 2024-12-31, test period blind)", "",
         (header + " " if header else "") +
         f"Extended strike list ({i['n_strikes']} dev strikes, {i['n_observed_all']} with an observed lifting post within 30 days). Strikes with at least "
         f"{min_obs} observed past durations are scored: {i['n_scored']} strikes ({i['n_observed_scored']} observed), {i['first_scored']:%Y-%m-%d} .. {i['last_scored']:%Y-%m-%d}. "
         f"CV: expanding window, K = {k} consecutive blocks; every strike is forecast at the end of its day from earlier strikes only, with their labels as known at that "
         "instant. MAE on observed strikes (days; mean over folds, SE over folds). LB = censoring-aware lower-bound MAE over all strikes (a censored strike counts "
         "max(0, time watched - forecast)); `const_30` shows that LB alone can be gamed. C = Harrell's C over strikes (0.5 = no ranking skill). "
         f"Paired columns: per-strike |error| difference vs `{base}` on observed strikes (negative = better), 95% CI by a moving-block bootstrap (blocks of {BLOCK} "
         "consecutive strikes, 2000 resamples, seed 42), Wilcoxon p. Strikes in a chain share one lifting post, so p-values and CIs are optimistic. "
         "Features: strike list (weapons_through for Air Force strikes, gap, strikes in 30 days, heating, temperature) and history of earlier labels as known before "
         "00:00 UTC of the strike day; whole-day counts (n_channels, n_attack_posts) are not features. No test-period number is computed.", "",
         "Folds: " + "; ".join(f"#{x['fold']}: {x['strikes']} strikes / {x['observed']} observed" for x in i["fold_sizes"]), "",
         f"Best CV MAE: `{s['best']}` {s['best_mae']:.2f} d (SE {s['best_se']:.2f}); 1-SE threshold {s['threshold']:.2f} d; {s['n_within']} designs within it. "
         f"**Selected (simplest within 1 SE): `{s['selected']}`** ({s['selected_mae']:.2f} d).", ""]
    head = ["design", "MAE_cv d", "SE", "LB_cv d", "C_cv", "C_pooled", f"diff vs {base} d [CI95]", "better strikes", "p Wilcoxon", "p LB"]
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    keep = list(t["method"].head(top)) + [m for m in BASELINES + REFERENCE_ONLY if m not in set(t["method"].head(top))]
    for r in t[t["method"].isin(keep)].itertuples():
        ci = f"{f(getattr(r, 'diff_vs_base_d', np.nan), 2)} [{f(getattr(r, 'diff_ci_lo', np.nan), 2)}, {f(getattr(r, 'diff_ci_hi', np.nan), 2)}]" if r.method != base else ""
        L.append(f"| {r.method} | {f(r.MAE_cv_d, 2)} | {f(r.MAE_cv_se_d, 2)} | {f(r.LB_cv_d, 2)} | {f(r.C_cv, 3)} | {f(r.C_pooled, 3)} | {ci} | "
                 f"{f(getattr(r, 'share_better', np.nan), 2)} | {f(getattr(r, 'p_wilcoxon', np.nan), 3)} | {f(getattr(r, 'p_wilcoxon_lb', np.nan), 3)} |")
    fam = t.assign(f=t["method"].str.extract(r"^(xgbaft|ipcw|aft|ridge)")[0]).dropna(subset=["f"])
    best_f = fam.loc[fam.groupby("f")["MAE_cv_d"].idxmin()].sort_values("MAE_cv_d")
    best_c = t[~t["method"].isin(REFERENCE_ONLY)].sort_values("C_cv", ascending=False).head(3)
    fs = t.assign(s=t["method"].str.extract(r"_(strike_hist_dmg|strike_dmg|strike_hist|dmg|hist|strike)(?:_|$)")[0]).dropna(subset=["s"])
    best_s = fs.loc[fs.groupby("s")["MAE_cv_d"].idxmin()].sort_values("MAE_cv_d")
    L += ["", "Best design per feature set (CV MAE; `dmg` = damage features: object classes named in the posts of the strike day, repeats, oblasts, queues): "
          + "; ".join(f"{x.s}: `{x.method}` {x.MAE_cv_d:.2f} d (SE {x.MAE_cv_se_d:.2f}) / C {x.C_cv:.3f}" for x in best_s.itertuples())]
    L += ["", "Best design per family (CV MAE): " + "; ".join(f"`{x.method}` {x.MAE_cv_d:.2f} d / C {x.C_cv:.3f}" for x in best_f.itertuples()),
          "Highest C_cv: " + "; ".join(f"`{x.method}` {x.C_cv:.3f} (SE {x.C_cv_se:.3f})" for x in best_c.itertuples()), ""]
    if i["groups"]:
        cols = [base, s["selected"], s["best"]]
        L += ["MAE (days, observed strikes, all scored dev strikes pooled) by strike type:", "",
              "| type | strikes | observed | " + " | ".join(dict.fromkeys(cols)) + " |", "|" + "---|" * (3 + len(dict.fromkeys(cols)))]
        for g in i["groups"]:
            L.append(f"| {g['group']} | {g['n']} | {g['n_observed']} | " + " | ".join(f(g[c], 2) for c in dict.fromkeys(cols)) + " |")
        L.append("")
    sel_row = b.loc[s["selected"]] if s["selected"] in b.index else None
    verdict = _verdict(res, sel_row, b)
    L += ["## Verdict (dev only)", ""] + verdict + [""]
    from pathlib import Path
    Path(path).write_text("\n".join(L), encoding="utf-8")


def _verdict(res: dict, sel_row, b: pd.DataFrame) -> list[str]:
    s, base = res["selection"], res["info"]["base"]
    cand = b.drop(index=[m for m in BASELINES + REFERENCE_ONLY if m in b.index])
    gain = cand.loc[cand["MAE_cv_d"].idxmin()]
    out = []
    if s["selected"] in BASELINES:
        out.append(f"The 1-SE rule selects the baseline `{s['selected']}`: no model is distinguishable from the best design within one SE of the CV MAE.")
    else:
        out.append(f"The 1-SE rule selects `{s['selected']}`.")
    cm = b.loc[base]
    out.append(f"Median baseline: MAE {cm.MAE_cv_d:.2f} d (SE {cm.MAE_cv_se_d:.2f}). Best model `{gain.name}`: {gain.MAE_cv_d:.2f} d (SE {gain.MAE_cv_se_d:.2f}), "
               f"per-strike difference {gain.diff_vs_base_d:+.2f} d, block-bootstrap CI [{gain.diff_ci_lo:+.2f}, {gain.diff_ci_hi:+.2f}], Wilcoxon p = {gain.p_wilcoxon:.3f}, "
               f"C {gain.C_cv:.3f} (SE {gain.C_cv_se:.3f}).")
    sig = cand[(cand["diff_ci_hi"] < 0) & (cand["p_wilcoxon"] < 0.05)]
    out.append(f"Designs whose bootstrap CI of the difference to the median lies entirely below 0 and Wilcoxon p < 0.05: {len(sig)} of {len(cand)} "
               "(p-values not corrected for the choice among designs; strikes in a chain are dependent).")
    return out


# ---------------------------------------------------------------------------------------- entry point
def run_duration_dev(cfg: Config, k: int | None = None) -> dict:
    tc = cfg.telegram
    k = int(k or tc.region_cv_folds)
    root = cfg._abs(cfg.paths.data_dir)
    pdir = cfg.telegram_processed_dir
    posts = pd.read_parquet(pdir / "telegram_posts.parquet")
    posts = add_derived(posts) if "impact_post" not in posts.columns else posts
    strikes = load_strikes(cfg.strike_list_path)
    vend = pd.Timestamp(tc.region_validation_end, tz="UTC")
    cut = vend + pd.Timedelta(hours=23)                                    # same boundary as the region dev mode
    strikes = strikes[pd.to_datetime(strikes["strike_at"], utc=True) <= cut].reset_index(drop=True)
    horizon = pd.Timedelta(days=tc.episode_horizon_days)
    posts = posts[posts["datetime"] <= vend.floor("D") + pd.Timedelta(days=1) + horizon]   # only to complete the window of the last dev strikes
    wpath = root / "raw" / "weather.parquet"
    weather = pd.read_parquet(wpath) if wpath.exists() else None
    prm = EpisodeParams(horizon_days=tc.episode_horizon_days, relapse_days=tc.relapse_days,
                        national_channels=tuple(tc.national_channels), pre_strike_days=tc.pre_strike_days)
    book = LabelBook(posts, strikes, prm)
    rawdir = root / "raw" / "telegram"
    dposts = load_damage_posts(rawdir, posts) if rawdir.exists() else None
    if dposts is None:
        log.warning("no raw telegram texts: damage features are 0 (dmg designs carry no information)")
    F = duration_features(strikes, book, weather, dposts)
    wf = walk_forward(strikes, book, F, tc.duration_alphas, int(tc.duration_dev_min_obs))
    res = dev_cv(wf, tc.duration_alphas, k)
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(d / "telegram_restoration_dev.csv", index=False)
    res["fold_table"].to_csv(d / "telegram_restoration_dev_folds.csv", index=False)
    write_dev_report(d / "telegram_restoration_dev.md", res, k, int(tc.duration_dev_min_obs))
    log.info("[duration dev] selected %s (best %s)", res["selection"]["selected"], res["selection"]["best"])
    # chain episodes: strikes with a gap <= 2 days to the previous one are merged; the episode is forecast at the issue time of its FIRST strike
    mask = episode_start_mask(strikes)
    ep = strikes[mask].reset_index(drop=True)
    ep_book = LabelBook(posts, ep, prm)
    Fe = episode_features(F, mask, ep, ep_book)
    wfe = walk_forward(ep, ep_book, Fe, tc.duration_alphas, int(tc.duration_dev_min_obs))
    rese = dev_cv(wfe, tc.duration_alphas, k)
    rese["table"].to_csv(d / "telegram_restoration_dev_episodes.csv", index=False)
    write_dev_report(d / "telegram_restoration_dev_episodes.md", rese, k, int(tc.duration_dev_min_obs),
                     header=f"EPISODE LEVEL: {len(strikes)} dev strikes merged into {len(ep)} chain episodes (gap <= 2 days; first strike = episode start; "
                            "features of the first strike only, history over earlier episodes only; training rows are earlier episodes). 'Strikes' below = episodes.")
    log.info("[duration dev, episodes] selected %s (best %s)", rese["selection"]["selected"], rese["selection"]["best"])
    res["episodes"] = rese
    return res
