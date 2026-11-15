"""Forecast of the restoration duration per (strike, oblast) from the oblast-level Telegram dataset (`deficit region-restoration`).

Target   `hours_to_resolution` of `ingestion/telegram_regional` (strike -> the oblast's restrictions are lifted / power is
         restored; episodes are watched until the NEXT strike day or 30 days, whichever is first). Right-censored otherwise.
Cohort   the (strike, oblast) pairs that the issue-time information already flags: an impact post for the oblast (consumers cut
         off, emergency outages or schedules) published before the issue time and not yet resolved by then.
Issue    end of the strike day (00:00 UTC of the next day). Features use posts published before that instant only.
         Strikes are processed in time order; the model of strike i is fitted on the cohort episodes of the strikes j < i. Their
         labels are final at that time: episode j stops at the next strike day, which is not later than the day of strike i.
         (A 30-day window would NOT be safe: its lifting post could be published after the issue time of a later strike.)

Baselines  global_median   median of the observed durations of all past episodes
           oblast_median   median of the observed durations of past episodes of the same oblast (global median if < `min_oblast` of them)
           geo_mean        exp(mean log duration) of past observed episodes = an intercept-only ridge (reference: shows how much of a
                           ridge gain is only the log-scale centre instead of the median)
           km_global       Kaplan-Meier median of all past episodes (uses the censored ones; reference only)
Models     ridge_<set>_a<alpha>  ridge on log(hours) of the observed episodes
           aft_<set>_a<alpha>    log-normal accelerated failure time model, right-censoring aware, L2 penalty
Feature sets (all known at the issue time):
  strike         strike strength log1p(weapons_through), days since the previous strike, strikes in the last 30 days,
                 heating season, mean temperature of the 24 h before the issue time
  strike_oblast  strike + oblast dummies
  strike_oblast_hist  strike_oblast + oblast history known at the issue time (see `add_history`): previous cohort episode of the
                 oblast (label as known at the issue time, observed flag, missing flag), number of prior episodes, days since
                 that strike, "restriction regime already in force before the strike day" (from posts only)
  strike_oblast_early / strike_oblast_hist_early  strike_oblast (+ hist) + "early impact" of the oblast (see `early_features`): what its posts
                 said within the first K hours (`telegram.region_early_hours`, 3 or 6, chosen on the dev CV only) of the strike day
  full           strike_oblast + severity in the oblast's posts of the strike day (max consumers without power, attack
                 reports, damaged object kinds, emergency outages, schedule posts, number of posts, oblasts hit nationwide) +
                 recent activity (impact posts in the 30 days before, days since the last impact post, regime already in force)
  full_no_oblast full without the dummies
Two label windows (`variant`), both reported:
  next_strike  the episode is watched until the next strike day (or 30 days): a restoration after the next strike belongs to the
               next episode. Strict, but most episodes are censored (strikes come every few days in 2025-26).
  30d          the episode is watched for 30 days whatever happens (a lifting post may then end several strikes at once, as in
               `telegram_episodes.parquet`). More observed episodes. Training labels are taken AS KNOWN at the issue time of the
               forecast strike (observed only if the resolution post is older than the issue time, else right-censored at
               "issue time - strike"), so no post after the issue time enters; the evaluated target is the final label.
The design is chosen on the VALIDATION strikes only (lowest MAE on observed episodes); nothing is tuned on the test strikes.

Scores: MAE on observed episodes (hours and days), a censoring-aware lower bound over all episodes and Harrell's C computed
within strikes only (does the forecast rank the oblasts of one strike correctly; ties count 0.5, so a constant forecast scores 0.5). Observed-only
MAE is biased in favour of short predictions (long episodes are censored more often); the lower bound is biased the other way (a
constant 30-day forecast scores well), so both are shown. Episodes of one strike are strongly dependent (the same national
lifting post ends several oblasts): the paired tests are run per episode (optimistic) and on per-strike means, and a
bootstrap over strikes gives a confidence interval of the MAE difference.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.telegram_extract import load_strikes
from ..ingestion.telegram_regional import OBLASTS
from .restoration import issue_time, km_median

log = logging.getLogger(__name__)

HORIZON_H = 30 * 24.0
MIN_H = 1.0
STRIKE_F = ["log_through", "gap_prev", "n_strikes_30", "heating", "temp_24h"]
SEVERITY_F = ["sev_log_consumers", "sev_attack_posts", "sev_object_kinds", "sev_emergency", "sev_schedule_posts", "sev_posts",
              "sev_oblasts_hit"]
ACTIVITY_F = ["act_impact_posts_30d", "act_days_since_impact", "act_regime_in_force"]
EARLY_F = ["early_log_consumers", "early_attack_posts", "early_impact_posts", "early_object_kinds", "early_emergency_schedule",
           "early_any_impact", "early_first_impact_h", "early_own_channel"]
EARLY_CONTINUOUS = ["early_log_consumers", "early_attack_posts", "early_impact_posts", "early_object_kinds", "early_first_impact_h"]
EARLY_HOURS = 3.0
HIST_F = ["hist_has_prev", "hist_prev_log_hours", "hist_prev_observed", "hist_n_prior", "hist_days_since_prev", "hist_under_restr"]
CONTINUOUS = set(STRIKE_F + SEVERITY_F + ACTIVITY_F + ["hist_prev_log_hours", "hist_n_prior", "hist_days_since_prev"] + EARLY_CONTINUOUS)
BASELINES = ["global_median", "oblast_median"]
REFERENCES = BASELINES + ["geo_mean", "km_global", "const_30d"]


def dummies() -> list[str]:
    return [f"ob_{o}" for o in OBLASTS]


def feature_sets() -> dict[str, list[str]]:
    return {"strike": STRIKE_F, "strike_oblast": STRIKE_F + dummies(), "strike_oblast_hist": STRIKE_F + dummies() + HIST_F,
            "strike_oblast_early": STRIKE_F + dummies() + EARLY_F, "strike_oblast_hist_early": STRIKE_F + dummies() + HIST_F + EARLY_F, "full": STRIKE_F + SEVERITY_F + ACTIVITY_F + dummies(),
            "full_no_oblast": STRIKE_F + SEVERITY_F + ACTIVITY_F}


# --------------------------------------------------------------------------------------------------
# features at the issue time
# --------------------------------------------------------------------------------------------------
def strike_table(strikes: pd.DataFrame, weather: pd.DataFrame | None) -> pd.DataFrame:
    """One row per strike (same order): strength and calendar/weather values known at the issue time."""
    t = pd.to_datetime(strikes["strike_at"], utc=True).reset_index(drop=True)
    out = pd.DataFrame({"strike_at": t, "idx": range(len(t))})
    out["issue"] = [issue_time(s) for s in t]
    through = strikes["weapons_through"].astype(float).reset_index(drop=True) if "weapons_through" in strikes else pd.Series(np.nan, index=out.index)
    out["log_through"] = np.log1p(through)
    out["gap_prev"] = (t.diff().dt.total_seconds() / 86400).clip(upper=60).fillna(60)
    out["n_strikes_30"] = [int(((t < s) & (t >= s - pd.Timedelta(days=30))).sum()) for s in t]
    out["heating"] = t.dt.month.isin([10, 11, 12, 1, 2, 3]).astype(float)
    temp = np.full(len(t), np.nan)
    if weather is not None and "temperature_C" in weather:
        w = weather["temperature_C"].dropna().sort_index()
        for k, iss in enumerate(out["issue"]):
            win = w[(w.index > iss - pd.Timedelta(hours=24)) & (w.index <= iss - pd.Timedelta(microseconds=1))]
            if len(win):
                temp[k] = float(win.mean())
    out["temp_24h"] = temp
    return out


def _impact_mask(f: pd.DataFrame) -> pd.Series:
    return f["attack"].astype(bool) | f["schedules"].astype(bool) | f["emergency"].astype(bool) | f["consumers_without_power"].notna()


def issue_features(f: pd.DataFrame, f_all: pd.DataFrame, strike_at, issue) -> dict:
    """Severity / activity of one oblast from its fact rows `f` (and all oblasts `f_all` for the breadth) published before
    `issue`. Nothing at or after `issue` is read."""
    day = pd.Timestamp(strike_at).floor("D")
    f = f[f["datetime"] < issue]
    today = f[f["datetime"] >= day]
    cons = pd.to_numeric(today["consumers_without_power"], errors="coerce")
    kinds = {o for lst in today["damaged_objects"] for o in lst}
    imp = f[_impact_mask(f)] if len(f) else f
    prev30 = imp[(imp["datetime"] >= day - pd.Timedelta(days=30)) & (imp["datetime"] < day)]
    prev = imp[imp["datetime"] < day]
    pre3 = imp[(imp["datetime"] >= day - pd.Timedelta(days=3)) & (imp["datetime"] < day)]
    a = f_all[(f_all["datetime"] >= day) & (f_all["datetime"] < issue) & f_all["attack"].astype(bool)]
    return {"sev_log_consumers": float(np.log1p(cons.max())) if cons.notna().any() else 0.0,
            "sev_attack_posts": int(today["attack"].astype(bool).sum()), "sev_object_kinds": len(kinds),
            "sev_emergency": float(today["emergency"].astype(bool).any()), "sev_schedule_posts": int(today["schedules"].astype(bool).sum()),
            "sev_posts": int(len(today)), "sev_oblasts_hit": int(a["oblast"].nunique()),
            "act_impact_posts_30d": int(len(prev30)),
            "act_days_since_impact": float(min((day - prev["datetime"].max()).total_seconds() / 86400, 60.0)) if len(prev) else 60.0,
            "act_regime_in_force": float(bool((pre3["schedules"].astype(bool) | pre3["emergency"].astype(bool)).any())) if len(pre3) else 0.0}


def early_features(f: pd.DataFrame, strike_at, issue, hours: float, own_channel: bool) -> dict:
    """Early impact of one oblast from its fact rows `f`: only posts in [start of the strike day, cut) with
    cut = min(strike_at + hours, issue). Nothing at or after `cut` is read, so a late post cannot change the values.
    (The strike hour in attacks.csv is a placeholder and real strikes happen at night, hence the window starts at the day start;
    `hours` counts from `strike_at`. For a late strike of the extended list the cut is clipped at the issue time.)
    Episodes already resolved before the cut are not in the cohort at all (the cohort needs 'unresolved at the issue time' and
    cut <= issue), so the label of every cohort row starts after the cut. `early_first_impact_h`: hours from the day start to
    the first impact post in the window (= window length if none; `early_any_impact` says so)."""
    day = pd.Timestamp(strike_at).floor("D")
    cut = min(pd.Timestamp(strike_at) + pd.Timedelta(hours=hours), pd.Timestamp(issue))
    w = f[(f["datetime"] >= day) & (f["datetime"] < cut)]
    imp = w[_impact_mask(w)] if len(w) else w
    cons = pd.to_numeric(w["consumers_without_power"], errors="coerce")
    kinds = {o for lst in w["damaged_objects"] for o in lst}
    first = float((imp["datetime"].min() - day).total_seconds() / 3600) if len(imp) else float((cut - day).total_seconds() / 3600)
    return {"early_log_consumers": float(np.log1p(cons.max())) if cons.notna().any() else 0.0,
            "early_attack_posts": int(w["attack"].astype(bool).sum()), "early_impact_posts": int(len(imp)), "early_object_kinds": len(kinds),
            "early_emergency_schedule": float((w["emergency"].astype(bool) | w["schedules"].astype(bool)).any()),
            "early_any_impact": float(len(imp) > 0), "early_first_impact_h": first, "early_own_channel": float(bool(own_channel))}


def add_history(out: pd.DataFrame, facts: pd.DataFrame) -> pd.DataFrame:
    """Regional history features of every cohort row, using only what is known at its issue time.
    Previous episode = the latest cohort episode of the same oblast with an EARLIER strike. Its duration is taken as known at
    `issue - 1 us` (`labels_as_of`): the observed duration if its resolution post is older than that, otherwise the elapsed time
    so far (flag hist_prev_observed = 0); the final label is never read. Missing (no previous episode): NaN + hist_has_prev = 0,
    filled later with a training-data median (`_Prep`). `hist_under_restr`: the oblast's last schedule/emergency post before the
    strike day is newer than its last lifting post (posts older than the strike day only)."""
    out = out.sort_values(["idx", "oblast"]).reset_index(drop=True)
    cols = {c: np.full(len(out), np.nan) for c in HIST_F}
    by_ob = {ob: g.sort_values("datetime") for ob, g in facts.groupby("oblast")} if len(facts) else {}
    for ob, g in out.groupby("oblast"):
        pos = list(g.index)
        fo = by_ob.get(ob)
        for k, i in enumerate(pos):
            r = out.loc[i]
            day = r["strike_at"].floor("D")
            cols["hist_n_prior"][i] = k
            cols["hist_has_prev"][i] = float(k > 0)
            if k:
                q = out.loc[pos[k - 1]]
                as_of = r["issue"] - pd.Timedelta(microseconds=1)
                obs = pd.notna(q["res_at"]) and q["res_at"] <= as_of
                h = q["res_hours"] if obs else min((as_of - q["strike_at"]).total_seconds() / 3600, q["end_hours"])
                cols["hist_prev_log_hours"][i] = float(np.log(max(h, MIN_H)))
                cols["hist_prev_observed"][i] = float(obs)
                cols["hist_days_since_prev"][i] = float(min((r["strike_at"] - q["strike_at"]).total_seconds() / 86400, 60.0))
            else:
                cols["hist_prev_observed"][i] = 0.0
            under = 0.0
            if fo is not None:
                old = fo[fo["datetime"] < day]
                restr = old.loc[old["schedules"].astype(bool) | old["emergency"].astype(bool), "datetime"]
                lift = old.loc[old["lifted"].astype(bool), "datetime"]
                if len(restr):
                    under = float(len(lift) == 0 or lift.max() < restr.max())
            cols["hist_under_restr"][i] = under
    for c, v in cols.items():
        out[c] = v
    return out


VARIANTS = ("next_strike", "30d")


def build_table(eps: pd.DataFrame, facts: pd.DataFrame, strikes: pd.DataFrame, weather: pd.DataFrame | None,
                validation_end: pd.Timestamp, variant: str = "next_strike", early_hours: float = EARLY_HOURS) -> pd.DataFrame:
    """Cohort episodes (see module docstring) with issue-time features, final label and split.
    `res_at` / `res_hours`: resolution post time and hours from the strike (NaT / NaN = censored); `end_hours`: hours from the
    strike to the end of the watched window."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    S = strike_table(strikes, weather)
    e = eps.merge(S, on="strike_at", how="inner")
    sfx = "" if variant == "next_strike" else "_30d"
    e["res_at"] = e["resolution_at" + sfx]
    e["res_hours"] = e["hours_to_resolution" + sfx]
    e["end_hours"] = (e["window_end" + sfx] - e["strike_at"]).dt.total_seconds() / 3600
    cohort = e["first_impact_at"].notna() & (e["first_impact_at"] < e["issue"]) & ~(e["res_at"].notna() & (e["res_at"] < e["issue"]))
    e = e[cohort].copy()
    by_ob = {ob: g.sort_values("datetime").reset_index(drop=True) for ob, g in facts.groupby("oblast")}
    empty = facts.iloc[0:0]
    F = pd.DataFrame([issue_features(by_ob.get(r.oblast, empty), facts, r.strike_at, r.issue) for r in e.itertuples()], index=e.index)
    E = pd.DataFrame([early_features(by_ob.get(r.oblast, empty), r.strike_at, r.issue, early_hours, r.coverage == "own_channel")
                      for r in e.itertuples()], index=e.index)
    out = pd.concat([e[["strike_at", "idx", "issue", "oblast", "coverage", "res_at", "res_hours", "end_hours"] + STRIKE_F], F, E], axis=1)
    for o in OBLASTS:
        out[f"ob_{o}"] = (out["oblast"] == o).astype(float)
    out["observed"] = out["res_at"].notna()                                 # final label
    out["hours"] = np.where(out["observed"], out["res_hours"], out["end_hours"]).astype(float)
    out["hours"] = out["hours"].clip(lower=MIN_H)
    out["split"] = np.where(out["strike_at"] <= validation_end + pd.Timedelta(hours=23), "validation", "test")
    out["variant"] = variant
    return add_history(out, facts)


def labels_as_of(tab: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Labels of the rows of `tab` as known at `as_of`: observed iff the resolution post is not newer than `as_of`; otherwise
    right-censored at the shorter of (as_of - strike) and the end of the watched window."""
    obs = tab["res_at"].notna() & (tab["res_at"] <= as_of)
    seen = np.minimum((as_of - tab["strike_at"]).dt.total_seconds() / 3600, tab["end_hours"])
    out = tab.copy()
    out["observed"] = obs
    out["hours"] = np.where(obs, tab["res_hours"], seen).astype(float)
    out["hours"] = out["hours"].clip(lower=MIN_H)
    return out


# --------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------
class _Prep:
    """Median imputation and standardisation of the continuous columns (statistics from the training rows only)."""

    def __init__(self, X: np.ndarray, cols: list[str]):
        self.cont = np.array([c in CONTINUOUS for c in cols])
        self.med = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0.0 for c in X.T])
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu = np.where(self.cont, Xi.mean(0), 0.0)
        sd = Xi.std(0)
        self.sd = np.where(self.cont & (sd > 1e-9), sd, 1.0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd


def fit_aft(X: np.ndarray, t: np.ndarray, event: np.ndarray, alpha: float, cols: list[str]):
    """Log-normal AFT with right censoring and an L2 penalty on the coefficients; returns a median predictor (hours)."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    pr = _Prep(X, cols)
    Z, y, ev = pr(X), np.log(np.clip(t, MIN_H, None)), event.astype(bool)
    p = Z.shape[1]

    def nll(w):
        b0, beta, sig = w[0], w[1:1 + p], np.exp(w[-1])
        z = (y - b0 - Z @ beta) / sig
        lsf = norm.logsf(z)
        ll = np.where(ev, norm.logpdf(z) - np.log(sig), lsf)
        h = np.exp(norm.logpdf(z) - lsf)                       # hazard of the standard normal
        d_mu = np.where(ev, z / sig, h / sig)                  # d ll / d mu
        d_ls = np.where(ev, z * z - 1.0, h * z)                # d ll / d log(sigma)
        g = np.concatenate([[d_mu.sum()], Z.T @ d_mu, [d_ls.sum()]])
        g[1:1 + p] -= 2 * alpha * beta
        return -(ll.sum() - alpha * float(beta @ beta)), -g

    w0 = np.zeros(p + 2)
    w0[0] = float(y[ev].mean()) if ev.any() else float(y.mean())
    w0[-1] = np.log(max(float(y[ev].std()) if ev.sum() > 1 else 1.0, 0.5))
    bounds = [(None, None)] * (p + 1) + [(np.log(0.3), np.log(3.0))]
    w = minimize(nll, w0, jac=True, method="L-BFGS-B", bounds=bounds).x
    return lambda Xn: np.clip(np.exp(w[0] + pr(Xn) @ w[1:1 + p]), MIN_H, HORIZON_H)


def fit_ridge(X: np.ndarray, t: np.ndarray, event: np.ndarray, alpha: float, cols: list[str]):
    """Ridge on log(hours) of the observed episodes only (the censored ones are dropped)."""
    from sklearn.linear_model import Ridge

    pr = _Prep(X[event], cols)
    m = Ridge(alpha=alpha).fit(pr(X[event]), np.log(np.clip(t[event], MIN_H, None)))
    return lambda Xn: np.clip(np.exp(m.predict(pr(Xn))), MIN_H, HORIZON_H)


def censor_survival(t: np.ndarray, event: np.ndarray):
    """Kaplan-Meier estimate of the CENSORING distribution (censored = the "event" here), fitted on the given rows only.
    Returns G(q) = P(C >= q) evaluated just before q (left limit); at a tie an observed duration counts as earlier than a censor time."""
    t, ev = np.asarray(t, float), np.asarray(event, bool)
    cs = np.unique(t[~ev])
    steps = np.array([1.0 - ((~ev) & (t == s)).sum() / max(int((t >= s).sum()), 1) for s in cs])
    cum = np.cumprod(steps) if len(cs) else np.array([])

    def G(q):
        q = np.atleast_1d(np.asarray(q, float))
        k = np.searchsorted(cs, q, side="left")                  # number of censor times strictly before q
        return np.where(k > 0, cum[np.maximum(k - 1, 0)] if len(cum) else 1.0, 1.0)
    return G


def fit_ipcw_ridge(X: np.ndarray, t: np.ndarray, event: np.ndarray, alpha: float, cols: list[str], gmin: float = 0.2):
    """Ridge on log(hours) of the observed episodes, weighted by 1 / G(t_i) (inverse probability of censoring weights; G = Kaplan-Meier of
    the censoring distribution of the TRAINING rows, floored at `gmin`; weights are scaled to mean 1 so `alpha` keeps its meaning)."""
    from sklearn.linear_model import Ridge

    w = 1.0 / np.maximum(censor_survival(t, event)(t[event]), gmin)
    w = w / w.mean()
    pr = _Prep(X[event], cols)
    m = Ridge(alpha=alpha).fit(pr(X[event]), np.log(np.clip(t[event], MIN_H, None)), sample_weight=w)
    return lambda Xn: np.clip(np.exp(m.predict(pr(Xn))), MIN_H, HORIZON_H)


def fit_xgb_aft(X: np.ndarray, t: np.ndarray, event: np.ndarray, depth: int, rounds: int, cols: list[str], eta: float = 0.1,
                reg_lambda: float = 10.0, sigma: float = 1.0, seed: int = 42):
    """Gradient-boosted trees with the Accelerated-Failure-Time objective (`survival:aft`, normal errors on log time).
    Interval labels: observed -> [t, t], censored -> [c, +inf). Small shallow trees, strong L2, no subsampling; predicts the median
    (exp of the margin). Imputation statistics come from the training rows only."""
    import xgboost as xgb

    pr = _Prep(X, cols)
    tt = np.clip(t, MIN_H, None)
    d = xgb.DMatrix(pr(X))
    d.set_float_info("label_lower_bound", tt)
    d.set_float_info("label_upper_bound", np.where(event, tt, np.inf))
    params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": "normal",
              "aft_loss_distribution_scale": sigma, "max_depth": depth, "learning_rate": eta, "reg_lambda": reg_lambda,
              "min_child_weight": 0.0, "tree_method": "hist", "nthread": 1, "seed": seed, "verbosity": 0}
    bst = xgb.train(params, d, rounds)
    return lambda Xn: np.clip(bst.predict(xgb.DMatrix(pr(Xn))), MIN_H, HORIZON_H)


def designs(alphas: list[float]) -> list[tuple[str, str, str, float]]:
    return [(f"{m}_{s}_a{a:g}", m, s, a) for m in ("ridge", "aft") for s in feature_sets() for a in alphas]


XGB_SETS = ("strike_oblast_hist", "full", "strike_oblast_hist_early")
XGB_GRID = [(1, 25), (1, 60), (2, 25), (2, 60)]                    # (max_depth, boosting rounds); eta 0.1, lambda 10 (dev CV only)
IPCW_SETS = ("strike", "strike_oblast", "strike_oblast_hist", "strike_oblast_early", "strike_oblast_hist_early")


def extra_designs(alphas: list[float]) -> list[tuple[str, str, str, object]]:
    """Censoring-aware designs added after the original ones (only used with `extra=True`; the original designs are unchanged)."""
    out = [(f"ipcw_{s}_a{a:g}", "ipcw", s, a) for s in IPCW_SETS for a in alphas]
    out += [(f"xgbaft_{s}_d{d}n{n}", "xgbaft", s, (d, n)) for s in XGB_SETS for d, n in XGB_GRID]
    return out


def all_methods(alphas: list[float], extra: bool = False) -> list[str]:
    return REFERENCES + [d[0] for d in designs(alphas)] + ([d[0] for d in extra_designs(alphas)] if extra else [])


def forecast_strike(train: pd.DataFrame, test: pd.DataFrame, alphas: list[float], min_oblast: int, extra: bool = False) -> pd.DataFrame:
    """Predictions of every method for the rows `test` from the rows `train` (all labels known at the issue time)."""
    t, ev = train["hours"].to_numpy(float), train["observed"].to_numpy(bool)
    glob = float(np.median(t[ev]))
    km = km_median(t, ev, HORIZON_H)
    obs_by = train[train["observed"]].groupby("oblast")["hours"].agg(["median", "size"])
    out = pd.DataFrame(index=test.index)
    out["global_median"] = glob
    out["oblast_median"] = [float(obs_by.loc[o, "median"]) if o in obs_by.index and obs_by.loc[o, "size"] >= min_oblast else glob
                            for o in test["oblast"]]
    out["geo_mean"] = float(np.clip(np.exp(np.mean(np.log(np.clip(t[ev], MIN_H, None)))), MIN_H, HORIZON_H))
    out["km_global"] = km
    out["const_30d"] = HORIZON_H
    for name, model, s, a in designs(alphas):
        cols = feature_sets()[s]
        fit = fit_aft if model == "aft" else fit_ridge
        out[name] = fit(train[cols].to_numpy(float), t, ev, a, cols)(test[cols].to_numpy(float))
    if extra:
        for name, model, s, a in extra_designs(alphas):
            cols = feature_sets()[s]
            Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
            out[name] = fit_ipcw_ridge(Xtr, t, ev, a, cols)(Xte) if model == "ipcw" else fit_xgb_aft(Xtr, t, ev, a[0], a[1], cols)(Xte)
    return out


# --------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------
def loss_lb(pred: pd.Series, actual: pd.Series, observed: pd.Series) -> pd.Series:
    """|error| for observed episodes; max(0, censor time - prediction) (lower bound of the true error) for censored ones."""
    return pd.Series(np.where(observed, (pred - actual).abs(), np.maximum(actual - pred, 0.0)), index=pred.index)


def harrell_c_within(pred: np.ndarray, t: np.ndarray, event: np.ndarray, group: np.ndarray) -> float:
    """Harrell's C over the comparable pairs of the same group (strike): a is observed and t[a] < t[b]; ties in the prediction 0.5."""
    num = den = 0.0
    for g in np.unique(group):
        k = np.flatnonzero(group == g)
        for a in k:
            if not event[a]:
                continue
            for b in k:
                if b != a and t[a] < t[b]:
                    den += 1
                    num += 1.0 if pred[a] < pred[b] else 0.5 if pred[a] == pred[b] else 0.0
    return num / den if den else np.nan


def score_table(tab: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        s = tab[(tab["split"] == split) & tab["evaluated"]]
        for m in methods:
            if s.empty or m not in s:
                continue
            ob = s["observed"]
            err = (s[m] - s["hours"]).abs()
            per_strike = err[ob].groupby(s.loc[ob, "idx"]).mean()
            rows.append({"split": split, "method": m, "n_strikes": int(s.loc[ob, "idx"].nunique()), "n_episodes": len(s),
                         "n_observed": int(ob.sum()), "MAE_h": float(err[ob].mean()), "MAE_days": float(err[ob].mean() / 24),
                         "RMSE_h": float(np.sqrt((err[ob] ** 2).mean())), "MAE_per_strike_h": float(per_strike.mean()),
                         "MAE_lower_bound_all_h": float(loss_lb(s[m], s["hours"], ob).mean()),
                         "C_within_strike": harrell_c_within(s[m].to_numpy(), s["hours"].to_numpy(), ob.to_numpy(), s["idx"].to_numpy())})
    return pd.DataFrame(rows)


def choose_on_validation(scores: pd.DataFrame) -> dict[str, str]:
    """Best baseline, best design per family and the best design overall by the validation MAE on observed episodes."""
    v = scores[scores["split"] == "validation"].set_index("method")["MAE_h"]
    chosen = {}
    base = v[[m for m in BASELINES if m in v.index]]
    if len(base):
        chosen["baseline"] = base.idxmin()
    mods = v[v.index.str.contains(r"_a\d", regex=True)]
    for fam in ("ridge", "aft"):
        sub = mods[mods.index.str.startswith(fam + "_")]
        if len(sub):
            chosen[fam] = sub.idxmin()
    if len(mods):
        chosen["overall"] = mods.idxmin()
    return chosen


def paired_tests(tab: pd.DataFrame, pairs: list[tuple[str, str]], n_boot: int = 2000, seed: int = 42, min_n: int = 8) -> pd.DataFrame:
    """Wilcoxon of |error| differences (method - baseline; negative = method is closer) on observed episodes:
    per episode, and on per-strike means; plus a bootstrap over strikes (strikes are the clusters)."""
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    rows = []
    for split in ("validation", "test"):
        s = tab[(tab["split"] == split) & tab["evaluated"] & tab["observed"]]
        for m, b in pairs:
            if m == b or m not in s or b not in s or s.empty:
                continue
            d = (s[m] - s["hours"]).abs() - (s[b] - s["hours"]).abs()
            ps = d.groupby(s["idx"]).mean()
            base_mae = float((s[b] - s["hours"]).abs().mean())
            boot = []
            ids = ps.index.to_numpy()
            g = {k: v.to_numpy() for k, v in d.groupby(s["idx"])}
            for _ in range(n_boot):
                pick = rng.choice(ids, size=len(ids), replace=True)
                boot.append(np.concatenate([g[k] for k in pick]).mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            row = {"split": split, "method": m, "baseline": b, "n_episodes": len(d), "n_strikes": len(ps),
                   "mean_diff_h": float(d.mean()), "rel_diff_pct": float(100 * d.mean() / base_mae),
                   "boot_ci95_lo_h": float(lo), "boot_ci95_hi_h": float(hi), "share_better_episodes": float((d < 0).mean()),
                   "p_episodes": float(wilcoxon(d).pvalue) if len(d) >= min_n and (d != 0).any() else np.nan,
                   "mean_diff_per_strike_h": float(ps.mean()), "share_better_strikes": float((ps < 0).mean()),
                   "p_per_strike": float(wilcoxon(ps).pvalue) if len(ps) >= min_n and (ps != 0).any() else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


def run_walk_forward(tab: pd.DataFrame, alphas: list[float], min_obs: int, min_oblast: int, extra: bool = False) -> pd.DataFrame:
    """Forecast of every cohort episode from the episodes of earlier strikes only, with their labels as known at the
    issue time of the forecast strike."""
    methods = all_methods(alphas, extra)
    pred = pd.DataFrame(np.nan, index=tab.index, columns=methods)
    tab = tab.copy()
    tab["evaluated"] = False
    tab["n_train"] = 0
    tab["n_train_observed"] = 0
    for i in sorted(tab["idx"].unique()):
        test = tab[tab["idx"] == i]
        train = labels_as_of(tab[tab["idx"] < i], test["issue"].iloc[0] - pd.Timedelta(microseconds=1))
        tab.loc[test.index, "n_train"] = len(train)
        tab.loc[test.index, "n_train_observed"] = int(train["observed"].sum())
        if int(train["observed"].sum()) < min_obs:
            continue
        pred.loc[test.index] = forecast_strike(train, test, alphas, min_oblast, extra).reindex(columns=methods)
        tab.loc[test.index, "evaluated"] = True
    return pd.concat([tab, pred], axis=1)


def per_oblast_scores(tab: pd.DataFrame, methods: list[str], min_n: int = 8) -> pd.DataFrame:
    s = tab[(tab["split"] == "test") & tab["evaluated"] & tab["observed"]]
    rows = []
    for o, g in s.groupby("oblast"):
        if len(g) >= min_n:
            rows.append({"oblast": o, "n_observed_test": len(g), **{f"MAE_h_{m}": float((g[m] - g["hours"]).abs().mean()) for m in methods}})
    return pd.DataFrame(rows).sort_values("n_observed_test", ascending=False) if rows else pd.DataFrame()


# --------------------------------------------------------------------------------------------------
# data completeness
# --------------------------------------------------------------------------------------------------
def channel_completeness(raw_dir: Path, channel_oblast: dict[str, str]) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(raw_dir).glob("*.parquet")):
        d = pd.read_parquet(f, columns=["datetime"])
        t = pd.to_datetime(d["datetime"], utc=True).sort_values()
        gaps = t.diff().dt.total_seconds().div(86400)
        rows.append({"channel": f.stem, "oblast": channel_oblast.get(f.stem, "(national)"), "posts": len(t),
                     "first": t.min().date() if len(t) else None, "last": t.max().date() if len(t) else None,
                     "max_gap_days": round(float(gaps.max()), 1) if len(t) > 1 else np.nan,
                     "file_mtime_utc": pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# step entry point
# --------------------------------------------------------------------------------------------------
def run_variant(tc, eps, facts, strikes, weather, variant: str, seed: int) -> dict:
    vend = pd.Timestamp(tc.region_validation_end, tz="UTC")
    tab = build_table(eps, facts, strikes, weather, vend, variant, tc.region_early_hours)
    tab = run_walk_forward(tab, tc.region_alphas, tc.region_min_obs, tc.region_min_oblast)
    methods = REFERENCES + [d[0] for d in designs(tc.region_alphas)]
    scores = score_table(tab, methods)
    chosen = choose_on_validation(scores)
    top = list(dict.fromkeys(chosen.values()))
    base = chosen.get("baseline", "global_median")
    pairs = [(m, base) for m in top if m != base] + [(m, b) for m in top if m != base for b in BASELINES if b != base]
    pairs += [(m, r) for m in top if m != base for r in ("geo_mean", "km_global") if m != r]
    tests = paired_tests(tab, pairs, seed=seed)
    obl = per_oblast_scores(tab, ["global_median", "oblast_median"] + [m for m in top if m not in BASELINES])
    log.info("[%s] cohort %d episodes (%d observed); evaluated %d; chosen on validation: %s", variant, len(tab),
             int(tab["observed"].sum()), int(tab["evaluated"].sum()), chosen)
    return {"variant": variant, "table": tab, "scores": scores, "chosen": chosen, "tests": tests, "oblast": obl}


def run_region_restoration(cfg: Config) -> dict[str, dict]:
    tc = cfg.telegram
    if tc.region_label_source != "telegram":
        raise ValueError("telegram.region_label_source other than 'telegram' is DEV ONLY (use region-restoration --dev)")
    root = cfg._abs(cfg.paths.data_dir)                                  # real data even if --synthetic is passed
    pdir = cfg.telegram_processed_dir
    eps = pd.read_parquet(pdir / "telegram_region_episodes.parquet")
    facts = pd.read_parquet(pdir / "telegram_region_posts.parquet")
    strikes = load_strikes(cfg.strike_list_path)
    wpath = root / "raw" / "weather.parquet"
    weather = pd.read_parquet(wpath) if wpath.exists() else None
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    res = {v: run_variant(tc, eps, facts, strikes, weather, v, cfg.tuning.seed) for v in VARIANTS}
    for v, r in res.items():
        r["table"].to_csv(d / f"telegram_region_predictions_{v}.csv", index=False)
        r["scores"].to_csv(d / f"telegram_region_scores_{v}.csv", index=False)
        r["tests"].to_csv(d / f"telegram_region_tests_{v}.csv", index=False)
        pd.Series(r["chosen"]).to_csv(d / f"telegram_region_chosen_{v}.csv", header=["design"], index_label="role")
    comp = channel_completeness(root / "raw" / "telegram", _channel_oblast())
    from .region_report import write_region_report
    write_region_report(cfg, eps, res, comp, d / "telegram_region_results.md")
    return res


def _channel_oblast() -> dict[str, str]:
    from ..config import PROJECT_ROOT
    from ..ingestion.telegram_regional import load_channel_oblasts
    return load_channel_oblasts(PROJECT_ROOT / "config" / "sources.yaml")
