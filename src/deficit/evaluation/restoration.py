"""Forecast of the DURATION OF THE RESTRICTION REGIME after a strike (`deficit restoration`).

Target: `days_to_resolution` of `ingestion/telegram_extract.compute_episode` (strike -> first Ukrenergo post saying that outage
schedules / emergency outages are not applied any more; strikes without a regime: first "all consumers have power" post).
Right-censored when no such post appeared within the 30-day episode window. 33 of the 77 strikes are observed.

Forecast issued at the END of the strike day (00:00 UTC of the next day) from posts published before that instant only.
Strikes are processed in time order (leave-later-out): the model for strike i is fitted on the strikes j < i with the label AS IT
WAS KNOWN AT THAT INSTANT (`label_as_of`): observed if the lifting post is older than the issue time, otherwise right-censored at
"issue time - strike". A later strike never enters, and neither does anything published after the issue time.

Predictors (the number is a median in days):
  median_obs     median of the observed durations known so far        (ignores censoring: biased low)
  km_median      Kaplan-Meier median of all labels known so far       (uses the censored ones, capped at the window)
  last_episode   the duration of the most recent strike whose label was known to be observed
  ridge_<set>_a  ridge regression of log(duration) on the observed episodes only
  aft_<set>_a    log-normal accelerated failure time model with right censoring and an L2 penalty
Feature sets: attack (strength of the strike, gap to the previous strike, strikes in the last 30 days, heating season),
tg (+ compact telegram severity), tg_regime (+ regime state at issue), tg_full (all telegram columns + regime), tg_all (as tg_regime
from all channels). A design (model, set, penalty) is chosen on the VALIDATION strikes only; test numbers are never used for choice.

Metrics: MAE in days on observed strikes, and a censoring-aware lower bound on all strikes (a censored strike counts
max(0, censor time - prediction), the smallest error it can have), Harrell's C, paired Wilcoxon of per-strike errors against the
baselines. Strikes in a chain share one lifting post, so the strikes are not independent and the p-values are optimistic.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..features.telegram import REGIME, TG_COLS, issue_features, regime_state
from ..ingestion.telegram_extract import EpisodeParams, add_derived, compute_episode, load_strikes

log = logging.getLogger(__name__)
ATTACK = ["log_through", "gap_prev", "n_strikes_30", "heating"]
TG_COMPACT = ["tg_gen_d0", "tg_regions_d0", "tg_cons_d0", "tg_emerg_d0", "tg_sched_d0", "tg_sched_days7", "tg_sched_days30",
              "tg_regions_max7"]
BASELINES = ["median_obs", "km_median", "last_episode"]
REFERENCES = BASELINES + ["const_30"]      # const_30 (always the window length) shows that the lower-bound metric alone can be gamed
HORIZON = 30.0
MIN_DAYS = 0.1


def feature_sets() -> dict[str, list[str]]:
    all_cols = lambda cols: [c + "_all" for c in cols]  # noqa: E731
    return {"attack": ATTACK, "tg": ATTACK + TG_COMPACT, "tg_regime": ATTACK + TG_COMPACT + REGIME,
            "tg_full": ATTACK + TG_COLS + REGIME, "tg_all": ATTACK + all_cols(TG_COMPACT) + all_cols(REGIME)}


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def issue_time(strike_at) -> pd.Timestamp:
    """End of the strike day. The strike hour in attacks.csv is a placeholder, so the whole day counts as the strike day."""
    return _utc(strike_at).floor("D") + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------------------- features
def strike_features(posts: pd.DataFrame, strikes: pd.DataFrame, feature_channels: list[str] | None, regime_lookback: int = 60,
                    all_channels: list[str] | None = None) -> pd.DataFrame:
    """One row per strike (same order as `strikes`), every value known at `issue_time(strike_at)`.
    `all_channels` = the pooled channel list for the `*_all` columns (None = skip them)."""
    t = pd.to_datetime(strikes["strike_at"], utc=True).reset_index(drop=True)
    days = t.dt.floor("D")
    idx = pd.date_range(days.min(), days.max(), freq="D", tz="UTC")
    out = pd.DataFrame(index=range(len(t)))
    variants = [("", feature_channels)] + ([("_all", all_channels)] if all_channels is not None else [])
    for suffix, ch in variants:
        tg = issue_features(posts, idx, ch).reindex(days.to_numpy()).reset_index(drop=True)
        reg = pd.DataFrame([regime_state(posts, issue_time(s), ch, regime_lookback) for s in t])
        for c in TG_COLS:
            out[c + suffix] = tg[c]
        for c in REGIME:
            out[c + suffix] = reg[c]
    through = strikes["weapons_through"].reset_index(drop=True) if "weapons_through" in strikes else pd.Series(np.nan, index=out.index)
    out["log_through"] = np.log1p(through.astype(float))
    gap = t.diff().dt.total_seconds() / 86400
    out["gap_prev"] = gap.clip(upper=60).fillna(60)
    out["n_strikes_30"] = [int(((t < s) & (t >= s - pd.Timedelta(days=30))).sum()) for s in t]
    out["heating"] = t.dt.month.isin([10, 11, 12, 1, 2, 3]).astype(float)
    return out


# ---------------------------------------------------------------------------------------- labels
def label_as_of(p: pd.DataFrame, nat: pd.DataFrame, strike_at, next_at, prm: EpisodeParams, as_of) -> tuple[float, bool, float]:
    """(days, observed, censor_days) of one strike using posts published at or before `as_of`.
    Observed: days to the resolution post. Censored: days the strike has been watched (<= the episode window)."""
    strike_at = _utc(strike_at)
    ep = compute_episode(p, strike_at, next_at, prm, resolution_posts=nat, as_of=_utc(as_of))
    watched = max((ep["window_end"] - strike_at).total_seconds() / 86400, 0.0)
    if ep["status"] == "observed":
        return float(ep["days_to_resolution"]), True, watched
    return watched, False, watched


def known_labels(p: pd.DataFrame, nat: pd.DataFrame, strikes: pd.DataFrame, i: int, prm: EpisodeParams):
    """Labels of the strikes before strike i as they were known at the issue time of strike i (only posts before it)."""
    as_of = issue_time(strikes["strike_at"].iloc[i]) - pd.Timedelta(microseconds=1)
    pi, ni = p[p["datetime"] <= as_of], nat[nat["datetime"] <= as_of]
    t, ev = [], []
    for j in range(i):
        nxt = strikes["strike_at"].iloc[j + 1] if j + 1 < len(strikes) else pd.NaT
        d, obs, _ = label_as_of(pi, ni, strikes["strike_at"].iloc[j], nxt, prm, as_of)
        t.append(d), ev.append(obs)
    return np.array(t, float), np.array(ev, bool)


# ---------------------------------------------------------------------------------------- models
def km_median(t: np.ndarray, event: np.ndarray, cap: float = HORIZON) -> float:
    """Kaplan-Meier median; if the curve never falls to 0.5 the window length `cap` is returned."""
    order = np.argsort(t, kind="stable")
    t, event = t[order], event[order]
    surv, at_risk = 1.0, len(t)
    for ti, e in zip(t, event):
        if e:
            surv *= 1.0 - 1.0 / at_risk
            if surv <= 0.5:
                return float(ti)
        at_risk -= 1
    return float(cap)


class _Scaler:
    def __init__(self, X: np.ndarray):
        self.med = np.array([np.median(c[~np.isnan(c)]) if (~np.isnan(c)).any() else 0.0 for c in X.T])
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu, sd = Xi.mean(0), Xi.std(0)
        self.sd = np.where(sd < 1e-9, 1.0, sd)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.where(np.isnan(X), self.med, X) - self.mu) / self.sd


def fit_aft(X: np.ndarray, t: np.ndarray, event: np.ndarray, alpha: float):
    """Log-normal AFT with right censoring and an L2 penalty on the (standardised) coefficients. Returns a median predictor."""
    from scipy.optimize import minimize
    from scipy.stats import norm

    sc = _Scaler(X)
    Z, y, ev = sc(X), np.log(np.clip(t, MIN_DAYS, None)), event.astype(bool)
    p = Z.shape[1]

    def nll(w):
        b0, beta, sig = w[0], w[1:1 + p], np.exp(w[-1])
        z = (y - b0 - Z @ beta) / sig
        ll = np.where(ev, norm.logpdf(z) - np.log(sig), norm.logsf(z))
        return -ll.sum() + alpha * float(beta @ beta)

    w0 = np.zeros(p + 2)
    w0[0], w0[-1] = float(y[ev].mean()), np.log(max(float(y[ev].std()), 0.5))
    bounds = [(None, None)] + [(None, None)] * p + [(np.log(0.3), np.log(2.0))]
    w = minimize(nll, w0, method="L-BFGS-B", bounds=bounds).x
    return lambda Xn: np.clip(np.exp(w[0] + sc(Xn) @ w[1:1 + p]), MIN_DAYS, HORIZON)


def fit_ridge(X: np.ndarray, t: np.ndarray, event: np.ndarray, alpha: float):
    """Ridge on log(duration) of the observed strikes only (censored strikes are dropped: biased towards short regimes)."""
    from sklearn.linear_model import Ridge

    sc = _Scaler(X[event])
    m = Ridge(alpha=alpha).fit(sc(X[event]), np.log(np.clip(t[event], MIN_DAYS, None)))
    return lambda Xn: np.clip(np.exp(m.predict(sc(Xn))), MIN_DAYS, HORIZON)


def designs(alphas: list[float]) -> list[tuple[str, str, str, float]]:
    return [(f"{m}_{s}_a{a:g}", m, s, a) for m in ("aft", "ridge") for s in feature_sets() for a in alphas]


def forecast_one(F: pd.DataFrame, i: int, t: np.ndarray, ev: np.ndarray, alphas: list[float]) -> dict[str, float]:
    """All predictors for strike i from the labels (t, ev) of the strikes before it."""
    out = {"median_obs": float(np.median(t[ev])), "km_median": km_median(t, ev),
           "last_episode": float(t[np.flatnonzero(ev)[-1]]), "const_30": HORIZON}
    for name, model, s, a in designs(alphas):
        cols = feature_sets()[s]
        X = F.loc[: i - 1, cols].to_numpy(float)
        fit = fit_aft if model == "aft" else fit_ridge
        out[name] = float(fit(X, t, ev, a)(F.loc[[i], cols].to_numpy(float))[0])
    return out


# ---------------------------------------------------------------------------------------- evaluation
def loss(pred: pd.Series, actual: pd.Series, observed: pd.Series, censor: pd.Series) -> pd.Series:
    """|error| for observed strikes; for censored ones max(0, censor time - prediction) (a lower bound of the true error)."""
    return pd.Series(np.where(observed, (pred - actual).abs(), np.maximum(censor - pred, 0.0)), index=pred.index)


def harrell_c(pred: np.ndarray, t: np.ndarray, event: np.ndarray) -> float:
    """Concordance of predicted and actual durations over the comparable pairs (ties in the prediction count 0.5)."""
    num = den = 0.0
    for a in range(len(t)):
        if not event[a]:
            continue
        for b in range(len(t)):
            if b != a and t[a] < t[b]:
                den += 1
                num += 1.0 if pred[a] < pred[b] else 0.5 if pred[a] == pred[b] else 0.0
    return num / den if den else np.nan


def score_table(tab: pd.DataFrame, methods: list[str], chosen: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    for split in ("validation", "test", "all"):
        s = tab if split == "all" else tab[tab["split"] == split]
        s = s[s["evaluated"]]
        for m in methods:
            if s.empty or m not in s:
                continue
            ok = s[m].notna()
            u = s[ok]
            if u.empty:
                continue
            e = loss(u[m], u["days"], u["observed"], u["censor_days"])
            ob = u["observed"]
            rows.append({"split": split, "method": m, "n_strikes": len(u), "n_observed": int(ob.sum()),
                         "MAE_observed_days": float((u.loc[ob, m] - u.loc[ob, "days"]).abs().mean()) if ob.any() else np.nan,
                         "MAE_lower_bound_all_days": float(e.mean()),
                         "C_index": harrell_c(u[m].to_numpy(), u["days"].to_numpy(), ob.to_numpy()),
                         "chosen_on_validation": bool(chosen and m in chosen.values())})
    return pd.DataFrame(rows)


def paired_tests(tab: pd.DataFrame, pairs: list[tuple[str, str]], min_n: int = 8) -> pd.DataFrame:
    """Paired Wilcoxon of per-strike errors for (method, baseline) pairs (negative mean = method is closer to the truth)."""
    from scipy.stats import wilcoxon

    rows = []
    for split in ("validation", "test", "all"):
        s = tab if split == "all" else tab[tab["split"] == split]
        s = s[s["evaluated"]]
        for m, b in pairs:
            if m == b or m not in s or b not in s:
                continue
            for what, sub in (("observed_abs_error", s[s["observed"]]), ("lower_bound_all", s)):
                u = sub.dropna(subset=[m, b])
                if what == "observed_abs_error":
                    diff = (u[m] - u["days"]).abs() - (u[b] - u["days"]).abs()
                else:
                    diff = loss(u[m], u["days"], u["observed"], u["censor_days"]) - loss(u[b], u["days"], u["observed"], u["censor_days"])
                if len(diff) < min_n or (diff == 0).all():
                    continue
                rows.append({"split": split, "method": m, "baseline": b, "metric": what, "n": len(diff),
                             "mean_diff_days": float(diff.mean()), "share_better": float((diff < 0).mean()),
                             "p_value": float(wilcoxon(diff).pvalue)})
    return pd.DataFrame(rows)


def choose_on_validation(scores: pd.DataFrame, alphas: list[float]) -> dict[str, str]:
    """Best design per model family (and overall) by the lower-bound MAE of the VALIDATION strikes."""
    v = scores[(scores["split"] == "validation") & scores["method"].str.contains(r"_a\d", regex=True)].set_index("method")
    chosen = {}
    for fam in ("aft", "ridge"):
        sub = v[v.index.str.startswith(fam + "_")]
        if len(sub):
            chosen[fam] = sub["MAE_lower_bound_all_days"].idxmin()
    if len(v):
        chosen["overall"] = v["MAE_lower_bound_all_days"].idxmin()
    return chosen


def run_restoration(cfg: Config) -> dict[str, pd.DataFrame]:
    tc = cfg.telegram
    posts = pd.read_parquet(cfg.telegram_processed_dir / "telegram_posts.parquet")   # scenario-scoped (telegram_extended: extended strike list)
    posts = add_derived(posts) if "impact_post" not in posts.columns else posts
    strikes = load_strikes(cfg.strike_list_path)
    prm = EpisodeParams(horizon_days=tc.episode_horizon_days, relapse_days=tc.relapse_days,
                        national_channels=tuple(tc.national_channels), pre_strike_days=tc.pre_strike_days)
    nat = posts[posts["channel"].isin(prm.national_channels)]
    pooled = sorted(posts["channel"].unique())
    F = strike_features(posts, strikes, tc.feature_channels, tc.regime_lookback_days, all_channels=pooled)
    n = len(strikes)
    full = []                                                           # ex-post labels (the targets)
    for i in range(n):
        nxt = strikes["strike_at"].iloc[i + 1] if i + 1 < n else pd.NaT
        full.append(label_as_of(posts, nat, strikes["strike_at"].iloc[i], nxt, prm, posts["datetime"].max()))
    rows = []
    for i in range(n):
        t, ev = known_labels(posts, nat, strikes, i, prm) if i else (np.array([]), np.array([], bool))
        days, observed, censor = full[i]
        row = {"strike_at": strikes["strike_at"].iloc[i], "split": "validation" if issue_time(strikes["strike_at"].iloc[i]) <= cfg.validation_end_ts + pd.Timedelta(hours=1) else "test",
               "days": days, "observed": observed, "censor_days": censor, "n_train": int(len(t)), "n_train_observed": int(ev.sum()),
               "evaluated": bool(ev.sum() >= tc.restoration_min_obs)}
        if row["evaluated"]:
            row.update(forecast_one(F, i, t, ev, tc.restoration_alphas))
        rows.append(row)
        log.info("strike %s: %d observed / %d known labels%s", row["strike_at"].date(), int(ev.sum()), len(t),
                 "" if row["evaluated"] else " (too few, skipped)")
    tab = pd.DataFrame(rows)
    methods = REFERENCES + [d[0] for d in designs(tc.restoration_alphas)]
    scores = score_table(tab, methods)
    chosen = choose_on_validation(scores, tc.restoration_alphas)
    scores = score_table(tab, methods, chosen)
    top = list(dict.fromkeys(chosen.values()))
    pairs = [(m, b) for m in top + ["km_median"] for b in REFERENCES]
    for m in top:                                                       # does the telegram part help? same model and penalty, attack features only
        fam, _, a = m.partition("_")
        pairs.append((m, f"{fam}_attack_{m.rsplit('_', 1)[1]}"))
    tests = paired_tests(tab, pairs)
    d = cfg.reports_dir
    tab.to_csv(d / "telegram_restoration_predictions.csv", index=False)
    scores.to_csv(d / "telegram_restoration_scores.csv", index=False)
    tests.to_csv(d / "telegram_restoration_tests.csv", index=False)
    pd.Series(chosen).to_csv(d / "telegram_restoration_chosen.csv", header=["design"], index_label="family")
    log.info("restoration duration: %d strikes, %d observed, %d evaluated (>= %d observed labels known); chosen on validation: %s",
             n, int(tab["observed"].sum()), int(tab["evaluated"].sum()), tc.restoration_min_obs, chosen)
    log.info("scores (lower is better, days):\n%s", scores.round(3).to_string(index=False))
    if len(tests):
        log.info("paired tests (Wilcoxon, strikes in a chain are dependent: p optimistic):\n%s", tests.round(4).to_string(index=False))
    return {"predictions": tab, "scores": scores, "tests": tests}
