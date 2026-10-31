"""Step 4 - change point detection on the daily-mean deficit series.

PELT (ruptures)  - offline, retrospective analysis only. Never used as a model input.
BOCPD            - online; its causal signal drives adaptive retraining (step 5).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import ruptures as rpt

from ..config import Config
from ..io import write_json, write_parquet
from .bocpd import alarms, bocpd

log = logging.getLogger(__name__)


def daily_series(master: pd.DataFrame, cfg: Config) -> pd.Series:
    return master["deficit"].resample(cfg.changepoints.resolution).mean()


def standardise(daily: pd.Series, cfg: Config) -> pd.Series:
    """z-score with *train-period* statistics only (causal for the online detector)."""
    train = daily[daily.index <= cfg.train_end_ts].dropna()
    if len(train) < 30:  # e.g. no pre-war data: fall back to the first year of the series
        train = daily.dropna().iloc[:365]
    sd = max(train.std() if len(train) > 1 else 0.0, cfg.changepoints.bocpd.scale_floor) or 1.0
    return (daily - train.mean()) / sd


def run_pelt(daily: pd.Series, cfg: Config) -> list[pd.Timestamp]:
    p = cfg.changepoints.pelt
    z = daily.interpolate(limit_direction="both")
    z = (z - z.mean()) / (z.std() or 1.0)
    algo = rpt.Pelt(model=p.model, min_size=p.min_size_days, jump=1).fit(z.to_numpy().reshape(-1, 1))
    bkps = algo.predict(pen=p.penalty)[:-1]
    return [daily.index[i] for i in bkps]


def run_bocpd(daily: pd.Series, cfg: Config) -> pd.DataFrame:
    b = cfg.changepoints.bocpd
    z = standardise(daily, cfg)
    sig = bocpd(z.to_numpy(), hazard=1.0 / b.hazard_days, max_run=b.max_run_length_days,
                recent=b.recent_window_days)
    sig.index = daily.index
    # a daily value for day d is complete only at the start of day d+1
    sig["available_at"] = sig.index + pd.Timedelta(days=1)
    return sig


def detection_scores(detections: list[pd.Timestamp], attacks: list[pd.Timestamp],
                     tolerance_days: int) -> dict:
    """Precision / recall / mean lag. A detection d matches attack a if -1 <= d - a <= tolerance."""
    tol = pd.Timedelta(days=tolerance_days)
    early = pd.Timedelta(days=1)  # daily resolution may place a break one day early
    det = pd.DatetimeIndex(detections)
    att = pd.DatetimeIndex(attacks)
    matched_det = [any((d - a >= -early) and (d - a <= tol) for a in att) for d in det]
    lags = []
    for a in att:
        after = [(d - a) for d in det if -early <= d - a <= tol]
        if after:
            lags.append(min(after, key=abs) / pd.Timedelta(days=1))
    precision = float(np.mean(matched_det)) if len(det) else float("nan")
    recall = len(lags) / len(att) if len(att) else float("nan")
    return {"n_detections": len(det), "n_attacks": len(att), "precision": precision,
            "recall": recall, "mean_lag_days": float(np.mean(lags)) if lags else float("nan")}


def detect(master: pd.DataFrame, attacks: pd.DataFrame, cfg: Config) -> dict:
    daily = daily_series(master, cfg)
    pelt_bkps = run_pelt(daily, cfg)
    sig = run_bocpd(daily, cfg)
    b = cfg.changepoints.bocpd
    al = alarms(sig, b.threshold, cfg.adaptive.cooldown_days)
    write_parquet(sig.drop(columns="available_at"), cfg.artifacts_dir / "bocpd_signal.parquet")

    major = attacks.loc[attacks["major"], "datetime"].dt.normalize().tolist()
    tol = cfg.changepoints.eval_tolerance_days
    # attacks before the first observation cannot be detected from this series
    major = [a for a in major if daily.index[0].normalize() <= a <= daily.index[-1] + pd.Timedelta(days=1)]
    until = cfg.changepoints.eval_until
    if until:  # the attack list is incomplete after this date: do not score detections there
        cut = pd.Timestamp(until, tz="UTC")
        major = [a for a in major if a <= cut]
        pelt_bkps = [d for d in pelt_bkps if d <= cut]
        al = al[al["alarm"] <= cut]
    result = {
        "pelt": {"breakpoints": [str(t.date()) for t in pelt_bkps],
                 "scores": detection_scores(pelt_bkps, major, tol)},
        "bocpd": {"alarms": [{"alarm": str(r.alarm.date()), "break_estimate": str(r.break_estimate.date()),
                              "cp_prob": round(float(r.cp_prob), 3)} for r in al.itertuples()],
                  "scores": detection_scores(al["alarm"].tolist(), major, tol),
                  "threshold": b.threshold},
    }
    write_json(result, cfg.artifacts_dir / "changepoints.json")
    log.info("PELT: %d breaks %s | BOCPD: %d alarms %s", len(pelt_bkps), result["pelt"]["scores"],
             len(al), result["bocpd"]["scores"])
    return result
