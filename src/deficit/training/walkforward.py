"""Step 5 - static and adaptive training with a strict walk-forward backtest.

Static   : fit once on the train period (2019-2021), predict everything after it.
Adaptive : refit at scheduled times and whenever a trigger fires. Triggers:
             * BOCPD alarm   - after a break at tau the model is trained on [tau - window, now)  (thesis §3.3);
             * known attack  - a strike from attacks.csv (xgb_attack), shorter recent window;
             * error drift   - CUSUM on the model's own daily MAE (xgb_drift), shorter recent window.

Timing rules (see README §Rules):
  * a model refitted at time T trains on target hours t < T (all observed by T);
  * it is used for forecasts issued at >= T, i.e. target hours t >= T + horizon;
  * a BOCPD alarm / an attack on day d is usable only from d + 1 00:00 UTC;
  * the drift detector at T sees only the errors of target hours < T (forecast and target both known).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..changepoints.bocpd import alarms as bocpd_alarms
from ..config import Config
from ..models.base import Forecaster

log = logging.getLogger(__name__)
DAY = pd.Timedelta(days=1)
MIN_TRIGGER_WINDOW = pd.Timedelta(days=7)   # 168 hourly targets: the refit minimum


@dataclass
class WalkForwardResult:
    forecasts: pd.DataFrame  # index datetime; y, yhat
    refits: pd.DataFrame     # one row per refit
    last_model: Forecaster


def fit_static(factory: Callable[[], Forecaster], frame: pd.DataFrame, cfg: Config,
               predict_from: pd.Timestamp) -> WalkForwardResult:
    model = factory()
    train = (frame.index <= cfg.train_end_ts)
    model.fit(frame, train)
    pred_mask = frame.index >= predict_from
    yhat = model.predict(frame, pred_mask)
    fc = pd.DataFrame({"y": frame.loc[pred_mask, "y"], "yhat": yhat}, index=frame.index[pred_mask])
    refits = pd.DataFrame([{"refit_at": cfg.train_end_ts, "train_start": frame.index[0],
                            "reason": "static", "n_train": int(train.sum())}])
    return WalkForwardResult(fc, refits, model)


def refit_schedule(start: pd.Timestamp, end: pd.Timestamp, every_days: int,
                   alarm_times: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, str]]:
    sched = {t: "scheduled" for t in pd.date_range(start, end, freq=f"{every_days}D")}
    for a in alarm_times:
        if start <= a <= end:
            sched[a] = "alarm"
    return sorted(sched.items())


@dataclass
class DriftParams:
    k: float = 0.5          # allowance: relative excess of the daily MAE tolerated per day
    h: float = 3.0          # alarm level of the cumulative excess
    ref_days: int = 60      # reference = mean daily MAE of the last `ref_days` live days
    min_days: int = 14      # live days needed before the detector may fire
    x_cap: float = 3.0      # one outlier day may add at most this much (relative excess)


class DriftDetector:
    """One-sided CUSUM on the relative excess of the model's own daily MAE.

    `update(m)` takes the MAE of one completed day of LIVE forecasts (forecast issued before the targets
    were known, targets now observed). The reference is the mean of the previous days only, so a
    sudden or persistent rise of the error accumulates in `s` until it crosses `h`.
    """

    def __init__(self, p: DriftParams):
        self.p = p
        self.hist: list[float] = []
        self.s = 0.0

    def update(self, day_mae: float) -> bool:
        if not np.isfinite(day_mae):
            return False
        p = self.p
        if len(self.hist) >= p.min_days:
            ref = max(float(np.mean(self.hist[-p.ref_days:])), 1e-12)
            x = min(day_mae / ref - 1.0, p.x_cap)
            self.s = max(0.0, self.s + x - p.k)
        self.hist.append(float(day_mae))
        return self.s > p.h

    def reset(self) -> None:
        self.s = 0.0


def attack_trigger_days(attack_times, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Midnights from which an attack is usable: the day after it happened (conservative)."""
    if attack_times is None or len(attack_times) == 0:
        return []
    t = pd.DatetimeIndex(pd.to_datetime(attack_times, utc=True)).floor("D") + DAY
    return sorted(x for x in set(t) if start <= x <= end)


def fit_adaptive(factory: Callable[[], Forecaster], frame: pd.DataFrame, cfg: Config,
                 signal: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp | None = None,
                 retrain_every_days: int | None = None, window_days: int | None = None,
                 threshold: float | None = None, use_alarms: bool = True, mode: str = "regime",
                 attack_times=None, drift: DriftParams | None = None,
                 trigger_window_days: int | None = None, trigger_hold_days: int = 0,
                 trigger_cooldown_days: int | None = None) -> WalkForwardResult:
    """Walk-forward backtest, stepped one day at a time.

    attack_times / drift enable the extra attack- and drift-triggered refits (use them with
    use_alarms=False so that only one kind of trigger is compared against the weekly rolling ablation).
    A triggered refit trains on the last `trigger_window_days`; scheduled refits within
    `trigger_hold_days` after a trigger keep that short window.
    """
    ad = cfg.adaptive
    h = pd.Timedelta(hours=cfg.features.horizon_hours)
    end = end or frame.index[-1]
    every = retrain_every_days or ad.retrain_every_days
    window = pd.Timedelta(days=window_days or ad.window_days)
    min_train = pd.Timedelta(days=ad.min_train_days)
    thr = cfg.changepoints.bocpd.threshold if threshold is None else threshold
    trig_window = pd.Timedelta(days=trigger_window_days) if trigger_window_days else None
    trig_hold = pd.Timedelta(days=trigger_hold_days)
    trig_cool = pd.Timedelta(days=ad.trigger_cooldown_days if trigger_cooldown_days is None
                             else trigger_cooldown_days)

    al = bocpd_alarms(signal, thr, ad.cooldown_days)
    if not use_alarms:   # ablation: scheduled refits on a fixed rolling window, change points ignored
        al = al.iloc[0:0]
    al["available_at"] = al["alarm"] + DAY
    alarm_av = [a for a in al["available_at"] if start <= a <= end]
    attack_days = set(attack_trigger_days(attack_times, start, end))
    detector = DriftDetector(drift) if drift is not None else None
    # the first refit happens `horizon` before `start` so that `start` itself is covered
    first = (start - h).floor("D")
    days = pd.date_range(first, end, freq="D")

    idx = frame.index
    y = frame["y"].to_numpy()
    y_known = ~np.isnan(y)
    yhat = pd.Series(np.nan, index=idx)
    rows, skipped, model = [], [], None
    pred_from = None                 # first target hour that has no forecast yet
    last_trigger, recent_until = None, None

    def flush(upto: pd.Timestamp) -> None:
        """Forecast the pending target hours [pred_from, upto) with the current model."""
        nonlocal pred_from
        if model is None or pred_from is None:
            return
        pm = (idx >= pred_from) & (idx < upto) & (idx <= end)
        if pm.any():
            yhat[pm] = model.predict(frame, pm)
        pred_from = max(pred_from, upto)

    for k, T in enumerate(days):
        # 1. (drift only) the detector needs the live forecasts of every target hour < T, made by the model
        #    that was live before T; otherwise forecasts are produced lazily, just before a refit
        if detector is not None:
            flush(T + h)

        # 2. which trigger fires at T?  (information available at T only)
        reason = "scheduled" if k % every == 0 else None
        if any(T - DAY < a <= T for a in alarm_av):
            reason = "alarm"
        trig_ok = last_trigger is None or T - last_trigger >= trig_cool
        if detector is not None:
            # completed day [T-1d, T): its forecasts were issued earlier and its targets are now observed
            dm = (idx >= T - DAY) & (idx < T) & y_known & yhat.notna().to_numpy()
            fired = dm.sum() >= 12 and detector.update(float(np.abs(y[dm] - yhat.to_numpy()[dm]).mean()))
            if fired and trig_ok:
                detector.reset()
                if reason != "alarm":
                    reason = "drift"
                last_trigger = T
        if T in attack_days and trig_ok and reason != "alarm":
            reason, last_trigger = "attack", T
        if reason is None:
            continue
        flush(T + h)     # the old model serves every target hour < T + h
        if reason in ("attack", "drift"):
            recent_until = T + trig_hold

        # 3. training window
        known = al[al["available_at"] <= T]
        if len(known):
            tau = known["break_estimate"].iloc[-1]
            if mode == "tau_w":      # literal formula of the theory chapter: [tau - w, now], keeps w days of the old regime
                train_start = min(tau - window, T - min_train)
            else:                   # regime only: from the estimated break, at most `window` days back, never fewer than min_train
                train_start = min(max(tau, T - window), T - min_train)
        elif not use_alarms:
            recent = trig_window is not None and (reason in ("attack", "drift")
                                                  or (recent_until is not None and T < recent_until))
            if recent:
                train_start = min(T - trig_window, T - MIN_TRIGGER_WINDOW)
            else:
                train_start = min(T - window, T - min_train)
        else:
            train_start = idx[0]  # no break detected yet -> all history
        train_mask = (idx >= train_start) & (idx < T)
        n_valid = int((train_mask & y_known).sum())
        if n_valid >= 24 * 7:
            model = factory().fit(frame, train_mask)
            if pred_from is None:
                pred_from = max(T + h, start)
            rows.append({"refit_at": T, "train_start": train_start, "reason": reason, "n_train": n_valid})
            if len(rows) % 25 == 1:
                log.info("  refit %d at %s (%s, %d rows)", len(rows), T, reason, n_valid)
        else:
            # too few observed targets in the window (e.g. an excluded period): an empty model would
            # silently predict a constant, so keep the previous model instead
            skipped.append({"at": T, "reason": reason, "valid_rows": n_valid})
    flush(end + DAY)

    sel = (idx >= start) & (idx <= end)
    fc = pd.DataFrame({"y": frame.loc[sel, "y"], "yhat": yhat[sel]})
    if skipped:
        log.warning("%d refits skipped for lack of observed targets (previous model kept), first at %s",
                    len(skipped), skipped[0]["at"])
    return WalkForwardResult(fc, pd.DataFrame(rows), model)
