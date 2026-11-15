"""Step 3 - feature engineering.

Row t describes the target hour t, and every feature is built only from information
available at the *issue time* t - horizon_hours (except calendar features, which are known,
and weather, which stands in for a weather forecast - see README §6).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from ..config import Config
from ..io import write_parquet
from ..preprocessing.master import split_label
from .calendar import calendar_features

log = logging.getLogger(__name__)

TARGET = "deficit"
WEATHER = ["temperature_C", "wind_speed_kmh"]


def _decayed(x: pd.Series, tau: float) -> pd.Series:
    """Causal exponential filter: F_t = x_t + exp(-1/tau) F_{t-1}."""
    a = np.exp(-1.0 / tau)
    return pd.Series(lfilter([1.0], [1.0, -a], x.fillna(0).to_numpy()), index=x.index)


def build_features(master: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    fc = cfg.features
    h = fc.horizon_hours
    y = master[TARGET]
    f = pd.DataFrame(index=master.index)
    f["y"] = y
    if "imputed_weekly" in master:  # artefact rows: not a training / evaluation target
        f.loc[master["imputed_weekly"] == 1, "y"] = np.nan

    lags = [k for k in fc.lags_hours if k >= h]
    dropped = sorted(set(fc.lags_hours) - set(lags))
    if dropped:
        log.info("lags %s < horizon %dh dropped to avoid leakage", dropped, h)
    for k in lags:
        f[f"lag_{k}h"] = y.shift(k)
    known = y.shift(h)  # last value available at issue time
    for w in fc.rolling_hours:
        f[f"rolling_mean_{w}h"] = known.rolling(w, min_periods=w // 2).mean()
        f[f"rolling_std_{w}h"] = known.rolling(w, min_periods=w // 2).std()
    for col in ("actual_load_MW", "net_import_MW", "outage_queues", "dam_price_ratio"):
        if col not in master:
            continue
        f[f"{col}_lag{h}h"] = master[col].shift(h)

    f = f.join(calendar_features(master.index, cfg.period.local_tz))
    for c in WEATHER:
        if c in master:
            f[c] = master[c]
    if "temperature_C" in master:
        f["heating_degree"] = (15 - master["temperature_C"]).clip(lower=0)

    # ---- attack features (known at issue time) --------------------------
    cap = master["capacity_lost_MW"]
    flag = master["attack_flag"]
    tau = fc.attack_decay_hours
    if cap.abs().sum() > 0:                      # capacity per attack is only known for synthetic data
        f["capacity_lost_MW"] = cap.shift(h).rolling(24, min_periods=1).sum()  # lost in last 24h
        f["attack_impulse_MW"] = np.exp(-h / tau) * _decayed(cap, tau).shift(h)
    if "weapons_through" in master and master["weapons_through"].sum() > 0:    # strength of the strike
        through = master["weapons_through"]
        f["weapons_through_24h"] = through.shift(h).rolling(24, min_periods=1).sum()
        f["weapons_through_impulse"] = np.exp(-h / tau) * _decayed(through, tau).shift(h)
    f["attack_count_impulse"] = np.exp(-h / tau) * _decayed(flag, tau).shift(h)
    last = pd.Series(np.where(flag > 0, np.arange(len(flag)), np.nan), index=flag.index).ffill().shift(h)
    f["hours_since_attack"] = (np.arange(len(flag)) - last).clip(upper=fc.attack_recency_cap_hours).fillna(fc.attack_recency_cap_hours)
    if "acled_energy_events" in master:
        f["acled_energy_events_7d"] = master["acled_energy_events"].shift(h).rolling(168, min_periods=1).sum()

    f["split"] = split_label(f.index, cfg)
    f = f.iloc[max(lags + [h]) + max(fc.rolling_hours):]  # drop warm-up rows
    write_parquet(f, cfg.processed_dir / "features.parquet")
    return f


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in ("y", "split")]
