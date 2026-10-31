"""Synthetic stand-in for the real sources, so the whole pipeline runs without API tokens.

The generator writes the *same files* the real ingestion step writes, into
paths.synthetic_dir. Numbers are invented but follow the structure described in the
thesis: a demand shock on 2022-02-24, attack waves in 2022-23 and 2024 that remove
generating capacity (partially repaired afterwards), and EU imports after synchronisation.
Never mix synthetic outputs with real results.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..io import write_parquet

log = logging.getLogger(__name__)

INVASION = pd.Timestamp("2022-02-24 03:00", tz="UTC")
EU_SYNC = pd.Timestamp("2022-03-16", tz="UTC")
# (date, capacity lost MW) - invented magnitudes on the thesis' wave dates
ATTACKS = [
    ("2022-02-24 03:00", 0), ("2022-10-10 05:00", 2500), ("2022-10-31 05:00", 1800),
    ("2022-11-15 12:00", 3000), ("2022-11-23 12:00", 4500), ("2022-12-16 06:00", 2500),
    ("2023-03-09 02:00", 1500), ("2024-03-22 03:00", 5000), ("2024-03-29 03:00", 2500),
    ("2024-04-11 03:00", 3000), ("2024-04-27 03:00", 1500), ("2024-05-08 03:00", 1500),
    ("2024-06-01 03:00", 1200), ("2024-08-26 03:00", 2500), ("2024-11-17 04:00", 2000),
    ("2024-11-28 04:00", 1500), ("2024-12-13 04:00", 1500), ("2024-12-25 04:00", 1200),
    ("2025-02-01 03:00", 1500), ("2025-03-07 03:00", 2000), ("2025-10-10 03:00", 2500),
    ("2025-11-08 03:00", 2000),
]


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.normal(0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def generate(cfg: Config, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(cfg.start_ts, cfg.end_ts, freq="h", name="datetime")
    n = len(idx)
    local = idx.tz_convert(cfg.period.local_tz)
    doy = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()

    # ---- weather ------------------------------------------------------
    temp = (8 - 12 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
            + _ar1(n, 0.995, 0.35, rng))
    wind = np.clip(12 + _ar1(n, 0.97, 1.2, rng), 0, None)

    # ---- demand -------------------------------------------------------
    daily = 1 + 0.08 * np.sin(2 * np.pi * (hour - 6) / 24) + 0.06 * np.exp(-((hour - 19) ** 2) / 6)
    weekly = np.where(dow >= 5, 0.93, 1.0)
    heat = 180 * np.clip(15 - temp, 0, None)
    cool = 120 * np.clip(temp - 23, 0, None)
    demand = (15500 * daily * weekly + heat + cool) * (1 + _ar1(n, 0.9, 0.004, rng))

    war = np.asarray(idx >= INVASION)
    # demand shock: -35% then slow partial recovery
    days_since = np.clip((idx - INVASION).total_seconds().to_numpy() / 86400, 0, None)
    shock = np.where(war, 0.65 + 0.08 * (1 - np.exp(-days_since / 400)), 1.0)
    demand = demand * shock

    # ---- available generation capacity --------------------------------
    capacity = np.full(n, 26000.0)
    capacity[np.asarray(idx >= pd.Timestamp("2022-09-11", tz="UTC"))] -= 5500  # ZNPP offline
    attacks = pd.DataFrame(ATTACKS, columns=["dt", "capacity_lost_MW"])
    attacks["dt"] = pd.to_datetime(attacks["dt"], utc=True)
    attacks = attacks[(attacks["dt"] >= idx[0]) & (attacks["dt"] <= idx[-1])].reset_index(drop=True)
    for dt, lost in zip(attacks["dt"], attacks["capacity_lost_MW"]):
        if lost == 0:
            continue
        h = (idx - dt).total_seconds().to_numpy() / 3600
        after = h >= 0
        permanent = 0.45 * lost
        repaired = 0.55 * lost * np.exp(-np.clip(h, 0, None) / (24 * 25))
        capacity[after] -= permanent + repaired[after]
    # reconstruction, distributed generation and batteries rebuild capacity slowly from 2023
    rebuild_days = np.clip((idx - pd.Timestamp("2023-01-01", tz="UTC")).total_seconds().to_numpy() / 86400, 0, None)
    capacity += 3000 * rebuild_days / 365
    # unplanned outages / maintenance noise
    capacity *= 1 + _ar1(n, 0.995, 0.004, rng)

    import_cap = np.where(np.asarray(idx >= EU_SYNC),
                          np.clip(700 + 1100 * (days_since - 20) / 900, 0, 1800), 0.0)
    shortfall = np.clip(demand - capacity, 0, None)
    net_import = np.minimum(import_cap, shortfall) + np.where(war, 0, rng.normal(0, 40, n))
    served_by_gen = np.minimum(demand, capacity)
    generation = served_by_gen * (1 + rng.normal(0, 0.004, n))  # reporting noise

    # ---- files in the same layout as the real ingestion ---------------
    load = pd.DataFrame({"actual_load_MW": demand}, index=idx)
    solar_shape = np.clip(np.sin(np.pi * (hour - 5) / 14), 0, None)
    solar = np.minimum(generation * 0.08, 1500 * solar_shape)
    rest = generation - solar
    gen = pd.DataFrame({"gen_nuclear_MW": rest * 0.55, "gen_fossil_hard_coal_MW": rest * 0.3,
                        "gen_hydro_water_reservoir_MW": rest * 0.15, "gen_solar_MW": solar}, index=idx)
    flows = pd.DataFrame(index=idx)
    shares = {"PL": 0.3, "SK": 0.25, "HU": 0.2, "RO": 0.2, "MD": 0.05}
    for nb, s in shares.items():
        flows[f"import_{nb}_MW"] = np.clip(net_import, 0, None) * s
        flows[f"export_{nb}_MW"] = np.clip(-net_import, 0, None) * s
    weather = pd.DataFrame({"temperature_C": temp, "wind_speed_kmh": wind}, index=idx)

    # ---- data-quality problems the preprocessing must handle ----------
    for _ in range(120):  # short technical gaps
        i = rng.integers(0, n - 4)
        load.iloc[i:i + rng.integers(1, 4)] = np.nan
    for _ in range(12):   # medium gaps
        i = rng.integers(0, n - 30)
        load.iloc[i:i + rng.integers(4, 24)] = np.nan
    for dt in attacks["dt"].iloc[4:8:3]:  # long outages right after two attacks
        i = idx.get_loc(dt)
        load.iloc[i:i + 30] = np.nan
        gen.iloc[i:i + 30] = np.nan

    raw = cfg.raw_dir
    write_parquet(load, raw / "entsoe_load.parquet")
    write_parquet(gen, raw / "entsoe_generation.parquet")
    write_parquet(flows, raw / "entsoe_flows.parquet")
    write_parquet(weather, raw / "weather.parquet")

    days = attacks["dt"].dt.normalize()
    acled = pd.Series(rng.poisson(3, len(days)) + 2, index=pd.DatetimeIndex(days, name="datetime"),
                      name="acled_energy_events")
    write_parquet(acled.to_frame(), raw / "acled_energy_daily.parquet")

    out = pd.DataFrame({
        "date": attacks["dt"].dt.strftime("%Y-%m-%d"),
        "time_utc": attacks["dt"].dt.strftime("%H:%M"),
        "description": "synthetic attack",
        "capacity_lost_MW": attacks["capacity_lost_MW"],
        "major": True, "source": "synthetic", "verified": True,
    })
    cfg.external_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(cfg.external_dir / "attacks.csv", index=False)
    log.info("synthetic data generated in %s", cfg.data_root)
