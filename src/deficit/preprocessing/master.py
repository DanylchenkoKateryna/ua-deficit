"""Step 2 - preprocessing: clean raw series and build the hourly master dataset.

Output: <data>/processed/master.parquet with, per UTC hour,
  actual_load_MW, generation_total_MW, net_import_MW, deficit (target),
  weather columns, attack columns, long_gap flag
and <data>/processed/deficit_events.csv (depth / recovery_days per major attack).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from ..config import Config
from ..evaluation.events import deficit_events, episodes
from ..features.calendar import calendar_features
from ..ingestion.attacks import load_attacks
from ..io import read_parquet, write_parquet
from .gaps import fill_gaps, gap_lengths, long_gap_starts

log = logging.getLogger(__name__)


def _regular(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    idx = pd.date_range(cfg.start_ts, cfg.end_ts, freq="h", name="datetime")
    return df[~df.index.duplicated()].reindex(idx)


def _clean(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (filled frame, long-gap mask, medium-gap mask) - masks are OR-ed over columns."""
    p = cfg.preprocessing
    out = {}
    long_any = pd.Series(False, index=df.index)
    medium_any = pd.Series(False, index=df.index)
    for c in df.columns:
        out[c], mask = fill_gaps(df[c], p.short_gap_hours, p.medium_gap_hours)
        long_any |= mask
        L = gap_lengths(df[c])
        medium_any |= (L > p.short_gap_hours) & (L <= p.medium_gap_hours)
    return pd.DataFrame(out, index=df.index), long_any, medium_any


def counterfactual_demand(master: pd.DataFrame, cfg: Config) -> pd.Series:
    """Peacetime demand model (calendar + weather, no lags) trained on the train period only."""
    X = calendar_features(master.index, cfg.period.local_tz)
    for c in ("temperature_C", "wind_speed_kmh"):
        if c in master:
            X[c] = master[c]
    y = master["actual_load_MW"]
    train = (master.index <= cfg.train_end_ts) & y.notna()
    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8)
    model.fit(X[train], y[train])
    return pd.Series(model.predict(X), index=master.index, name="expected_load_MW")


def _entsoe_master(cfg: Config) -> pd.DataFrame:
    """Hourly master from ENTSO-E load / generation / flows + weather; deficit by thesis section 2.2."""
    raw = cfg.raw_dir
    load = _regular(read_parquet(raw / "entsoe_load.parquet"), cfg)
    gen = _regular(read_parquet(raw / "entsoe_generation.parquet"), cfg)
    flows = _regular(read_parquet(raw / "entsoe_flows.parquet"), cfg)
    weather = _regular(read_parquet(raw / "weather.parquet"), cfg)

    load, long_load, med_load = _clean(load, cfg)
    gen, long_gen, med_gen = _clean(gen, cfg)
    # a flow direction that is simply not reported is zero, not missing
    flows = flows.fillna(0.0)
    weather = weather.interpolate(limit_direction="both")

    m = pd.DataFrame(index=load.index)
    m["actual_load_MW"] = load["actual_load_MW"]
    # all-types sum: NaN if any type is missing (a partial sum would look like a deficit)
    gen_cols = [c for c in gen.columns if gen[c].notna().any()]
    m["generation_total_MW"] = gen[gen_cols].sum(axis=1, skipna=False)
    imp = flows.filter(like="import_").sum(axis=1)
    exp = flows.filter(like="export_").sum(axis=1)
    m["net_import_MW"] = imp - exp
    m = m.join(weather)
    m["long_gap"] = (long_load | long_gen).astype(int)
    # load and generation filled independently from the previous week do not balance, so the
    # derived deficit there is an artefact: keep the row for lags, but never as a target
    m["imputed_weekly"] = (med_load | med_gen).astype(int)

    if cfg.target.definition == "balance":
        m["deficit"] = m["actual_load_MW"] - (m["generation_total_MW"] + m["net_import_MW"])
    else:
        m["expected_load_MW"] = counterfactual_demand(m, cfg)
        m["deficit"] = m["expected_load_MW"] - m["actual_load_MW"]
    return m


def _outage_master(cfg: Config) -> pd.DataFrame:
    """Hourly master from the Ukrenergo outage dataset + weather. Target = share of regions under outages.

    Unknown hours stay NaN (no imputation: unknown is not zero) and suspect periods are excluded
    from the target when outages.exclude_suspect is set.
    """
    out = _regular(read_parquet(cfg.raw_dir / "outages_hourly.parquet"), cfg)
    weather = _regular(read_parquet(cfg.raw_dir / "weather.parquet"), cfg).interpolate(limit_direction="both")
    m = out[["outage_on", "outage_queues", "suspect"]].copy()
    m["deficit"] = out["outage_extent"]
    if cfg.outages.exclude_suspect:
        m.loc[out["suspect"] == 1, "deficit"] = np.nan
    m = m.join(weather)
    m["long_gap"] = (gap_lengths(m["deficit"]) > cfg.preprocessing.medium_gap_hours).astype(int)
    return m


def _trade_master(cfg: Config) -> pd.DataFrame:
    """Hourly master with net imports as the target (a measured, MW-valued deficit proxy) + weather."""
    trade = _regular(read_parquet(cfg.raw_dir / "trade_hourly.parquet"), cfg)
    weather = _regular(read_parquet(cfg.raw_dir / "weather.parquet"), cfg).interpolate(limit_direction="both")
    m = trade[["import_MW", "export_MW", "net_import_MW"]].copy()
    m["deficit"] = m["net_import_MW"]
    m = m.join(weather)
    m["long_gap"] = (gap_lengths(m["deficit"]) > cfg.preprocessing.medium_gap_hours).astype(int)
    return m


def _balance_master(cfg: Config) -> pd.DataFrame:
    """Hourly master from the real consumption series (balance file) + weather.

    Target `deficit` = expected peacetime demand - actual consumption (MW): the demand that disappeared or was
    not served. The expectation is a calendar + weather model trained on the training period only.
    """
    bal = _regular(read_parquet(cfg.raw_dir / "balance_hourly.parquet"), cfg)
    weather = _regular(read_parquet(cfg.raw_dir / "weather.parquet"), cfg).interpolate(limit_direction="both")
    m = pd.DataFrame(index=bal.index)
    m["actual_load_MW"] = bal["consumption_MW"]
    m["net_import_MW"] = bal["flow_MW"]
    m = m.join(weather)
    m["expected_load_MW"] = counterfactual_demand(m, cfg)
    m["deficit"] = m["expected_load_MW"] - m["actual_load_MW"]
    m["long_gap"] = (gap_lengths(m["deficit"]) > cfg.preprocessing.medium_gap_hours).astype(int)
    return m


def _dam_master(cfg: Config) -> pd.DataFrame:
    """Hourly master with the day-ahead price relative to the price cap as the target (scarcity indicator) + weather."""
    dam = _regular(read_parquet(cfg.raw_dir / "dam_hourly.parquet"), cfg)
    weather = _regular(read_parquet(cfg.raw_dir / "weather.parquet"), cfg).interpolate(limit_direction="both")
    m = dam[["dam_price", "dam_purchase_MWh"]].copy()
    m["deficit"] = dam["dam_price_ratio"]
    m = m.join(weather)
    m["long_gap"] = (gap_lengths(m["deficit"]) > cfg.preprocessing.medium_gap_hours).astype(int)
    return m


def build_master(cfg: Config) -> pd.DataFrame:
    d = cfg.target.definition
    m = (_outage_master(cfg) if d == "outage_extent" else _trade_master(cfg) if d == "net_import" else
         _balance_master(cfg) if d == "demand_loss" else _dam_master(cfg) if d == "dam_price" else _entsoe_master(cfg))
    dam_path = cfg.raw_dir / "dam_hourly.parquet"
    if cfg.features.use_dam and d != "dam_price" and dam_path.exists():   # known the day before delivery
        m["dam_price_ratio"] = _regular(read_parquet(dam_path), cfg)["dam_price_ratio"]
    trade_path = cfg.raw_dir / "trade_hourly.parquet"
    if d == "outage_extent" and trade_path.exists():     # imports as a covariate of the outage target
        m["net_import_MW"] = _regular(read_parquet(trade_path), cfg)["net_import_MW"]
    raw = cfg.raw_dir

    # ---- attacks ------------------------------------------------------
    attacks = load_attacks(cfg)
    at = attacks.groupby("datetime").agg(attack_flag=("major", "size"), capacity_lost_MW=("capacity_lost_MW", "sum"),
                                         weapons_fired=("weapons_fired", "sum"), weapons_through=("weapons_through", "sum"))
    m = m.join(at, how="left")
    acols = ["attack_flag", "capacity_lost_MW", "weapons_fired", "weapons_through"]
    m[acols] = m[acols].fillna(0)

    acled_path = raw / "acled_energy_daily.parquet"
    if acled_path.exists():
        daily = read_parquet(acled_path)["acled_energy_events"]
        # a day's count is only known at the end of that day -> stamp it on 23:00
        hourly = daily.copy()
        hourly.index = hourly.index + pd.Timedelta(hours=23)
        m["acled_energy_events"] = hourly.reindex(m.index).fillna(0)

    # ---- report ---------------------------------------------------------
    gaps = long_gap_starts(m["long_gap"].astype(bool))
    pd.DataFrame(gaps, columns=["start_utc", "hours"]).assign(
        note="long gap - check whether it follows an attack").to_csv(
        cfg.interim_dir / "long_gap_candidates.csv", index=False)
    log.info("master: %d rows, %d long gaps, deficit NaN share %.2f%%",
             len(m), len(gaps), 100 * m["deficit"].isna().mean())

    write_parquet(m, cfg.processed_dir / "master.parquet")
    t = cfg.target
    ev = deficit_events(m["deficit"], attacks, t.recovery_threshold, t.recovery_max_days,
                        cfg.evaluation.crisis_window_days)
    ev.to_csv(cfg.processed_dir / "deficit_events.csv", index=False)
    eps = episodes(m["deficit"], t.episode_threshold, t.episode_gap_days)
    eps.to_csv(cfg.processed_dir / "deficit_episodes.csv", index=False)
    log.info("%d deficit episodes found in the target series", len(eps))
    return m


def split_label(index: pd.DatetimeIndex, cfg: Config) -> np.ndarray:
    return np.where(index <= cfg.train_end_ts, "train",
                    np.where(index <= cfg.validation_end_ts, "validation", "test"))
