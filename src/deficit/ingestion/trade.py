"""Step 1e - hourly cross-border electricity trade (Energy Map, "Hourly electricity imports and exports").

Manual download, saved as <data>/raw/<trade.file>. Long format: one row per
date, hour ("HH:00-HH:00", local Kyiv time), energy_system, country, customs_regime (імпорт/експорт), amount (MWh).
Since 24 Feb 2022 the source is the regulator dashboard and ENTSO-E (Ukrenergo stopped publishing).

Output <data>/raw/trade_hourly.parquet (UTC hourly index):
  import_MW, export_MW   totals over all neighbours and energy systems
  net_import_MW          import_MW - export_MW  (positive = Ukraine imports; a deficit proxy: a shortage
                         that the neighbouring systems could cover)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import Config
from ..io import write_parquet
from .outages import HOUR_RE

log = logging.getLogger(__name__)
REQUIRED = {"date", "hour", "energy_system", "country", "customs_regime", "amount"}
REGIME = {"імпорт": "import_MW", "експорт": "export_MW", "import": "import_MW", "export": "export_MW"}


def hourly_trade(raw: pd.DataFrame, local_tz: str) -> tuple[pd.DataFrame, dict]:
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    start_hour = d["hour"].astype(str).str.strip().str.extract(HOUR_RE)[0]
    ok = d["date"].notna() & start_hour.notna()
    stats = {"rows": len(d), "dropped_unparseable": int((~ok).sum())}
    d = d[ok].copy()
    d["h"] = start_hour[ok].astype(int)
    d["col"] = d["customs_regime"].astype(str).str.strip().str.lower().map(REGIME)
    unknown = d["col"].isna().sum()
    if unknown:
        log.warning("%d rows with unknown customs_regime ignored", unknown)
    d = d[d["col"].notna()]
    # On the DST-end day the local hour 03:00-04:00 occurs twice: the n-th row of an identical
    # (hour, system, country, regime) key is the n-th occurrence of that hour.
    key = ["date", "h", "energy_system", "country", "customs_regime"]
    d["occ"] = d.groupby(key).cumcount()
    wide = d.pivot_table(index=["date", "h", "occ"], columns="col", values="amount", aggfunc="sum").reset_index()
    for c in ("import_MW", "export_MW"):
        if c not in wide:
            wide[c] = 0.0
    local = wide["date"] + pd.to_timedelta(wide["h"], unit="h")
    stamp = pd.DatetimeIndex(local).tz_localize(local_tz, ambiguous=(wide["occ"] == 0).to_numpy(), nonexistent="NaT")
    out = wide[["import_MW", "export_MW"]].copy()
    out.index = stamp.tz_convert("UTC")
    out = out[out.index.notna()]
    stats["duplicate_utc_hours"] = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="first")].sort_index().fillna(0.0)
    out["net_import_MW"] = out["import_MW"] - out["export_MW"]
    return out, stats


def fetch_trade(cfg: Config) -> None:
    path = cfg.raw_dir / cfg.trade.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - download the hourly imports/exports CSV from Energy Map "
                                "(README §Data sources) and save it there")
    raw = pd.read_csv(path)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    out, stats = hourly_trade(raw, cfg.period.local_tz)
    idx = pd.date_range(out.index.min(), out.index.max(), freq="h", name="datetime")
    stats.update({"first": str(out.index.min()), "last": str(out.index.max()), "hours": len(out),
                  "missing_hours_inside": int(len(idx) - len(out)),
                  "net_import_MW_min": float(out["net_import_MW"].min()),
                  "net_import_MW_max": float(out["net_import_MW"].max())})
    log.info("trade: %s", stats)
    write_parquet(out.reindex(idx), cfg.raw_dir / "trade_hourly.parquet")
