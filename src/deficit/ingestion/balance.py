"""Step 1f - hourly electricity balance of the IPS of Ukraine (Energy Map, Ukrenergo data), 2014-01 .. 2022-10-27.

Manual download, saved as <data>/raw/<balance.file>. Long format: date, time ("HH:00-HH:00", local Kyiv time),
balance_component (генерація / міждержавні перетоки / споживання), balance_subcomponent, amount (MW).

Output <data>/raw/balance_hourly.parquet (UTC hourly index):
  generation_MW   nuclear + thermal + CHP + renewables + pumped storage + hydro
  flow_MW         sum of cross-border flows; positive = import (checked against the trade file: corr 0.90)
  pumping_MW      pumped-storage pumping, negative as in the source
  consumption_MW  generation + flows + pumping  (national consumption without pumping)

The "Бурштинський енергоострів" generation is NOT included: the source lists it only from 2022-01-01, which would
add an artificial step of about 0.8 GW to the consumption series.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..io import write_parquet
from .outages import HOUR_RE

log = logging.getLogger(__name__)
REQUIRED = {"date", "time", "balance_component", "balance_subcomponent", "amount"}
ISLAND = "Бурштинський енергоострів"
GEN, FLOW, PUMP = "генерація", "міждержавні перетоки", "споживання"


def hourly_balance(raw: pd.DataFrame, local_tz: str) -> tuple[pd.DataFrame, dict]:
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    start_hour = d["time"].astype(str).str.strip().str.extract(HOUR_RE)[0]
    ok = d["date"].notna() & start_hour.notna() & d["amount"].notna()
    stats = {"rows": len(d), "dropped_unparseable": int((~ok).sum())}
    d = d[ok].copy()
    d["h"] = start_hour[ok].astype(int)
    comp = d["balance_component"].astype(str).str.strip().str.lower()
    sub = d["balance_subcomponent"].astype(str).str.strip()
    d["group"] = None
    d.loc[(comp == GEN) & (sub != ISLAND), "group"] = "generation_MW"
    d.loc[comp == FLOW, "group"] = "flow_MW"
    d.loc[(comp == PUMP) & sub.str.contains("ГАЕС"), "group"] = "pumping_MW"
    stats["island_rows_ignored"] = int(((comp == GEN) & (sub == ISLAND)).sum())
    d = d[d["group"].notna()]
    # DST-end day: the n-th row of an identical (hour, component, subcomponent) key is the n-th occurrence of that hour
    d["occ"] = d.groupby(["date", "h", "balance_component", "balance_subcomponent"]).cumcount()
    wide = d.pivot_table(index=["date", "h", "occ"], columns="group", values="amount", aggfunc="sum").reset_index()
    for c in ("generation_MW", "flow_MW", "pumping_MW"):
        if c not in wide:
            wide[c] = 0.0
    local = wide["date"] + pd.to_timedelta(wide["h"], unit="h")
    stamp = pd.DatetimeIndex(local).tz_localize(local_tz, ambiguous=(wide["occ"] == 0).to_numpy(), nonexistent="NaT")
    out = wide[["generation_MW", "flow_MW", "pumping_MW"]].copy()
    out.index = stamp.tz_convert("UTC")
    out = out[out.index.notna()]
    stats["duplicate_utc_hours"] = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out["consumption_MW"] = out["generation_MW"] + out["flow_MW"] + out["pumping_MW"]
    return out, stats


def fetch_balance(cfg: Config) -> None:
    path = cfg.raw_dir / cfg.balance.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - download the hourly balance CSV from Energy Map "
                                "(README §7) and save it there")
    raw = pd.read_csv(path)
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    out, stats = hourly_balance(raw, cfg.period.local_tz)
    idx = pd.date_range(out.index.min(), out.index.max(), freq="h", name="datetime")
    stats.update({"first": str(out.index.min()), "last": str(out.index.max()), "hours": len(out),
                  "missing_hours_inside": int(len(idx) - len(out)),
                  "consumption_MW_min": float(out["consumption_MW"].min()),
                  "consumption_MW_max": float(out["consumption_MW"].max())})
    log.info("balance: %s", stats)
    write_parquet(out.reindex(idx), cfg.raw_dir / "balance_hourly.parquet")
