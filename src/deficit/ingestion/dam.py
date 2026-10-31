"""Step 1h - day-ahead market (DAM) results, hourly (Energy Map, Market Operator data), 2019-07 onwards.

Manual download, saved as <data>/raw/<dam.file>: date, hour (1..24, 23/25 on DST days), energy_system, price (UAH/MWh),
amount_sales, amount_purchase, demand, price_cap. The United Power System rows ("ОЕС України", later "... (синхронізована з
ENTSO-E)") are used; the Burshtyn island is not.

Output <data>/raw/dam_hourly.parquet (UTC hourly index):
  dam_price_ratio   price / price cap, 0..1: close to 1 = the market is short of supply (scarcity indicator)
  dam_price         UAH/MWh
  dam_purchase_MWh  volume bought on the DAM
The price for hour t is published the day before delivery (about 13:30), so its value 24 h earlier is always known.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from ..io import write_parquet

log = logging.getLogger(__name__)
REQUIRED = {"date", "hour", "energy_system", "price", "amount_purchase", "price_cap"}


def hourly_dam(raw: pd.DataFrame, local_tz: str) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"DAM file lacks columns {sorted(missing)}")
    d = raw[raw["energy_system"].astype(str).str.startswith("ОЕС")].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["hour"] = pd.to_numeric(d["hour"], errors="coerce")
    d = d[d["date"].notna() & d["hour"].notna() & d["price"].notna() & d["price_cap"].notna()]
    # hour h of a delivery day = h-1 elapsed hours after local midnight (a DST day has 23 or 25 hours)
    start = d["date"].dt.tz_localize(local_tz).dt.tz_convert("UTC")
    stamp = start + pd.to_timedelta(d["hour"] - 1, unit="h")
    out = pd.DataFrame({"dam_price": d["price"].to_numpy(),
                        "dam_price_ratio": (d["price"] / d["price_cap"]).clip(upper=1.0).to_numpy(),
                        "dam_purchase_MWh": pd.to_numeric(d["amount_purchase"], errors="coerce").to_numpy()},
                       index=pd.DatetimeIndex(stamp, name="datetime"))
    stats = {"rows": len(raw), "ues_rows": len(d), "duplicate_hours": int(out.index.duplicated().sum())}
    out = out[~out.index.duplicated(keep="last")].sort_index()     # the ENTSO-E-synchronised series replaces the old one
    return out, stats


def fetch_dam(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / cfg.dam.file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - download 'DAM trading results' from Energy Map")
    out, stats = hourly_dam(pd.read_csv(path), cfg.period.local_tz)
    write_parquet(out, cfg.raw_dir / "dam_hourly.parquet")
    log.info("DAM: %s; %d hours %s .. %s, mean price/cap %.2f", stats, len(out), out.index[0], out.index[-1],
             out["dam_price_ratio"].mean())
    return out
