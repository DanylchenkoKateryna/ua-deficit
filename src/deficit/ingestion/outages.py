"""Step 1d - Ukrenergo planned-outage dataset (Energy Map, "Electricity consumption limitation measures").

Manual download (the site needs a login for the file, so this step only converts it):
  save the CSV as <data>/raw/<outages.file>.
Columns: date, hour ("HH:00-HH:00", local Kyiv time), outage_use (так/ні), outage_amount
(number of outage queues, a number or a range such as "від 0,5 до 3"; often empty), number
(spatial extent: "в усіх областях" = in all regions, ...).

Output <data>/raw/outages_hourly.parquet (UTC hourly index) with
  outage_on      1 if outages were applied that hour, else 0 (NaN where the hour is unknown)
  outage_extent  share of regions affected, 0 when there are no outages (see outages.extent_scores)
  outage_queues  midpoint of the reported queue range (NaN when not reported)
  suspect        1 inside outages.suspect_periods
Unknown is never turned into zero: days missing from the file stay NaN.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from ..config import Config
from ..io import write_parquet

log = logging.getLogger(__name__)

HOUR_RE = re.compile(r"^(\d{1,2}):00\s*-\s*\d{1,2}:00$")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def parse_amount(value) -> float:
    """'2' -> 2.0, 'від 0,5 до 3' -> 1.75 (midpoint), empty -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    nums = [float(x.replace(",", ".")) for x in NUM_RE.findall(str(value))]
    return float(np.mean(nums)) if nums else np.nan


def to_utc_hourly(df: pd.DataFrame, local_tz: str) -> tuple[pd.DataFrame, dict]:
    """Attach a UTC timestamp to each row. Rows that cannot be placed in time are dropped.

    The file has local clock hours, so on the DST-end day the hour 03:00-04:00 occurs twice (first
    occurrence = summer time, second = winter time) and on the DST-start day one hour is absent.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    start_hour = d["hour"].astype(str).str.strip().str.extract(HOUR_RE)[0]
    ok = d["date"].notna() & start_hour.notna()
    stats = {"rows": len(d), "dropped_unparseable": int((~ok).sum())}
    d = d[ok].copy()
    local = d["date"] + pd.to_timedelta(start_hour[ok].astype(int), unit="h")
    first_occurrence = ~local.duplicated(keep="first")
    stamp = pd.DatetimeIndex(local).tz_localize(local_tz, ambiguous=first_occurrence.to_numpy(),
                                                nonexistent="NaT")
    d.index = stamp.tz_convert("UTC")
    d = d[d.index.notna()]
    stats["duplicate_utc_hours"] = int(d.index.duplicated().sum())
    return d[~d.index.duplicated(keep="first")].sort_index(), stats


def load_outages(cfg: Config) -> tuple[pd.DataFrame, dict]:
    path = cfg.raw_dir / cfg.outages.file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - download the outage CSV from Energy Map "
                                "(README §7) and save it there")
    raw = pd.read_csv(path)
    missing = {"date", "hour", "outage_use", "outage_amount", "number"} - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    d, stats = to_utc_hourly(raw, cfg.period.local_tz)

    on = d["outage_use"].astype(str).str.strip().str.lower().map({"так": 1.0, "ні": 0.0, "yes": 1.0, "no": 0.0})
    scores = {k.strip().lower(): v for k, v in cfg.outages.extent_scores.items()}
    extent = d["number"].astype(str).str.strip().str.lower().map(scores)
    unknown = d.loc[(on == 1) & extent.isna(), "number"].dropna().unique()
    if len(unknown):
        log.warning("outage extent phrases without a score (rows set to NaN): %s", list(unknown))
    out = pd.DataFrame({
        "outage_on": on,
        "outage_extent": np.where(on == 0, 0.0, extent),
        "outage_queues": d["outage_amount"].map(parse_amount),
    }, index=d.index)
    out.loc[on == 0, "outage_queues"] = 0.0

    idx = pd.date_range(cfg.start_ts, cfg.end_ts, freq="h", name="datetime")
    out = out.reindex(idx)
    local_date = out.index.tz_convert(cfg.period.local_tz).tz_localize(None).normalize()
    suspect = np.zeros(len(out), dtype=bool)
    for a, b in cfg.outages.suspect_periods:
        suspect |= (local_date >= pd.Timestamp(a)) & (local_date <= pd.Timestamp(b))
    out["suspect"] = suspect.astype(int)

    stats.update({"hours_in_period": len(out), "hours_with_data": int(out["outage_on"].notna().sum()),
                  "hours_unknown": int(out["outage_on"].isna().sum()),
                  "outage_hours": int((out["outage_on"] == 1).sum()),
                  "outage_hours_without_queues": int(((out["outage_on"] == 1) & out["outage_queues"].isna()).sum()),
                  "suspect_hours": int(out["suspect"].sum())})
    return out, stats


def fetch_outages(cfg: Config) -> None:
    out, stats = load_outages(cfg)
    log.info("outages: %s", stats)
    if stats["dropped_unparseable"]:
        log.warning("%d rows had no valid date/hour (e.g. spreadsheet formulas) and were dropped; "
                    "their days are NaN. Re-download the file if this matters.", stats["dropped_unparseable"])
    write_parquet(out, cfg.raw_dir / "outages_hourly.parquet")
