"""Step 1c - ACLED conflict events, filtered to strikes on energy infrastructure.

Two ways to get the data (README §Data sources):
  1. manual export from the ACLED Explorer saved as <data>/raw/acled_export.csv  (preferred), or
  2. the ACLED API with ACLED_EMAIL / ACLED_PASSWORD in .env (OAuth password grant).

The output is a list of *candidates*: ACLED notes must be verified by hand before an
event is copied into data/external/attacks.csv (thesis §2.1).
"""
from __future__ import annotations

import logging
import os
import re

import pandas as pd
import requests

from ..config import Config
from ..io import write_parquet

log = logging.getLogger(__name__)

TOKEN_URL = os.getenv("ACLED_TOKEN_URL", "https://acleddata.com/oauth/token")
READ_URL = os.getenv("ACLED_READ_URL", "https://acleddata.com/api/acled/read")
FIELDS = ["event_id_cnty", "event_date", "event_type", "sub_event_type", "admin1", "location",
          "latitude", "longitude", "notes", "fatalities"]


def _token() -> str:
    email, pwd = os.getenv("ACLED_EMAIL"), os.getenv("ACLED_PASSWORD")
    if not (email and pwd):
        raise RuntimeError("ACLED_EMAIL / ACLED_PASSWORD not set and no acled_export.csv found")
    r = requests.post(TOKEN_URL, data={"username": email, "password": pwd,
                                       "grant_type": "password", "client_id": "acled"}, timeout=60)
    r.raise_for_status()
    return r.json()["access_token"]


def _fetch_api(cfg: Config) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {_token()}"}
    rows, page = [], 1
    while True:
        params = {"country": cfg.acled.country,
                  "event_date": f"{cfg.period.start}|{cfg.period.end}",
                  "event_date_where": "BETWEEN",
                  "event_type": ":OR:event_type=".join(cfg.acled.event_types),
                  "fields": "|".join(FIELDS), "limit": 5000, "page": page, "_format": "json"}
        r = requests.get(READ_URL, params=params, headers=headers, timeout=180)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        log.info("ACLED page %d: %d rows", page, len(data))
        page += 1
    return pd.DataFrame(rows)


def filter_energy(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df[df["event_type"].isin(cfg.acled.event_types)].copy()
    pattern = re.compile("|".join(re.escape(k) for k in cfg.acled.energy_keywords), re.IGNORECASE)
    df = df[df["notes"].fillna("").str.contains(pattern)]
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["matched_keywords"] = df["notes"].str.findall(pattern).map(lambda m: ", ".join(sorted({x.lower() for x in m})))
    return df.sort_values("event_date")


def fetch_acled(cfg: Config) -> None:
    export = cfg.raw_dir / "acled_export.csv"
    if export.exists():
        log.info("using manual ACLED export %s", export)
        df = pd.read_csv(export)
    else:
        df = _fetch_api(cfg)
    energy = filter_energy(df, cfg)
    daily = energy.groupby(energy["event_date"].dt.normalize()).size().rename("acled_energy_events")
    daily.index = pd.DatetimeIndex(daily.index).tz_localize("UTC").rename("datetime")
    write_parquet(daily.to_frame(), cfg.raw_dir / "acled_energy_daily.parquet")
    out = cfg.interim_dir / "acled_energy_candidates.csv"
    energy.assign(verified=False).to_csv(out, index=False)
    log.info("wrote %d energy-strike candidates to %s - verify them manually", len(energy), out)
