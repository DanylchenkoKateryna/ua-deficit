"""Curated attack chronology (Ministry of Energy reports + verified ACLED events).

File: <data>/external/attacks.csv with columns
  date, time_utc, description, capacity_lost_MW, major, source, verified
capacity_lost_MW is the key covariate and must be filled from official reports.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)

REQUIRED = ["date", "major"]


def load_attacks(cfg: Config) -> pd.DataFrame:
    path = cfg.external_dir / "attacks.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - the strike list is part of the repository")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"attacks.csv lacks columns {missing}")
    time = df.get("time_utc", pd.Series("00:00", index=df.index)).fillna("00:00")
    df["datetime"] = pd.to_datetime(df["date"] + " " + time, utc=True).dt.floor("h")
    df["major"] = df["major"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["verified"] = df.get("verified", False)
    df["verified"] = df["verified"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "capacity_lost_MW" not in df.columns:
        df["capacity_lost_MW"] = float("nan")
    for c in ("weapons_fired", "weapons_through"):         # strength of the strike (see ingestion/strikes.py)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df.columns else 0.0
    n_nan = int(df["capacity_lost_MW"].isna().sum())
    if n_nan and not (df["weapons_through"] > 0).any():
        log.warning("attacks.csv: %d/%d rows have no capacity_lost_MW - treated as 0. "
                    "Fill them from Ministry of Energy reports.", n_nan, len(df))
    df["capacity_lost_MW"] = pd.to_numeric(df["capacity_lost_MW"], errors="coerce").fillna(0.0)
    if not df["verified"].all():
        log.warning("attacks.csv: %d rows are not verified", int((~df["verified"]).sum()))
    return df.sort_values("datetime").reset_index(drop=True)
