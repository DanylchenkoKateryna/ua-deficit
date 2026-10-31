"""Step 1g - massive strikes on critical infrastructure (Energy Map, from Air Force summaries) -> attacks.csv.

Manual download, saved as <data>/raw/<strikes.file>. One row per weapon type and date:
  date, weapon_type, weapon_subtype, weapon_fired, weapon_shot_down, weapon_failed, notes, link

One row per date goes to <data>/external/attacks.csv, together with the full-scale invasion of 2022-02-24:
  weapons_fired    missiles + drones launched that day
  weapons_through  launched - shot down - failed: the weapons that reached Ukrainian territory (strength of the strike)
capacity_lost_MW stays empty: no open source gives the capacity lost per strike.
The time of day is unknown; 06:00 UTC is used (most massive strikes land at night or in the morning).
"""
from __future__ import annotations

import logging
import shutil

import pandas as pd

from ..config import Config

log = logging.getLogger(__name__)
REQUIRED = {"date", "weapon_type", "weapon_fired", "weapon_shot_down"}
INVASION = {"date": "2022-02-24", "time_utc": "03:00", "description": "Full-scale invasion - demand shock",
            "capacity_lost_MW": None, "major": True, "source": "public reporting", "verified": True,
            "weapons_fired": 0, "weapons_through": 0, "missiles_fired": 0}


def daily_strikes(raw: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"strikes file lacks columns {sorted(missing)}")
    d = raw.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()]
    for c in ("weapon_fired", "weapon_shot_down", "weapon_failed"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0) if c in d.columns else 0.0
    d["through"] = (d["weapon_fired"] - d["weapon_shot_down"] - d["weapon_failed"]).clip(lower=0)
    d["missiles"] = d["weapon_fired"].where(d["weapon_type"].astype(str).str.contains("ракет"), 0.0)
    if "link" not in d.columns:
        d["link"] = ""
    g = d.groupby("date").agg(weapons_fired=("weapon_fired", "sum"), weapons_through=("through", "sum"),
                              missiles_fired=("missiles", "sum"), source=("link", "first"))
    return g.reset_index()


def build_attacks(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / cfg.strikes.file
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - download 'Information on massive strikes on critical infrastructure' from Energy Map")
    g = daily_strikes(pd.read_csv(path))
    out = pd.DataFrame({
        "date": g["date"].dt.strftime("%Y-%m-%d"), "time_utc": "06:00",
        "description": ("Massive strike: " + g["missiles_fired"].astype(int).astype(str) + " missiles, "
                        + (g["weapons_fired"] - g["missiles_fired"]).astype(int).astype(str) + " drones, "
                        + g["weapons_through"].astype(int).astype(str) + " reached targets"),
        "capacity_lost_MW": None, "major": True, "source": g["source"], "verified": True,
        "weapons_fired": g["weapons_fired"].astype(int), "weapons_through": g["weapons_through"].astype(int),
        "missiles_fired": g["missiles_fired"].astype(int)})
    out = pd.concat([pd.DataFrame([INVASION]), out], ignore_index=True)
    target = cfg.external_dir / "attacks.csv"
    cfg.external_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and "weapons_through" not in pd.read_csv(target, nrows=1).columns:
        shutil.copy(target, cfg.external_dir / "attacks_manual.csv")     # keep the hand-made list
    out.to_csv(target, index=False)
    log.info("attacks.csv: %d strike days %s .. %s (+ the invasion), %d weapons fired, %d reached targets", len(g),
             g["date"].min().date(), g["date"].max().date(), int(g["weapons_fired"].sum()), int(g["weapons_through"].sum()))
    return out
