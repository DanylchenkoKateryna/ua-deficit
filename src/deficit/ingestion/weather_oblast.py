"""Daily weather at the oblast centre (Open-Meteo archive API, CC BY 4.0, https://open-meteo.com), one `<Oblast>.json` per oblast in
`data/raw/weather_oblast/` (file stem = project oblast code, `telegram_regional.OBLASTS`; `_coords.csv` lists the coordinates).
Daily UTC values: temperature_2m_mean / min / max (C), wind_gusts_10m_max and wind_speed_10m_max (km/h), snowfall_sum (cm),
precipitation_sum (mm).  Files whose stem is not an oblast code are ignored.
Weather is a feature only; the value of day D is observed by the end of D, so it may be used at the issue time (end of D) but nothing
of day D + 1 or later (README §6: observed weather stands in for a forecast).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .telegram_regional import OBLASTS

log = logging.getLogger(__name__)

VARS = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "wind_gusts_10m_max", "wind_speed_10m_max", "snowfall_sum",
        "precipitation_sum"]
COLS = ["oblast", "date"] + VARS


def load_weather_oblast(folder: Path) -> pd.DataFrame:
    """One row per (oblast, day): `date` is a naive UTC day, missing values stay NaN."""
    frames = []
    for p in sorted(Path(folder).glob("*.json")):
        if p.stem not in OBLASTS:
            continue
        d = json.loads(p.read_text(encoding="utf-8")).get("daily", {})
        if not d.get("time"):
            log.warning("weather_oblast: %s has no daily data", p.name)
            continue
        f = pd.DataFrame({"oblast": p.stem, "date": pd.to_datetime(d["time"])})
        for v in VARS:
            f[v] = pd.to_numeric(pd.Series(d.get(v) or [None] * len(f)), errors="coerce").to_numpy()
        frames.append(f)
    if not frames:
        return pd.DataFrame(columns=COLS)
    return pd.concat(frames, ignore_index=True)[COLS]
