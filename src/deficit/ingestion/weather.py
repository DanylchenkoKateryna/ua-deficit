"""Step 1b - Open-Meteo historical weather (no registration, <=10k requests/day)."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import requests

from ..config import Config
from ..io import write_parquet

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_year(cfg: Config, start: str, end: str) -> list[pd.DataFrame]:
    locs = cfg.weather.locations
    params = {
        "latitude": ",".join(str(l.lat) for l in locs),
        "longitude": ",".join(str(l.lon) for l in locs),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(cfg.weather.variables),
        "timezone": "GMT",
    }
    for attempt in range(4):
        r = requests.get(ARCHIVE_URL, params=params, timeout=120)
        if r.status_code == 429:  # rate limited - back off
            time.sleep(10 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
        r.raise_for_status()
    payload = r.json()
    payload = payload if isinstance(payload, list) else [payload]
    frames = []
    for res in payload:
        h = res["hourly"]
        df = pd.DataFrame({v: h[v] for v in cfg.weather.variables},
                          index=pd.to_datetime(h["time"], utc=True))
        frames.append(df)
    return frames


def weighted_average(frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    stacked = np.stack([f.to_numpy(dtype=float) for f in frames])  # (loc, time, var)
    mask = ~np.isnan(stacked)
    num = np.nansum(stacked * w[:, None, None], axis=0)
    den = (mask * w[:, None, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(den > 0, num / den, np.nan)
    return pd.DataFrame(avg, index=frames[0].index, columns=frames[0].columns)


def fetch_weather(cfg: Config) -> None:
    start = pd.Timestamp(cfg.weather.start or cfg.period.start)
    end = pd.Timestamp(cfg.weather.end or cfg.period.end)
    years = range(start.year, end.year + 1)
    parts = []
    for y in years:
        s = max(pd.Timestamp(f"{y}-01-01"), start).date().isoformat()
        e = min(pd.Timestamp(f"{y}-12-31"), end).date().isoformat()
        frames = _fetch_year(cfg, s, e)
        parts.append(weighted_average(frames, [l.weight for l in cfg.weather.locations]))
        log.info("Open-Meteo %s..%s ok", s, e)
    df = pd.concat(parts)
    df.columns = [c.replace("temperature_2m", "temperature_C").replace("wind_speed_10m", "wind_speed_kmh")
                  for c in df.columns]
    df.index.name = "datetime"
    write_parquet(df, cfg.raw_dir / "weather.parquet")
