"""Small I/O helpers. Every time series is stored as Parquet with a UTC DatetimeIndex."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("deficit")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def to_utc_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out = df.copy()
    out.index = idx.rename("datetime")
    return out.sort_index()


def write_parquet(df: pd.DataFrame | pd.Series, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df.to_parquet(path)
    log.info("wrote %s (%d rows)", path, len(df))


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the previous pipeline step first")
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = to_utc_index(df)
    return df


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
