"""Step 1a - ENTSO-E Transparency Platform (actual load, generation per type, cross-border flows).

Needs the ENTSOE_API_KEY environment variable (free token, see README).
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from ..config import Config
from ..io import to_utc_index, write_parquet

log = logging.getLogger(__name__)


def _client():
    from entsoe import EntsoePandasClient

    key = os.getenv("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError("ENTSOE_API_KEY is not set. Put it in .env (see .env.example) "
                           "or run the pipeline with --synthetic.")
    return EntsoePandasClient(api_key=key)


def _chunks(cfg: Config):
    tz = cfg.period.local_tz
    start = pd.Timestamp(cfg.period.start, tz=tz)
    end = pd.Timestamp(cfg.period.end, tz=tz) + pd.Timedelta(days=1)
    step = pd.Timedelta(days=cfg.entsoe.chunk_days)
    t = start
    while t < end:
        yield t, min(t + step, end)
        t += step


def _query(fn, label: str, cfg: Config, **kw) -> pd.DataFrame | None:
    from entsoe.exceptions import NoMatchingDataError

    parts = []
    for a, b in _chunks(cfg):
        try:
            parts.append(fn(start=a, end=b, **kw))
            log.info("ENTSO-E %s %s..%s ok", label, a.date(), b.date())
        except NoMatchingDataError:
            log.warning("ENTSO-E %s %s..%s: no data", label, a.date(), b.date())
    if not parts:
        return None
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")]
    return to_utc_index(out)


def _hourly(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    # ENTSO-E may deliver 15-min resolution; everything downstream is hourly.
    return df.resample("h").mean()


def fetch_load(cfg: Config) -> None:
    c = _client()
    df = _query(c.query_load, "load", cfg, country_code=cfg.entsoe.area)
    if df is None:
        raise RuntimeError(f"No load data for area {cfg.entsoe.area}; try area: UA in config.yaml")
    df = df.iloc[:, [0]] if isinstance(df, pd.DataFrame) else df.to_frame()
    df.columns = ["actual_load_MW"]
    write_parquet(_hourly(df), cfg.raw_dir / "entsoe_load.parquet")


def fetch_generation(cfg: Config) -> None:
    c = _client()
    df = _query(c.query_generation, "generation", cfg, country_code=cfg.entsoe.area)
    if df is None:
        raise RuntimeError(f"No generation data for area {cfg.entsoe.area}")
    if isinstance(df.columns, pd.MultiIndex):
        # keep produced energy only; 'Actual Consumption' (pumping) is dropped
        df = df.xs("Actual Aggregated", axis=1, level=1, drop_level=True)
    df.columns = [f"gen_{str(c).lower().replace(' ', '_').replace('/', '_')}_MW" for c in df.columns]
    write_parquet(_hourly(df), cfg.raw_dir / "entsoe_generation.parquet")


def fetch_flows(cfg: Config) -> None:
    """Physical cross-border flows in both directions for every neighbour."""
    c = _client()
    cols = {}
    for nb in cfg.entsoe.neighbours:
        imp = _query(c.query_crossborder_flows, f"flow {nb}->UA", cfg,
                     country_code_from=nb, country_code_to=cfg.entsoe.area)
        exp = _query(c.query_crossborder_flows, f"flow UA->{nb}", cfg,
                     country_code_from=cfg.entsoe.area, country_code_to=nb)
        if imp is not None:
            cols[f"import_{nb}_MW"] = _hourly(imp.squeeze())
        if exp is not None:
            cols[f"export_{nb}_MW"] = _hourly(exp.squeeze())
    if not cols:
        raise RuntimeError("No cross-border flow data returned")
    write_parquet(pd.DataFrame(cols), cfg.raw_dir / "entsoe_flows.parquet")


def fetch_all(cfg: Config) -> None:
    fetch_load(cfg)
    fetch_generation(cfg)
    fetch_flows(cfg)
