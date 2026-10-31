"""Quick check of what ENTSO-E actually publishes for Ukraine (run before a full download).

For one day per sampled month it asks for actual load, generation per type and cross-border flows and
records how many rows came back. Output: <artifacts>/entsoe_coverage.csv and a summary by year in the log.
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from .entsoe_client import _client

log = logging.getLogger(__name__)


def _rows(fn, **kw) -> tuple[int, str]:
    from entsoe.exceptions import NoMatchingDataError

    try:
        r = fn(**kw)
        n = int(r.notna().sum().sum()) if hasattr(r, "notna") else len(r)
        return n, "ok" if n else "empty"
    except NoMatchingDataError:
        return 0, "no_data"
    except Exception as e:  # noqa: BLE001 - network / HTTP errors are part of the answer
        return 0, f"error:{type(e).__name__}"


def run_coverage(cfg: Config) -> pd.DataFrame:
    c = _client()
    e = cfg.entsoe
    tz = cfg.period.local_tz
    months = pd.date_range(cfg.period.start, cfg.period.end, freq="MS")[:: e.coverage_step_months]
    rows = []
    for m in months:
        day = m + pd.Timedelta(days=14)
        a, b = pd.Timestamp(day, tz=tz), pd.Timestamp(day, tz=tz) + pd.Timedelta(days=1)
        for area in e.coverage_areas:
            n_load, s_load = _rows(c.query_load, country_code=area, start=a, end=b)
            n_gen, s_gen = _rows(c.query_generation, country_code=area, start=a, end=b)
            nb = e.neighbours[0]
            n_flow, s_flow = _rows(c.query_crossborder_flows, country_code_from=nb, country_code_to=area, start=a, end=b)
            rows.append({"month": m.strftime("%Y-%m"), "area": area, "load_values": n_load, "load_status": s_load,
                         "generation_values": n_gen, "generation_status": s_gen,
                         f"flow_{nb}_to_area_values": n_flow, "flow_status": s_flow})
            log.info("%s %-7s load=%s generation=%s flow=%s", m.strftime("%Y-%m"), area, s_load, s_gen, s_flow)
    df = pd.DataFrame(rows)
    out = cfg.artifacts_dir / "entsoe_coverage.csv"
    df.to_csv(out, index=False)
    df["year"] = df["month"].str[:4]
    summ = df.groupby(["area", "year"]).agg(
        sampled=("month", "size"), load_ok=("load_status", lambda s: int((s == "ok").sum())),
        generation_ok=("generation_status", lambda s: int((s == "ok").sum())),
        flow_ok=("flow_status", lambda s: int((s == "ok").sum())))
    log.info("ENTSO-E coverage (sampled months with data; one day per sampled month):\n%s", summ.to_string())
    log.info("wrote %s", out)
    return df
