"""Missing-value strategy from thesis §2.2.

  gap <= short  (3 h)   -> linear interpolation
  short < gap <= medium (24 h) -> same hour one week earlier
  gap > medium          -> left NaN and flagged (possible consequence of an attack)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = 168


def gap_lengths(s: pd.Series) -> pd.Series:
    """Length (in rows) of the NaN run each NaN belongs to; 0 for observed values."""
    isna = s.isna()
    run_id = (~isna).cumsum()
    lengths = isna.groupby(run_id).transform("sum")
    return lengths.where(isna, 0).astype(int)


def fill_gaps(s: pd.Series, short: int = 3, medium: int = 24) -> tuple[pd.Series, pd.Series]:
    """Return (filled series, boolean mask of long gaps). Index must be regular hourly."""
    lengths = gap_lengths(s)
    out = s.copy()

    interp = s.interpolate(method="linear", limit_area="inside")
    short_mask = (lengths > 0) & (lengths <= short)
    out[short_mask] = interp[short_mask]

    medium_mask = (lengths > short) & (lengths <= medium)
    weekly = s.shift(WEEK)
    out[medium_mask] = weekly[medium_mask]
    # previous week missing too -> interpolate as a last resort
    still = medium_mask & out.isna()
    out[still] = interp[still]

    long_mask = lengths > medium
    return out, long_mask


def long_gap_starts(mask: pd.Series) -> list[tuple[pd.Timestamp, int]]:
    """(start timestamp, length in hours) for each flagged long gap."""
    m = mask.to_numpy()
    if not m.any():
        return []
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(mask.index[a], int(b - a)) for a, b in zip(starts, ends)]
