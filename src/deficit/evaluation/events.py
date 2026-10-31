"""Depth and duration of deficit episodes after each attack (the thesis' two target quantities).

  depth          - peak daily-mean deficit in the crisis window after the attack
  peak_hourly    - peak hourly deficit in the crisis window
  cumulative     - sum of positive hourly deficit in the crisis window (unit x hours)
  recovery_days  - days from the attack until the daily-mean deficit first drops to
                   <= recovery_threshold (0 if it never exceeded it); capped at
                   recovery_max_days, `censored` if the next major attack came first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def deficit_events(deficit: pd.Series, attacks: pd.DataFrame, threshold: float,
                   max_days: int, window_days: int) -> pd.DataFrame:
    major = attacks[attacks["major"]].sort_values("datetime").reset_index(drop=True)
    daily = deficit.resample("D").mean()
    rows = []
    for i, a in major.iterrows():
        t0 = a["datetime"]
        nxt = major["datetime"].iloc[i + 1] if i + 1 < len(major) else None
        win = deficit[(deficit.index >= t0) & (deficit.index < t0 + pd.Timedelta(days=window_days))]
        dwin = daily[(daily.index >= t0.normalize()) &
                     (daily.index < t0.normalize() + pd.Timedelta(days=max_days))]
        if win.dropna().empty:
            continue
        below = np.where(dwin.to_numpy() <= threshold)[0]
        recovered = len(below) > 0
        rec_days = int(below[0]) if recovered else max_days
        censored = (not recovered) or (
            nxt is not None and t0.normalize() + pd.Timedelta(days=rec_days) > nxt)
        dcrisis = dwin.iloc[:window_days]
        rows.append({
            "attack_datetime": t0,
            "description": a.get("description", ""),
            "capacity_lost_MW": a["capacity_lost_MW"],
            "depth": float(dcrisis.max()),
            "peak_hourly": float(win.max()),
            "cumulative": float(win.clip(lower=0).sum()),
            "recovery_days": rec_days,
            "censored": bool(censored),
        })
    return pd.DataFrame(rows)


def episodes(series: pd.Series, threshold: float, gap_days: int, min_days: int = 1) -> pd.DataFrame:
    """Deficit episodes found in a (possibly forecast) hourly series, independent of any attack list.

    A day belongs to an episode if its daily mean exceeds `threshold`; episodes separated by fewer than
    `gap_days` calm days are merged. Unknown days (NaN) neither start nor end an episode.
      depth         - highest daily mean inside the episode
      duration_days - calendar days from first to last episode day
      cumulative    - sum of the hourly deficit over the episode
    """
    daily = series.resample("D").mean()
    active = (daily > threshold).fillna(False).to_numpy()
    days = daily.index
    spans, i, n = [], 0, len(days)
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i                                    # last active day of the current episode
        k = i + 1
        while k < n and k - j <= gap_days:       # allow short calm gaps inside an episode
            if active[k]:
                j = k
            k += 1
        spans.append((i, j))
        i = j + 1
    rows = []
    for a, b in spans:
        if b - a + 1 < min_days:
            continue
        seg = daily.iloc[a:b + 1]
        hourly = series[(series.index >= days[a]) & (series.index < days[b] + pd.Timedelta(days=1))]
        rows.append({"start": days[a], "end": days[b], "duration_days": int(b - a + 1),
                     "depth": float(seg.max()), "cumulative": float(hourly.clip(lower=0).sum())})
    return pd.DataFrame(rows, columns=["start", "end", "duration_days", "depth", "cumulative"])


def match_episodes(actual: pd.DataFrame, predicted: pd.Series, threshold: float) -> pd.DataFrame:
    """For every actual episode, the depth and duration that a forecast series implies over the same span.

    Duration is counted as the days in the span whose forecast daily mean exceeds the threshold, so a model
    that only catches part of an episode gets a shorter duration; depth is its highest daily forecast mean.
    """
    daily = predicted.resample("D").mean()
    rows = []
    for e in actual.itertuples():
        seg = daily[(daily.index >= e.start) & (daily.index <= e.end)]
        rows.append({"start": e.start, "pred_depth": float(seg.max()) if seg.notna().any() else np.nan,
                     "pred_duration_days": int((seg > threshold).sum())})
    return pd.DataFrame(rows)
