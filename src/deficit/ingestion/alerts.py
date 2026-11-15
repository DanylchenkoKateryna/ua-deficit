"""Air-raid alert intervals per oblast (Vadimkin/ukrainian-air-raid-sirens-dataset, `official_data_en.csv`, MIT licence,
https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset).

Raw columns: oblast, raion, hromada, level (oblast | raion | hromada), started_at, finished_at (UTC), source.  Names look like
'Kharkivska oblast' and 'Kyiv City'; they are mapped to the project's oblast codes (`telegram_regional.OBLASTS`).  Since 12.2025 many
records are raion-level, so the alert state of an oblast is the UNION of the intervals of the oblast itself and of all its raions and
hromadas (overlaps are never double counted).  Caveat: a hromada-level alert in a large oblast counts as "oblast under alert".

`AlertIndex` answers, for a window [lo, hi), the hours under alert and the number of separate alerts starting in it.  The query is
clipped to `hi`, so intervals that are still running at the issue time contribute only the part before it: nothing after `hi` is read
(a record that starts at or after `hi` has no effect; the end of a record that started before `hi` matters only up to `hi`).
Alerts are features only; they are never a label.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .telegram_regional import OBLASTS

log = logging.getLogger(__name__)

NAME_TO_OBLAST: dict[str, str] = {
    "Cherkaska oblast": "Cherkasy", "Chernihivska oblast": "Chernihiv", "Chernivetska oblast": "Chernivtsi",
    "Dnipropetrovska oblast": "Dnipropetrovsk", "Donetska oblast": "Donetsk", "Ivano-Frankivska oblast": "Ivano-Frankivsk",
    "Kharkivska oblast": "Kharkiv", "Khersonska oblast": "Kherson", "Khmelnytska oblast": "Khmelnytskyi",
    "Kirovohradska oblast": "Kirovohrad", "Kyiv City": "Kyiv_city", "Kyivska oblast": "Kyiv", "Luhanska oblast": "Luhansk",
    "Lvivska oblast": "Lviv", "Mykolaivska oblast": "Mykolaiv", "Odeska oblast": "Odesa", "Poltavska oblast": "Poltava",
    "Rivnenska oblast": "Rivne", "Sumska oblast": "Sumy", "Ternopilska oblast": "Ternopil", "Vinnytska oblast": "Vinnytsia",
    "Volynska oblast": "Volyn", "Zakarpatska oblast": "Zakarpattia", "Zaporizka oblast": "Zaporizhzhia", "Zhytomyrska oblast": "Zhytomyr",
}
HOUR = np.timedelta64(1, "h")


def map_oblast(name) -> str | None:
    code = NAME_TO_OBLAST.get(str(name).strip())
    return code if code in OBLASTS else None


def _ns(t) -> np.datetime64:
    t = pd.Timestamp(t)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return np.datetime64(t, "ns")


def union_intervals(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Union of intervals (datetime64 arrays): sorted, disjoint; overlapping or touching intervals are merged."""
    if not len(start):
        return start, end
    o = np.argsort(start, kind="stable")
    s, e = start[o], end[o]
    run_end = np.maximum.accumulate(e)
    new = np.r_[True, s[1:] > run_end[:-1]]                 # a gap before this interval starts a new block
    ix = np.flatnonzero(new)
    return s[ix], np.maximum.reduceat(e, ix)


class AlertIndex:
    """Merged alert intervals per oblast with cumulative coverage for O(log n) window queries."""

    def __init__(self, intervals: pd.DataFrame):
        """`intervals`: columns oblast (code), start, end (UTC, naive or tz-aware)."""
        self.iv: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for o, g in intervals.groupby("oblast"):
            s = pd.to_datetime(g["start"], utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]")
            e = pd.to_datetime(g["end"], utc=True).dt.tz_convert(None).to_numpy("datetime64[ns]")
            ms, me = union_intervals(s, e)
            cum = np.r_[0.0, np.cumsum((me - ms) / HOUR)]
            self.iv[o] = (ms, me, cum)

    def _cov(self, o: str, t) -> float:
        """Hours under alert in (-inf, t)."""
        if o not in self.iv:
            return 0.0
        ms, me, cum = self.iv[o]
        t = _ns(t)
        k = int(np.searchsorted(ms, t, "left"))            # blocks starting before t
        if k == 0:
            return 0.0
        last = min((t - ms[k - 1]) / HOUR, (me[k - 1] - ms[k - 1]) / HOUR)
        return float(cum[k - 1] + last)

    def hours(self, o: str, lo, hi) -> float:
        return self._cov(o, hi) - self._cov(o, lo)

    def count(self, o: str, lo, hi) -> int:
        """Number of separate (merged) alerts that START in [lo, hi)."""
        if o not in self.iv:
            return 0
        ms = self.iv[o][0]
        return int(np.searchsorted(ms, _ns(hi), "left") - np.searchsorted(ms, _ns(lo), "left"))


def load_alerts(path: Path) -> pd.DataFrame:
    """Raw file -> one row per record with columns oblast (code), start, end (UTC); unknown oblast names and records with end <= start are dropped."""
    df = pd.read_csv(path, usecols=["oblast", "started_at", "finished_at"])
    df["code"] = df["oblast"].map(map_oblast)
    bad = df["code"].isna()
    if bad.any():
        log.warning("alerts: %d records with an unknown oblast name dropped (%s)", int(bad.sum()), sorted(df.loc[bad, "oblast"].astype(str).unique())[:5])
    out = pd.DataFrame({"oblast": df["code"], "start": pd.to_datetime(df["started_at"], utc=True, errors="coerce"),
                        "end": pd.to_datetime(df["finished_at"], utc=True, errors="coerce")})
    ok = out["oblast"].notna() & out["start"].notna() & out["end"].notna() & (out["end"] > out["start"])
    if (~ok & ~bad).any():
        log.warning("alerts: %d records with a missing time or end <= start dropped", int((~ok & ~bad).sum()))
    return out[ok].reset_index(drop=True)
