"""Night-light response of every oblast to every strike (Black Marble VNP46A2 crops), the raw material of the night-light label.

For a strike at ``strike_at`` the first night whose overpass (granule date + ``OBS_HOUR_UTC``) is after the strike is night k = 0.
For every oblast and k = -28..MAX_K:

    ratio_k = sum(ntl_k[G]) / sum(base[G]),   G = lit pixels of the oblast that are good (high quality, no snow) at night k

``base`` = per-pixel median of the good observations of the 28 nights before night 0 (at least ``MIN_BASE_OBS``), so the reference is
the pre-strike state of each pixel; ``lit`` = pixels with a pre-war (2021-12-31 .. 2022-02-23) median radiance >= ``LIT_NW``.  A night
counts only if G covers >= ``MIN_COVER`` of the oblast's lit pixels with a valid base (else ratio = NaN: clouds / summer twilight).
Ratios use only observations of night k and of nights before the strike (no look-ahead into later nights).

HOLD-OUT: strikes after ``DEV_END`` are a reserved blind test (never computed here unless ``include_holdout``; the CLI never sets it).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion import black_marble as bm
from ..ingestion.telegram_extract import load_strikes

log = logging.getLogger("deficit")

DEV_END = pd.Timestamp("2026-06-26", tz="UTC")      # last strike day of development; later strikes = reserved blind test
PREWAR = ("2021-12-31", "2022-02-23")
LIT_NW = 2.0
BASE_NIGHTS = 28
MIN_BASE_OBS = 3
MIN_COVER = 0.3
MAX_K = 30
CITY_NW = 10.0                # robust variant: pixels with a pre-strike base >= 10 nW (towns/cities)
MIN_CITY_PIX = 20


def _nights(crops: Path) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.Timestamp(p.stem.split("_")[1]) for p in crops.glob("VNP46A2_*.npz")))


def _load(crops: Path, night: pd.Timestamp, idx: np.ndarray) -> np.ndarray:
    """float32 radiance (nW) of the pixels ``idx`` (flat indices), NaN where not good (quality, fill or snow)."""
    z = np.load(crops / f"VNP46A2_{night:%Y%m%d}.npz")
    ntl, qf, snow = z["ntl"].ravel()[idx], z["qf"].ravel()[idx], z["snow"].ravel()[idx]
    good = np.isin(qf, bm.GOOD_QF) & (ntl != bm.FILL16) & (snow != 1)
    return np.where(good, ntl.astype(np.float32) / 10, np.nan)


def lit_pixels(crops: Path, nights: pd.DatetimeIndex, lab: np.ndarray) -> np.ndarray:
    """Flat indices of the pixels inside an oblast that were lit before the war."""
    inside = np.flatnonzero(lab.ravel() > 0)
    pre = [n for n in nights if pd.Timestamp(PREWAR[0]) <= n <= pd.Timestamp(PREWAR[1])]
    stack = np.vstack([_load(crops, n, inside) for n in pre])
    ok = (np.isfinite(stack).sum(0) >= MIN_BASE_OBS)
    med = np.full(len(inside), np.nan, dtype=np.float32)
    med[ok] = np.nanmedian(stack[:, ok], axis=0)
    return inside[np.nan_to_num(med, nan=0.0) >= LIT_NW]


def first_night(strike_at: pd.Timestamp, nights: pd.DatetimeIndex) -> int | None:
    obs = nights.tz_localize("UTC") + pd.Timedelta(hours=bm.OBS_HOUR_UTC)
    i = int(np.searchsorted(obs.values, strike_at.to_datetime64(), side="right"))
    return i if i < len(nights) else None


def response_table(data_dir: Path, strikes: pd.DataFrame, include_holdout: bool = False) -> pd.DataFrame:
    crops = data_dir / "raw" / "black_marble" / "crops"
    z = np.load(data_dir / "raw" / "black_marble" / "oblast_raster.npz")
    lab, names = z["lab"], list(z["names"])
    nights = _nights(crops)
    st = strikes if include_holdout else strikes[strikes["strike_at"] < DEV_END + pd.Timedelta(days=1)]
    idx = lit_pixels(crops, nights, lab)
    obl = lab.ravel()[idx].astype(int) - 1
    log.info("lit pixels %d; strikes %d; nights %d", len(idx), len(st), len(nights))
    # full-night calendar (missing nights = all NaN) so k counts nights, not files
    cal = pd.date_range(nights.min(), nights.max(), freq="D")
    have = set(nights)
    cache: dict[pd.Timestamp, np.ndarray] = {}

    def get(n):
        if n not in cache:
            cache[n] = _load(crops, n, idx) if n in have else np.full(len(idx), np.nan, dtype=np.float32)
            if len(cache) > 2 * (BASE_NIGHTS + MAX_K + 2):
                cache.pop(next(iter(cache)))
        return cache[n]

    rows = []
    for s in st.itertuples():
        i0 = first_night(s.strike_at, cal)
        if i0 is None or i0 - BASE_NIGHTS < 0:
            continue
        base_stack = np.vstack([get(cal[j]) for j in range(i0 - BASE_NIGHTS, i0)])
        nb = np.isfinite(base_stack).sum(0)
        base = np.full(len(idx), np.nan, dtype=np.float32)
        okb = nb >= MIN_BASE_OBS
        base[okb] = np.nanmedian(base_stack[:, okb], axis=0)
        n_valid = np.bincount(obl[okb], minlength=len(names))
        for k in range(-BASE_NIGHTS, MAX_K + 1):
            j = i0 + k
            if j >= len(cal):
                break
            v = get(cal[j])
            g = okb & np.isfinite(v)
            num = np.bincount(obl[g], weights=v[g], minlength=len(names))
            den = np.bincount(obl[g], weights=base[g], minlength=len(names))
            cnt = np.bincount(obl[g], minlength=len(names))
            cover = np.divide(cnt, n_valid, out=np.zeros(len(names)), where=n_valid > 0)
            ratio = np.where((cover >= MIN_COVER) & (den > 0), num / np.where(den > 0, den, 1), np.nan)
            # robust variant: median of per-pixel ratios over the city pixels (base >= CITY_NW)
            gc = g & (base >= CITY_NW)
            med = pd.Series(v[gc] / base[gc]).groupby(obl[gc]).median().reindex(range(len(names))).to_numpy()
            cc = np.bincount(obl[gc], minlength=len(names))
            nc = np.bincount(obl[okb & (base >= CITY_NW)], minlength=len(names))
            cov_c = np.divide(cc, nc, out=np.zeros(len(names)), where=nc > 0)
            med = np.where((cov_c >= MIN_COVER) & (cc >= MIN_CITY_PIX), med, np.nan)
            for o in range(len(names)):
                rows.append((s.strike_at, names[o], k, cal[j], ratio[o], cover[o], n_valid[o], med[o], cov_c[o]))
    return pd.DataFrame(rows, columns=["strike_at", "oblast", "k", "night", "ratio", "cover", "n_lit_base", "ratio_city_med",
                                       "cover_city"])


def run_ntl_response(cfg) -> Path:
    data_dir = cfg._abs(cfg.paths.data_dir)
    tab = response_table(data_dir, load_strikes(cfg.strike_list_path))
    out = data_dir / "processed" / "black_marble" / "ntl_response_dev.parquet"
    tab.to_parquet(out, index=False)
    log.info("wrote %s (%d rows, %d strikes)", out, len(tab), tab["strike_at"].nunique())
    return out
