"""NASA Black Marble VNP46A2 (daily BRDF/moon-corrected night lights, 15", collection 002) over Ukraine, aggregated per oblast.

Source: NASA LAADS DAAC (https://ladsweb.modaps.eosdis.nasa.gov), granules found via the public CMR search API, downloaded with the
user's Earthdata token (env ``EARTHDATA_TOKEN``; never printed or logged).  Oblast boundaries: geoBoundaries gbOpen UKR ADM1
(ODbL, OpenStreetMap), ``data/raw/boundaries/geoBoundaries-UKR-ADM1_simplified.geojson``.

Tiles h20v03, h20v04, h21v03, h21v04 cover Ukraine.  For every night all tiles are downloaded one at a time, cropped to the Ukraine
box (lon 22..40, lat 44..52.5) and mosaicked; the full granules are deleted right after processing.  Kept for re-use (no re-download
needed): ``data/raw/black_marble/crops/VNP46A2_<YYYYMMDD>.npz`` (compressed uint16 ``ntl`` and ``gap`` = BRDF-corrected and gap-filled
radiance x10 nW/cm2/sr with 65535 = fill, uint8 ``qf`` = Mandatory_Quality_Flag, uint8 ``snow`` = Snow_Flag) and a manifest
``_manifest.csv`` with the granule names and SHA-256 of the original files, so every number can be re-downloaded and re-checked.

Output table ``data/processed/black_marble/oblast_daily.csv``: date, oblast, n_pix, n_good, share_good, mean_ntl_good,
sum_ntl_good, mean_gap, snow_share.  ``good`` = Mandatory_Quality_Flag in {0, 1} (high quality) and a valid radiance.

Timing: the Suomi-NPP overpass is at about 01:30 local solar time, i.e. about 23:30 UTC of the granule date over Ukraine (±40 min
west-east).  A value for date D is therefore known at about D 23:30 UTC; ``OBS_HOUR_UTC`` is used by anything that turns it into a
label or a feature (no look-ahead: a value may only be used after that time).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("deficit")

CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = "VNP46A2"
TILES = ("h20v03", "h20v04", "h21v03", "h21v04")
PX = 240                      # pixels per degree (15 arc-seconds)
TILE_PX = 2400
LON0, LON1, LAT0, LAT1 = 22.0, 40.0, 44.0, 52.5
NROW, NCOL = int((LAT1 - LAT0) * PX), int((LON1 - LON0) * PX)
FILL16, FILL8 = 65535, 255
OBS_HOUR_UTC = 23.5
GRID = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/"
NTL_FILL = -999.9             # float32 fill of the radiance layers in collection 002 (scale_factor 1, nW/cm2/sr)
LAYERS = {"ntl": "DNB_BRDF-Corrected_NTL", "gap": "Gap_Filled_DNB_BRDF-Corrected_NTL", "qf": "Mandatory_Quality_Flag",
          "snow": "Snow_Flag"}
GOOD_QF = (0, 1)
SLEEP = 1.0
TIMEOUT = 120
MAX_RETRIES = 5
MANIFEST_COLS = ["granule", "date", "tile", "url", "size_mb", "sha256", "status", "processed_at"]
DAILY_COLS = ["date", "oblast", "n_pix", "n_good", "share_good", "mean_ntl_good", "sum_ntl_good", "mean_gap", "snow_share"]

# geoBoundaries shapeName -> project oblast key (telegram_extract.REGIONS); Sevastopol is merged into Crimea
NAME_MAP = {"Autonomous Republic of Crimea": "Crimea", "Sevastopol": "Crimea", "Kyiv": "Kyiv_city", "Odessa Oblast": "Odesa",
            "Zaporizhia Oblast": "Zaporizhzhia"}


def oblast_key(shape_name: str) -> str:
    return NAME_MAP.get(shape_name, shape_name.replace(" Oblast", ""))


def tile_box(tile: str) -> tuple[float, float]:
    """(west lon, north lat) of a Black Marble 10x10 degree tile."""
    h, v = int(tile[1:3]), int(tile[4:6])
    return -180.0 + 10 * h, 90.0 - 10 * v


def tile_window(tile: str) -> tuple[slice, slice, slice, slice] | None:
    """(tile rows, tile cols, mosaic rows, mosaic cols) where the tile overlaps the Ukraine box."""
    west, north = tile_box(tile)
    lat_hi, lat_lo = min(north, LAT1), max(north - 10, LAT0)
    lon_lo, lon_hi = max(west, LON0), min(west + 10, LON1)
    if lat_hi <= lat_lo or lon_hi <= lon_lo:
        return None
    tr = slice(round((north - lat_hi) * PX), round((north - lat_lo) * PX))
    tc = slice(round((lon_lo - west) * PX), round((lon_hi - west) * PX))
    mr = slice(round((LAT1 - lat_hi) * PX), round((LAT1 - lat_lo) * PX))
    mc = slice(round((lon_lo - LON0) * PX), round((lon_hi - LON0) * PX))
    return tr, tc, mr, mc


def empty_mosaic() -> dict[str, np.ndarray]:
    return {k: np.full((NROW, NCOL), FILL16 if k in ("ntl", "gap") else FILL8, dtype=np.uint16 if k in ("ntl", "gap") else np.uint8)
            for k in LAYERS}


def read_tile_into(path: Path, tile: str, mosaic: dict[str, np.ndarray]) -> None:
    import h5py
    w = tile_window(tile)
    if w is None:
        return
    tr, tc, mr, mc = w
    with h5py.File(path, "r") as f:
        for k, name in LAYERS.items():
            mosaic[k][mr, mc] = to_store(f[GRID + name][tr, tc], k)


def to_store(a: np.ndarray, layer: str) -> np.ndarray:
    """Radiance float32 (nW) -> uint16 x10 (0.1 nW steps, 65535 = fill, capped at 6553.4 nW); flags unchanged."""
    if layer not in ("ntl", "gap") or a.dtype == np.uint16:
        return a
    a = a.astype(np.float64)
    bad = ~np.isfinite(a) | (a < 0) | np.isclose(a, NTL_FILL)
    out = np.clip(np.round(np.where(bad, 0, a) * 10), 0, FILL16 - 1).astype(np.uint16)
    out[bad] = FILL16
    return out


# ----------------------------------------------------------------------------------------------- oblast raster
def oblast_raster(geojson: Path) -> tuple[np.ndarray, list[str]]:
    """uint8 raster (NROW x NCOL) of oblast index + 1 (0 = outside), pixel centres tested against the polygons."""
    from matplotlib.path import Path as MPath
    gj = json.loads(Path(geojson).read_text(encoding="utf-8"))
    names = sorted({oblast_key(f["properties"]["shapeName"]) for f in gj["features"]})
    lab = np.zeros((NROW, NCOL), dtype=np.uint8)
    for f in gj["features"]:
        idx = names.index(oblast_key(f["properties"]["shapeName"])) + 1
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = np.asarray(poly[0])
            holes = [np.asarray(h) for h in poly[1:]]
            x0, y0 = ring.min(0)
            x1, y1 = ring.max(0)
            c0, c1 = max(int((x0 - LON0) * PX), 0), min(int((x1 - LON0) * PX) + 1, NCOL)
            r0, r1 = max(int((LAT1 - y1) * PX), 0), min(int((LAT1 - y0) * PX) + 1, NROW)
            if c1 <= c0 or r1 <= r0:
                continue
            cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
            pts = np.column_stack([LON0 + (cc.ravel() + 0.5) / PX, LAT1 - (rr.ravel() + 0.5) / PX])
            inside = MPath(ring).contains_points(pts)
            for h in holes:
                inside &= ~MPath(h).contains_points(pts)
            sub = lab[r0:r1, c0:c1].ravel()
            sub[inside] = idx
            lab[r0:r1, c0:c1] = sub.reshape(r1 - r0, c1 - c0)
    return lab, names


def oblast_stats(day: str, m: dict[str, np.ndarray], lab: np.ndarray, names: list[str]) -> list[dict]:
    ntl, gap, qf, snow = m["ntl"], m["gap"], m["qf"], m["snow"]
    good = np.isin(qf, GOOD_QF) & (ntl != FILL16)
    rows = []
    for i, o in enumerate(names, start=1):
        sel = lab == i
        n = int(sel.sum())
        g = sel & good
        ng = int(g.sum())
        gv = gap[sel]
        gv = gv[gv != FILL16]
        sv = snow[sel]
        sv = sv[sv != FILL8]
        rows.append(dict(date=day, oblast=o, n_pix=n, n_good=ng, share_good=ng / n if n else np.nan,
                         mean_ntl_good=float(ntl[g].mean()) / 10 if ng else np.nan,
                         sum_ntl_good=float(ntl[g].sum(dtype=np.float64)) / 10 if ng else np.nan,
                         mean_gap=float(gv.mean()) / 10 if len(gv) else np.nan,
                         snow_share=float((sv == 1).mean()) if len(sv) else np.nan))
    return rows


# ----------------------------------------------------------------------------------------------- CMR + download
def list_granules(start: str, end: str, session=requests) -> pd.DataFrame:
    rows, page = [], 1
    while True:
        r = session.get(CMR_URL, params={"short_name": SHORT_NAME, "bounding_box": f"{LON0},{LAT0},{LON1},{LAT1}",
                                         "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z", "page_size": 2000, "page_num": page},
                        timeout=60)
        r.raise_for_status()
        ent = r.json()["feed"]["entry"]
        for g in ent:
            gid = g["producer_granule_id"]
            tile = next((t for t in TILES if f".{t}." in gid), None)
            urls = [l["href"] for l in g.get("links", []) if l.get("href", "").endswith(".h5") and l["href"].startswith("https")]
            if tile and urls:
                d = datetime.strptime(gid.split(".")[1][1:], "%Y%j").date().isoformat()
                rows.append(dict(granule=gid, date=d, tile=tile, url=urls[0], size_mb=float(g.get("granule_size", 0))))
        if len(ent) < 2000:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["granule", "date", "tile", "url", "size_mb"])
    return df.sort_values(["date", "tile"]).drop_duplicates(["date", "tile"], keep="last").reset_index(drop=True)


class AuthError(RuntimeError):
    pass


def download(url: str, dest: Path, token: str, session=requests) -> str:
    """Stream to ``dest``; return SHA-256.  Retries on network errors / 5xx; raises AuthError on 401/403."""
    last = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            with session.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=TIMEOUT) as r:
                if r.status_code in (401, 403):
                    raise AuthError(f"HTTP {r.status_code}: Earthdata token rejected or expired")
                if r.status_code == 200:
                    h = hashlib.sha256()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            h.update(chunk)
                    return h.hexdigest()
                last = f"HTTP {r.status_code}"
        except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            last = type(e).__name__                    # incl. "Connection broken: IncompleteRead" mid-stream
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"download failed after retries: {last}")


OUTAGE_WAIT = 600             # server outage (e.g. a run of HTTP 502): wait 10 min, up to OUTAGE_TRIES times (~1 h)
OUTAGE_TRIES = 6


def download_patient(url: str, dest: Path, token: str, session=requests) -> str:
    for k in range(OUTAGE_TRIES + 1):
        try:
            return download(url, dest, token, session)
        except RuntimeError as e:
            if isinstance(e, AuthError) or k == OUTAGE_TRIES:
                raise
            log.warning("%s; server outage? waiting %d s (%d/%d)", e, OUTAGE_WAIT, k + 1, OUTAGE_TRIES)
            time.sleep(OUTAGE_WAIT)


def _append(path: Path, rows: list[dict], cols: list[str]) -> None:
    pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=not path.exists(), index=False)


def collect(data_dir: Path, start: str, end: str, token: str, limit_days: int | None = None, session=requests) -> dict:
    raw = data_dir / "raw" / "black_marble"
    crops, tmp = raw / "crops", raw / "_tmp"
    out = data_dir / "processed" / "black_marble"
    for d in (crops, tmp, out):
        d.mkdir(parents=True, exist_ok=True)
    mpath, dpath = raw / "_manifest.csv", out / "oblast_daily.csv"
    lab_path = raw / "oblast_raster.npz"
    if lab_path.exists():
        z = np.load(lab_path, allow_pickle=False)
        lab, names = z["lab"], list(z["names"])
    else:
        lab, names = oblast_raster(data_dir / "raw" / "boundaries" / "geoBoundaries-UKR-ADM1_simplified.geojson")
        np.savez_compressed(lab_path, lab=lab, names=np.array(names))
    done = set(pd.read_csv(dpath, usecols=["date"])["date"]) if dpath.exists() else set()
    gr = list_granules(start, end, session)
    days = [d for d in sorted(gr["date"].unique()) if d not in done]
    log.info("granules %d, nights %d, to process %d (%.1f GB to download)",len(gr), gr["date"].nunique(), len(days),
             gr[gr["date"].isin(days)]["size_mb"].sum() / 1024)
    n = failed = 0
    t0 = time.time()
    for d in days:
        if limit_days is not None and n >= limit_days:
            break
        mos, man = empty_mosaic(), []
        for g in gr[gr["date"] == d].itertuples():
            f = tmp / g.granule
            sha = download_patient(g.url, f, token, session)
            try:
                read_tile_into(f, g.tile, mos)
                status = "ok"
            except Exception as e:           # corrupt file: logged, the tile stays fill
                status = f"read_error:{type(e).__name__}"
            f.unlink(missing_ok=True)
            man.append(dict(granule=g.granule, date=d, tile=g.tile, url=g.url, size_mb=g.size_mb, sha256=sha, status=status,
                            processed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            time.sleep(SLEEP)
        if any(r["status"] != "ok" for r in man):
            _append(mpath, man, MANIFEST_COLS)
            failed += 1
            log.warning("%s: tile read error, night skipped (%d in a row)", d, failed)
            if failed >= 3:
                raise RuntimeError("3 nights in a row could not be read; stopping (see _manifest.csv)")
            continue
        failed = 0
        np.savez_compressed(crops / f"VNP46A2_{d.replace('-', '')}.npz", **mos)
        _append(mpath, man, MANIFEST_COLS)
        _append(dpath, oblast_stats(d, mos, lab, names), DAILY_COLS)
        n += 1
        el = time.time() - t0
        log.info("[%d/%d] %s tiles=%d  %.0f s/night, ETA %.1f h", n, len(days), d, len(man), el / n, el / n * (len(days) - n) / 3600)
    return dict(processed=n, remaining=len(days) - n)


def run_collect(cfg, start: str | None = None, end: str | None = None, limit_days: int | None = None, plan_only: bool = False) -> None:
    data_dir = cfg._abs(cfg.paths.data_dir)
    start = start or "2022-01-01"
    end = end or date.today().isoformat()
    if plan_only:
        gr = list_granules(start, end)
        print(f"granules {len(gr)}, nights {gr['date'].nunique()}, {gr['size_mb'].sum() / 1024:.0f} GB")
        print(gr.groupby("tile").size().to_string())
        return
    token = os.environ.get("EARTHDATA_TOKEN", "")
    if not token:
        raise SystemExit("EARTHDATA_TOKEN is not set (pass the .env file to docker with --env-file)")
    print(collect(data_dir, start, end, token, limit_days))
