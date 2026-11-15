"""Black Marble collector: tile geometry, mosaic placement, oblast raster and statistics, download handling (no network)."""
import json

import numpy as np
import pytest

from deficit.ingestion import black_marble as bm

h5py = pytest.importorskip("h5py")


def test_tile_windows_cover_box_exactly():
    cover = np.zeros((bm.NROW, bm.NCOL), dtype=int)
    for t in bm.TILES:
        tr, tc, mr, mc = bm.tile_window(t)
        assert tr.stop - tr.start == mr.stop - mr.start and tc.stop - tc.start == mc.stop - mc.start
        cover[mr, mc] += 1
    assert (cover == 1).all()
    assert bm.tile_window("h10v05") is None


def _fake_tile(path, tile, value):
    west, north = bm.tile_box(tile)
    a = np.full((bm.TILE_PX, bm.TILE_PX), value, dtype=np.uint16)
    # mark the pixel at lon 30.0..30+1/240, lat 50.0-1/240..50.0 (inside every tile that contains it)
    r, c = round((north - 50.0) * bm.PX), round((30.0 - west) * bm.PX)
    if 0 <= r < bm.TILE_PX and 0 <= c < bm.TILE_PX:
        a[r, c] = 999
    rad = (a / 10).astype(np.float32)          # collection 002 stores float32 nW; the store keeps x10 uint16
    rad[-1, -1] = bm.NTL_FILL
    with h5py.File(path, "w") as f:
        for k, name in bm.LAYERS.items():
            f.create_dataset(bm.GRID + name, data=rad if k in ("ntl", "gap") else np.zeros_like(a, dtype=np.uint8))


def test_mosaic_georeference(tmp_path):
    m = bm.empty_mosaic()
    for i, t in enumerate(bm.TILES):
        p = tmp_path / f"{t}.h5"
        _fake_tile(p, t, 10 * (i + 1))
        bm.read_tile_into(p, t, m)
    r, c = round((bm.LAT1 - 50.0) * bm.PX), round((30.0 - bm.LON0) * bm.PX)
    assert m["ntl"][r, c] == 999
    assert m["ntl"][0, 0] == 10          # NW corner: h20v03
    assert m["ntl"][-1, -1] == 40        # SE corner: h21v04
    # the fill pixel sits in each tile's SE corner: it lands in the box for the v03 tiles (lat 50) only, not for v04 (lat 40)
    r50 = round((bm.LAT1 - 50.0) * bm.PX) - 1
    assert m["ntl"][r50, round((30.0 - bm.LON0) * bm.PX) - 1] == bm.FILL16
    assert m["ntl"][r50, -1] == bm.FILL16
    assert (m["ntl"] == bm.FILL16).sum() == 2


def test_to_store_conversion():
    a = np.array([0.0, 0.04, 12.34, 7000.0, bm.NTL_FILL, np.nan], dtype=np.float32)
    assert bm.to_store(a, "ntl").tolist() == [0, 0, 123, bm.FILL16 - 1, bm.FILL16, bm.FILL16]
    q = np.array([0, 2, 255], dtype=np.uint8)
    assert bm.to_store(q, "qf") is q


def _square_geojson(path):
    sq = lambda x0, y0, x1, y1: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]
    gj = {"type": "FeatureCollection", "features": [
        {"properties": {"shapeName": "Kyiv Oblast"}, "geometry": {"type": "Polygon", "coordinates": sq(30, 50, 31, 51)}},
        {"properties": {"shapeName": "Odessa Oblast"}, "geometry": {"type": "MultiPolygon", "coordinates": [sq(30, 46, 30.5, 46.5)]}},
        {"properties": {"shapeName": "Sevastopol"}, "geometry": {"type": "Polygon", "coordinates": sq(33.5, 44.5, 33.75, 44.75)}},
    ]}
    path.write_text(json.dumps(gj), encoding="utf-8")


def test_oblast_raster_and_stats(tmp_path):
    g = tmp_path / "b.geojson"
    _square_geojson(g)
    lab, names = bm.oblast_raster(g)
    assert names == ["Crimea", "Kyiv", "Odesa"]
    assert (lab == names.index("Kyiv") + 1).sum() == bm.PX * bm.PX
    assert (lab == names.index("Odesa") + 1).sum() == (bm.PX // 2) ** 2
    m = bm.empty_mosaic()
    m["ntl"][:] = 50          # 5.0 nW
    m["gap"][:] = 50
    m["qf"][:] = 0
    m["snow"][:] = 0
    sel = lab == names.index("Kyiv") + 1
    q = m["qf"].copy()
    q[np.where(sel)[0][:bm.PX], np.where(sel)[1][:bm.PX]] = 2      # one row of Kyiv pixels poor quality
    m["qf"] = q
    st = {r["oblast"]: r for r in bm.oblast_stats("2023-01-01", m, lab, names)}
    assert st["Kyiv"]["n_pix"] == bm.PX ** 2 and st["Kyiv"]["n_good"] == bm.PX ** 2 - bm.PX
    assert st["Kyiv"]["mean_ntl_good"] == pytest.approx(5.0)
    assert st["Odesa"]["share_good"] == 1.0 and st["Odesa"]["snow_share"] == 0.0


class _Resp:
    def __init__(self, code, body=b""):
        self.status_code, self.body = code, body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        yield self.body


class _Sess:
    def __init__(self, codes):
        self.codes, self.calls = list(codes), []

    def get(self, url, headers=None, stream=False, timeout=None):
        self.calls.append(headers)
        return _Resp(self.codes.pop(0), b"abc")


def test_download_auth_and_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.time, "sleep", lambda s: None)
    with pytest.raises(bm.AuthError):
        bm.download("https://x/y.h5", tmp_path / "a", "TOKEN", _Sess([401]))
    s = _Sess([503, 200])
    sha = bm.download("https://x/y.h5", tmp_path / "b", "TOKEN", s)
    assert sha == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert s.calls[0]["Authorization"] == "Bearer TOKEN"


def test_download_patient_waits_out_outage(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.time, "sleep", lambda s: None)
    n = bm.MAX_RETRIES + 1
    s = _Sess([502] * (2 * n) + [200])          # two full retry rounds fail (outage), then the server is back
    assert bm.download_patient("https://x/y.h5", tmp_path / "c", "T", s)
    with pytest.raises(bm.AuthError):
        bm.download_patient("https://x/y.h5", tmp_path / "d", "T", _Sess([401]))
