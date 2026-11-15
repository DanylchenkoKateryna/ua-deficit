"""Night-light response table: night alignment, pre-strike baseline, ratio, cloud handling, no look-ahead, hold-out exclusion."""
import numpy as np
import pandas as pd
import pytest

from deficit.evaluation import ntl_response as nr
from deficit.ingestion import black_marble as bm

STRIKE = pd.Timestamp("2022-03-30 10:00", tz="UTC")


def _setup(tmp_path, override=None):
    raw = tmp_path / "raw" / "black_marble"
    crops = raw / "crops"
    crops.mkdir(parents=True)
    lab = np.array([[1, 1, 0], [2, 2, 0]], dtype=np.uint8)          # oblast A = row 0, B = row 1, column 2 outside
    np.savez_compressed(raw / "oblast_raster.npz", lab=lab, names=np.array(["A", "B"]))
    for n in pd.date_range("2021-12-31", "2022-05-15"):
        ntl = np.full(lab.shape, 100, dtype=np.uint16)                # 10 nW everywhere
        qf = np.zeros(lab.shape, dtype=np.uint8)
        k = (n - pd.Timestamp("2022-03-30")).days
        if 0 <= k <= 2:
            ntl[0, :] = 50                                            # A at half light for 3 nights
        if n == pd.Timestamp("2022-04-05"):
            qf[0, :] = 2                                              # A clouded on k = 6
        if override:
            override(n, ntl, qf)
        np.savez_compressed(crops / f"VNP46A2_{n:%Y%m%d}.npz", ntl=ntl, gap=ntl, qf=qf, snow=np.zeros_like(qf))
    return tmp_path


def _strikes():
    return pd.DataFrame({"strike_at": [STRIKE, pd.Timestamp("2026-08-01", tz="UTC")], "major": [True, True]})


def test_ratio_and_alignment(tmp_path):
    t = nr.response_table(_setup(tmp_path), _strikes())
    a = t[t.oblast == "A"].set_index("k")
    assert a.loc[0, "night"] == pd.Timestamp("2022-03-30")          # overpass 23:30 UTC is after a 10:00 UTC strike
    assert a.loc[0, "ratio"] == pytest.approx(0.5) and a.loc[2, "ratio"] == pytest.approx(0.5)
    assert a.loc[3, "ratio"] == pytest.approx(1.0) and a.loc[-1, "ratio"] == pytest.approx(1.0)
    assert np.isnan(a.loc[6, "ratio"])                                # clouded night is missing, not zero
    np.testing.assert_allclose(t[t.oblast == "B"]["ratio"].dropna(), 1.0)


def test_holdout_excluded(tmp_path):
    t = nr.response_table(_setup(tmp_path), _strikes())
    assert t["strike_at"].max() <= nr.DEV_END


def test_no_lookahead(tmp_path):
    """Changing nights after k = 0 must not change ratio at k = 0 (the base uses only pre-strike nights)."""
    base = nr.response_table(_setup(tmp_path / "a"), _strikes())

    def later(n, ntl, qf):
        if n > pd.Timestamp("2022-03-30"):
            ntl[:] = 999
    alt = nr.response_table(_setup(tmp_path / "b", later), _strikes())
    k0 = lambda t: t[t.k <= 0].sort_values(["oblast", "k"])["ratio"].to_numpy()
    np.testing.assert_allclose(k0(base), k0(alt))


def test_strike_after_overpass_moves_to_next_night():
    nights = pd.date_range("2022-03-28", "2022-04-02")
    assert nights[nr.first_night(pd.Timestamp("2022-03-30 23:45", tz="UTC"), nights)] == pd.Timestamp("2022-03-31")
    assert nights[nr.first_night(pd.Timestamp("2022-03-30 23:00", tz="UTC"), nights)] == pd.Timestamp("2022-03-30")
