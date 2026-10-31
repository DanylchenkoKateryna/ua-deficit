import numpy as np
import pandas as pd

from deficit.evaluation.events import deficit_events
from deficit.evaluation.metrics import mae, mape, rmse
from deficit.preprocessing.gaps import fill_gaps, gap_lengths, long_gap_starts


def _series(n=24 * 21):
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(np.arange(n, dtype=float), index=idx)


def test_gap_lengths():
    s = _series(10)
    s.iloc[[2, 3, 7]] = np.nan
    assert gap_lengths(s).tolist() == [0, 0, 2, 2, 0, 0, 0, 1, 0, 0]


def test_short_gap_interpolated():
    s = _series()
    s.iloc[200:203] = np.nan
    out, long_ = fill_gaps(s)
    assert np.allclose(out.iloc[200:203], [200, 201, 202])
    assert not long_.any()


def test_medium_gap_uses_previous_week():
    s = _series()
    s.iloc[300:310] = np.nan
    out, long_ = fill_gaps(s)
    assert np.allclose(out.iloc[300:310], np.arange(300, 310) - 168)
    assert not long_.any()


def test_long_gap_left_nan_and_flagged():
    s = _series()
    s.iloc[400:430] = np.nan
    out, long_ = fill_gaps(s)
    assert out.iloc[400:430].isna().all()
    assert long_.sum() == 30
    assert long_gap_starts(long_) == [(s.index[400], 30)]


def test_metrics():
    y, p = np.array([100, -100, 10, np.nan]), np.array([110, -90, 0, 5])
    assert mae(y, p) == 10
    assert np.isclose(rmse(y, p), 10)
    assert np.isclose(mape(y, p, min_abs=50), 10)  # |10| is below min_abs and skipped


def test_deficit_events_depth_and_recovery():
    idx = pd.date_range("2022-01-01", periods=24 * 60, freq="h", tz="UTC")
    d = pd.Series(-100.0, index=idx)
    d["2022-01-10":"2022-01-14 23:00"] = 500.0  # 5 deficit days after the attack
    attacks = pd.DataFrame({"datetime": [pd.Timestamp("2022-01-10", tz="UTC")],
                            "capacity_lost_MW": [1000.0], "major": [True], "description": ["x"]})
    ev = deficit_events(d, attacks, threshold=0, max_days=30, window_days=14)
    assert ev.loc[0, "depth"] == 500
    assert ev.loc[0, "recovery_days"] == 5
    assert not ev.loc[0, "censored"]
