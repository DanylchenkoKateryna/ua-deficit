import numpy as np
import pandas as pd

from deficit.changepoints.bocpd import alarms, bocpd
from deficit.changepoints.detect import detection_scores


def test_bocpd_detects_mean_shift():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    sig = bocpd(x, hazard=1 / 100, max_run=300, recent=3)
    sig.index = pd.date_range("2022-01-01", periods=len(x), freq="D", tz="UTC")
    al = alarms(sig, threshold=0.7, cooldown=7)
    assert len(al) >= 1
    first = al.iloc[0]
    assert 200 <= sig.index.get_loc(first["alarm"]) <= 206       # fires within a few days
    assert abs(sig.index.get_loc(first["break_estimate"]) - 200) <= 3
    assert (sig["cp_prob"].iloc[20:195] < 0.7).all()             # quiet before the break


def test_bocpd_is_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 300)
    a = bocpd(x, 1 / 100, 300, 3)["cp_prob"].to_numpy()
    y = x.copy(); y[200:] += 10
    b = bocpd(y, 1 / 100, 300, 3)["cp_prob"].to_numpy()
    assert np.allclose(a[:200], b[:200])  # future data never changes past signal


def test_detection_scores():
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    s = detection_scores([ts("2022-10-12"), ts("2023-06-01")], [ts("2022-10-10"), ts("2024-03-22")], 14)
    assert s["precision"] == 0.5 and s["recall"] == 0.5 and s["mean_lag_days"] == 2
