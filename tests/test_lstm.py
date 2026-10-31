"""LSTM smoke test - skipped when the optional [lstm] extra is not installed."""
import numpy as np
import pytest

pytest.importorskip("keras")

from deficit.features.build import build_features  # noqa: E402
from deficit.ingestion.synthetic import generate  # noqa: E402
from deficit.models.lstm_model import LSTMForecaster  # noqa: E402
from deficit.preprocessing.master import build_master  # noqa: E402


def test_lstm_fit_predict(cfg):
    generate(cfg)
    frame = build_features(build_master(cfg), cfg)
    params = {**cfg.models.lstm.model_dump(), "sequence_hours": 24, "units": 8, "epochs": 1}
    n = len(frame)
    train = np.zeros(n, bool); train[: n // 2] = True
    test = np.zeros(n, bool); test[n // 2: n // 2 + 500] = True
    m = LSTMForecaster(params).fit(frame, train)
    yhat = m.predict(frame, test)
    assert yhat.shape == (500,) and np.isfinite(yhat).all()
