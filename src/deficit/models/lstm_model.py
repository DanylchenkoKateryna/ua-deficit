"""LSTM on sequences of issue-time-safe feature rows (thesis §1.5, §3.2).

The input for target row t is the window of feature rows [t-L+1 .. t]; each of those rows only
contains information known at its own issue time (<= t - horizon), so no leakage.
Requires the optional extra:  pip install -e ".[lstm]"
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from ..features.build import feature_columns
from .base import Forecaster, trainable

log = logging.getLogger(__name__)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _dataset_class():
    import keras

    class Windows(keras.utils.PyDataset):
        """Builds (batch, L, n_features) windows on the fly to keep memory small."""

        def __init__(self, arr, ends, L, y=None, batch_size=128, shuffle=False, seed=0, **kw):
            super().__init__(**kw)
            self.arr, self.ends, self.L, self.y = arr, np.asarray(ends), L, y
            self.bs, self.shuffle = batch_size, shuffle
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(self.ends))

        def __len__(self):
            return int(np.ceil(len(self.ends) / self.bs))

        def on_epoch_end(self):
            if self.shuffle:  # shuffles *windows* between epochs; each window stays ordered
                self.rng.shuffle(self.order)

        def __getitem__(self, i):
            sel = self.ends[self.order[i * self.bs:(i + 1) * self.bs]]
            offs = np.arange(-self.L + 1, 1)
            x = self.arr[sel[:, None] + offs[None, :]]
            return (x, self.y[sel]) if self.y is not None else x

    return Windows


class LSTMForecaster(Forecaster):
    name = "lstm"

    def __init__(self, params: dict, seed: int = 42):
        self.p = dict(params)
        self.seed = seed
        self.model = None
        self.columns: list[str] = []

    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.columns].to_numpy(dtype=np.float32)
        x = (x - self.mu) / self.sd
        return np.nan_to_num(x, nan=0.0).astype(np.float32)

    def fit(self, frame: pd.DataFrame, train_mask: np.ndarray) -> "LSTMForecaster":
        import keras

        keras.utils.set_random_seed(self.seed)
        L = self.p["sequence_hours"]
        m = trainable(frame, train_mask)
        m[:L - 1] = False  # need a full look-back window
        self.columns = feature_columns(frame)
        tr = frame.loc[m, self.columns].to_numpy(dtype=np.float32)
        self.mu = np.nanmean(tr, axis=0)
        self.sd = np.nanstd(tr, axis=0)
        self.mu = np.nan_to_num(self.mu)
        self.sd = np.where(np.nan_to_num(self.sd) > 1e-6, np.nan_to_num(self.sd), 1.0)
        yv = frame["y"].to_numpy(dtype=np.float32)
        self.y_mu, self.y_sd = float(np.nanmean(yv[m])), float(np.nanstd(yv[m]) or 1.0)
        arr = self._matrix(frame)
        y_scaled = np.nan_to_num((yv - self.y_mu) / self.y_sd).astype(np.float32)

        self.model = keras.Sequential([
            keras.Input(shape=(L, len(self.columns))),
            keras.layers.LSTM(self.p["units"]),
            keras.layers.Dropout(self.p["dropout"]),
            keras.layers.Dense(1),
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(self.p["learning_rate"]), loss="mse")
        Windows = _dataset_class()
        ds = Windows(arr, np.where(m)[0], L, y_scaled, self.p["batch_size"], shuffle=True, seed=self.seed)
        self.model.fit(ds, epochs=self.p["epochs"], verbose=0)
        return self

    def predict(self, frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
        L = self.p["sequence_hours"]
        ends = np.where(np.asarray(mask))[0]
        out = np.full(len(ends), np.nan, dtype=float)
        ok = ends >= L - 1
        if ok.any():
            Windows = _dataset_class()
            ds = Windows(self._matrix(frame), ends[ok], L, None, 1024)
            out[ok] = self.model.predict(ds, verbose=0).ravel() * self.y_sd + self.y_mu
        return out
