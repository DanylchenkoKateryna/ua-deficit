"""Seasonal-naive baseline: the value one week earlier (lag_168h), which is known at issue time.

The simplest forecast a reviewer will ask about. It needs no training and cannot be fooled by a trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Forecaster


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def fit(self, frame: pd.DataFrame, train_mask: np.ndarray) -> "SeasonalNaive":
        if "lag_168h" not in frame:
            raise ValueError("seasonal_naive needs the lag_168h feature (features.lags_hours must include 168)")
        return self

    def predict(self, frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
        return frame.loc[np.asarray(mask), "lag_168h"].to_numpy(dtype=float)
