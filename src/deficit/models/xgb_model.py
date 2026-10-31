"""XGBoost on lag / rolling / calendar / weather / attack features (thesis §1.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..features.build import feature_columns
from .base import Forecaster, trainable


class XGBForecaster(Forecaster):
    name = "xgboost"

    def __init__(self, params: dict, seed: int = 42, halflife_days: float | None = None):
        self.params = dict(params)
        self.seed = seed
        self.halflife_days = halflife_days   # recency weighting: a row loses half its weight every N days of age
        self.model: xgb.XGBRegressor | None = None
        self.columns: list[str] = []

    def fit(self, frame: pd.DataFrame, train_mask: np.ndarray) -> "XGBForecaster":
        m = trainable(frame, train_mask)
        self.columns = feature_columns(frame)
        self.model = xgb.XGBRegressor(**self.params, random_state=self.seed, n_jobs=-1,
                                      tree_method="hist")
        self.model.fit(frame.loc[m, self.columns], frame.loc[m, "y"], sample_weight=self._weights(frame.index[m]))
        return self

    def _weights(self, index: pd.DatetimeIndex) -> np.ndarray | None:
        """Weight 0.5 ** (age / halflife); age is measured from the newest training row, so it only
        uses the training window itself (no look-ahead)."""
        if not self.halflife_days:
            return None
        age_days = (index.max() - index) / pd.Timedelta(days=1)
        return np.power(0.5, np.asarray(age_days) / self.halflife_days)

    def predict(self, frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
        return self.model.predict(frame.loc[np.asarray(mask), self.columns])

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.columns).sort_values(ascending=False)
