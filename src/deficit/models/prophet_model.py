"""Prophet baseline - the 'null hypothesis' model.

Trained once on peacetime data and never retrained, so its error growth after each attack
wave quantifies the structural breaks.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import Forecaster, trainable

log = logging.getLogger(__name__)
REGRESSORS = ["temperature_C", "heating_degree"]


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(self, changepoints: list[pd.Timestamp] | None = None):
        self.changepoints = changepoints  # optional known event dates (e.g. attacks)
        self.model = None
        self.regressors: list[str] = []

    def _df(self, frame: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
        sub = frame.loc[mask]
        df = pd.DataFrame({"ds": sub.index.tz_convert(None), "y": sub["y"].to_numpy()})
        for r in self.regressors:
            df[r] = sub[r].to_numpy()
        return df

    def fit(self, frame: pd.DataFrame, train_mask: np.ndarray) -> "ProphetForecaster":
        from prophet import Prophet

        self.regressors = [r for r in REGRESSORS if r in frame]
        m = trainable(frame, train_mask)
        cps = None
        if self.changepoints:
            lo, hi = frame.index[m].min(), frame.index[m].max()
            cps = [c.tz_convert(None) for c in self.changepoints if lo < c < hi] or None
        self.model = Prophet(changepoints=cps, daily_seasonality=True, weekly_seasonality=True,
                             yearly_seasonality=True)
        self.model.add_country_holidays("UA")
        for r in self.regressors:
            self.model.add_regressor(r)
        self.model.fit(self._df(frame, m))
        return self

    def predict(self, frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
        df = self._df(frame, np.asarray(mask)).drop(columns="y")
        return self.model.predict(df)["yhat"].to_numpy()
