"""Common forecaster interface.

Every model receives the *full* feature frame plus a boolean mask, so sequence models can
look back over rows that are not themselves training targets. Implementations must only
read rows at or before the rows they fit / predict (features are already issue-time safe).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, frame: pd.DataFrame, train_mask: np.ndarray) -> "Forecaster": ...

    @abstractmethod
    def predict(self, frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray: ...

    def feature_importance(self) -> pd.Series | None:
        return None


def trainable(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Rows usable as training targets: inside the mask and with an observed target."""
    return np.asarray(mask) & frame["y"].notna().to_numpy()
