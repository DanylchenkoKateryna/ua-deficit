"""MAE, RMSE, MAPE."""
from __future__ import annotations

import numpy as np


def _clean(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = ~(np.isnan(y) | np.isnan(yhat))
    return y[ok], yhat[ok]


def mae(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.mean(np.abs(y - yhat))) if len(y) else float("nan")


def rmse(y, yhat) -> float:
    y, yhat = _clean(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else float("nan")


def mape(y, yhat, min_abs: float = 0.0) -> float:
    """MAPE in %, skipping |y| < min_abs: the deficit crosses zero, where MAPE is undefined."""
    y, yhat = _clean(y, yhat)
    ok = np.abs(y) >= max(min_abs, 1e-9)
    if not ok.any():
        return float("nan")
    return float(100 * np.mean(np.abs((y[ok] - yhat[ok]) / y[ok])))


def all_metrics(y, yhat, min_abs: float) -> dict:
    y_, _ = _clean(y, yhat)
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "MAPE_pct": mape(y, yhat, min_abs),
            "n_hours": int(len(y_))}
