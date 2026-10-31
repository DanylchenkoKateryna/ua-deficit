"""Calendar features, computed in local (Kyiv) time - they are known in advance, no leakage."""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def calendar_features(index: pd.DatetimeIndex, local_tz: str) -> pd.DataFrame:
    local = index.tz_convert(local_tz)
    ua = holidays.country_holidays("UA", years=range(local.year.min(), local.year.max() + 1))
    hol = np.array([d in ua for d in local.date])
    return pd.DataFrame({
        "hour": local.hour,
        "dayofweek": local.dayofweek,
        "month": local.month,
        "is_weekend": (local.dayofweek >= 5).astype(int),
        "is_holiday": hol.astype(int),
        "hour_sin": np.sin(2 * np.pi * local.hour / 24),
        "hour_cos": np.cos(2 * np.pi * local.hour / 24),
        "doy_sin": np.sin(2 * np.pi * local.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * local.dayofyear / 365.25),
    }, index=index)
