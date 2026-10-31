"""Small result readers shared by make_figures.py and content_ch3.py (pandas + scipy only, no python-docx)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SCENARIOS = ("demand", "net_import", "outages", "price")


def read_csv_or_none(path: Path):
    """A CSV, or None if the file is missing or empty (the post-event tests file is empty when there are too few events)."""
    try:
        if path.exists() and path.stat().st_size > 5:
            return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        pass
    return None


def stars(p) -> str:
    if p is None or p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def forecasts(root: Path, sc: str) -> dict[str, pd.DataFrame]:
    d = root / "artifacts" / sc / "forecasts"
    return {p.stem: pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))}


def vs_reference(root: Path, sc: str, ref: str = "xgb_rolling", split: str = "test", fcs=None) -> pd.DataFrame:
    """Paired Wilcoxon test on DAILY mean absolute errors of every model against `ref` (same procedure as the project's
    wilcoxon.csv, whose reference is xgb_adaptive): columns model, ref_MAE, model_MAE, rel_pct (model vs ref), p_value."""
    fcs = fcs or forecasts(root, sc)
    if ref not in fcs:
        return pd.DataFrame()
    base = fcs[ref]
    mask = (base["split"] == split).to_numpy()
    daily = lambda fc: (fc["y"] - fc["yhat"]).abs().resample("D").mean()
    r = daily(base[mask])
    rows = []
    for name, fc in fcs.items():
        if name == ref:
            continue
        o = daily(fc[fc.index.isin(base.index[mask])])
        pair = pd.concat([r, o], axis=1, keys=["ref", "other"]).dropna()
        if len(pair) < 10:
            continue
        try:
            p = float(wilcoxon(pair["ref"], pair["other"]).pvalue)
        except ValueError:
            p = np.nan
        rows.append({"model": name, "n_days": len(pair), "ref_MAE": pair["ref"].mean(), "model_MAE": pair["other"].mean(),
                     "rel_pct": 100.0 * (pair["other"].mean() - pair["ref"].mean()) / pair["ref"].mean(), "p_value": p})
    return pd.DataFrame(rows)


# Reader-facing model names (the program identifiers appear only in the two model tables of chapter 1).
NAME = {"seasonal_naive": "сезонний наївний", "prophet_static": "статичний Prophet", "xgb_static": "статичний XGBoost",
        "xgb_rolling": "XGBoost, щотижня", "xgb_rolling_freq": "XGBoost, кожні 3 доби",
        "xgb_drift": "XGBoost, за зростанням похибки", "xgb_attack": "XGBoost, після удару",
        "xgb_adaptive": "XGBoost, за тривогою BOCPD", "xgb_adaptive_tau_w": "XGBoost, BOCPD зі ширшим вікном",
        "xgb_expanding": "XGBoost на всій історії", "persistence": "персистентність", "baseline28": "середнє за 28 діб",
        "ridge_depth": "регресія Ridge", "xgb_direct": "Ridge + XGBoost",
        "xgb_direct_t": "Ridge + XGBoost з ознаками напруженості", "xgb_aft": "модель виживання XGBoost",
        "xgb_cls": "класифікатор XGBoost"}
# after an attack the rolling model is refitted once, on the last 180 days
NAME_POST = {**NAME, "xgb_rolling": "XGBoost, ковзне вікно"}


def cap(s: str) -> str:
    """Capitalise the first letter (for table cells and legends)."""
    return s[:1].upper() + s[1:]
