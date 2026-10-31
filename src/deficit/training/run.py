"""Train every enabled model and store its out-of-sample forecasts (validation + test)."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from ..config import Config
from ..io import read_json, read_parquet, write_json, write_parquet
from ..models.xgb_model import XGBForecaster
from ..preprocessing.master import split_label
from .walkforward import DriftParams, WalkForwardResult, fit_adaptive, fit_static

log = logging.getLogger(__name__)


def apply_tuned(cfg: Config) -> Config:
    """Overlay artifacts/tuned.json (written by `deficit tune`) onto the config."""
    path = cfg.artifacts_dir / "tuned.json"
    if not path.exists():
        return cfg
    t = read_json(path)
    cfg = cfg.model_copy(deep=True)
    if "xgboost" in t:
        cfg.models.xgboost = cfg.models.xgboost.model_copy(update=t["xgboost"])
    if "window_days" in t:
        cfg.adaptive.window_days = int(t["window_days"])
    if "bocpd_threshold" in t:
        cfg.changepoints.bocpd.threshold = float(t["bocpd_threshold"])
    log.info("applied tuned parameters from %s", path)
    return cfg


def apply_tuned_triggers(cfg: Config) -> Config:
    """Overlay artifacts/tuned_triggers.json (`deficit tune-triggers`, validation period only)."""
    path = cfg.artifacts_dir / "tuned_triggers.json"
    if not path.exists():
        return cfg
    t = read_json(path)
    cfg = cfg.model_copy(deep=True)
    cfg.adaptive = cfg.adaptive.model_copy(update=t.get("adaptive", {}))
    log.info("applied tuned trigger parameters from %s", path)
    return cfg


NO_ALARM_SUFFIXES = ("_rolling", "_attack", "_drift", "_recency", "_freq")   # BOCPD alarms are ignored


def trigger_kwargs(name: str, cfg: Config) -> dict:
    """Extra fit_adaptive arguments for the attack-, drift-triggered and recency-weighted models."""
    ad = cfg.adaptive
    if name.endswith("_attack"):
        from ..ingestion.attacks import load_attacks
        return {"attack_times": load_attacks(cfg)["datetime"].tolist(),
                "trigger_window_days": ad.attack_window_days, "trigger_hold_days": ad.attack_hold_days}
    if name.endswith("_drift"):
        return {"drift": DriftParams(k=ad.drift_k, h=ad.drift_h, ref_days=ad.drift_ref_days,
                                     min_days=ad.drift_min_days),
                "trigger_window_days": ad.drift_window_days, "trigger_hold_days": ad.drift_hold_days}
    if name.endswith("_recency"):      # long window, the recent days carry the weight
        return {"window_days": ad.recency_window_days}
    return {}


def _factory(name: str, cfg: Config):
    seed = cfg.tuning.seed
    if name == "seasonal_naive":
        from ..models.naive_model import SeasonalNaive
        return lambda: SeasonalNaive()
    if name.startswith("xgb"):
        halflife = cfg.adaptive.recency_halflife_days if name.endswith("_recency") else None
        return lambda: XGBForecaster(cfg.models.xgboost.model_dump(), seed, halflife_days=halflife)
    if name.startswith("lstm"):
        from ..models.lstm_model import LSTMForecaster
        return lambda: LSTMForecaster(cfg.models.lstm.model_dump(), seed)
    if name.startswith("prophet"):
        from ..models.prophet_model import ProphetForecaster
        return lambda: ProphetForecaster()
    raise ValueError(f"unknown model {name}")


def blend_static_rolling(cfg: Config, static: str = "xgb_static", rolling: str = "xgb_rolling"):
    """xgb_blend = a * static + (1 - a) * rolling, with a chosen on the VALIDATION split only.

    Needs both forecasts on disk. Costs no extra training; its refit count is the rolling model's plus the
    one static fit.
    """
    fdir = cfg.artifacts_dir / "forecasts"
    s, r = read_parquet(fdir / f"{static}.parquet"), read_parquet(fdir / f"{rolling}.parquet")
    df = pd.DataFrame({"y": r["y"], "static": s["yhat"].reindex(r.index),
                       "rolling": r["yhat"], "split": r["split"]})
    val = df[(df["split"] == "validation")].dropna()
    grid = np.linspace(0, 1, 11)
    scores = [float(np.mean(np.abs(val["y"] - (a * val["static"] + (1 - a) * val["rolling"])))) for a in grid]
    a = float(grid[int(np.argmin(scores))])
    yhat = a * df["static"] + (1 - a) * df["rolling"]
    fc = pd.DataFrame({"y": df["y"], "yhat": yhat})
    write_json({"static_weight": a, "grid": dict(zip(map(float, grid), scores))}, cfg.artifacts_dir / "blend.json")
    log.info("xgb_blend: static weight %.1f chosen on validation (MAE %.4g)", a, min(scores))
    refits = pd.concat([pd.read_csv(fdir / f"{static}_refits.csv"), pd.read_csv(fdir / f"{rolling}_refits.csv")],
                       ignore_index=True)
    return WalkForwardResult(fc, refits, None)


def lstm_available() -> bool:
    try:
        import keras  # noqa: F401
        return True
    except Exception:
        return False


def train_all(cfg: Config, models: list[str] | None = None) -> dict[str, pd.DataFrame]:
    cfg = apply_tuned_triggers(apply_tuned(cfg))
    frame = read_parquet(cfg.processed_dir / "features.parquet")
    signal = read_parquet(cfg.artifacts_dir / "bocpd_signal.parquet")
    start = cfg.train_end_ts + pd.Timedelta(hours=1)
    names = models or cfg.models.enabled
    if any(n.startswith("lstm") for n in names) and not lstm_available():
        log.warning("keras not installed - skipping LSTM models (pip install -e \".[lstm]\")")
        names = [n for n in names if not n.startswith("lstm")]

    out = {}
    for name in names:
        t0 = time.time()
        log.info("training %s", name)
        factory = None if name.endswith("_blend") else _factory(name, cfg)
        if name.endswith("_blend"):
            res = blend_static_rolling(cfg)
        elif name.endswith("_static") or name == "seasonal_naive":
            res = fit_static(factory, frame, cfg, start)
        else:
            every = (cfg.models.lstm.retrain_every_days if name.startswith("lstm") else
                     cfg.adaptive.control_every_days if name.endswith("_freq") else None)
            res = fit_adaptive(factory, frame, cfg, signal, start, retrain_every_days=every,
                               use_alarms=not name.endswith(NO_ALARM_SUFFIXES),
                               mode="tau_w" if name.endswith("_tau_w") else "regime",
                               **trigger_kwargs(name, cfg))
        fc = res.forecasts.assign(split=lambda d: split_label(d.index, cfg), model=name)
        write_parquet(fc, cfg.artifacts_dir / "forecasts" / f"{name}.parquet")
        res.refits.to_csv(cfg.artifacts_dir / "forecasts" / f"{name}_refits.csv", index=False)
        imp = res.last_model.feature_importance() if res.last_model else None
        if imp is not None:
            imp.rename("importance").to_csv(cfg.reports_dir / f"feature_importance_{name}.csv")
        log.info("%s done in %.0fs (%d refits, %.1f%% forecasts missing)", name, time.time() - t0,
                 len(res.refits), 100 * np.isnan(fc["yhat"]).mean())
        out[name] = fc
    return out
