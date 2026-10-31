"""Hyper-parameter tuning on the VALIDATION period only (2022-2023). The test period is never touched.

1. XGBoost hyper-parameters - Optuna, TimeSeriesSplit over validation rows.
2. Adaptive window and BOCPD threshold - small grid, xgb_adaptive walk-forward over validation.
Result -> artifacts/tuned.json, applied automatically by `deficit train`.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..config import Config
from ..evaluation.metrics import mae
from ..features.build import feature_columns
from ..io import read_parquet, write_json
from ..models.xgb_model import XGBForecaster
from .walkforward import fit_adaptive

log = logging.getLogger(__name__)


def tune_xgb(frame: pd.DataFrame, cfg: Config) -> dict:
    val = frame[(frame["split"] == "validation") & frame["y"].notna()]
    X, y = val[feature_columns(frame)], val["y"]
    tss = TimeSeriesSplit(n_splits=cfg.tuning.cv_splits, gap=cfg.features.horizon_hours)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
        }
        errs = []
        for tr, te in tss.split(X):
            m = XGBForecaster(params, cfg.tuning.seed)
            sub = val.iloc[np.r_[tr, te]]
            mask = np.zeros(len(sub), bool); mask[: len(tr)] = True
            m.fit(sub, mask)
            errs.append(mae(sub["y"].to_numpy()[~mask], m.predict(sub, ~mask)))
        return float(np.mean(errs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.tuning.seed))
    study.optimize(objective, n_trials=cfg.tuning.n_trials, show_progress_bar=False)
    log.info("best XGB MAE %.1f with %s", study.best_value, study.best_params)
    return study.best_params


def tune_adaptive(frame: pd.DataFrame, signal: pd.DataFrame, cfg: Config, xgb_params: dict,
                  windows=(30, 60, 90, 180), thresholds=(0.5, 0.7, 0.9)) -> dict:
    start = cfg.train_end_ts + pd.Timedelta(hours=1)
    end = cfg.validation_end_ts
    factory = lambda: XGBForecaster(xgb_params, cfg.tuning.seed)  # noqa: E731
    results = []
    for w in windows:
        for thr in thresholds:
            res = fit_adaptive(factory, frame, cfg, signal, start, end,
                               retrain_every_days=max(14, cfg.adaptive.retrain_every_days),
                               window_days=w, threshold=thr)
            fc = res.forecasts.dropna()
            score = mae(fc["y"].to_numpy(), fc["yhat"].to_numpy())
            results.append({"window_days": w, "bocpd_threshold": thr, "val_MAE": score})
            log.info("window=%d threshold=%.2f -> validation MAE %.1f", w, thr, score)
    best = min(results, key=lambda r: r["val_MAE"])
    return {"window_days": best["window_days"], "bocpd_threshold": best["bocpd_threshold"],
            "grid": results}


def tune(cfg: Config) -> dict:
    frame = read_parquet(cfg.processed_dir / "features.parquet")
    signal = read_parquet(cfg.artifacts_dir / "bocpd_signal.parquet")
    xgb_params = {**cfg.models.xgboost.model_dump(), **tune_xgb(frame, cfg)}
    ad = tune_adaptive(frame, signal, cfg, xgb_params)
    out = {"xgboost": xgb_params, **ad}
    write_json(out, cfg.artifacts_dir / "tuned.json")
    return out


def _val_run(factory, frame, cfg, signal, start, end, **kw) -> dict:
    res = fit_adaptive(factory, frame, cfg, signal, start, end, use_alarms=False, **kw)
    fc = res.forecasts.dropna()
    return {"val_MAE": float(mae(fc["y"].to_numpy(), fc["yhat"].to_numpy())), "n_refits": int(len(res.refits)),
            "n_triggered": int(res.refits["reason"].isin(["attack", "drift"]).sum())}


def tune_triggers(cfg: Config) -> dict:
    """Choose the attack- and drift-trigger parameters on the VALIDATION period only.

    Every candidate is a walk-forward run over validation (never the test period) of the weekly rolling
    schedule plus the extra trigger. The detector (k, h) is tuned first with the shortest candidate window, then
    the refit window and hold time given the detector. Ties go to the candidate with fewer refits.
    Result -> artifacts/<scenario>/tuned_triggers.json, applied by `deficit train`.
    """
    from .run import apply_tuned
    from ..ingestion.attacks import load_attacks
    from .walkforward import DriftParams

    cfg = apply_tuned(cfg)
    frame = read_parquet(cfg.processed_dir / "features.parquet")
    signal = read_parquet(cfg.artifacts_dir / "bocpd_signal.parquet")
    factory = lambda: XGBForecaster(cfg.models.xgboost.model_dump(), cfg.tuning.seed)  # noqa: E731
    start, end = cfg.train_end_ts + pd.Timedelta(hours=1), cfg.validation_end_ts
    ad = cfg.adaptive
    attacks = load_attacks(cfg)["datetime"].tolist()
    wins = sorted({x for x in (14, 30, 60, ad.window_days) if x <= ad.window_days})
    holds = (0, 7)
    key = lambda r: (round(r["val_MAE"], 12), r["n_refits"])  # noqa: E731

    base = _val_run(factory, frame, cfg, signal, start, end)
    log.info("validation, weekly rolling (window %d): MAE %.4g, %d refits", ad.window_days, base["val_MAE"], base["n_refits"])

    grid_a = []
    for win in wins:
        for hold in holds:
            r = {"window_days": win, "hold_days": hold,
                 **_val_run(factory, frame, cfg, signal, start, end, attack_times=attacks,
                            trigger_window_days=win, trigger_hold_days=hold)}
            grid_a.append(r)
            log.info("attack  win=%d hold=%d -> val MAE %.4g (%d refits)", win, hold, r["val_MAE"], r["n_refits"])
    best_a = min(grid_a, key=key)

    grid_d = []
    for dk in (0.25, 0.5, 1.0):
        for dh in (2.0, 4.0, 8.0):
            p = DriftParams(k=dk, h=dh, ref_days=ad.drift_ref_days, min_days=ad.drift_min_days)
            r = {"k": dk, "h": dh, "window_days": wins[0], "hold_days": 0,
                 **_val_run(factory, frame, cfg, signal, start, end, drift=p,
                            trigger_window_days=wins[0], trigger_hold_days=0)}
            grid_d.append(r)
            log.info("drift k=%.2f h=%.1f -> val MAE %.4g (%d refits, %d triggered)", dk, dh, r["val_MAE"], r["n_refits"], r["n_triggered"])
    best_d = min(grid_d, key=key)
    p = DriftParams(k=best_d["k"], h=best_d["h"], ref_days=ad.drift_ref_days, min_days=ad.drift_min_days)
    for win in wins:
        for hold in holds:
            if (win, hold) == (wins[0], 0):
                continue
            r = {"k": best_d["k"], "h": best_d["h"], "window_days": win, "hold_days": hold,
                 **_val_run(factory, frame, cfg, signal, start, end, drift=p,
                            trigger_window_days=win, trigger_hold_days=hold)}
            grid_d.append(r)
            log.info("drift win=%d hold=%d -> val MAE %.4g (%d refits)", win, hold, r["val_MAE"], r["n_refits"])
    best_d = min((r for r in grid_d if (r["k"], r["h"]) == (best_d["k"], best_d["h"])), key=key)

    out = {"adaptive": {"attack_window_days": best_a["window_days"], "attack_hold_days": best_a["hold_days"],
                        "drift_window_days": best_d["window_days"], "drift_hold_days": best_d["hold_days"],
                        "drift_k": best_d["k"], "drift_h": best_d["h"]},
           "rolling_val": base, "best_attack": best_a, "best_drift": best_d,
           "grid_attack": grid_a, "grid_drift": grid_d}
    write_json(out, cfg.artifacts_dir / "tuned_triggers.json")
    return tune_recency(cfg, out)


def tune_recency(cfg: Config, out: dict | None = None) -> dict:
    """Half-life of the recency weights (xgb_recency) on the VALIDATION period; merged into tuned_triggers.json."""
    from .run import apply_tuned
    from ..io import read_json

    cfg = apply_tuned(cfg)
    path = cfg.artifacts_dir / "tuned_triggers.json"
    out = out if out is not None else read_json(path)
    frame = read_parquet(cfg.processed_dir / "features.parquet")
    signal = read_parquet(cfg.artifacts_dir / "bocpd_signal.parquet")
    start, end = cfg.train_end_ts + pd.Timedelta(hours=1), cfg.validation_end_ts
    ad = cfg.adaptive
    grid = []
    for hl in (7, 14, 30, 60):
        factory = lambda hl=hl: XGBForecaster(cfg.models.xgboost.model_dump(), cfg.tuning.seed, halflife_days=hl)  # noqa: E731
        r = {"halflife_days": hl, **_val_run(factory, frame, cfg, signal, start, end, window_days=ad.recency_window_days)}
        grid.append(r)
        log.info("recency half-life %d d -> val MAE %.4g", hl, r["val_MAE"])
    best = min(grid, key=lambda r: (round(r["val_MAE"], 12), -r["halflife_days"]))
    out["adaptive"]["recency_halflife_days"] = best["halflife_days"]
    out["grid_recency"], out["best_recency"] = grid, best
    write_json(out, path)
    return out
