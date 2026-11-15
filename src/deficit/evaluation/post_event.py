"""Forecast of the DEPTH and DURATION of the deficit after an attack or an alarm (the thesis title).

At an issue day d0 (an attack day, or the day BOCPD raises an alarm) the model sees only the daily deficit up to d0
and predicts the daily mean for each of the next H days. From that trajectory
  depth    = highest predicted daily mean,
  duration = days until the prediction stays at or below the episode threshold for 3 days in a row.
The same rules applied to the real days after d0 give the actual depth and duration.

Methods (all use only information available at d0):
  persistence     the last observed day, kept constant
  baseline28      the 28-day mean before d0 ("nothing changes")
  xgb_static      XGBoost trained once on the training period
  xgb_expanding   XGBoost retrained at d0 on all targets known by d0
  xgb_rolling     XGBoost retrained at d0 on the last `window_days` days of known targets
  blend           the mean of persistence and xgb_rolling
  xgb_cls         duration only: a classifier trained like xgb_rolling predicts P(day k is above the threshold);
                  the episode is over at the first run of three days with P < 0.5
One stacked model with the horizon k as a feature is used instead of H separate models.

Direct methods (they skip the smoothed path: one row per historical issue day i, one target per quantity, all taken
from the real window y[i+1..i+H]: depth = its maximum, duration = the rule above (censored at H), deficit days = number of
days above the threshold):
  ridge_depth     depth only: ridge regression on the recent levels (y0, m7, max7, m28, max28); extrapolates to new regimes
  xgb_direct      ridge + XGBoost trees on its residual for the depth; XGBoost (absolute error) for duration and deficit days
  xgb_aft         duration only: XGBoost survival:aft with right-censored durations (window never recovered)
  xgb_direct_t, xgb_aft_t   the same trees with the system-tension features added (season, 7-day day-ahead price / cap,
                  days above the threshold, current run length, 28-day maximum); an ablation reported next to the others
  xgb_cls         also forecasts the deficit days: the sum of the per-day probabilities
  *_g, *_tg       the same direct models with Telegram severity features added (features/telegram.py: damaged generation /
                  substations, regions hit, consumers without power, emergency outages, schedules, lifting statements on the issue
                  day and over 7 / 30 days). `_g` = base + telegram, `_tg` = tension + telegram. The old sets stay for comparison.
                  Only posts published before the end of the issue day are read (`telegram_features`).
LEAKAGE RULE: an example issued at day i is used for a forecast at d0 only if its whole window is observed, i + H <= d0
(`known_rows`; tests/test_post_event.py scrambles the days after d0 and checks that nothing changes).
Features at d0 use days <= d0 only; the day-ahead price of day d is published on d-1, so its values up to d0 are known.
"""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

from ..config import Config  # noqa: E402
from ..ingestion.attacks import load_attacks  # noqa: E402
from ..features.telegram import TG_COLS, TG_LIN, issue_features as telegram_features  # noqa: E402
from ..io import read_json, read_parquet  # noqa: E402

log = logging.getLogger(__name__)
PATH_METHODS = ["persistence", "baseline28", "xgb_static", "xgb_expanding", "xgb_rolling", "blend"]   # predict a daily path
METHODS = PATH_METHODS + ["ridge_depth", "xgb_direct", "xgb_direct_t", "ridge_depth_g", "xgb_direct_g", "xgb_direct_tg"]   # all methods that forecast the depth
DURATION_ONLY = ["xgb_cls", "xgb_aft", "xgb_aft_t", "xgb_aft_g", "xgb_aft_tg"]             # duration (and deficit days) but no depth
TG_SETS = {"g": "base", "tension_g": "tension"}                  # telegram feature set -> the set it extends
ALL_METHODS = METHODS + DURATION_ONLY
FEATURES = ["k", "y0", "m3", "m7", "m28", "max7", "slope", "std7", "temp7", "since_attack", "n_attack_30", "through7", "through30", "tdoy_sin", "tdoy_cos", "tdow"]
BASE_DIRECT = ["y0", "m3", "m7", "m28", "max7", "slope", "std7", "temp7", "since_attack", "n_attack_30", "through7", "through30"]
LINEAR_COLS = ["y0", "m7", "max7", "m28", "max28"]
TENSION = ["doy_sin", "doy_cos", "n_above7", "run_above", "max28", "dam7"]                              # added by direct_features


def load_dam_daily(cfg: Config, index: pd.DatetimeIndex) -> pd.Series | None:
    """Daily mean day-ahead price / cap, or None (no DAM file, or the DAM price is the target itself)."""
    path = cfg.raw_dir / "dam_hourly.parquet"
    if cfg.target.definition == "dam_price" or not path.exists():
        return None
    return read_parquet(path)["dam_price_ratio"].resample("D").mean().reindex(index)


def daily_table(master: pd.DataFrame, dam: pd.Series | None = None) -> pd.DataFrame:
    d = pd.DataFrame({"y": master["deficit"].resample("D").mean()})
    if dam is not None:
        d["dam"] = dam.reindex(d.index)
    d["temp"] = master["temperature_C"].resample("D").mean() if "temperature_C" in master else 0.0
    d["attack"] = (master["attack_flag"].resample("D").sum() > 0).astype(float)
    d["through"] = master["weapons_through"].resample("D").sum() if "weapons_through" in master else 0.0   # strength of the strikes
    return d


def issue_features(d: pd.DataFrame) -> pd.DataFrame:
    """Features known at the END of each day (index = issue day)."""
    y = d["y"]
    f = pd.DataFrame(index=d.index)
    f["y0"] = y
    f["m3"] = y.rolling(3, min_periods=2).mean()
    f["m7"] = y.rolling(7, min_periods=4).mean()
    f["m28"] = y.rolling(28, min_periods=14).mean()
    f["max7"] = y.rolling(7, min_periods=4).max()
    f["slope"] = f["m3"] - y.shift(3).rolling(7, min_periods=4).mean()
    f["std7"] = y.rolling(7, min_periods=4).std()
    f["temp7"] = d["temp"].rolling(7, min_periods=4).mean()
    last = pd.Series(np.where(d["attack"] > 0, np.arange(len(d)), np.nan), index=d.index).ffill()
    f["since_attack"] = (np.arange(len(d)) - last).clip(upper=30).fillna(30)       # capped: a growing clock is never seen in training
    f["n_attack_30"] = d["attack"].rolling(30, min_periods=1).sum()
    f["through7"] = d["through"].rolling(7, min_periods=1).sum()
    f["through30"] = d["through"].rolling(30, min_periods=1).sum()
    return f


def direct_features(d: pd.DataFrame, thr: float) -> pd.DataFrame:
    """issue_features plus system-tension features, all known at the END of the issue day (backward windows only)."""
    f = issue_features(d)
    y = d["y"]
    f["doy_sin"] = np.sin(2 * np.pi * d.index.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * d.index.dayofyear / 365.25)
    above = (y > thr).astype(float).where(y.notna())
    f["n_above7"] = above.rolling(7, min_periods=4).sum()
    run, cur = np.zeros(len(d)), 0
    for i, a in enumerate(above.to_numpy()):                  # consecutive days above the threshold up to and including the day
        cur = cur + 1 if a == 1 else 0
        run[i] = min(cur, 30)
    f["run_above"] = run
    f["max28"] = y.rolling(28, min_periods=14).max()
    if "dam" in d:
        f["dam7"] = d["dam"].rolling(7, min_periods=3).mean()   # price of day d is published on d-1: values up to d0 are known
    return f


def direct_targets(y: np.ndarray, H: int, thr: float, min_share: float) -> pd.DataFrame:
    """Per issue day i: depth, duration, censoring and deficit days of the window y[i+1..i+H]. NaN unless the window is
    entirely inside the data (the examples of the last H days have no complete window)."""
    n = len(y)
    out = pd.DataFrame({"depth": np.nan, "duration": np.nan, "censored": np.nan, "ndays": np.nan, "end_i": np.arange(n) + H})
    cols = {c: out[c].to_numpy().copy() for c in out}
    for i in range(n - H):
        w = y[i + 1: i + 1 + H]
        avail = int(np.sum(~np.isnan(w)))
        if avail < min_share * H:
            continue
        dep, dur, cens = depth_duration(w, thr)
        cols["depth"][i], cols["duration"][i], cols["censored"][i] = dep, dur, float(cens)
        cols["ndays"][i] = np.nansum(w > thr) / avail * H
    return pd.DataFrame(cols)


def direct_frame(d: pd.DataFrame, H: int, thr: float, pe, tg: pd.DataFrame | None = None) -> pd.DataFrame:
    """All direct features and the direct targets of every issue day (row position = issue day position).
    attrs: "base" = tree features of the standard set, "tension" = base + season / DAM price / run lengths;
    with `tg` (telegram issue features, same index as `d`): "g" = base + telegram, "tension_g" = tension + telegram.
    attrs["lin_cols"] = the columns of the ridge of each set."""
    F = direct_features(d, thr).reset_index(drop=True)
    T = direct_targets(d["y"].to_numpy(), H, thr, pe.direct_min_avail_share)
    parts = [F, T]
    sets = {"base": list(BASE_DIRECT), "tension": BASE_DIRECT + [c for c in TENSION if c in F]}
    lin = {k: list(LINEAR_COLS) for k in sets}
    if tg is not None:
        parts.append(tg.reset_index(drop=True)[TG_COLS])
        for name, src in TG_SETS.items():
            sets[name] = sets[src] + TG_COLS
            lin[name] = LINEAR_COLS + TG_LIN
    out = pd.concat(parts, axis=1)
    out.attrs["feature_sets"], out.attrs["lin_cols"] = sets, lin
    return out


def known_rows(frame: pd.DataFrame, i0: int, window_days: int | None = None) -> pd.DataFrame:
    """Training examples usable at issue day i0: the whole window i+1..i+H is observed, i.e. i + H <= i0."""
    k = frame[(frame["end_i"] <= i0) & frame["depth"].notna() & frame["y0"].notna() & frame["m28"].notna()]
    if window_days is not None:
        k = k[k.index > i0 - window_days]
    return k


def _booster_params(pe, xcfg: dict, seed: int, objective: str) -> dict:
    return dict(n_estimators=pe.direct_n_estimators, max_depth=pe.direct_max_depth, learning_rate=pe.direct_learning_rate,
                min_child_weight=pe.direct_min_child_weight, subsample=xcfg.get("subsample", 0.8),
                colsample_bytree=xcfg.get("colsample_bytree", 0.8), objective=objective, random_state=seed,
                n_jobs=1, tree_method="hist")                       # a few thousand rows: one thread is fastest


def _fit_aft(X: pd.DataFrame, dur: np.ndarray, cens: np.ndarray, pe, xcfg: dict, seed: int):
    """survival:aft on days-to-recovery (+1 so that the label is positive); a window that never recovered is right-censored."""
    dm = xgb.DMatrix(X)
    dm.set_float_info("label_lower_bound", dur + 1.0)
    dm.set_float_info("label_upper_bound", np.where(cens > 0, np.inf, dur + 1.0))
    params = {"objective": "survival:aft", "eval_metric": "aft-nloglik", "aft_loss_distribution": pe.aft_distribution,
              "aft_loss_distribution_scale": pe.aft_scale, "learning_rate": pe.direct_learning_rate,
              "max_depth": pe.direct_max_depth, "min_child_weight": pe.direct_min_child_weight,
              "subsample": xcfg.get("subsample", 0.8), "colsample_bytree": xcfg.get("colsample_bytree", 0.8),
              "tree_method": "hist", "seed": seed, "verbosity": 0, "nthread": 1}
    return xgb.train(params, dm, num_boost_round=pe.direct_n_estimators)


class DirectModel:
    """Direct models fitted on the examples known at one issue day (see `known_rows`), for one tree feature set.
      ridge        depth = ridge regression on the recent levels (y0, m7, max7, m28, max28)
      depth        ridge + XGBoost trees on its residual
      duration     XGBoost, absolute error (median days to recovery; a window that never recovered counts as H)
      ndays        XGBoost, absolute error (days above the threshold in the window)
      aft          XGBoost survival:aft, right-censored durations
    """

    def __init__(self, known: pd.DataFrame, feats: list[str], H: int, pe, xcfg: dict, seed: int,
                 lin_cols: list[str] | None = None):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.feats, self.H, self.lin_cols = feats, H, list(lin_cols or LINEAR_COLS)
        X = known[feats]
        self.ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=pe.depth_ridge_alpha))
        self.ridge.fit(known[self.lin_cols], known["depth"])
        fit = lambda obj, target: xgb.XGBRegressor(**_booster_params(pe, xcfg, seed, obj)).fit(X, target)
        self.depth = fit(pe.depth_objective, known["depth"].to_numpy() - self.ridge.predict(known[self.lin_cols]))
        self.duration = fit(pe.duration_objective, known["duration"])
        self.ndays = fit(pe.duration_objective, known["ndays"])
        self.aft = _fit_aft(X, known["duration"].to_numpy(), known["censored"].to_numpy(), pe, xcfg, seed)

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        H = self.H
        ridge = self.ridge.predict(x[self.lin_cols])
        xf = x[self.feats]
        return pd.DataFrame({"ridge_depth": ridge, "depth": ridge + self.depth.predict(xf),
                             "duration": np.clip(np.round(self.duration.predict(xf)), 0, H),
                             "ndays": np.clip(self.ndays.predict(xf), 0, H),
                             "aft_duration": np.clip(np.round(self.aft.predict(xgb.DMatrix(xf)) - 1.0), 0, H)}, index=x.index)


def direct_forecast(d: pd.DataFrame, i0: int, H: int, thr: float, pe, xcfg: dict, seed: int,
                    frame: pd.DataFrame | None = None, feature_set: str = "base") -> dict[str, float]:
    """Depth, duration (regression and AFT) and deficit days for the window after issue day i0.
    Only examples with i + H <= i0 are used for training, and only features of day i0 for the forecast."""
    frame = direct_frame(d, H, thr, pe) if frame is None else frame
    known = known_rows(frame, i0, pe.direct_window_days)
    x0 = frame.loc[[i0]]
    if len(known) < pe.direct_min_rows or x0[["y0", "m28"]].isna().any(axis=None):
        return {k: np.nan for k in ("ridge_depth", "depth", "duration", "ndays", "aft_duration")}
    return DirectModel(known, frame.attrs["feature_sets"][feature_set], H, pe, xcfg, seed,
                       frame.attrs.get("lin_cols", {}).get(feature_set)).predict(x0).iloc[0].to_dict()


def stacked(d: pd.DataFrame, H: int) -> pd.DataFrame:
    f = issue_features(d)
    n = len(d)
    parts = []
    y = d["y"].to_numpy()
    for k in range(1, H + 1):
        g = f.iloc[: n - k].copy()
        g["k"] = k
        g["target"] = y[k:]
        g["issue_i"] = np.arange(n - k)
        g["target_i"] = np.arange(k, n)
        tdate = d.index[k:]
        g["tdoy_sin"] = np.sin(2 * np.pi * tdate.dayofyear / 365.25)
        g["tdoy_cos"] = np.cos(2 * np.pi * tdate.dayofyear / 365.25)
        g["tdow"] = tdate.dayofweek
        parts.append(g)
    return pd.concat(parts).dropna(subset=["y0", "target", "m7", "m28"])


def duration_from_flags(recovered: np.ndarray) -> int:
    """Days until three consecutive recovered days (all of them if that never happens)."""
    r = np.asarray(recovered, bool)
    for k in range(len(r) - 2):
        if r[k] and r[k + 1] and r[k + 2]:
            return k
    return len(r)


def depth_duration(path: np.ndarray, threshold: float) -> tuple[float, int, bool]:
    """(depth, duration in days, censored). NaN days are unknown."""
    p = np.asarray(path, float)
    avail = int(np.sum(~np.isnan(p)))
    if avail == 0:
        return np.nan, 0, True
    depth = float(np.nanmax(p))
    ok = np.where(np.isnan(p), np.nan, (p <= threshold).astype(float))
    for k in range(len(p) - 2):
        if ok[k] == 1 and ok[k + 1] == 1 and ok[k + 2] == 1:
            return depth, k, False
    return depth, avail, True


def _fit(train: pd.DataFrame, params: dict, seed: int):
    if len(train) < 300:
        return None
    m = xgb.XGBRegressor(**params, random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(train[FEATURES], train["target"])
    return m


def _cls_probs(train: pd.DataFrame, X0: pd.DataFrame, thr: float, params: dict, seed: int) -> np.ndarray | None:
    """P(daily mean above the threshold) for the horizon days; None if the training window has too few examples."""
    if len(train) < 300:
        return None
    yb = (train["target"] > thr).astype(int)
    if yb.nunique() < 2:                                  # the window never (or always) crossed the threshold
        return np.full(len(X0), float(yb.iloc[0]))
    p = {k: v for k, v in params.items() if k != "objective"}
    m = xgb.XGBClassifier(**p, random_state=seed, n_jobs=-1, tree_method="hist")
    m.fit(train[FEATURES], yb)
    return m.predict_proba(X0[FEATURES])[:, 1]


def _events(cfg: Config, d: pd.DataFrame, first_i: int) -> list[tuple[str, int]]:
    out = []
    attacks = load_attacks(cfg)
    for t in attacks.loc[attacks["major"], "datetime"]:
        day = t.normalize()
        if day in d.index:
            out.append(("attack", int(d.index.get_loc(day))))
    cp_path = cfg.artifacts_dir / "changepoints.json"
    if cp_path.exists():
        for a in read_json(cp_path)["bocpd"]["alarms"]:
            day = pd.Timestamp(a["alarm"], tz="UTC").normalize()
            if day in d.index:
                out.append(("alarm", int(d.index.get_loc(day))))
    return sorted({e for e in out if e[1] >= first_i}, key=lambda e: (e[1], e[0]))


def paired_tests(table: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank test of each method against persistence on the per-event errors."""
    from scipy.stats import wilcoxon

    rows = []
    t = table.drop_duplicates(["issue_day"]) if len(table) else table      # an attack and an alarm on one day are one event
    for m in ALL_METHODS:
        if m == "persistence" or t.empty:
            continue
        diffs = {}
        if f"{m}_depth" in t:
            diffs["depth"] = (t[f"{m}_depth"] - t["actual_depth"]).abs() - (t["persistence_depth"] - t["actual_depth"]).abs()
        if f"{m}_duration_days" in t:
            diffs["duration"] = (t[f"{m}_duration_days"] - t["actual_duration_days"]).abs() \
                - (t["persistence_duration_days"] - t["actual_duration_days"]).abs()
        if f"{m}_ndays" in t and "actual_ndays" in t:
            diffs["deficit_days"] = (t[f"{m}_ndays"] - t["actual_ndays"]).abs() - (t["persistence_ndays"] - t["actual_ndays"]).abs()
        if f"{m}_traj_MAE" in t:
            diffs["trajectory"] = t[f"{m}_traj_MAE"] - t["persistence_traj_MAE"]
        for what, diff in diffs.items():
            diff = diff.dropna()
            if len(diff) < 8 or (diff == 0).all():
                continue
            rows.append({"method": m, "metric": what, "n_events": len(diff), "mean_error_minus_persistence": float(diff.mean()),
                         "share_better": float((diff < 0).mean()), "p_value": float(wilcoxon(diff).pvalue)})
    return pd.DataFrame(rows)


def telegram_vs_base_tests(table: pd.DataFrame) -> pd.DataFrame:
    """Do the telegram features help? Paired Wilcoxon of each telegram method against the same method without them
    (per-event absolute errors; negative mean = telegram is closer to the truth)."""
    from scipy.stats import wilcoxon

    pairs = [("ridge_depth_g", "ridge_depth"), ("xgb_direct_g", "xgb_direct"), ("xgb_direct_tg", "xgb_direct_t"),
             ("xgb_aft_g", "xgb_aft"), ("xgb_aft_tg", "xgb_aft_t")]
    t = table.drop_duplicates(["issue_day"]) if len(table) else table
    rows = []
    for new, old in pairs:
        for what, suffix, act in (("depth", "_depth", "actual_depth"), ("duration", "_duration_days", "actual_duration_days"),
                                  ("deficit_days", "_ndays", "actual_ndays")):
            if f"{new}{suffix}" not in t or f"{old}{suffix}" not in t:
                continue
            diff = ((t[f"{new}{suffix}"] - t[act]).abs() - (t[f"{old}{suffix}"] - t[act]).abs()).dropna()
            if len(diff) < 8 or (diff == 0).all():
                continue
            rows.append({"method": new, "reference": old, "metric": what, "n_events": len(diff),
                         "mean_error_minus_reference": float(diff.mean()), "share_better": float((diff < 0).mean()),
                         "p_value": float(wilcoxon(diff).pvalue)})
    return pd.DataFrame(rows)


def split_of(issue_day: str, cfg: Config) -> str:
    """Events issued up to the end of the validation period are validation events (model choices were made on them only)."""
    return "validation" if pd.Timestamp(issue_day, tz="UTC") <= cfg.validation_end_ts else "test"


def _mae(t: pd.DataFrame, col: str, actual: str) -> float:
    if col not in t or actual not in t:
        return np.nan
    e = (t[col] - t[actual]).abs().dropna()
    return float(e.mean()) if len(e) else np.nan


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """MAE per split (validation / test / all), trigger (attack / alarm / all) and method."""
    summ = []
    for split in ("validation", "test", "all"):
        ts = table if split == "all" else table[table["split"] == split]
        for trig in ("attack", "alarm", "all"):
            t = ts if trig == "all" else ts[ts["trigger"] == trig]
            if t.empty:
                continue
            for m in ALL_METHODS:
                traj = t[f"{m}_traj_MAE"].mean() if f"{m}_traj_MAE" in t else np.nan
                summ.append({"split": split, "trigger": trig, "method": m, "n_events": len(t),
                             "depth_MAE": _mae(t, f"{m}_depth", "actual_depth"),
                             "duration_MAE_days": _mae(t, f"{m}_duration_days", "actual_duration_days"),
                             "deficit_days_MAE": _mae(t, f"{m}_ndays", "actual_ndays"),
                             "trajectory_MAE": float(traj) if pd.notna(traj) else np.nan})
    return pd.DataFrame(summ)


def run_post_event(cfg: Config) -> dict[str, pd.DataFrame]:
    if cfg.post_event.strike_list:
        raise ValueError("post_event.strike_list is read only by `deficit events-dev` (train+validation); `events` also scores the test period")
    pe = cfg.post_event
    H, thr = pe.horizon_days, cfg.target.episode_threshold
    master = read_parquet(cfg.processed_dir / "master.parquet")
    d = daily_table(master, load_dam_daily(cfg, master["deficit"].resample("D").mean().index))
    S = stacked(d, H)
    feats = issue_features(d)
    train_end_i = int((d.index <= cfg.train_end_ts.normalize()).sum() - 1)
    xcfg = cfg.models.xgboost.model_dump()
    params = dict(xcfg)
    params["n_estimators"] = min(params["n_estimators"], 300)
    seed = cfg.tuning.seed
    tg = None
    tpath = cfg._abs(cfg.paths.data_dir) / "processed" / "telegram_posts.parquet"      # real posts even if --synthetic is passed
    if cfg.telegram.use_in_post_event and tpath.exists() and not cfg.synthetic:
        tg = telegram_features(pd.read_parquet(tpath), d.index, cfg.telegram.feature_channels)
        log.info("telegram issue features: %d columns, channels %s, first covered day %s", len(TG_COLS),
                 cfg.telegram.feature_channels, tg.dropna(how="all").index.min().date())
    frame = direct_frame(d, H, thr, pe, tg)
    log.info("post-event direct tension features: %s", frame.attrs["feature_sets"]["tension"])

    static = _fit(S[S["target_i"] <= train_end_i], params, seed)
    events = _events(cfg, d, first_i=max(train_end_i + 1, 60))
    rows, paths = [], {}
    for trig, i0 in events:
        y_act = d["y"].to_numpy()[i0 + 1: i0 + 1 + H]
        y_act = np.concatenate([y_act, np.full(H - len(y_act), np.nan)])
        a_depth, a_dur, a_cens = depth_duration(y_act, thr)
        a_nd = float(np.nansum(y_act > thr) / np.sum(~np.isnan(y_act)) * H) if np.any(~np.isnan(y_act)) else np.nan
        f0 = feats.iloc[i0]
        if f0[["y0", "m7", "m28"]].isna().any():
            continue
        X0 = pd.DataFrame([f0] * H)
        X0["k"] = np.arange(1, H + 1)
        tdate = d.index[i0] + pd.to_timedelta(X0["k"].to_numpy(), unit="D")
        X0["tdoy_sin"], X0["tdoy_cos"], X0["tdow"] = np.sin(2 * np.pi * tdate.dayofyear / 365.25), np.cos(2 * np.pi * tdate.dayofyear / 365.25), tdate.dayofweek
        known = S[S["target_i"] <= i0]
        preds = {"persistence": np.full(H, f0["y0"]), "baseline28": np.full(H, f0["m28"])}
        for name, model in (("xgb_static", static), ("xgb_expanding", _fit(known, params, seed)),
                            ("xgb_rolling", _fit(known[known["issue_i"] >= i0 - pe.window_days], params, seed))):
            preds[name] = model.predict(X0[FEATURES]) if model is not None else np.full(H, np.nan)
        preds["blend"] = 0.5 * (preds["persistence"] + preds["xgb_rolling"])
        probs = _cls_probs(known[known["issue_i"] >= i0 - pe.window_days], X0, thr, params, seed)
        direct = direct_forecast(d, i0, H, thr, pe, xcfg, seed, frame, "base")
        direct_t = direct_forecast(d, i0, H, thr, pe, xcfg, seed, frame, "tension")
        direct_g = direct_forecast(d, i0, H, thr, pe, xcfg, seed, frame, "g") if tg is not None else None
        direct_tg = direct_forecast(d, i0, H, thr, pe, xcfg, seed, frame, "tension_g") if tg is not None else None
        day = d.index[i0].date().isoformat()
        row = {"trigger": trig, "issue_day": day, "split": split_of(day, cfg), "actual_depth": a_depth,
               "actual_duration_days": a_dur, "actual_censored": a_cens, "actual_ndays": a_nd, "last_day_level": float(f0["y0"])}
        for m in PATH_METHODS:
            p_depth, p_dur, _ = depth_duration(preds[m], thr)
            ok = ~np.isnan(y_act) & ~np.isnan(preds[m])
            row[f"{m}_depth"], row[f"{m}_duration_days"] = p_depth, p_dur
            row[f"{m}_ndays"] = float(np.sum(preds[m] > thr)) if not np.isnan(preds[m]).any() else np.nan
            row[f"{m}_traj_MAE"] = float(np.mean(np.abs(preds[m][ok] - y_act[ok]))) if ok.any() else np.nan
        row["ridge_depth_depth"] = direct["ridge_depth"]
        for m, r in (("xgb_direct", direct), ("xgb_direct_t", direct_t)):
            row[f"{m}_depth"], row[f"{m}_duration_days"], row[f"{m}_ndays"] = r["depth"], r["duration"], r["ndays"]
        row["xgb_aft_duration_days"], row["xgb_aft_t_duration_days"] = direct["aft_duration"], direct_t["aft_duration"]
        if tg is not None:
            row["ridge_depth_g_depth"] = direct_g["ridge_depth"]
            for m, r in (("xgb_direct_g", direct_g), ("xgb_direct_tg", direct_tg)):
                row[f"{m}_depth"], row[f"{m}_duration_days"], row[f"{m}_ndays"] = r["depth"], r["duration"], r["ndays"]
            row["xgb_aft_g_duration_days"], row["xgb_aft_tg_duration_days"] = direct_g["aft_duration"], direct_tg["aft_duration"]
        row["xgb_cls_duration_days"] = duration_from_flags(probs < 0.5) if probs is not None else np.nan
        row["xgb_cls_ndays"] = float(np.sum(probs)) if probs is not None else np.nan
        rows.append(row)
        paths[(trig, day)] = (y_act, preds)
    table = pd.DataFrame(rows)
    table.to_csv(cfg.reports_dir / "post_event.csv", index=False)
    traj = []
    for (trig, day), (y_act, preds) in paths.items():
        for k in range(H):
            traj.append({"trigger": trig, "issue_day": day, "k": k + 1, "actual": y_act[k], **{m: preds[m][k] for m in PATH_METHODS}})
    pd.DataFrame(traj).to_csv(cfg.reports_dir / "post_event_paths.csv", index=False)

    summary = summarize(table) if len(table) else pd.DataFrame()
    summary.to_csv(cfg.reports_dir / "post_event_summary.csv", index=False)
    tests = []
    for split in ("validation", "test", "all"):
        t = table if split == "all" else table[table["split"] == split]
        res = paired_tests(t)
        if len(res):
            tests.append(res.assign(split=split))
    tests = pd.concat(tests, ignore_index=True) if tests else pd.DataFrame()
    tests.to_csv(cfg.reports_dir / "post_event_tests.csv", index=False)
    if tg is not None and len(table):
        tgt = pd.concat([telegram_vs_base_tests(table if s == "all" else table[table["split"] == s]).assign(split=s)
                         for s in ("validation", "test", "all")], ignore_index=True)
        tgt.to_csv(cfg.reports_dir / "post_event_telegram_tests.csv", index=False)
        if len(tgt):
            log.info("telegram vs the same method without telegram features:\n%s", tgt.round(4).to_string(index=False))
    if len(tests):
        log.info("post-event paired tests (Wilcoxon on per-event absolute errors; events overlap, p-values optimistic):\n%s",
                 tests.round(4).to_string(index=False))
    _plot(cfg, paths, table, thr, H)
    if len(summary):
        log.info("post-event forecast (H=%d days, threshold %s %s):\n%s", H, thr, cfg.target.unit,
                 summary[summary["trigger"] == "all"].round(3).to_string(index=False))
    return {"events": table, "summary": summary, "tests": tests}


def _plot(cfg, paths, table, thr, H):
    if not paths:
        return
    keys = list(paths)[:6]
    cols = 3
    rows = int(np.ceil(len(keys) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(13, 3.3 * rows), squeeze=False)
    for ax, key in zip(axs.ravel(), keys):
        y_act, preds = paths[key]
        x = np.arange(1, H + 1)
        ax.plot(x, y_act, color="black", lw=1.8, label="actual")
        for m, c in (("persistence", "#8c8c8c"), ("xgb_static", "#4C78A8"), ("xgb_rolling", "#D62728")):
            ax.plot(x, preds[m], color=c, lw=1.1, label=m)
        ax.axhline(thr, color="#bbb", ls=":")
        ax.set_title(f"{key[0]} {key[1]}", fontsize=10)
        ax.set_xlabel("days after the issue day")
    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")
    axs[0][0].set_ylabel(cfg.target.unit)
    axs[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.reports_dir / "figures" / "post_event.png", dpi=130)
    plt.close(fig)
