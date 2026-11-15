"""`deficit events-dev`: post-event depth / duration on the EXTENDED strike list, train + validation only (test-blind).

The test period is never touched: the daily table is cut at `split.validation_end` before anything is built, and events are
kept only if their whole 30-day window lies inside the cut (so labels never come from the test period either).

Events
  A  the major strikes of attacks.csv (the current post-event setup, without the BOCPD alarms)
  B  A plus the derived strikes of `data/external/attacks_extended.csv` (post_event.strike_list), kept greedily in time order
     when at least `min_event_gap_days` after every earlier kept event (major strikes are always kept)
  event time = strike time (derived rows: time of the first attack-report post); the issue day is that UTC day, as in post_event.py.

Strike features of the extended variant (all known at the end of the issue day; `n_channels` / `n_attack_posts` are whole-day
counts of the derived rows and are never read):
  since_attack, n_attack_30     over all rows of the list (major + derived)
  through7 / through30          weapons_through of the MAJOR rows only; NaN when the window holds strikes but none is major
  derived7 / derived30          indicator: number of derived strikes in the window

Model variants (same direct models as post_event.py: ridge_depth, xgb_direct, xgb_aft)
  orig  all issue days as training rows, strike features of attacks.csv (the current model)
  ext   all issue days as training rows, strike features of the extended list
  ev    trained only on the issue days that are strike days of the extended list (`dev_min_event_rows` known ones needed)
Baselines: persistence, baseline28, event_median (median depth / duration of the strike days known at the issue day;
`event_median_A` over the major strikes only, `event_median_B` over all strikes of the list).

Each event is forecast by models fitted on examples known at its issue day (i + H <= i0), so the pooled errors ARE a
rolling-origin (expanding window) evaluation; the report also gives them per block of consecutive events.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.attacks import load_attacks
from ..io import read_parquet
from .post_event import (DirectModel, daily_table, depth_duration, direct_frame, known_rows, load_dam_daily)

log = logging.getLogger(__name__)

DIRECT = ["ridge_depth", "xgb_direct", "xgb_aft"]
VARIANTS = ["orig", "ext", "ev"]
EXT_STRIKE_COLS = ["derived7", "derived30"]


# ---------------------------------------------------------------- strike list and events
def load_strike_list(path) -> pd.DataFrame:
    """Rows of the extended list: datetime (hour floor, UTC), day, major, derived, weapons_through (NaN = unknown).
    Whole-day counts (n_channels, n_attack_posts) are dropped on purpose."""
    df = pd.read_csv(path)
    time = df["time_utc"].fillna("00:00") if "time_utc" in df else "00:00"
    out = pd.DataFrame({"datetime": pd.to_datetime(df["date"] + " " + time, utc=True).dt.floor("h")})
    out["day"] = out["datetime"].dt.normalize()
    out["major"] = df["major"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
    out["derived"] = (df["derived"].astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()
                      if "derived" in df else np.zeros(len(df), bool))
    thr = pd.to_numeric(df["weapons_through"], errors="coerce") if "weapons_through" in df else pd.Series(np.nan, index=df.index)
    out["weapons_through"] = np.where(out["derived"], np.nan, thr.to_numpy())      # derived rows: strength unknown
    return out.sort_values("datetime").reset_index(drop=True)


def select_events(strikes: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """Event days, chronologically: every major strike, and a derived strike only if it is at least `gap_days` after every EARLIER kept event
    (a later strike never changes an earlier decision).
    Returns day, major, derived; sorted by day. Depends only on strike times and the major flag."""
    s = strikes.sort_values("datetime").drop_duplicates("day", keep="first")
    kept_days = []
    for r in s.itertuples():
        if r.major or all((r.day - k).days >= gap_days for k in kept_days):
            kept_days.append(r.day)
    ev = s[s["day"].isin(kept_days)][["day", "major", "derived"]]
    return ev.sort_values("day").reset_index(drop=True)


def ext_daily(d: pd.DataFrame, strikes: pd.DataFrame) -> pd.DataFrame:
    """Daily table with strike columns rebuilt from the extended list. through = known weapons_through of major rows (0 if none),
    through_known = indicator that the day holds a major row, derived = number of derived rows."""
    out = d.copy()
    g = strikes.groupby("day")
    out["attack"] = (g.size().reindex(out.index).fillna(0) > 0).astype(float)
    out["through"] = strikes[strikes["major"]].groupby("day")["weapons_through"].sum().reindex(out.index).fillna(0.0)
    out["major_day"] = (strikes[strikes["major"]].groupby("day").size().reindex(out.index).fillna(0) > 0).astype(float)
    out["derived"] = strikes[strikes["derived"]].groupby("day").size().reindex(out.index).fillna(0.0)
    return out


def ext_frame(d_ext: pd.DataFrame, H: int, thr: float, pe) -> pd.DataFrame:
    """direct_frame of the extended daily table + derived7/30; through7/30 become NaN when the window has strikes but no major one."""
    fr = direct_frame(d_ext, H, thr, pe)
    a7, a30 = d_ext["attack"].rolling(7, min_periods=1).sum().to_numpy(), d_ext["attack"].rolling(30, min_periods=1).sum().to_numpy()
    m7, m30 = d_ext["major_day"].rolling(7, min_periods=1).sum().to_numpy(), d_ext["major_day"].rolling(30, min_periods=1).sum().to_numpy()
    fr["through7"] = np.where((a7 > 0) & (m7 == 0), np.nan, fr["through7"])
    fr["through30"] = np.where((a30 > 0) & (m30 == 0), np.nan, fr["through30"])
    fr["derived7"] = d_ext["derived"].rolling(7, min_periods=1).sum().to_numpy()
    fr["derived30"] = d_ext["derived"].rolling(30, min_periods=1).sum().to_numpy()
    for k in ("base", "tension"):
        fr.attrs["feature_sets"][k] = fr.attrs["feature_sets"][k] + EXT_STRIKE_COLS
    return fr


# ---------------------------------------------------------------- forecasts
def _nan_pred() -> dict:
    return {"ridge_depth": np.nan, "depth": np.nan, "duration": np.nan, "aft_duration": np.nan}


def forecast_variants(frame_o: pd.DataFrame, frame_e: pd.DataFrame, event_days: np.ndarray, i0: int, H: int, pe, xcfg: dict,
                      seed: int, major_days: np.ndarray) -> dict[str, dict]:
    """Predictions of the three variants + the event-median baselines at issue day position i0 (training examples with i + H <= i0 only)."""
    out = {}
    for name, fr in (("orig", frame_o), ("ext", frame_e)):
        known = known_rows(fr, i0, pe.direct_window_days)
        x0 = fr.loc[[i0]]
        if len(known) < pe.direct_min_rows or x0[["y0", "m28"]].isna().any(axis=None):
            out[name] = _nan_pred()
        else:
            out[name] = DirectModel(known, fr.attrs["feature_sets"]["base"], H, pe, xcfg, seed,
                                    fr.attrs["lin_cols"]["base"]).predict(x0).iloc[0].to_dict()
    known = known_rows(frame_e, i0, pe.direct_window_days)
    kev = known[known.index.isin(event_days)]
    x0 = frame_e.loc[[i0]]
    if len(kev) < pe.dev_min_event_rows or x0[["y0", "m28"]].isna().any(axis=None):
        out["ev"] = _nan_pred()
    else:
        out["ev"] = DirectModel(kev, frame_e.attrs["feature_sets"]["base"], H, pe, xcfg, seed,
                                frame_e.attrs["lin_cols"]["base"]).predict(x0).iloc[0].to_dict()
    for tag, days in (("A", major_days), ("B", event_days)):
        k = known[known.index.isin(days)]
        out[f"event_median_{tag}"] = ({"depth": float(k["depth"].median()), "duration": float(np.round(k["duration"].median()))}
                                      if len(k) >= 3 else {"depth": np.nan, "duration": np.nan})
    return out


def evaluate_events(cfg: Config) -> dict:
    """Run every event of the union of all gaps once; returns the pieces `report` needs."""
    pe, H, thr = cfg.post_event, cfg.post_event.horizon_days, cfg.target.episode_threshold
    master = read_parquet(cfg.processed_dir / "master.parquet")
    cut = cfg.validation_end_ts
    master = master[master.index <= cut]                                   # the test period is never loaded into the daily table
    d = daily_table(master, load_dam_daily(cfg, master["deficit"].resample("D").mean().index))
    assert d.index.max() <= cut.normalize(), "test period leaked into the dev daily table"
    path = cfg._abs(Path(pe.strike_list))
    strikes = load_strike_list(path)
    strikes = strikes[strikes["datetime"] <= cut]
    d_ext = ext_daily(d, strikes)
    xcfg, seed = cfg.models.xgboost.model_dump(), cfg.tuning.seed
    frame_o, frame_e = direct_frame(d, H, thr, pe), ext_frame(d_ext, H, thr, pe)

    train_end_i = int((d.index <= cfg.train_end_ts.normalize()).sum() - 1)
    gaps = [pe.min_event_gap_days] + [g for g in pe.dev_gap_sensitivity if g != pe.min_event_gap_days]
    selections = {g: select_events(strikes, g) for g in gaps}
    major = strikes[strikes["major"]].drop_duplicates("day")
    major_days_idx = np.array(sorted(d.index.get_loc(x) for x in major["day"] if x in d.index))
    ev_all = strikes.drop_duplicates("day")                               # training days of the event-trained variant: every strike day of the list
    union = sorted(set().union(*[set(s["day"]) for s in selections.values()]))
    y = d["y"].to_numpy()
    rows = []
    for day in union:
        if day not in d.index:
            continue
        i0 = int(d.index.get_loc(day))
        if i0 <= train_end_i or i0 < 60 or i0 + H >= len(d):
            continue                                                       # window must lie inside train+validation
        w = y[i0 + 1: i0 + 1 + H]
        if np.sum(~np.isnan(w)) < pe.direct_min_avail_share * H or frame_e.loc[i0, ["y0", "m7", "m28"]].isna().any():
            continue
        a_depth, a_dur, a_cens = depth_duration(w, thr)
        ev_idx = np.array(sorted(d.index.get_loc(x) for x in ev_all["day"] if x in d.index))
        f = forecast_variants(frame_o, frame_e, ev_idx[ev_idx <= i0], i0, H, pe, xcfg, seed, major_days_idx[major_days_idx <= i0])
        row = {"issue_day": day.date().isoformat(), "is_major": bool(day in set(major["day"])),
               "actual_depth": a_depth, "actual_duration": a_dur, "actual_censored": bool(a_cens), "y0": float(y[i0])}
        for g in gaps:
            row[f"in_gap{g}"] = bool(day in set(selections[g]["day"]))
        for name, hz in (("persistence", np.full(H, y[i0])), ("baseline28", np.full(H, frame_e.loc[i0, "m28"]))):
            dep, dur, _ = depth_duration(hz, thr)
            row[f"{name}|depth"], row[f"{name}|duration"] = dep, dur
        for v in VARIANTS:
            p = f[v]
            row[f"ridge_depth|{v}|depth"] = p["ridge_depth"]
            row[f"xgb_direct|{v}|depth"], row[f"xgb_direct|{v}|duration"] = p["depth"], p["duration"]
            row[f"xgb_aft|{v}|duration"] = p["aft_duration"]
        for t in ("A", "B"):
            row[f"event_median_{t}|depth"], row[f"event_median_{t}|duration"] = f[f"event_median_{t}"]["depth"], f[f"event_median_{t}"]["duration"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("issue_day").reset_index(drop=True)
    return {"table": table, "gaps": gaps, "strikes": strikes, "d": d}


# ---------------------------------------------------------------- statistics
def block_bootstrap_ci(diff: np.ndarray, block: int, n_boot: int, seed: int = 42) -> tuple[float, float]:
    """95% CI of the mean of a time-ordered per-event difference (moving blocks of `block` consecutive events)."""
    n = len(diff)
    if n == 0:
        return (np.nan, np.nan)
    b = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(t: pd.DataFrame, method: str, metric: str, ref: str, block: int, n_boot: int) -> dict | None:
    """Per-event absolute error of `method` minus that of `ref` (negative = method better) on the rows of `t` where both exist."""
    from scipy.stats import wilcoxon

    col, rcol, act = f"{method}|{metric}", f"{ref}|{metric}", f"actual_{metric}"
    if col not in t or rcol not in t:
        return None
    ok = t[[col, rcol, act]].notna().all(axis=1)
    if ok.sum() < 3:
        return {"method": method, "metric": metric, "ref": ref, "n": int(ok.sum())}
    e_m, e_r = (t.loc[ok, col] - t.loc[ok, act]).abs().to_numpy(), (t.loc[ok, rcol] - t.loc[ok, act]).abs().to_numpy()
    diff = e_m - e_r
    lo, hi = block_bootstrap_ci(diff, block, n_boot)
    p = float(wilcoxon(diff).pvalue) if len(diff) >= 8 and not (diff == 0).all() else np.nan
    return {"method": method, "metric": metric, "ref": ref, "n": int(ok.sum()), "MAE": float(e_m.mean()), "MAE_ref": float(e_r.mean()),
            "mean_diff": float(diff.mean()), "ci_lo": lo, "ci_hi": hi, "share_better": float((diff < 0).mean()), "p_wilcoxon": p}


def method_list() -> list[tuple[str, str]]:
    ms = [("persistence", "depth"), ("persistence", "duration"), ("baseline28", "depth"), ("baseline28", "duration")]
    for t in ("A", "B"):
        ms += [(f"event_median_{t}", "depth"), (f"event_median_{t}", "duration")]
    for v in VARIANTS:
        ms += [(f"ridge_depth|{v}", "depth"), (f"xgb_direct|{v}", "depth"), (f"xgb_direct|{v}", "duration"), (f"xgb_aft|{v}", "duration")]
    return ms


def table_vs(t: pd.DataFrame, ref: str, block: int, n_boot: int, label: str) -> pd.DataFrame:
    rows = []
    for m, metric in method_list():
        if m == ref:
            continue
        r = compare(t, m, metric, ref, block, n_boot)
        if r:
            rows.append({"set": label, **r})
    return pd.DataFrame(rows)


def fold_table(t: pd.DataFrame, n_folds: int, methods: list[tuple[str, str]], ref: str = "persistence") -> pd.DataFrame:
    """MAE of each method per block of consecutive events (rolling-origin report), next to persistence."""
    rows = []
    if t.empty:
        return pd.DataFrame()
    fold = np.minimum((np.arange(len(t)) * n_folds) // len(t), n_folds - 1)
    for k in range(n_folds):
        tk = t[fold == k]
        for m, metric in methods:
            col, act = f"{m}|{metric}", f"actual_{metric}"
            if col not in tk:
                continue
            ok = tk[[col, f"{ref}|{metric}", act]].notna().all(axis=1)
            if ok.sum() == 0:
                continue
            e = (tk.loc[ok, col] - tk.loc[ok, act]).abs()
            er = (tk.loc[ok, f"{ref}|{metric}"] - tk.loc[ok, act]).abs()
            rows.append({"fold": k + 1, "from": tk["issue_day"].iloc[0], "to": tk["issue_day"].iloc[-1], "n": int(ok.sum()),
                         "method": m, "metric": metric, "MAE": float(e.mean()), "MAE_persistence": float(er.mean()),
                         "better_than_persistence": bool(e.mean() < er.mean())})
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)\n"
    cols = list(df.columns)
    f = lambda x: "" if (isinstance(x, float) and np.isnan(x)) else (f"{x:.4g}" if isinstance(x, float) else str(x))
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]) + "\n"


def run_post_event_dev(cfg: Config) -> dict:
    pe = cfg.post_event
    if not pe.strike_list:
        raise ValueError("events-dev needs post_event.strike_list (use --config config/net_import_extended.yaml)")
    res = evaluate_events(cfg)
    table, gaps = res["table"], res["gaps"]
    out = cfg.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "post_event_extended_dev_events.csv", index=False)
    parts, folds = [], []
    sets = {"A: major strikes (current setup)": table[table["is_major"]]}
    for g in gaps:
        sets[f"B(gap={g}): extended events"] = table[table[f"in_gap{g}"]]
    sup = table[table["ridge_depth|ev|depth"].notna()]
    sets[f"B(gap={gaps[0]}) common support: events where the event-trained variant has >= {pe.dev_min_event_rows} known strike days"] = sup[sup[f"in_gap{gaps[0]}"]]
    sets[f"B(gap={gaps[0]}) common support, major strikes only"] = sup[sup[f"in_gap{gaps[0]}"] & sup["is_major"]]
    for label, t in sets.items():
        parts.append(table_vs(t, "persistence", pe.dev_block, pe.dev_bootstrap, label))
        parts.append(table_vs(t, "baseline28", pe.dev_block, pe.dev_bootstrap, label).assign(ref="baseline28"))
        ft = fold_table(t, pe.dev_folds, [("persistence", "depth"), ("persistence", "duration"), ("xgb_direct|orig", "depth"), ("xgb_direct|ext", "depth"),
                                          ("ridge_depth|ext", "depth"), ("xgb_direct|ext", "duration"), ("xgb_aft|ext", "duration"),
                                          ("xgb_direct|orig", "duration"), ("event_median_B", "duration"), ("event_median_B", "depth")])
        folds.append(ft.assign(set=label))
    tests = pd.concat([p for p in parts if len(p)], ignore_index=True)
    tests.to_csv(out / "post_event_extended_dev_tests.csv", index=False)
    pd.concat([f for f in folds if len(f)], ignore_index=True).to_csv(out / "post_event_extended_dev_folds.csv", index=False)
    counts = {label: {"events": len(t), "major": int(t["is_major"].sum()), "derived": int((~t["is_major"]).sum()),
                      "censored": int(t["actual_censored"].sum())} for label, t in sets.items()}
    log.info("events per set: %s", counts)
    for label in sets:
        s = tests[(tests["set"] == label) & (tests["ref"] == "persistence")]
        log.info("%s vs persistence\n%s", label, s.drop(columns=["set", "ref"]).round(3).to_string(index=False))
    (out / "post_event_extended_dev_tables.md").write_text(
        "\n".join(f"## {label}\n\nevents: {counts[label]}\n\n### vs persistence\n\n" + _md(tests[(tests['set'] == label) & (tests['ref'] == 'persistence')].drop(columns=['set', 'ref']))
                  + "\n### vs baseline28\n\n" + _md(tests[(tests['set'] == label) & (tests['ref'] == 'baseline28')].drop(columns=['set', 'ref'])) for label in sets),
        encoding="utf-8")
    return {"table": table, "tests": tests, "counts": counts}
