"""Step 6 - evaluation and output: metric tables, crisis-week comparison, Wilcoxon test,
depth/duration per attack, figures."""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from ..config import Config  # noqa: E402
from ..ingestion.attacks import load_attacks  # noqa: E402
from ..io import read_json, read_parquet  # noqa: E402
from .events import deficit_events, episodes, match_episodes  # noqa: E402
from .metrics import all_metrics  # noqa: E402

log = logging.getLogger(__name__)


def load_forecasts(cfg: Config) -> dict[str, pd.DataFrame]:
    d = cfg.artifacts_dir / "forecasts"
    return {p.stem: read_parquet(p) for p in sorted(d.glob("*.parquet"))}


def crisis_mask(index: pd.DatetimeIndex, attacks: pd.DataFrame, days: int) -> np.ndarray:
    m = np.zeros(len(index), bool)
    for t in attacks.loc[attacks["major"], "datetime"]:
        m |= (index >= t) & (index < t + pd.Timedelta(days=days))
    return m


def metrics_table(fcs: dict[str, pd.DataFrame], attacks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    ev = cfg.evaluation
    rows = []
    for name, fc in fcs.items():
        crisis = crisis_mask(fc.index, attacks, ev.crisis_window_days)
        for split in ("validation", "test"):
            s = (fc["split"] == split).to_numpy()
            for subset, mask in (("all", s), ("crisis_14d", s & crisis), ("non_crisis", s & ~crisis)):
                sub = fc[mask]
                rows.append({"model": name, "split": split, "subset": subset,
                             **all_metrics(sub["y"], sub["yhat"], ev.mape_min_abs)})
    return pd.DataFrame(rows)


def wilcoxon_table(fcs: dict[str, pd.DataFrame], attacks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test on *daily* MAE (hourly errors are autocorrelated)."""
    ref = cfg.evaluation.wilcoxon_reference
    if ref not in fcs:
        return pd.DataFrame()
    def daily_abs(fc):
        return (fc["y"] - fc["yhat"]).abs().resample("D").mean()

    rows = []
    for split in ("validation", "test"):
        for subset in ("all", "crisis_14d"):
            base = fcs[ref]
            mask = (base["split"] == split).to_numpy()
            if subset == "crisis_14d":
                mask = mask & crisis_mask(base.index, attacks, cfg.evaluation.crisis_window_days)
            r = daily_abs(base[mask])
            for name, fc in fcs.items():
                if name == ref:
                    continue
                o = daily_abs(fc[fc.index.isin(base.index[mask])])
                pair = pd.concat([r, o], axis=1, keys=["ref", "other"]).dropna()
                if len(pair) < 10:
                    continue
                stat, p = wilcoxon(pair["ref"], pair["other"])
                rows.append({"split": split, "subset": subset, "reference": ref, "model": name,
                             "n_days": len(pair), "ref_daily_MAE": pair["ref"].mean(),
                             "model_daily_MAE": pair["other"].mean(), "statistic": stat, "p_value": p,
                             "ref_better": bool(pair["ref"].mean() < pair["other"].mean())})
    return pd.DataFrame(rows)


def events_table(fcs: dict[str, pd.DataFrame], attacks: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Actual vs predicted depth and recovery duration for every major attack after the train period."""
    t = cfg.target
    after = attacks[attacks["datetime"] > cfg.train_end_ts]
    any_fc = next(iter(fcs.values()))
    actual = deficit_events(any_fc["y"], after, t.recovery_threshold, t.recovery_max_days,
                            cfg.evaluation.crisis_window_days)
    if actual.empty:
        return actual
    out = actual.add_prefix("actual_").rename(columns={"actual_attack_datetime": "attack_datetime",
                                                      "actual_description": "description",
                                                      "actual_capacity_lost_MW": "capacity_lost_MW"})
    for name, fc in fcs.items():
        pred = deficit_events(fc["yhat"], after, t.recovery_threshold, t.recovery_max_days,
                              cfg.evaluation.crisis_window_days)
        if pred.empty:
            continue
        pred = pred.set_index("attack_datetime")[["depth", "recovery_days"]].add_prefix(f"{name}_")
        out = out.merge(pred, left_on="attack_datetime", right_index=True, how="left")
    return out


def events_summary(ev: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for m in models:
        if f"{m}_depth" not in ev:
            continue
        rows.append({"model": m,
                     "depth_MAE": (ev[f"{m}_depth"] - ev["actual_depth"]).abs().mean(),
                     "recovery_MAE_days": (ev[f"{m}_recovery_days"] - ev["actual_recovery_days"]).abs().mean(),
                     "n_events": int(ev[f"{m}_depth"].notna().sum())})
    return pd.DataFrame(rows)


def episodes_tables(fcs: dict[str, pd.DataFrame], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual deficit episodes (found in the observed target) vs the depth / duration each model implies."""
    t = cfg.target
    first = next(iter(fcs.values()))
    actual = episodes(first["y"], t.episode_threshold, t.episode_gap_days)
    if actual.empty:
        return actual, pd.DataFrame()
    actual["split"] = np.where(actual["start"] <= cfg.validation_end_ts, "validation", "test")
    table = actual.rename(columns={"depth": "actual_depth", "duration_days": "actual_duration_days"})
    table = table.drop(columns="cumulative")
    for name, fc in fcs.items():
        m = match_episodes(actual, fc["yhat"], t.episode_threshold).drop(columns="start")
        m.columns = [f"{name}_depth", f"{name}_duration_days"]
        table = pd.concat([table, m], axis=1)
    rows = []
    for split in ("validation", "test"):
        sub = table[table["split"] == split]
        for name in fcs:
            if sub.empty:
                continue
            days = fcs[name][["y", "yhat"]].resample("D").mean().dropna()
            days = days[(days.index <= cfg.validation_end_ts) == (split == "validation")]
            act, prd = days["y"] > t.episode_threshold, days["yhat"] > t.episode_threshold
            tp = int((act & prd).sum())
            rows.append({"split": split, "model": name, "n_episodes": len(sub),
                         "deficit_days_actual": int(act.sum()), "deficit_days_predicted": int(prd.sum()),
                         "day_precision": tp / prd.sum() if prd.sum() else np.nan,
                         "day_recall": tp / act.sum() if act.sum() else np.nan,
                         "depth_MAE": (sub[f"{name}_depth"] - sub["actual_depth"]).abs().mean(),
                         "duration_MAE_days": (sub[f"{name}_duration_days"] - sub["actual_duration_days"]).abs().mean(),
                         "mean_actual_duration_days": sub["actual_duration_days"].mean()})
    return table, pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def plot_changepoints(master: pd.DataFrame, attacks: pd.DataFrame, cfg: Config) -> None:
    cp = read_json(cfg.artifacts_dir / "changepoints.json")
    sig = read_parquet(cfg.artifacts_dir / "bocpd_signal.parquet")
    daily = master["deficit"].resample("D").mean()
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[3, 1])
    ax[0].plot(daily.index, daily, lw=0.8, color="#333", label="daily-mean deficit")
    for i, t in enumerate(attacks.loc[attacks["major"], "datetime"]):
        ax[0].axvline(t, color="#c0392b", alpha=0.35, lw=1, label="attack" if i == 0 else None)
    for i, b in enumerate(cp["pelt"]["breakpoints"]):
        ax[0].axvline(pd.Timestamp(b, tz="UTC"), color="#2471a3", ls="--", lw=1,
                      label="PELT break" if i == 0 else None)
    ax[0].set_ylabel(cfg.target.unit); ax[0].legend(loc="upper left"); ax[0].set_title("Deficit, attacks and detected breaks")
    ax[1].plot(sig.index, sig["cp_prob"], lw=0.8, color="#7d3c98")
    ax[1].axhline(cp["bocpd"]["threshold"], color="#999", ls=":")
    ax[1].set_ylabel("BOCPD P(break)"); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(cfg.reports_dir / "figures" / "changepoints.png", dpi=130); plt.close(fig)


def plot_forecasts(fcs: dict[str, pd.DataFrame], attacks: pd.DataFrame, cfg: Config) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    first = next(iter(fcs.values()))
    actual = first["y"].resample("D").mean()
    ax.plot(actual.index, actual, color="black", lw=1.2, label="actual (daily mean)")
    for name, fc in fcs.items():
        d = fc["yhat"].resample("D").mean()
        ax.plot(d.index, d, lw=0.9, label=name)
    ax.axvline(cfg.validation_end_ts, color="#999", ls=":")
    ax.set_title("Daily-mean deficit forecasts (validation | test)"); ax.set_ylabel(cfg.target.unit); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(cfg.reports_dir / "figures" / "forecasts.png", dpi=130); plt.close(fig)

    # zoom on the first crisis windows in the test period
    test_attacks = attacks[(attacks["major"]) & (attacks["datetime"] > cfg.validation_end_ts)]["datetime"][:3]
    for i, t in enumerate(test_attacks):
        lo, hi = t - pd.Timedelta(days=3), t + pd.Timedelta(days=cfg.evaluation.crisis_window_days)
        fig, ax = plt.subplots(figsize=(12, 4))
        sl = first.loc[lo:hi]
        ax.plot(sl.index, sl["y"], color="black", lw=1.2, label="actual")
        for name, fc in fcs.items():
            s = fc.loc[lo:hi]
            ax.plot(s.index, s["yhat"], lw=0.9, label=name)
        ax.axvline(t, color="#c0392b", lw=1)
        ax.set_title(f"Crisis window after {t:%Y-%m-%d}"); ax.set_ylabel(cfg.target.unit); ax.legend(ncol=3, fontsize=8)
        fig.tight_layout(); fig.savefig(cfg.reports_dir / "figures" / f"crisis_{i + 1}.png", dpi=130); plt.close(fig)


def plot_metrics(table: pd.DataFrame, cfg: Config) -> None:
    t = table[(table["split"] == "test") & table["subset"].isin(["all", "crisis_14d"])]
    piv = t.pivot(index="model", columns="subset", values="MAE")
    ax = piv.plot.bar(figsize=(9, 4), rot=20)
    ax.set_ylabel(f"MAE, {cfg.target.unit}"); ax.set_title("Test MAE: all hours vs first 14 days after attacks")
    plt.tight_layout(); plt.savefig(cfg.reports_dir / "figures" / "mae_comparison.png", dpi=130); plt.close()


def evaluate(cfg: Config) -> dict[str, pd.DataFrame]:
    attacks = load_attacks(cfg)
    master = read_parquet(cfg.processed_dir / "master.parquet")
    fcs = load_forecasts(cfg)
    if not fcs:
        raise RuntimeError("no forecasts found - run `deficit train` first")
    rd = cfg.reports_dir
    table = metrics_table(fcs, attacks, cfg); table.to_csv(rd / "metrics.csv", index=False)
    wt = wilcoxon_table(fcs, attacks, cfg); wt.to_csv(rd / "wilcoxon.csv", index=False)
    ev = events_table(fcs, attacks, cfg); ev.to_csv(rd / "events_depth_duration.csv", index=False)
    es = events_summary(ev, list(fcs)); es.to_csv(rd / "events_summary.csv", index=False)
    ep, eps = episodes_tables(fcs, cfg)
    ep.to_csv(rd / "episodes_depth_duration.csv", index=False)
    eps.to_csv(rd / "episodes_summary.csv", index=False)
    if not eps.empty:
        log.info("episodes (depth in %s, duration in days):\n%s", cfg.target.unit, eps.round(3).to_string(index=False))
    plot_changepoints(master, attacks, cfg)
    plot_forecasts(fcs, attacks, cfg)
    plot_metrics(table, cfg)
    test = table[(table["split"] == "test")].pivot(index="model", columns="subset", values="MAE")
    log.info("test MAE (%s):\n%s", cfg.target.unit, test.round(3).to_string())
    return {"metrics": table, "wilcoxon": wt, "events": ev, "events_summary": es,
            "episodes": ep, "episodes_summary": eps}
