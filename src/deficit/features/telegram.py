"""Features from public Telegram posts, known at an issue time (README §6).

Every value for the issue day d uses only posts with `datetime < d + 1 day`, i.e. published before the END of day d
(the forecast is issued at 00:00 UTC of the next day; a post stamped exactly at that instant is not used).
Windows only look backwards. The posts table is `data/processed/telegram_posts.parquet` (`deficit extract`).

Coverage: the national channel (ukrenergo) is continuous since 2022-02, the oblast channels start later, so the default
channel set is the national one (`telegram.feature_channels`); pooled channels would make "number of regions" and
"number of schedule posts" jump when a channel joins. Days before the first post are NaN (unknown), not zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = ["tg_gen_d0", "tg_sub_d0", "tg_regions_d0", "tg_cons_d0", "tg_emerg_d0", "tg_sched_d0", "tg_reports_d0", "tg_lifted_d0"]
RECENT = ["tg_sched_days7", "tg_emerg_days7", "tg_gen_days7", "tg_regions_max7", "tg_lifted_days7",
          "tg_sched_days30", "tg_emerg_days30", "tg_gen_days30", "tg_regions_max30"]
TG_COLS = DAY + RECENT
TG_LIN = ["tg_cons_d0", "tg_regions_d0", "tg_gen_d0", "tg_sub_d0", "tg_emerg_d0", "tg_sched_d0"]   # the few columns the ridge sees
REGIME = ["tg_regime_age", "tg_days_since_lift"]


def _has(series: pd.Series, *kinds: str) -> pd.Series:
    return series.map(lambda x: any(k in x for k in kinds))


def _subset(posts: pd.DataFrame, channels: list[str] | None) -> pd.DataFrame:
    return posts if not channels else posts[posts["channel"].isin(channels)]


def daily_flags(posts: pd.DataFrame, channels: list[str] | None = None) -> pd.DataFrame:
    """One row per UTC day with posts: what the posts of that day (and only that day) say."""
    p = _subset(posts, channels)
    if p.empty:
        return pd.DataFrame(columns=["gen", "sub", "regions", "cons", "emerg", "sched", "reports", "lifted"])
    q = pd.DataFrame({"date": p["datetime"].dt.floor("D"),
                      "gen": _has(p["damaged_objects"], "generation"),
                      "sub": _has(p["damaged_objects"], "substation", "transmission"),
                      "cons": p["consumers_without_power"].fillna(0.0),
                      "emerg": p["emergency_outages"].astype(bool), "sched": p["outage_schedules_announced"].astype(bool),
                      "reports": p["is_attack_report"].astype(int), "lifted": p["schedules_lifted"].astype(bool)})
    g = q.groupby("date")
    out = pd.DataFrame({"gen": g["gen"].any(), "sub": g["sub"].any(), "cons": g["cons"].max(), "emerg": g["emerg"].any(),
                        "sched": g["sched"].any(), "reports": g["reports"].sum(), "lifted": g["lifted"].any()}).astype(float)
    regs = pd.Series({d: len({r for lst in x for r in lst}) for d, x in p.groupby(p["datetime"].dt.floor("D"))["regions_hit"]})
    out["regions"] = regs.reindex(out.index).astype(float)
    return out


def issue_features(posts: pd.DataFrame, index: pd.DatetimeIndex, channels: list[str] | None = None) -> pd.DataFrame:
    """TG_COLS for every issue day in `index` (tz-aware UTC midnights); row d depends on posts published before d + 1 day."""
    p = _subset(posts, channels)
    p = p[p["datetime"] < index.max() + pd.Timedelta(days=1)]        # nothing after the last issue time is ever read
    if p.empty:
        return pd.DataFrame(np.nan, index=index, columns=TG_COLS)
    days = daily_flags(p, channels).reindex(pd.date_range(min(index.min(), p["datetime"].min().floor("D")), index.max(), freq="D",
                                                          tz="UTC")).fillna(0.0)
    f = pd.DataFrame(index=days.index)
    f["tg_gen_d0"], f["tg_sub_d0"], f["tg_regions_d0"] = days["gen"], days["sub"], days["regions"]
    f["tg_cons_d0"] = np.log1p(days["cons"])
    f["tg_emerg_d0"], f["tg_sched_d0"], f["tg_reports_d0"], f["tg_lifted_d0"] = days["emerg"], days["sched"], days["reports"], days["lifted"]
    for w in (7, 30):
        r = days.rolling(w, min_periods=1)
        f[f"tg_sched_days{w}"], f[f"tg_emerg_days{w}"], f[f"tg_gen_days{w}"] = r["sched"].sum(), r["emerg"].sum(), r["gen"].sum()
        f[f"tg_regions_max{w}"] = r["regions"].max()
        if w == 7:
            f["tg_lifted_days7"] = r["lifted"].sum()
    f = f[TG_COLS]
    f[f.index < p["datetime"].min().floor("D")] = np.nan             # before the first post: unknown, not "nothing happened"
    return f.reindex(index)


def regime_state(posts: pd.DataFrame, issue_time, channels: list[str] | None = None, lookback_days: int = 60) -> dict[str, float]:
    """Restriction-regime state at `issue_time` from posts before it: days since the first restriction post after the last
    "restrictions lifted" post (0 = no regime in force), and days since that lifting post (`lookback_days` if none)."""
    t = pd.Timestamp(issue_time)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    p = _subset(posts, channels)
    p = p[(p["datetime"] < t) & (p["datetime"] >= t - pd.Timedelta(days=lookback_days))]
    lifts = p.loc[p["schedules_lifted"].astype(bool), "datetime"]
    last_lift = lifts.max() if len(lifts) else pd.NaT
    restr = p[p["outage_schedules_announced"].astype(bool) | p["emergency_outages"].astype(bool)]
    if pd.notna(last_lift):
        restr = restr[restr["datetime"] > last_lift]
    age = (t - restr["datetime"].min()).total_seconds() / 86400 if len(restr) else 0.0
    since = (t - last_lift).total_seconds() / 86400 if pd.notna(last_lift) else float(lookback_days)
    return {"tg_regime_age": float(age), "tg_days_since_lift": float(since)}
