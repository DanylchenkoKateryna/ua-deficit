"""Extended strike list for the Telegram regional pipeline only (`deficit derive-strikes`).

data/external/attacks.csv holds only the largest waves (Air Force summaries); strikes on energy infrastructure happen on
15-20 days per month. This module adds days that are visible in the continuously collected national / oblenergo channels.

Event definition (fixed before any restoration / outage outcome was looked at; outcome fields are never read):
  * candidate = a UTC day with >= 1 post with ``is_attack_report`` in the continuous channels (``CONTINUOUS_CHANNELS``);
  * ``strike_at`` = timestamp of the first such post of that day (UTC, minute precision);
  * strength rank (budget selection in window mode) = ``n_channels``, then ``n_attack_posts`` (both counted on that day);
  * every day is a separate row (no merging of chained days); days already present in attacks.csv are NOT duplicated:
    those rows are kept as they are (major=true, weapons_* ...);
  * ``near_major`` = derived day within +-1 day of a row of attacks.csv.
Extra filter parameters (``min_channels``, ``min_posts``) default to "no filter".

The file is read only by the Telegram regional steps (collect --windows-from-attacks, extract, region-restoration) when
``telegram.strike_list`` points to it (config/telegram_extended.yaml). The forecasting pipeline keeps using attacks.csv.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CONTINUOUS_CHANNELS = ("ukrenergo", "dtek_ua", "Lvivoblenergo", "sumyoblenergo", "khmelnytskoblenergo", "ztoe_ua")
DERIVED_SOURCE = "telegram_attack_reports"
EXTRA_COLUMNS = ["derived", "n_channels", "n_attack_posts", "near_major"]
NEAR_DAYS = 1


def derive_days(posts: pd.DataFrame, channels: tuple[str, ...] = CONTINUOUS_CHANNELS) -> pd.DataFrame:
    """One row per UTC day with an attack-report post: date, strike_at, n_channels, n_attack_posts, first_channel."""
    p = posts[posts["channel"].isin(channels) & posts["is_attack_report"].fillna(False).astype(bool)].copy()
    p["datetime"] = pd.to_datetime(p["datetime"], utc=True)
    p = p.sort_values(["datetime", "channel", "post_id"], kind="mergesort")
    p["date"] = p["datetime"].dt.strftime("%Y-%m-%d")
    g = p.groupby("date", sort=True)
    out = pd.DataFrame({"strike_at": g["datetime"].first(), "first_channel": g["channel"].first(),
                        "n_channels": g["channel"].nunique(), "n_attack_posts": g.size()})
    return out.reset_index()


def build_extended(attacks: pd.DataFrame, posts: pd.DataFrame, channels: tuple[str, ...] = CONTINUOUS_CHANNELS,
                   min_channels: int = 1, min_posts: int = 1) -> pd.DataFrame:
    """attacks.csv rows (kept verbatim, as strings) + derived rows; sorted by date. See the module docstring."""
    base = attacks.copy()
    days = derive_days(posts, channels)
    cnt = days.set_index("date")
    base["derived"] = False
    base["n_channels"] = base["date"].map(cnt["n_channels"]).fillna(0).astype(int)
    base["n_attack_posts"] = base["date"].map(cnt["n_attack_posts"]).fillna(0).astype(int)
    base["near_major"] = False

    have = set(base["date"])
    new = days[~days["date"].isin(have)]
    new = new[(new["n_channels"] >= min_channels) & (new["n_attack_posts"] >= min_posts)]
    major_days = pd.to_datetime(base["date"]).reset_index(drop=True)
    rows = []
    for r in new.itertuples(index=False):
        d = pd.Timestamp(r.date)
        near = bool(((major_days - d).abs() <= pd.Timedelta(days=NEAR_DAYS)).any())
        rows.append({"date": r.date, "time_utc": r.strike_at.strftime("%H:%M"),
                     "description": f"Derived: {r.n_attack_posts} attack-report post(s) in {r.n_channels} channel(s), "
                                    f"first in {r.first_channel}",
                     "capacity_lost_MW": "", "major": "False", "source": DERIVED_SOURCE, "verified": "False",
                     "weapons_fired": "", "weapons_through": "", "missiles_fired": "",
                     "derived": True, "n_channels": int(r.n_channels), "n_attack_posts": int(r.n_attack_posts),
                     "near_major": near})
    out = pd.concat([base, pd.DataFrame(rows, columns=list(base.columns))], ignore_index=True)
    out = out.sort_values(["date", "time_utc"], kind="mergesort").reset_index(drop=True)
    return out


def run_derive(data_root: Path, out_path: Path | None = None, channels: tuple[str, ...] = CONTINUOUS_CHANNELS,
               min_channels: int = 1, min_posts: int = 1) -> pd.DataFrame:
    """data/external/attacks.csv + data/processed/telegram_posts.parquet -> data/external/attacks_extended.csv."""
    data_root = Path(data_root)
    attacks = pd.read_csv(data_root / "external" / "attacks.csv", dtype=str, keep_default_na=False, encoding="utf-8")
    posts = pd.read_parquet(data_root / "processed" / "telegram_posts.parquet",
                            columns=["channel", "post_id", "datetime", "is_attack_report"])
    out = build_extended(attacks, posts, channels, min_channels, min_posts)
    out_path = out_path or (data_root / "external" / "attacks_extended.csv")
    out.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    d = out[out["derived"]]
    log.info("wrote %s: %d rows (%d from attacks.csv, %d derived, %d derived near a major strike)", out_path, len(out),
             int((~out["derived"]).sum()), len(d), int(d["near_major"].sum()))
    return out
