"""Quality report for the Telegram extraction: coverage per year/channel, strike-episode yield, and precision/recall
against a hand-labelled gold sample.

Gold labels: tests/fixtures/telegram_gold.csv, one row per labelled post (channel, post_id, stratum + one column per
field, filled by reading the post text, NOT by looking at extractor output). Written to
reports/telegram_extraction_quality.md by `deficit extract`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT

log = logging.getLogger(__name__)

GOLD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "telegram_gold.csv"
NOTES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "telegram_gold_notes.md"

# gold column -> (kind, extracted column)
FIELDS: list[tuple[str, str, str]] = [
    ("attack", "bool", "is_attack_report"),
    ("objects", "set", "damaged_objects"),
    ("regions", "set", "regions"),
    ("regions_affected", "set", "regions_affected"),
    ("consumers", "num", "consumers_without_power"),
    ("consumers_restored", "num", "consumers_restored"),
    ("settlements", "num", "settlements_without_power"),
    ("schedules", "bool", "outage_schedules_announced"),
    ("hours", "set", "schedule_hours"),
    ("queues_max", "num", "schedule_queues_max"),
    ("emergency", "bool", "emergency_outages"),
    ("lifted", "bool", "schedules_lifted"),
    ("mw", "mw", "mw"),
    ("restored_any", "bool", "restoration_power"),
    ("restored_full", "bool", "restoration_full"),
    ("repairs", "bool", "restoration_repairs"),
    ("cons_pct", "num", "consumption_change_pct"),
    ("eta_h", "num", "eta_hours"),
]


def _gold_set(v) -> set[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "none"):
        return set()
    return {x.strip() for x in str(v).split(";") if x.strip()}


def _gold_num(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "none"):
        return None
    return float(v)


def _pred_mw(row) -> set[str]:
    out = set()
    for cls in ("deficit", "import", "export", "limit", "capacity_lost"):
        v = row.get(f"{cls}_mw")
        if v is not None and not pd.isna(v):
            out.add(f"{cls}:{v:g}")
    return out


def _num_eq(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) <= 1e-3 * max(1.0, abs(b))


def score_gold(gold: pd.DataFrame, posts: pd.DataFrame, strata: list[str] | None = None) -> pd.DataFrame:
    """Per-field TP/FP/FN, precision and recall of the extractor on the gold sample.

    A gold row scores only the fields in its `scored` column ("all" or a ';'-separated list), so the targeted
    supplements (posts picked because they contain consumer counts, ETAs, MW values) do not count unlabelled fields
    as negatives. `strata` restricts the rows (e.g. ["random"]).
    """
    g = gold.rename(columns={c: f"gold_{c}" for c in gold.columns if c not in ("channel", "post_id")})
    m = g.merge(posts, on=["channel", "post_id"], how="left")
    if m["datetime"].isna().any():
        missing = m.loc[m["datetime"].isna(), ["channel", "post_id"]].values.tolist()
        raise ValueError(f"gold posts missing from the extracted frame (collect them first): {missing[:5]}")
    if strata is not None:
        m = m[m["gold_stratum"].isin(strata)]
    rows = []
    for name, kind, col in FIELDS:
        tp = fp = fn = 0
        for _, r in m.iterrows():
            sc = r.get("gold_scored", "all")
            if isinstance(sc, str) and sc != "all" and name not in sc.split(";"):
                continue
            g = r[f"gold_{name}"]
            if kind == "bool":
                gold_v = bool(int(g)) if not pd.isna(g) else False
                if name == "restored_any":
                    pred = r["restoration_power"] is not None and not pd.isna(r["restoration_power"])
                elif name == "restored_full":
                    pred = r["restoration_power"] == "full"
                else:
                    pred = bool(r[col])
                tp += gold_v and pred
                fp += pred and not gold_v
                fn += gold_v and not pred
            elif kind in ("set", "mw"):
                gs = _gold_set(g)
                ps = _pred_mw(r) if kind == "mw" else set(map(str, r[col]))
                tp += len(gs & ps)
                fp += len(ps - gs)
                fn += len(gs - ps)
            else:
                gv = _gold_num(g)
                pv = r[col]
                pv = None if pv is None or pd.isna(pv) else float(pv)
                if gv is not None and _num_eq(pv, gv):
                    tp += 1
                else:
                    fp += pv is not None
                    fn += gv is not None
        n_rows = sum(1 for _, r in m.iterrows()
                     if not isinstance(r.get("gold_scored", "all"), str) or r["gold_scored"] == "all" or name in r["gold_scored"].split(";"))
        rows.append({"field": name, "posts": n_rows, "tp": int(tp), "fp": int(fp), "fn": int(fn), "gold_n": int(tp + fn),
                     "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows)


def coverage_table(posts: pd.DataFrame) -> pd.DataFrame:
    """Share of posts per year and channel in which each field was extracted."""
    p = posts.copy()
    p["year"] = p["datetime"].dt.year
    flags = {
        "attack_report": p["is_attack_report"], "schedules": p["outage_schedules_announced"],
        "emergency": p["emergency_outages"], "lifted": p["schedules_lifted"],
        "consumers": p["consumers_without_power"].notna(), "restored": p["restoration_power"].notna() | p["restoration_repairs"],
        "region": p["regions"].map(len) > 0, "MW": p[["deficit_mw", "import_mw", "export_mw", "limit_mw", "capacity_lost_mw"]].notna().any(axis=1),
        "cons_change": p["consumption_change_pct"].notna(), "eta": p["eta_hours"].notna(),
    }
    for k, v in flags.items():
        p[k] = v
    g = p.groupby(["channel", "year"])
    out = g.size().rename("posts").to_frame()
    for k in flags:
        out[k] = g[k].sum()
    return out.reset_index()


def _md(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if v is None or (isinstance(v, float) and np.isnan(v))
                         else floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def episode_summary(eps: pd.DataFrame) -> str:
    a = eps[eps["scope"] == "ALL"].copy()
    a["year"] = a["strike_at"].dt.year
    a["lifted_obs"] = a["lifted_at"].notna()
    a["sustained"] = a["lifted_sustained_at"].notna()
    a["standalone"] = (a["status"] == "observed") & a["resolved_before_next_strike"]
    tab = a.groupby("year").agg(strikes=("strike_at", "size"), restrictions=("restrictions_observed", "sum"),
                                lifted_observed=("lifted_obs", "sum"), sustained_3d=("sustained", "sum"),
                                duration_observed=("status", lambda s: int((s == "observed").sum())),
                                resolved_before_next_strike=("standalone", "sum"),
                                median_days=("days_to_resolution", "median")).reset_index()
    tot = {"year": "all", "strikes": len(a), "restrictions": int(a["restrictions_observed"].sum()), "lifted_observed": int(a["lifted_obs"].sum()),
           "sustained_3d": int(a["sustained"].sum()), "duration_observed": int((a["status"] == "observed").sum()),
           "resolved_before_next_strike": int(a["standalone"].sum()), "median_days": a["days_to_resolution"].median()}
    tab = pd.concat([tab, pd.DataFrame([tot])], ignore_index=True)
    kinds = a["resolution_kind"].value_counts().to_dict()
    txt = (f"Strikes in attacks.csv: {len(a)} ({int(a['major'].sum())} flagged major). A restriction regime (schedules or emergency "
           f"outages) was announced in the posts after {int(a['restrictions_observed'].sum())}. An **observed restoration duration** "
           f"(first post after the first restriction post saying restrictions are not applied / cancelled; if no restrictions were "
           f"announced, first \"all consumers have power\" post after the first outage post) exists for "
           f"**{int((a['status'] == 'observed').sum())} of {len(a)} strikes** ({kinds}); {int(a['standalone'].sum())} of them ended before the next "
           f"strike in the list, the others overlap a chain of strikes and share one lifting post. `sustained_3d` counts lifting "
           f"posts not followed by new restrictions within {3} days (the stricter definition). Windows: strike day 00:00 UTC to +30 days "
           f"(right-censored afterwards). Columns of `telegram_episodes.parquet` are listed in the module docstring; every `*_at` column is "
           f"the publication time of the post that determined the value.\n\n")
    return txt + _md(tab, "{:.1f}")


REGION_GOLD_PATH = PROJECT_ROOT / "tests" / "fixtures" / "telegram_region_gold.csv"
REGION_NOTES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "telegram_region_gold_notes.md"


def region_section(region_posts: pd.DataFrame | None, region_episodes: pd.DataFrame | None,
                   gold_path: Path = REGION_GOLD_PATH) -> list[str]:
    """Markdown block for the oblast-level dataset: yield per oblast and precision/recall on the regional gold sample."""
    from .telegram_regional import score_region_gold
    parts = ["## Oblast-level dataset (`telegram_region_episodes.parquet`)", ""]
    if region_episodes is None or not len(region_episodes):
        return parts + ["_not built_", ""]
    e = region_episodes
    imp = e[e["outage"]]
    parts += [f"{len(e)} (strike, oblast) rows = {e['strike_at'].nunique()} strikes x {e['oblast'].nunique()} oblasts. Impact "
              f"(consumers cut off, emergency outages or schedules in a post of that oblast) is reported for **{len(imp)}** of them; "
              f"an **observed restoration duration** exists for **{int((~imp['censored']).sum())}** "
              f"({int((imp['resolution_source'] == 'oblast').sum())} from posts about the oblast, "
              f"{int((imp['resolution_source'] == 'national_lift').sum())} from the national lifting post), "
              f"{int(imp['censored'].sum())} are right-censored. Rows with `coverage = text_only` have no channel of their own "
              f"(impact only from mentions in other channels; 'no impact' is weak evidence there).", ""]
    tab = (e.groupby("oblast").agg(strikes_with_own_channel=("coverage", lambda c: int((c == "own_channel").sum())))
           .join(imp.groupby("oblast").agg(impact=("outage", "sum"), observed=("censored", lambda c: int((~c).sum())),
                                           censored=("censored", "sum"), median_hours=("hours_to_resolution", "median")))
           .fillna({"impact": 0, "observed": 0, "censored": 0}).reset_index())
    parts += [_md(tab, "{:.1f}"), ""]
    if gold_path.exists() and region_posts is not None:
        gold = pd.read_csv(gold_path, dtype={"channel": str})
        try:
            sc = score_region_gold(gold, region_posts)
            parts += ["### Precision / recall on hand-labelled regional posts", "",
                      f"{len(gold)} posts ({gold['stratum'].value_counts().to_dict() if 'stratum' in gold else ''}; channels "
                      f"{gold['channel'].value_counts().to_dict()}), labelled by reading the text before looking at extractor "
                      "output (`tests/fixtures/telegram_region_gold.csv`). Items are (oblast) sets per post; `consumers` items are "
                      "`oblast:value` and must match exactly. The strata are enriched (see the notes below), so these are not population "
                      "precision / recall; fields with few positives are anecdotal.", "", _md(sc, "{:.2f}"), ""]
            if REGION_NOTES_PATH.exists():
                parts += [REGION_NOTES_PATH.read_text(encoding="utf-8").strip(), ""]
        except KeyError as e:
            parts += ["### Precision / recall on hand-labelled regional posts", "", f"Not computed: missing column {e}", ""]
    return parts


def write_quality_report(posts: pd.DataFrame, episodes: pd.DataFrame | None, reports_dir: Path,
                         gold_path: Path = GOLD_PATH, notes_path: Path = NOTES_PATH,
                         region_posts: pd.DataFrame | None = None, region_episodes: pd.DataFrame | None = None) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "telegram_extraction_quality.md"
    parts = ["# Telegram extraction quality", "",
             f"Generated from `data/raw/telegram/*.parquet` ({len(posts)} posts, channels: "
             f"{', '.join(sorted(posts['channel'].unique()))}). Rule-based extractor: "
             "`src/deficit/ingestion/telegram_extract.py`; every extracted value keeps its matched span in the `spans` column.", ""]
    if gold_path.exists():
        gold = pd.read_csv(gold_path, dtype={"channel": str})
        try:
            sc = score_gold(gold, posts)
            strat = gold["stratum"].value_counts().to_dict() if "stratum" in gold else {}
            parts += ["## Precision / recall against hand labels", "",
                      f"Gold sample: {len(gold)} posts ({', '.join(f'{k}: {v}' for k, v in strat.items())}); channels "
                      f"{gold['channel'].value_counts().to_dict()}. Labels were written by reading each post, before looking at "
                      "extractor output (see notes). Sets (objects, regions, hours, MW) are scored per element; numbers must match "
                      "exactly (0.1 %); `gold_n` is the number of true items. **These numbers are optimistic**: the extractor was tuned on the main "
                      "sample (enriched + random) after a first blind pass whose scores are listed below, and only the targeted posts were "
                      "partly held out; fields with few positives (eta_h, lifted, consumers) are anecdotal.", "", _md(sc, "{:.2f}"), ""]
        except ValueError as e:     # collector still running: some gold posts not in the data yet
            parts += ["## Precision / recall against hand labels", "", f"Not computed: {e}", ""]
    else:
        parts += ["## Precision / recall against hand labels", "", f"`{gold_path}` not found.", ""]
    parts += ["## Coverage per channel and year", "",
              "Number of posts in which each field was extracted (a post can be counted in several columns).", "",
              _md(coverage_table(posts), "{:.0f}"), ""]
    if episodes is not None and len(episodes):
        parts += ["## Strike impact episodes", "", episode_summary(episodes), ""]
    if region_episodes is not None:
        parts += region_section(region_posts, region_episodes)
    if notes_path.exists():
        parts += [notes_path.read_text(encoding="utf-8").strip(), ""]
    out.write_text("\n".join(parts), encoding="utf-8")
    log.info("wrote %s", out)
    return out
