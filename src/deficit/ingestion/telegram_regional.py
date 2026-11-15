"""Oblast-level strike-impact dataset from the Telegram posts: one row per (strike, oblast).

Input : the extracted posts (`telegram_extract.extract_frame` + `add_derived`), the raw texts (needed to re-read the sentences
        that name a given oblast) and `channel -> oblast` from `config/sources.yaml` (key `oblast`).
Output: data/processed/telegram_region_posts.parquet     one row per (post, oblast): what the post says about that oblast
        data/processed/telegram_region_episodes.parquet  one row per (strike in attacks.csv, oblast)

How a post is mapped to oblasts
  * `own`  : the channel belongs to one oblast (an oblenergo / a regional DTEK network / a regional administration): the
             post-level fields of the extractor are the facts of that oblast, unless the post names another oblast (a repost of a
             national digest, a list of oblasts): then it is read like a national post (source `text` for every oblast named).
  * `text` : every oblast named in the text (also in regional channels, and in all posts of the national channels). The
             sentences that name the oblast (plus the line before a list item, "... повернули світло для:") are re-read with
             the same extractor, so a digest that lists several oblasts yields one fact row per oblast.  Counts of consumers
             are kept only when the sentence names exactly one oblast (`multi_region` marks the others).
  Facts per row: attack (new strike / damage reported for the oblast), consumers_without_power, schedules, emergency,
  lifted (restrictions cancelled / not applied), restored ("full" / "partial" power restoration), consumers_restored and the
  text spans that produced them (JSON, offsets in the raw post text).

Episode of one (strike, oblast) uses the same definitions as `telegram_extract.compute_episode` (window = strike day 00:00 UTC
to +horizon days, only posts <= that window; everything is an ex-post label, never a feature):
  * outage / restriction evidence = the oblast had consumers cut off, emergency outages or schedules in a post;
  * restoration = first post after the first restriction that cancels the schedules (`restrictions_lifted`); if the oblast
    had no restriction regime, the first "all consumers have power" post after the first outage post (`power_restored`);
  * `resolution_source`: `oblast` (a post about that oblast) or `national_lift` (the oblast had restrictions but only the
    national channel announced that they are not applied any more);
  * `status`: observed / censored (impact seen, no restoration inside the window) / no_impact_reported;
  * `coverage`: `own_channel` when the oblast has a channel whose stored posts start before the window and that posted within
    30 days of the strike day, else `text_only`
    (only mentions in other channels: "no impact reported" is then weak evidence).
  * window channels (`window_collected: true` in sources.yaml, collected with `deficit collect --windows-from-attacks`): the
    channel is only observed in the windows of `data/raw/telegram/_windows.csv`.  For a strike with a finished window the
    oblast is `own_channel` and its episode is RIGHT-CENSORED at the end of that window (`window_end_reason`
    = `collection_window`): a restoration post published later is not in the data, so "no restoration seen" only means "none
    in the first days".  Facts of a window channel outside the window of the strike are ignored for that strike's episode (they
    are keyword-search leftovers).  A strike without window gets no own-channel coverage from that channel (`text_only`), and
    a channel with `continuous_from` is treated as an ordinary continuous channel from that date on.
Leakage: nothing here is a model input; the observed durations are labels.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .telegram_extract import (EpisodeParams, REGIONS, compute_episode, extract_post, normalize, segments)

log = logging.getLogger(__name__)

OBLASTS: list[str] = sorted(k for k in REGIONS if k != "Crimea")
FACT_SPAN_FIELDS = ("is_attack_report", "consumers_without_power", "outage_schedules_announced", "emergency_outages",
                    "schedules_lifted", "restoration_power", "consumers_restored")
FACT_COLS = ["channel", "post_id", "datetime", "url", "oblast", "source", "attack", "consumers_without_power", "schedules",
             "emergency", "lifted", "restoration_power", "consumers_restored", "multi_region", "damaged_objects", "spans"]


def load_channel_oblasts(sources_path: Path) -> dict[str, str]:
    """`channel -> oblast code` for every Telegram source in sources.yaml that has an `oblast` key."""
    import yaml
    src = yaml.safe_load(Path(sources_path).read_text(encoding="utf-8")) or {}
    out = {}
    for c in src.get("telegram", []) or []:
        ob = c.get("oblast")
        if ob:
            if ob not in OBLASTS:
                raise ValueError(f"sources.yaml: unknown oblast code {ob!r} for {c.get('name')} (valid: {OBLASTS})")
            out[c["name"]] = ob
    return out


def load_window_info(sources_path: Path, data_root: Path | None = None) -> dict[str, dict]:
    """`channel -> {"continuous_from": Timestamp | None, "windows": DataFrame[window_start, window_end, strikes]}` for the
    channels flagged `window_collected` in sources.yaml (windows from `_windows.csv`; empty frame if none were collected)."""
    import yaml
    src = yaml.safe_load(Path(sources_path).read_text(encoding="utf-8")) or {}
    log_df = None
    if data_root is not None:
        p = Path(data_root) / "raw" / "telegram" / "_windows.csv"
        if p.exists():
            log_df = pd.read_csv(p)
            for c in ("window_start", "window_end"):
                log_df[c] = pd.to_datetime(log_df[c], utc=True)
    out = {}
    for c in src.get("telegram", []) or []:
        if not c.get("window_collected"):
            continue
        cf = c.get("continuous_from")
        w = (log_df[log_df["channel"] == c["name"]][["window_start", "window_end", "strikes"]].reset_index(drop=True)
             if log_df is not None else pd.DataFrame(columns=["window_start", "window_end", "strikes"]))
        out[c["name"]] = {"continuous_from": pd.Timestamp(cf, tz="UTC") if cf else None, "windows": w}
    return out


def _strike_window(info: dict, strike_day: pd.Timestamp):
    """(start, end) of the finished windows that hold the strike day, else None.  A later run with more strikes can log
    a merged window that overlaps an older one; all of them hold the strike day, so their union is one interval."""
    w = info["windows"]
    key = strike_day.strftime("%Y-%m-%d")
    hit = w[w["strikes"].astype(str).map(lambda x: key in x.split("|"))] if len(w) else w
    return (hit["window_start"].min(), hit["window_end"].max()) if len(hit) else None


# --------------------------------------------------------------------------------------------------
# post -> per-oblast facts
# --------------------------------------------------------------------------------------------------
def _own_fact(row, oblast: str) -> dict:
    spans = json.loads(row["spans"]) if isinstance(row.get("spans"), str) and row["spans"] else {}
    return {"channel": row["channel"], "post_id": row["post_id"], "datetime": row["datetime"], "url": row.get("url"),
            "oblast": oblast, "source": "own", "attack": bool(row["is_attack_report"]),
            "consumers_without_power": row["consumers_without_power"], "schedules": bool(row["outage_schedules_announced"]),
            "emergency": bool(row["emergency_outages"]), "lifted": bool(row["schedules_lifted"]),
            "restoration_power": row["restoration_power"], "consumers_restored": row["consumers_restored"],
            "multi_region": False, "damaged_objects": list(row["damaged_objects"]),
            "spans": json.dumps({k: v for k, v in spans.items() if k in FACT_SPAN_FIELDS}, ensure_ascii=False)}


def _snippets(t: str, segs: list[tuple[int, int]], hits: list[tuple[int, int]],
              seg_regions: dict[int, set[str]] | None = None) -> list[tuple[int, int, int]]:
    """Contiguous text slices (start, end, segment_index) of the sentences containing the hits; a sentence that ends with ':'
    (list header) before a hit sentence is included.  When the hit sentence names exactly one oblast, the next sentence is
    read together with it if it names no oblast at all ("... ударів по об'єктах на Одещині. На ранок без світла залишаються
    90 тисяч споживачів.": the follow-up sentence is about the same oblast)."""
    seg_regions = seg_regions or {}
    seen, out = set(), []
    for pos, _ in hits:
        for k, (s, e) in enumerate(segs):
            if s <= pos < e and k not in seen:
                seen.add(k)
                start = s
                if k and t[slice(*segs[k - 1])].rstrip().endswith(":"):
                    start = segs[k - 1][0]
                if k + 1 < len(segs) and len(seg_regions.get(k, ())) == 1 and not seg_regions.get(k + 1):
                    e = segs[k + 1][1]
                out.append((start, e, k))
    return out


def _text_facts(row, text: str, own: str | None) -> list[dict]:
    """One fact per oblast named in `text` (other than the channel's own), from the sentences naming it."""
    regions = [r for r in row["regions"] if r != own and r in REGIONS]
    if not regions:
        return []
    t = normalize(text)
    segs = segments(t)
    hits: dict[str, list[tuple[int, int]]] = {}
    for name in regions:
        for p in REGIONS[name]:
            hits.setdefault(name, []).extend((m.start(), m.end()) for m in p.finditer(t))
    seg_regions: dict[int, set[str]] = {}
    for name, hh in hits.items():
        for pos, _ in hh:
            for k, (s, e) in enumerate(segs):
                if s <= pos < e:
                    seg_regions.setdefault(k, set()).add(name)
    out = []
    for name in regions:
        agg = {"attack": bool(row["is_attack_report"]) and name in set(row["regions_affected"]), "consumers_without_power": None,
               "schedules": False, "emergency": False, "lifted": False, "restoration_power": None, "consumers_restored": None,
               "multi_region": False}
        spans: dict[str, list[dict]] = {}
        snips = _snippets(t, segs, hits.get(name, []), seg_regions)
        if len(row["regions"]) == 1 and name in seg_regions.get(0, ()) and len(segs) > 1:
            # a post about one oblast whose first line (headline) names it: "Одеська область: ДТЕК повертає світло ...";
            # the sentences below do not repeat the name but belong to it
            snips = [(0, len(t), 0)]
        for start, end, k in snips:
            r = extract_post(text[start:end])
            multi = len(seg_regions.get(k, set())) > 1
            sp = json.loads(r["spans"]) if r["spans"] else {}
            agg["multi_region"] |= multi
            if not multi:
                for f in ("consumers_without_power", "consumers_restored"):
                    if r[f] is not None and (agg[f] is None or r[f] > agg[f]):
                        agg[f] = r[f]
            agg["schedules"] |= bool(r["outage_schedules_announced"])
            agg["emergency"] |= bool(r["emergency_outages"])
            agg["lifted"] |= bool(r["schedules_lifted"])
            if r["restoration_power"] and (agg["restoration_power"] != "full"):
                agg["restoration_power"] = r["restoration_power"]
            for f, lst in sp.items():
                if f in FACT_SPAN_FIELDS:
                    spans.setdefault(f, []).extend({**x, "start": x["start"] + start, "end": x["end"] + start} for x in lst)
        if not (agg["attack"] or agg["consumers_without_power"] is not None or agg["schedules"] or agg["emergency"]
                or agg["lifted"] or agg["restoration_power"]):
            continue
        if agg["attack"]:
            for pos, end in hits.get(name, [])[:1]:
                spans.setdefault("region", []).append({"start": pos, "end": end, "text": text[pos:end], "value": name})
        out.append({"channel": row["channel"], "post_id": row["post_id"], "datetime": row["datetime"], "url": row.get("url"),
                    "oblast": name, "source": "text", **agg, "damaged_objects": list(row["damaged_objects"]) if agg["attack"] else [],
                    "spans": json.dumps(spans, ensure_ascii=False)})
    return out


def region_facts(posts: pd.DataFrame, raw: pd.DataFrame, channel_oblast: dict[str, str]) -> pd.DataFrame:
    """One row per (post, oblast) with the extracted facts (`own` and `text` sources, see module docstring)."""
    txt = raw[["channel", "post_id", "text"]].drop_duplicates(["channel", "post_id"])
    m = posts.merge(txt, on=["channel", "post_id"], how="left")
    rows = []
    for row in m.to_dict("records"):
        own = channel_oblast.get(row["channel"])
        named = [r for r in row["regions"] if r in REGIONS]
        if own and any(r != own for r in named):
            # a regional channel that reposts a national digest (Lvivoblenergo, Sumyoblenergo copy Ukrenergo posts) or lists
            # several oblasts: the post is not "about the own oblast", so every oblast, the own one included, is read
            # sentence by sentence like in a national channel
            rows.extend(_text_facts(row, row.get("text") or "", None))
            continue
        if own:
            rows.append(_own_fact(row, own))
        if len(row["regions"]):
            rows.extend(_text_facts(row, row.get("text") or "", own))
    if not rows:
        return pd.DataFrame(columns=FACT_COLS)
    df = pd.DataFrame(rows, columns=FACT_COLS)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df.sort_values(["datetime", "oblast", "channel", "post_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------------------------------
def _as_posts(f: pd.DataFrame) -> pd.DataFrame:
    """Facts of one oblast in the column layout `compute_episode` expects."""
    p = pd.DataFrame({
        "datetime": f["datetime"], "channel": f["channel"], "is_attack_report": f["attack"].astype(bool),
        "outage_schedules_announced": f["schedules"].astype(bool), "emergency_outages": f["emergency"].astype(bool),
        "schedules_lifted": f["lifted"].astype(bool), "restoration_power": f["restoration_power"],
        "consumers_without_power": pd.to_numeric(f["consumers_without_power"], errors="coerce"),
    })
    p["impact_post"] = p["outage_schedules_announced"] | p["emergency_outages"] | p["consumers_without_power"].notna()
    for c in ("settlements_without_power", "deficit_mw", "import_mw", "limit_mw", "capacity_lost_mw", "schedule_queues_max"):
        p[c] = np.nan
    p["regions_hit"] = [[] for _ in range(len(p))]
    p["nationwide"] = False
    p["damaged_objects"] = list(f["damaged_objects"]) if len(f) else []
    return p.reset_index(drop=True)


def _evidence(f: pd.DataFrame, role: str, at, *fields: str) -> dict | None:
    """The fact row published at `at` and the span text of the first of `fields` that has a span in it."""
    if pd.isna(at) or f.empty:
        return None
    r = f[f["datetime"] == at]
    if r.empty:
        return None
    r = r.iloc[0]
    sp = json.loads(r["spans"]) if r["spans"] else {}
    field = next((x for x in fields if sp.get(x)), None)
    items = sp.get(field) or sp.get("region") or []
    return {"role": role, "at": str(at), "channel": r["channel"], "post_id": int(r["post_id"]), "url": r["url"],
            "source": r["source"], "field": field, "text": items[0]["text"] if items else None}


KEEP = ["window_start", "window_end", "n_posts", "n_channels", "n_attack_reports", "impact_observed", "first_impact_at",
        "last_impact_at", "first_schedule_at", "schedule_days", "emergency_outages", "first_emergency_at",
        "max_consumers_without_power", "max_consumers_without_power_at", "damaged_objects", "restrictions_observed",
        "first_restriction_at", "last_restriction_at", "restriction_days", "lifted_at", "lifted_sustained_at", "relapse",
        "first_outage_at", "restored_at", "resolution_at", "resolution_kind", "days_to_resolution",
        "days_to_restrictions_lifted", "days_to_power_restored", "next_strike_gap_days", "resolved_before_next_strike", "status"]


def _episode(p: pd.DataFrame, p_nat: pd.DataFrame, f: pd.DataFrame, f_nat: pd.DataFrame, strike_at, nxt, prm: EpisodeParams,
             cutoff) -> tuple[dict, str | None, pd.DataFrame]:
    """compute_episode for one oblast, posts limited to `cutoff` (None = horizon only); returns (episode, resolution
    source, fact table that holds the evidence)."""
    ep = compute_episode(p, strike_at, nxt, prm, as_of=cutoff)
    src = "oblast" if pd.notna(ep["lifted_at"]) else None
    ef = f
    if ep["restrictions_observed"] and pd.isna(ep["lifted_at"]) and len(p_nat):
        ep2 = compute_episode(p_nat, strike_at, nxt, prm, as_of=cutoff)
        # a national lifting post only counts for an oblast that had a restriction regime of its own
        restr2 = p_nat[p_nat["outage_schedules_announced"] | p_nat["emergency_outages"]]
        if pd.notna(ep2["lifted_at"]) and len(restr2):
            ep, src, ef = ep2, "national_lift", f_nat
    return ep, src, ef


def _active(channel_times, ch: str, start: pd.Timestamp, days: int) -> bool:
    """The channel posted at least once within `days` of the strike day (a channel whose stored history ends years before the
    strike does not cover it); always True when the post times are not given."""
    if channel_times is None:
        return True
    t = channel_times.get(ch)
    return bool(t is not None and ((t >= start - pd.Timedelta(days=days)) & (t <= start + pd.Timedelta(days=days))).any())


def build_region_episodes(facts: pd.DataFrame, strikes: pd.DataFrame, channel_oblast: dict[str, str],
                          channel_first_post: dict[str, pd.Timestamp], national_lifts: pd.DataFrame | None = None,
                          prm: EpisodeParams | None = None, channel_times: dict[str, pd.DatetimeIndex] | None = None,
                          active_days: int = 30, window_info: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per (strike, oblast in OBLASTS); see the module docstring for the definitions."""
    prm = prm or EpisodeParams()
    by_ob = {ob: g.sort_values("datetime").reset_index(drop=True) for ob, g in facts.groupby("oblast")} if len(facts) else {}
    nat = national_lifts if national_lifts is not None else pd.DataFrame(columns=["datetime", "channel"])
    nat_f = pd.DataFrame({"datetime": nat["datetime"], "channel": nat["channel"], "post_id": 0, "url": None, "oblast": "",
                          "source": "national", "attack": False, "consumers_without_power": np.nan, "schedules": False,
                          "emergency": False, "lifted": True, "restoration_power": None, "consumers_restored": np.nan,
                          "multi_region": False, "damaged_objects": [[] for _ in range(len(nat))], "spans": "{}"})
    winfo = window_info or {}
    own_ch = {}
    for ch, ob in channel_oblast.items():
        own_ch.setdefault(ob, []).append(ch)
    strikes = strikes.reset_index(drop=True)
    rows = []
    for ob in OBLASTS:
        f = by_ob.get(ob, pd.DataFrame(columns=FACT_COLS))
        p = _as_posts(f) if len(f) else _as_posts(pd.DataFrame({c: pd.Series(dtype=object) for c in FACT_COLS}).assign(
            datetime=pd.Series(dtype="datetime64[ns, UTC]")))
        f_nat = pd.concat([f, nat_f], ignore_index=True).sort_values("datetime").reset_index(drop=True) if len(nat_f) else f
        p_nat = _as_posts(f_nat) if len(f_nat) else p
        chans = own_ch.get(ob, [])
        for i, st in strikes.iterrows():
            nxt = strikes["strike_at"].iloc[i + 1] if i + 1 < len(strikes) else pd.NaT
            # primary episode: watched until the NEXT strike day starts (posts after it belong to the next episode) or the
            # horizon; `*_30d` columns keep the plain 30-day window (a lifting post then may follow a later strike)
            cutoff = nxt.floor("D") if pd.notna(nxt) else None
            day = st["strike_at"].floor("D")
            # window channels: observed only inside the strike's window (see module docstring)
            wonly = [c for c in chans if c in winfo and not (winfo[c]["continuous_from"] is not None
                                                             and day >= winfo[c]["continuous_from"])]
            f_i, p_i, f_nat_i, p_nat_i, coll_end, win_cov = f, p, f_nat, p_nat, None, []
            if wonly:
                keep = ~f["channel"].isin(wonly) if len(f) else pd.Series([], dtype=bool)
                for c in wonly:
                    sw = _strike_window(winfo[c], day)
                    if sw is not None and len(f):
                        keep |= (f["channel"] == c) & (f["datetime"] >= sw[0]) & (f["datetime"] < sw[1])
                    if sw is not None:
                        win_cov.append(c)
                        coll_end = sw[1] if coll_end is None else max(coll_end, sw[1])
                f_i = f[keep].reset_index(drop=True) if len(f) else f
                p_i = _as_posts(f_i) if len(f_i) else p.iloc[0:0]
                f_nat_i = (pd.concat([f_i, nat_f], ignore_index=True).sort_values("datetime").reset_index(drop=True)
                           if len(nat_f) else f_i)
                p_nat_i = _as_posts(f_nat_i) if len(f_nat_i) else p_i
            cut_pri = cutoff if coll_end is None else (coll_end if cutoff is None else min(cutoff, coll_end))
            ep, src, ef = _episode(p_i, p_nat_i, f_i, f_nat_i, st["strike_at"], nxt, prm, cut_pri)
            ep30, src30, _ = _episode(p_i, p_nat_i, f_i, f_nat_i, st["strike_at"], nxt, prm, coll_end)
            start = ep["window_start"]
            covered = [c for c in chans if c not in wonly
                       and channel_first_post.get(c, pd.Timestamp.max.tz_localize("UTC")) <= start + pd.Timedelta(days=1)
                       and _active(channel_times, c, start, active_days)] + win_cov
            win = f_i[(f_i["datetime"] >= ep["window_start"]) & (f_i["datetime"] <= ep["window_end"])] if len(f_i) else f_i
            own_n = int((win["source"] == "own").sum()) if len(win) else 0
            row = {"strike_at": st["strike_at"], "major": bool(st.get("major", True)), "description": st.get("description"),
                   "oblast": ob, "coverage": "own_channel" if covered else "text_only", "own_channels": ";".join(covered),
                   "own_posts_in_window": own_n, "text_posts_in_window": int((win["source"] == "text").sum()) if len(win) else 0}
            row["first_post_at"] = win["datetime"].min() if len(win) else pd.NaT
            row.update({k: ep[k] for k in KEEP})
            row["resolution_source"] = src if pd.notna(ep["resolution_at"]) else None
            if pd.notna(ep["resolution_at"]) and ep["resolution_kind"] == "power_restored":
                row["resolution_source"] = "oblast"
            row["hours_to_resolution"] = ep["days_to_resolution"] * 24 if pd.notna(ep["days_to_resolution"]) else np.nan
            row["hours_first_impact_to_resolution"] = ((ep["resolution_at"] - ep["first_impact_at"]).total_seconds() / 3600
                                                       if pd.notna(ep["resolution_at"]) and pd.notna(ep["first_impact_at"]) else np.nan)
            row["censored"] = bool(ep["status"] == "censored")
            row["censor_at"] = ep["window_end"] if row["censored"] else pd.NaT
            row["censor_hours"] = max((ep["window_end"] - st["strike_at"]).total_seconds() / 3600, 0.0) if row["censored"] else np.nan
            if coll_end is not None and ep["window_end"] == coll_end and (cutoff is None or coll_end <= cutoff):
                row["window_end_reason"] = "collection_window"
            else:
                row["window_end_reason"] = "next_strike" if (cutoff is not None and ep["window_end"] == cutoff) else "horizon"
            row["collection_window_end"] = coll_end if coll_end is not None else pd.NaT
            row["resolution_at_30d"] = ep30["resolution_at"]
            row["window_end_30d"] = ep30["window_end"]
            row["status_30d"] = ep30["status"]
            row["hours_to_resolution_30d"] = ep30["days_to_resolution"] * 24 if pd.notna(ep30["days_to_resolution"]) else np.nan
            row["censor_hours_30d"] = (max((ep30["window_end"] - st["strike_at"]).total_seconds() / 3600, 0.0)
                                       if ep30["status"] == "censored" else np.nan)
            row["outage"] = bool(ep["impact_observed"])
            row["schedules"] = bool(pd.notna(ep["first_schedule_at"]))
            ev = [_evidence(ef, "first_impact", ep["first_impact_at"], "consumers_without_power",
                            "outage_schedules_announced", "emergency_outages"),
                  _evidence(ef, "max_consumers", ep["max_consumers_without_power_at"], "consumers_without_power"),
                  _evidence(ef, "first_restriction", ep["first_restriction_at"], "outage_schedules_announced",
                            "emergency_outages"),
                  _evidence(ef, "lifted", ep["lifted_at"], "schedules_lifted"),
                  _evidence(f, "restored", ep["restored_at"], "restoration_power")]
            row["spans_used"] = json.dumps([e for e in ev if e], ensure_ascii=False)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["strike_at", "oblast"]).reset_index(drop=True)


def run_regional(posts: pd.DataFrame, raw: pd.DataFrame, strikes: pd.DataFrame, channel_oblast: dict[str, str],
                 out_dir: Path, prm: EpisodeParams | None = None,
                 window_info: dict[str, dict] | None = None) -> dict[str, pd.DataFrame]:
    """posts + raw texts -> telegram_region_posts.parquet and telegram_region_episodes.parquet in `out_dir`."""
    prm = prm or EpisodeParams()
    facts = region_facts(posts, raw, channel_oblast)
    first = posts.groupby("channel")["datetime"].min().to_dict()
    times = {c: pd.DatetimeIndex(g["datetime"]).sort_values() for c, g in posts.groupby("channel")}
    nat = posts[posts["channel"].isin(prm.national_channels) & posts["schedules_lifted"]][["datetime", "channel"]]
    eps = build_region_episodes(facts, strikes, channel_oblast, first, nat, prm, channel_times=times,
                                window_info=window_info)
    out_dir.mkdir(parents=True, exist_ok=True)
    facts.to_parquet(out_dir / "telegram_region_posts.parquet")
    eps.to_parquet(out_dir / "telegram_region_episodes.parquet")
    imp = eps[eps["outage"]]
    log.info("regional: %d fact rows; %d (strike, oblast) episodes with impact, observed restoration %d, censored %d "
             "(wrote %s)", len(facts), len(imp), int((imp["status"] == "observed").sum()), int(imp["censored"].sum()),
             out_dir / "telegram_region_episodes.parquet")
    return {"region_posts": facts, "region_episodes": eps}


# --------------------------------------------------------------------------------------------------
# quality: hand-labelled regional posts
# --------------------------------------------------------------------------------------------------
GOLD_FIELDS = [("attack", "attack"), ("consumers", "consumers"), ("schedules", "schedules"), ("emergency", "emergency"),
               ("lifted", "lifted"), ("restored_full", "restored_full"), ("restored_any", "restored_any")]


def _gold_items(v) -> set[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "none"):
        return set()
    return {x.strip() for x in str(v).split(";") if x.strip()}


def _pred_items(g: pd.DataFrame, field: str, emergency_is_schedule: bool = True) -> set[str]:
    """Predicted (oblast) items of one gold field.  The gold definition of `schedules` includes emergency outage schedules
    (GAV) and emergency outages announced for an oblast, so `emergency` implies `schedules` unless switched off."""
    out = set()
    for r in g.itertuples():
        if field == "attack" and r.attack:
            out.add(r.oblast)
        elif field == "consumers" and r.consumers_without_power is not None and not pd.isna(r.consumers_without_power):
            out.add(f"{r.oblast}:{r.consumers_without_power:g}")
        elif field == "schedules" and (r.schedules or (emergency_is_schedule and r.emergency)):
            out.add(r.oblast)
        elif field == "emergency" and r.emergency:
            out.add(r.oblast)
        elif field == "lifted" and r.lifted:
            out.add(r.oblast)
        elif field == "restored_full" and r.restoration_power == "full":
            out.add(r.oblast)
        elif field == "restored_any" and isinstance(r.restoration_power, str):
            out.add(r.oblast)
    return out


def score_region_gold(gold: pd.DataFrame, facts: pd.DataFrame, raw_keys: pd.DataFrame | None = None,
                      emergency_is_schedule: bool = True) -> pd.DataFrame:
    """Per-field TP/FP/FN of (oblast) sets: a post counts for an oblast only when the extractor and the label agree on
    that oblast. Gold columns: channel, post_id, stratum, one column per GOLD_FIELDS name with ';'-separated oblast codes
    (`consumers`: `Oblast:value`)."""
    keyed = {k: g for k, g in facts.groupby(["channel", "post_id"])} if len(facts) else {}
    empty = pd.DataFrame(columns=FACT_COLS)
    rows = []
    for name, field in GOLD_FIELDS:
        tp = fp = fn = 0
        for r in gold.itertuples():
            g = keyed.get((r.channel, r.post_id), empty)
            gs, ps = _gold_items(getattr(r, name)), _pred_items(g, field, emergency_is_schedule)
            tp += len(gs & ps)
            fp += len(ps - gs)
            fn += len(gs - ps)
        rows.append({"field": name, "tp": tp, "fp": fp, "fn": fn, "gold_n": tp + fn,
                     "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan})
    return pd.DataFrame(rows)
