"""Alternative (strike x oblast) restoration label from the Energy Map event records (`ingestion/energy_map.py`), DEV ONLY.

Label (ex-post, never a feature)
  Pair (strike, oblast) gets a label iff a blackout is reported for the oblast on the strike day D or D + 1 (any source;
  `has_blackout`, point records only: period records with number_days > 1 are ignored).  b0 = the first such record day.
  Restoration = the first record of the oblast with `has_restored` on a day STRICTLY after b0 (the same-day record may refer to an
  earlier event; the share of pairs that also have a restoration record on b0 is reported as a sensitivity).  Observed if it falls
  within `window_days` (30) of D, else right-censored at the end of day D + window_days.
  Time resolution is one day: the restoration time of a record dated d is taken as d 12:00 UTC for the duration
  (`hours_to_resolution_30d`) and as the END of day d (d + 1 day 00:00) for "when was it known" (`resolution_at_30d`), so
  `region_restoration.labels_as_of` never sees a restoration earlier than the day ended.  `first_impact_at` is set to the strike time:
  the cohort (blackout reported on D or D + 1) is a label-side definition, so these dev results are conditional on "a blackout was
  reported", unlike the Telegram cohort that only uses posts known at the issue time.
  `next_strike` columns exist so that `region_restoration.build_table` works for both variants (nearly all censored: the next strike is
  usually one or two days away); only the `30d` variant is used.
Caveats: a record with restored numbers is the first RESTORATION REPORT (maybe partial), not "all consumers have power"; the files are
  curated event records, so a missing record means "not reported", not "no outage"; frontline records are dominated by shelling.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion import energy_map as em
from ..ingestion.telegram_extract import load_strikes
from ..ingestion.telegram_regional import OBLASTS
from . import region_dev as rd
from . import region_restoration as rr

log = logging.getLogger(__name__)

WINDOW_DAYS = 30
MIN_OBSERVED = 150            # dev pairs with an observed restoration needed to run the CV (fixed in advance)
DAY = pd.Timedelta(days=1)


def _dates(r: pd.DataFrame, oblast: str, col: str) -> np.ndarray:
    d = r.loc[(r["oblast"] == oblast) & r[col], "date"]
    return np.sort(d.unique())


def build_labels(strikes: pd.DataFrame, records: pd.DataFrame, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    """One row per labelled (strike, oblast); `strikes` has a tz-aware UTC `strike_at` (sorted). See the module docstring."""
    r = records[~records["is_period"]]
    st = pd.to_datetime(strikes["strike_at"], utc=True).reset_index(drop=True)
    major = strikes["major"].reset_index(drop=True) if "major" in strikes else pd.Series(True, index=st.index)
    day = st.dt.floor("D").dt.tz_localize(None)
    nxt = day.shift(-1)
    black = {o: _dates(r, o, "has_blackout") for o in OBLASTS}
    rest = {o: _dates(r, o, "has_restored") for o in OBLASTS}
    link_r = r[r["has_restored"]].sort_values("date").drop_duplicates(["oblast", "date"]).set_index(["oblast", "date"])["link"]
    rows = []
    for i in range(len(st)):
        d0, w_end = day[i], day[i] + (window_days + 1) * DAY
        for o in OBLASTS:
            b = black[o]
            hit = b[(b >= np.datetime64(d0)) & (b <= np.datetime64(d0 + DAY))]
            if not len(hit):
                continue
            b0 = pd.Timestamp(hit[0])
            rd_ = rest[o]
            later = rd_[(rd_ > np.datetime64(b0)) & (rd_ < np.datetime64(w_end))]
            res = pd.Timestamp(later[0]) if len(later) else pd.NaT
            n_end = min(nxt[i], w_end) if pd.notna(nxt[i]) else w_end
            res_ns = res if (pd.notna(res) and res < n_end) else pd.NaT
            sub = r[(r["oblast"] == o) & r["has_blackout"] & (r["date"] >= d0) & (r["date"] <= d0 + DAY)]
            cons = sub["blackout_consumers"].max()
            rows.append({
                "strike_at": st[i], "idx_all": i, "major": bool(major[i]), "oblast": o, "coverage": "text_only", "blackout_day": b0,
                "blackout_consumers": cons, "blackout_sources": "+".join(sorted(sub["source"].unique())),
                "first_impact_at": st[i],
                "resolution_at_30d": (res + DAY).tz_localize("UTC") if pd.notna(res) else pd.NaT,
                "hours_to_resolution_30d": (res + pd.Timedelta(hours=12) - st[i].tz_localize(None)).total_seconds() / 3600 if pd.notna(res) else np.nan,
                "window_end_30d": w_end.tz_localize("UTC"),
                "resolution_at": (res_ns + DAY).tz_localize("UTC") if pd.notna(res_ns) else pd.NaT,
                "hours_to_resolution": (res_ns + pd.Timedelta(hours=12) - st[i].tz_localize(None)).total_seconds() / 3600 if pd.notna(res_ns) else np.nan,
                "window_end": n_end.tz_localize("UTC"),
                "restored_same_day": bool((rd_ == np.datetime64(b0)).any()),
                "restored_link": link_r.get((o, res), None) if pd.notna(res) else None,
                "status_30d": "observed" if pd.notna(res) else "censored",
            })
    cols = ["strike_at", "idx_all", "major", "oblast", "coverage", "blackout_day", "blackout_consumers", "blackout_sources", "first_impact_at",
            "resolution_at_30d", "hours_to_resolution_30d", "window_end_30d", "resolution_at", "hours_to_resolution", "window_end",
            "restored_same_day", "restored_link", "status_30d"]
    return pd.DataFrame(rows, columns=cols)


def strike_types(path: Path) -> pd.DataFrame:
    """strike_at -> type ('air_force' = a strike of the Air Force list, 'derived' = from attack reports of the channels)."""
    df = pd.read_csv(path)
    time = df["time_utc"].fillna("00:00") if "time_utc" in df.columns else "00:00"
    df["strike_at"] = pd.to_datetime(df["date"].astype(str) + " " + time, utc=True)
    t = np.where(df["derived"].astype(str).str.lower().isin(["true", "1"]), "derived", "air_force") if "derived" in df else "air_force"
    return pd.DataFrame({"strike_at": df["strike_at"], "type": t}).drop_duplicates("strike_at")


def _strikes_with_type(cfg: Config) -> pd.DataFrame:
    s = load_strikes(cfg.strike_list_path)
    return s.merge(strike_types(cfg.strike_list_path), on="strike_at", how="left")


def episodes_for_cfg(cfg: Config, strikes: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    """Episode table of the configured label source (`telegram.region_label_source`) in the format of `telegram_region_episodes.parquet`
    that `region_restoration.build_table` expects.  `energy_map` is DEV ONLY."""
    tc = cfg.telegram
    pdir = cfg.telegram_processed_dir
    source = source or tc.region_label_source
    if source == "telegram":
        return pd.read_parquet(pdir / "telegram_region_episodes.parquet")
    if source != "energy_map":
        raise ValueError(f"telegram.region_label_source must be telegram or energy_map, got {source!r}")
    rec = em.load_energy_map(cfg._abs(cfg.paths.data_dir) / "raw" / "energy_map")
    eps = build_labels(strikes, rec)
    tg = pd.read_parquet(pdir / "telegram_region_episodes.parquet")[["strike_at", "oblast", "coverage"]]
    eps = eps.drop(columns="coverage").merge(tg, on=["strike_at", "oblast"], how="left")
    eps["coverage"] = eps["coverage"].fillna("text_only")
    return eps


# --------------------------------------------------------------------------------------------------
# descriptive coverage and comparison with Telegram (dev period only for anything about durations)
# --------------------------------------------------------------------------------------------------
def _group_label(o: str) -> str:
    return "frontline" if o in em.FRONTLINE else "rear"


def coverage_tables(labels: pd.DataFrame, strikes: pd.DataFrame, tg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Counts only (no durations): pairs with an Energy Map label by year, group and strike type, next to the Telegram cohort."""
    s = strikes.assign(year=pd.to_datetime(strikes["strike_at"], utc=True).dt.year)
    lab = labels.merge(s[["strike_at", "type", "year"]], on="strike_at", how="left")
    lab["group"] = lab["oblast"].map(_group_label)
    tgl = tg[tg["status_30d"].isin(["observed", "censored"])].merge(s[["strike_at", "type", "year"]], on="strike_at", how="left")
    tgl["group"] = tgl["oblast"].map(_group_label)
    n_ob = {g: sum(_group_label(o) == g for o in OBLASTS) for g in ("frontline", "rear")}

    def agg(by):
        a = lab.groupby(by).agg(em_pairs=("oblast", "size"), em_observed=("status_30d", lambda x: int((x == "observed").sum())))
        b = tgl.groupby(by).agg(tg_pairs=("oblast", "size"), tg_observed=("status_30d", lambda x: int((x == "observed").sum())))
        out = a.join(b, how="outer").fillna(0).astype(int)
        return out.reset_index()
    out = {"year": agg(["year"]), "group": agg(["group"]), "type": agg(["type"]), "year_group": agg(["year", "group"]),
           "year_type": agg(["year", "type"]), "oblast": agg(["oblast"])}
    out["oblast"]["group"] = out["oblast"]["oblast"].map(_group_label)
    out["n_oblasts"] = pd.Series(n_ob)
    return out


def compare_with_telegram(labels: pd.DataFrame, tg: pd.DataFrame, cut: pd.Timestamp) -> dict:
    """Dev-period comparison on pairs that have BOTH an Energy Map label and a Telegram label (status observed / censored, 30 d)."""
    from scipy.stats import pearsonr, spearmanr
    e = labels[labels["strike_at"] <= cut]
    t = tg[(tg["strike_at"] <= cut) & tg["status_30d"].isin(["observed", "censored"])]
    m = e.merge(t[["strike_at", "oblast", "status_30d", "hours_to_resolution_30d"]], on=["strike_at", "oblast"], suffixes=("_em", "_tg"))
    out = {"n_em_dev": len(e), "n_tg_dev": len(t), "n_overlap": len(m)}
    if not len(m):
        return out
    ct = pd.crosstab(m["status_30d_em"], m["status_30d_tg"]).reindex(index=["observed", "censored"], columns=["observed", "censored"], fill_value=0)
    n = ct.to_numpy().sum()
    po = (ct.iloc[0, 0] + ct.iloc[1, 1]) / n
    pe = (ct.sum(1) * ct.sum(0)).sum() / n ** 2
    out.update({"crosstab": ct, "agree": float(po), "kappa": float((po - pe) / (1 - pe)) if pe < 1 else np.nan})
    both = m[(m["status_30d_em"] == "observed") & (m["status_30d_tg"] == "observed")]
    out["n_both_obs"] = len(both)
    if len(both) >= 5:
        a, b = both["hours_to_resolution_30d_em"].to_numpy(float), both["hours_to_resolution_30d_tg"].to_numpy(float)
        out.update({"spearman": float(spearmanr(a, b)[0]), "pearson_log": float(pearsonr(np.log(np.clip(a, 1, None)), np.log(np.clip(b, 1, None)))[0]),
                    "median_em_h": float(np.median(a)), "median_tg_h": float(np.median(b))})
    by = {}
    for g, mm in m.assign(group=m["oblast"].map(_group_label)).groupby("group"):
        c = mm[(mm["status_30d_em"] == "observed") & (mm["status_30d_tg"] == "observed")]
        by[g] = {"n": len(mm), "agree": float((mm["status_30d_em"] == mm["status_30d_tg"]).mean()), "n_both_obs": len(c),
                 "spearman": float(spearmanr(c["hours_to_resolution_30d_em"], c["hours_to_resolution_30d_tg"])[0]) if len(c) >= 5 else np.nan}
    out["by_group"] = by
    return out


# --------------------------------------------------------------------------------------------------
# dev CV
# --------------------------------------------------------------------------------------------------
def dev_cv_for_subset(tab: pd.DataFrame, tc, k: int, exclude: tuple[str, ...] = ()) -> dict | None:
    t = tab[~tab["oblast"].isin(exclude)] if exclude else tab
    dev = t[t["split"] == "validation"]
    if int(dev["observed"].sum()) < MIN_OBSERVED:
        return {"skipped": True, "n_observed": int(dev["observed"].sum()), "n_episodes": len(dev)}
    return rd.dev_cv(t, tc.region_alphas, int(tc.region_dev_min_obs), tc.region_min_oblast, k, thresholds=(float(tc.region_long_primary),))


def _f(x, nd=1):
    return "" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _md(df: pd.DataFrame) -> list[str]:
    cols = list(df.columns)
    return ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] + \
           ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]


def _cv_section(name: str, r: dict) -> list[str]:
    L = [f"### {name}", ""]
    if r.get("skipped"):
        return L + [f"Skipped: only {r['n_observed']} observed dev pairs (< {MIN_OBSERVED}).", ""]
    i, s, t, rk = r["info"], r["selection"], r["table"], r["ranking"]
    L += [f"Dev pairs {i['n_dev_episodes']} ({i['n_dev_observed']} observed); strikes scored {i['n_strikes_scored']} of {i['n_strikes_total']} "
          f"({i['first_scored']:%Y-%m-%d} .. {i['last_scored']:%Y-%m-%d}). Folds: " +
          "; ".join(f"{f['strikes']} strikes / {f['observed']} observed" for f in i["fold_sizes"]), ""]
    L += [f"Best CV MAE `{s['best']}` {s['best_mae']:.1f} h (SE {s['best_se']:.1f}); 1-SE selected `{s['selected']}` ({s['selected_mae']:.1f} h).", ""]
    byc = t[~t["method"].isin(rd.REFERENCE_ONLY)].sort_values("C_cv", ascending=False)
    top = list(byc["method"].head(6))
    keep = top + [m for m in rd.SELECTABLE_BASELINES + rd.REFERENCE_ONLY if m not in top]
    L += ["Primary metric: mean C within strike over folds (SE over folds). Top 6 by C plus baselines:", ""]
    L += rd._md_table(t.set_index("method").loc[keep].reset_index(), i["base"]) + [""]
    q = rk[rk["method"].isin(list(top[:3]) + ["oblast_median", "global_median"])]
    L += ["Bootstrap over strikes (dev, descriptive; p-values not corrected for the choice among designs):", ""]
    L += _md(pd.DataFrame({"design": q["method"], "C_within [95% CI]": [f"{x.C_within:.3f} [{x.C_within_lo:.3f}, {x.C_within_hi:.3f}]" for x in q.itertuples()],
                           "p (C>0.5)": [f"{x.p_C_within_gt_half:.3f}" for x in q.itertuples()]})) + [""]
    return L


def run_energy_map_dev(cfg: Config, k: int | None = None) -> dict:
    """Coverage, comparison with Telegram and the dev CV with the Energy Map target; writes reports/<scenario>/energy_map_labels_dev.md."""
    tc = cfg.telegram
    k = int(k or tc.region_cv_folds)
    root = cfg._abs(cfg.paths.data_dir)
    pdir = cfg.telegram_processed_dir
    strikes = _strikes_with_type(cfg)
    rec = em.load_energy_map(root / "raw" / "energy_map")
    labels = build_labels(strikes, rec)
    tg = pd.read_parquet(pdir / "telegram_region_episodes.parquet")
    vend = pd.Timestamp(tc.region_validation_end, tz="UTC")
    cut = vend + pd.Timedelta(hours=23)
    cov = coverage_tables(labels, strikes, tg)
    cmp_ = compare_with_telegram(labels, tg, cut)
    labels.to_parquet(pdir / "energy_map_region_labels.parquet", index=False)

    facts = pd.read_parquet(pdir / "telegram_region_posts.parquet")
    wpath = root / "raw" / "weather.parquet"
    weather = pd.read_parquet(wpath) if wpath.exists() else None
    sdev = strikes[pd.to_datetime(strikes["strike_at"], utc=True) <= cut].reset_index(drop=True)      # test-period strikes never enter
    eps = episodes_for_cfg(cfg, strikes, "energy_map")
    eps = eps[pd.to_datetime(eps["strike_at"], utc=True) <= cut]
    tab = rr.build_table(eps, facts, sdev, weather, vend, "30d", tc.region_early_hours)
    tab = tab[tab["split"] == "validation"]
    res = {"all oblasts": dev_cv_for_subset(tab, tc, k), "excluding front-line (Donetsk, Luhansk, Zaporizhzhia, Kherson)": dev_cv_for_subset(tab, tc, k, em.FRONTLINE)}
    write_report(cfg.reports_dir / "energy_map_labels_dev.md", rec, labels, cov, cmp_, res, tab, strikes, cut, k)
    return {"labels": labels, "coverage": cov, "comparison": cmp_, "cv": res}


def write_report(path, rec, labels, cov, cmp_, res, tab, strikes, cut, k) -> None:
    ev = labels[labels["strike_at"] <= cut]
    L = ["# Energy Map labels for the oblast restoration forecast: DEV report (test period blind)", "",
         "Source: Energy Map / DiXi Group, https://energy-map.info (free account; files `*_oblasti_osr.csv` = regional administrations and DSOs, "
         "`*_oblasti_minenergo.csv` = Ministry of Energy). These are human-curated event records, not a continuous series: a missing record means "
         "'not reported', not 'no outage'.", "",
         "Label (see `energy_map_labels.py`): a (strike, oblast) pair is labelled iff a blackout is reported on the strike day or the day after; "
         f"restoration = first later record with restored consumers/settlements within {WINDOW_DAYS} days (observed) else censored. Period records "
         "(number_days > 1) are ignored. The notes never contain an 'all restored' statement, so the label is the FIRST RESTORATION REPORT "
         "(possibly partial), a different quantity from the Telegram label (schedules lifted / all consumers have power). Time resolution is one day. "
         "The cohort is defined by the reported blackout (label side), so results are conditional on it.", "",
         "Splits decided BEFORE looking at any model result: all oblasts, and excluding the front-line oblasts Donetsk, Luhansk, Zaporizhzhia, "
         "Kherson (their records are dominated by daily shelling outages). Both are reported whatever the outcome. No model metric for strikes "
         "from 2025-01-01 is computed; below the 2025+ numbers are label counts only.", "",
         f"Records loaded: {len(rec)} ({int((rec['source'] == 'osr').sum())} osr, {int((rec['source'] == 'minenergo').sum())} minenergo); "
         f"{int(rec['is_period'].sum())} period records ignored; with blackout {int((rec['has_blackout'] & ~rec['is_period']).sum())}, "
         f"with restoration {int((rec['has_restored'] & ~rec['is_period']).sum())}.", "",
         "## 1. Coverage (counts only)", "",
         f"Strikes: {len(strikes)}; (strike x oblast) pairs with an Energy Map label: {len(labels)} "
         f"({int((labels['status_30d'] == 'observed').sum())} observed, {int((labels['status_30d'] == 'censored').sum())} censored). "
         f"{int(labels['restored_same_day'].sum())} pairs also have a restoration record on the first blackout day (not used as restoration). "
         "`em_*` = Energy Map, `tg_*` = Telegram pairs with status observed/censored (30 d window) on the same grid.", ""]
    for key, title in (("year", "By year"), ("group", "By region group (front-line = Donetsk, Luhansk, Zaporizhzhia, Kherson; rear = other 20)"),
                       ("type", "By strike type (air_force = Air Force list, derived = from attack-report posts)"),
                       ("year_group", "Year x group"), ("year_type", "Year x strike type"), ("oblast", "By oblast")):
        L += [f"### {title}", ""] + _md(cov[key]) + [""]
    dev_obs = int((ev["status_30d"] == "observed").sum())
    L += [f"Dev period (strikes up to {cut:%Y-%m-%d}): {len(ev)} labelled pairs, {dev_obs} observed "
          f"({int((ev[ev['oblast'].isin(em.FRONTLINE)]['status_30d'] == 'observed').sum())} front-line, "
          f"{int((ev[~ev['oblast'].isin(em.FRONTLINE)]['status_30d'] == 'observed').sum())} rear). Threshold for running the CV: >= {MIN_OBSERVED} observed.", "",
          "## 2. Comparison with the Telegram labels (dev period, overlapping pairs)", ""]
    c = cmp_
    L += [f"Dev pairs: Energy Map {c['n_em_dev']}, Telegram (observed/censored) {c['n_tg_dev']}, overlap {c['n_overlap']}.", ""]
    if c.get("crosstab") is not None:
        ct = c["crosstab"]
        L += [f"Observed/censored agreement: {100 * c['agree']:.1f}% (Cohen kappa {c['kappa']:.3f}). Rows Energy Map, columns Telegram:", ""]
        L += _md(ct.reset_index().rename(columns={"status_30d_em": "EM \\ TG"})) + [""]
        L += [f"Both observed: {c['n_both_obs']} pairs; " + (f"Spearman of durations {c['spearman']:.3f}, Pearson of log durations {c['pearson_log']:.3f}; "
              f"median duration Energy Map {c['median_em_h']:.0f} h vs Telegram {c['median_tg_h']:.0f} h." if "spearman" in c else "too few for a correlation."), ""]
        for g, v in c.get("by_group", {}).items():
            L.append(f"- {g}: {v['n']} overlapping pairs, agreement {100 * v['agree']:.1f}%, both observed {v['n_both_obs']}, Spearman {_f(v['spearman'], 3)}")
        L.append("")
    L += ["## 3. Dev CV with the Energy Map target", "",
          f"Rolling-origin CV exactly as `region_dev.dev_cv` (expanding window over strikes, K = {k}, warm-up `region_dev_min_obs`, 1-SE rule on MAE, "
          "mean C within strike as primary metric, MAE paired against `global_median`). Window 30 d only (the next-strike variant is almost all "
          "censored here). Features are the same Telegram/strike/oblast/history features as in the Telegram runs; only the label (and the cohort) differs.", ""]
    for name, r in res.items():
        L += _cv_section(name, r)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L), encoding="utf-8")
