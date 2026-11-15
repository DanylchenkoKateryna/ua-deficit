"""Do air-raid alerts and weather of the oblast help the Energy Map restoration forecast?  (DEV ONLY: CV over all strikes up to
`energy_map_test_end`, exactly as `energy_map_improve`; the whole period is development data; the blind test of
`docs/preregistration_energy_map_v2_draft.md` is not touched.)

Features per (strike, oblast), all at the issue time (end of the strike day D, UTC); nothing after D enters:
  al_h_day    hours under alert in the oblast on D (clipped to the issue time; = 24 x the share of D under alert, so no separate share column)
  al_h_prev   hours under alert on D - 1
  al_h_7d     hours under alert on D - 7 .. D - 1
  al_n_day    number of separate alerts (merged intervals) that start on D
  w_tmin_d    minimum temperature of D at the oblast centre;   w_tmin_3d  minimum over D - 2 .. D
  w_frost_d   1 if w_tmin_d < -5 C;                              w_frost_3d 1 if w_tmin_3d < -5 C
  w_gust_d    maximum wind gust of D;                            w_gust_3d  maximum over D - 2 .. D
  w_snow_d    snowfall of D;                                     w_snow_3d  snowfall sum over D - 2 .. D
Weather of D is observed by the end of D (acceptable at the issue time, README §6).  Alerts: union of oblast / raion / hromada
intervals (`ingestion/alerts.py`), a still-running alert counts only up to the issue time.
Feature sets: <base>_al (+ alerts), <base>_w (+ weather), <base>_alw (+ both) for the previous best designs of `energy_map_improve`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.alerts import AlertIndex, load_alerts
from ..ingestion.weather_oblast import load_weather_oblast
from . import energy_map_improve as ei
from . import energy_map_test as emt
from . import region_dev as rd
from . import region_restoration as rr

log = logging.getLogger(__name__)

AL_F = ["al_h_day", "al_h_prev", "al_h_7d", "al_n_day"]
W_F = ["w_tmin_d", "w_tmin_3d", "w_frost_d", "w_frost_3d", "w_gust_d", "w_gust_3d", "w_snow_d", "w_snow_3d"]
FROST_C = -5.0
DAY = pd.Timedelta(days=1)
rr.CONTINUOUS.update(set(AL_F) | {"w_tmin_d", "w_tmin_3d", "w_gust_d", "w_gust_3d", "w_snow_d", "w_snow_3d"})      # frost flags stay 0/1

# base designs of `energy_map_improve` (best by C / by MAE / the C-selected tree model) and the extra sets
BASES = {"ipcw": ("ipcw", "so_all", 100.0), "ridge": ("ridge", "so_em", 100.0), "xgbaft": ("xgbaft", "so_em", (2, 25))}
SUFFIX = {"": [], "_al": AL_F, "_w": W_F, "_alw": AL_F + W_F}
OLD_MODEL = ei.OLD_MODEL


def new_sets() -> dict[str, list[str]]:
    out = {}
    for fam, (_, s, _) in BASES.items():
        for suf, extra in SUFFIX.items():
            if suf:
                out[s + suf] = ei.SETS[s] + extra
    out["so_alw"] = ei.SETS["so"] + AL_F + W_F
    return out


def _cols(s: str) -> list[str]:
    return list(ei.SETS[s]) if s in ei.SETS else list(new_sets()[s])      # column lists travel with the design (joblib workers do not share module state)


def design_names(fam: str, s: str, p) -> str:
    return f"{fam}_{s}_a{p:g}" if fam != "xgbaft" else f"{fam}_{s}_d{p[0]}n{p[1]}"


def designs() -> list[tuple[str, str, str, object]]:
    out = []
    for fam, (f, s, p) in BASES.items():
        for suf in SUFFIX:
            out.append((design_names(f, s + suf, p), f, _cols(s + suf), p))
    out.append(("ridge_so_alw_a100", "ridge", _cols("so_alw"), 100.0))
    out.append((OLD_MODEL, "ridge", _cols("so"), 100.0))
    return out


# --------------------------------------------------------------------------------------------------
# features at the issue time
# --------------------------------------------------------------------------------------------------
def alert_features(tab: pd.DataFrame, idx: AlertIndex) -> pd.DataFrame:
    """Needs `oblast` and `issue` (tz-aware UTC end of the strike day).  Every window ends at or before `issue`."""
    rows = []
    for o, iss in zip(tab["oblast"], pd.to_datetime(tab["issue"], utc=True)):
        d0 = iss - DAY
        rows.append((idx.hours(o, d0, iss), idx.hours(o, d0 - DAY, d0), idx.hours(o, d0 - 7 * DAY, d0), float(idx.count(o, d0, iss))))
    return pd.DataFrame(rows, columns=AL_F, index=tab.index)


def weather_features(tab: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Daily weather of the strike day D and of D - 2, D - 1 (a missing day gives NaN, imputed by the training median)."""
    by = {o: g.set_index("date").sort_index() for o, g in weather.groupby("oblast")}
    rows = []
    for o, iss in zip(tab["oblast"], pd.to_datetime(tab["issue"], utc=True)):
        D = (iss - DAY).tz_convert(None)
        w = by.get(o)
        days = [D - 2 * DAY, D - DAY, D]
        if w is None:
            rows.append([np.nan] * len(W_F))
            continue
        x = w.reindex(days)
        tmin_d, tmin_3 = x["temperature_2m_min"].iloc[-1], x["temperature_2m_min"].min()
        rows.append([tmin_d, tmin_3, float(tmin_d < FROST_C) if pd.notna(tmin_d) else np.nan,
                     float(tmin_3 < FROST_C) if pd.notna(tmin_3) else np.nan,
                     x["wind_gusts_10m_max"].iloc[-1], x["wind_gusts_10m_max"].max(),
                     x["snowfall_sum"].iloc[-1], x["snowfall_sum"].sum(min_count=1)])
    return pd.DataFrame(rows, columns=W_F, index=tab.index)


def add_alert_weather_features(tab: pd.DataFrame, idx: AlertIndex, weather: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([tab, alert_features(tab, idx), weather_features(tab, weather)], axis=1)


# --------------------------------------------------------------------------------------------------
# paired comparison and run
# --------------------------------------------------------------------------------------------------
def paired_vs(sc: ei.Scorer, folds: list[list], m: str, base: str) -> dict:
    sc.prep([m, base])
    c, a = sc.fold_stats(m, folds)
    cb, ab = sc.fold_stats(base, folds)
    dc, da = c - cb, a - ab
    return {"dC": float(np.nanmean(dc)), "dC_se": ei._se(dc), "dC_better": int((dc > 0).sum()), "dMAE": float(np.nanmean(da)),
            "dMAE_se": ei._se(da), "dMAE_better": int((da < 0).sum()), "n": len(folds)}


def run_alerts_weather(cfg: Config, k: int | None = None, n_boot: int = 2000, seed: int = 42, n_jobs: int = 6) -> dict:
    tc = cfg.telegram
    k = int(k or 8)
    root = cfg._abs(cfg.paths.data_dir) / "raw"
    tab = ei.build_dev_table(cfg)
    al = load_alerts(root / "alerts" / "official_data_en.csv")
    wx = load_weather_oblast(root / "weather_oblast")
    tab = add_alert_weather_features(tab, AlertIndex(al), wx)
    des = designs()
    wf = ei.walk_forward(tab, des, int(tc.region_dev_min_obs), int(tc.region_min_oblast), n_jobs)
    ev = wf[wf["evaluated"]].copy()
    names = [d[0] for d in des]
    folds = rd.make_folds(list(ev["idx"].unique()), k)
    sc = ei.Scorer(ev)
    t = ei.cv_table(sc, folds, ei.REFERENCES + names)
    pv = {}
    for fam, (f, s, p) in BASES.items():
        base = design_names(f, s, p)
        for suf in ("_al", "_w", "_alw"):
            m = design_names(f, s + suf, p)
            pv[m] = {"base": base, **paired_vs(sc, folds, m, base)}
    pv["ridge_so_alw_a100"] = {"base": OLD_MODEL, **paired_vs(sc, folds, "ridge_so_alw_a100", OLD_MODEL)}
    boot = {}
    for m, v in pv.items():
        boot[m] = {"om": emt.paired_c_bootstrap(ev, m, "oblast_median", n_boot, seed), "base": emt.paired_c_bootstrap(ev, m, v["base"], n_boot, seed),
                   "mae_om": ei.paired_mae_bootstrap(ev, m, "oblast_median", n_boot, seed), "mae_base": ei.paired_mae_bootstrap(ev, m, v["base"], n_boot, seed)}
    cover = {"pairs": len(tab), "al_nonzero_day": float((tab["al_h_day"] > 0).mean()), "al_mean_h_day": float(tab["al_h_day"].mean()),
             "w_missing": float(tab["w_tmin_d"].isna().mean()), "frost_d": float(tab["w_frost_d"].mean()),
             "alerts_records": len(al), "weather_oblasts": int(wx["oblast"].nunique()), "weather_last": wx["date"].max(),
             "alerts_first": al["start"].min(), "alerts_last": al["start"].max()}
    out = {"table": t, "pv": pv, "boot": boot, "folds": folds, "cover": cover, "ev": ev,
           "info": {"pairs": len(tab), "observed": int(tab["observed"].sum()), "strikes": int(tab["idx"].nunique()), "scored": int(ev["idx"].nunique()),
                    "first": ev["strike_at"].min(), "last": ev["strike_at"].max(), "k": k, "n_boot": n_boot, "min_obs": int(tc.region_dev_min_obs),
                    "fold_sizes": [(len(ids), int(ev["idx"].isin(ids).sum()), int((ev["idx"].isin(ids) & ev["observed"]).sum())) for ids in folds]}}
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    t.to_csv(d / "energy_map_alerts_weather_dev_cv.csv", index=False)
    write_report(d / "energy_map_alerts_weather_dev.md", out)
    return out


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
def _f(x, nd=3):
    return "" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _verdict(v: dict, b: dict) -> str:
    """Rule fixed in the code: 'helps' needs mean dC > 2 SE, >= 6/8 folds better and a bootstrap CI of dC (vs the own base) above 0."""
    ok_c = v["dC"] > 2 * (v["dC_se"] or 0) and v["dC_better"] >= 6 and b["base"]["diff_lo"] > 0
    ok_m = v["dMAE"] < -2 * (v["dMAE_se"] or 0) and v["dMAE_better"] >= 6 and b["mae_base"]["hi"] < 0
    bad_c = b["base"]["diff_hi"] < 0
    bad_m = b["mae_base"]["lo"] > 0
    return ("C helps" if ok_c else "C hurts" if bad_c else "C no clear effect") + "; " + ("MAE helps" if ok_m else "MAE hurts" if bad_m else "MAE no clear effect")


def write_report(path, o: dict) -> None:
    t, i, pv, bt, cv = o["table"], o["info"], o["pv"], o["boot"], o["cover"]
    x = t.set_index("method")
    L = ["# Alerts and weather as features of the Energy Map restoration forecast (DEV; rolling-origin CV over ALL strikes up to 2026-06-26)", "",
         "Same label, cohort, walk-forward, folds and metrics as `energy_map_improve_dev.md` (the frozen test of `docs/preregistration_energy_map.md` is spent; "
         "the whole period is development data; the blind test of `docs/preregistration_energy_map_v2_draft.md` is untouched). Only the feature sets differ.", "",
         "Sources (downloaded 2026-11-09): air-raid alerts, Vadimkin/ukrainian-air-raid-sirens-dataset `official_data_en.csv` (MIT), https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset; "
         "daily weather at the oblast centre, Open-Meteo archive API (CC BY 4.0), https://open-meteo.com.", "",
         f"Dev pairs {i['pairs']} ({i['observed']} observed) in {i['strikes']} strikes; scored {i['scored']} strikes ({i['first']:%Y-%m-%d} .. {i['last']:%Y-%m-%d}); "
         f"{i['k']} folds (strikes / pairs / observed): " + "; ".join(f"{a} / {b} / {c}" for a, b, c in i["fold_sizes"]) + ".", "",
         "## 1. Features (issue time = end of the strike day D, UTC)", "",
         "- Alerts (`al_*`): union of the intervals of the oblast, its raions and hromadas (no double counting; a hromada alert counts as the oblast being under alert), "
         "clipped to the issue time: `al_h_day` hours under alert on D (= 24 x the share of D under alert, so no separate share column), `al_h_prev` on D-1, "
         "`al_h_7d` on D-7..D-1, `al_n_day` number of separate alerts starting on D.",
         "- Weather (`w_*`) at the oblast centre: minimum temperature of D and the minimum over D-2..D, frost flags (< -5 C) for both, maximum gust of D and of D-2..D, "
         "snowfall of D and the sum over D-2..D. Weather of D is observed by the end of D (acceptable, README §6). Nothing after D is used (`tests/test_no_leakage.py`).",
         f"- Coverage: {cv['alerts_records']} alert records ({cv['alerts_first']:%Y-%m-%d} .. {cv['alerts_last']:%Y-%m-%d} starts), weather for {cv['weather_oblasts']} oblasts up to "
         f"{cv['weather_last']:%Y-%m-%d}; over the {cv['pairs']} dev pairs the mean `al_h_day` is {cv['al_mean_h_day']:.1f} h (alert on D in {100 * cv['al_nonzero_day']:.0f}% of pairs), "
         f"frost on D in {100 * cv['frost_d']:.0f}%, missing weather {100 * cv['w_missing']:.1f}%.",
         "- Designs: the previous best designs `ipcw_so_all_a100` (best MAE), `ridge_so_em_a100`, `xgbaft_so_em_d2n25` (selected by C) and the same with `_al` (+alerts), `_w` (+weather), "
         "`_alw` (+both); `ridge_so_alw_a100` = the old `ridge_so_a100` + both. Only 12 designs + 3 references, no tuning of alphas or tree sizes on the new sets. "
         "The rows are fixed in advance, so there is no selection among them, but the numbers still come from the same dev CV as the earlier designs.", "",
         "## 2. CV table (8 folds)", "",
         "Cells: mean over folds (SE over folds); dC > 0 and dMAE < 0 are better; 'folds better' counts folds. 'vs base' = paired against the design without the new features.", "",
         "| design | C_cv (SE) | dC vs oblast_median (SE), folds better | MAE_cv h (SE) | dMAE vs oblast_median h (SE), folds better | dC vs base (SE), folds better | dMAE vs base h (SE), folds better |",
         "|---|---|---|---|---|---|---|"]
    n = len(o["folds"])
    for fam, (f, s, p) in BASES.items():
        for suf in SUFFIX:
            m = design_names(f, s + suf, p)
            r = x.loc[m]
            if suf:
                v = pv[m]
                pb = f"{_f(v['dC'])} ({_f(v['dC_se'])}), {v['dC_better']}/{n}"
                pm = f"{_f(v['dMAE'], 1)} ({_f(v['dMAE_se'], 1)}), {v['dMAE_better']}/{n}"
            else:
                pb = pm = "(base)"
            L.append(f"| {m} | {_f(r['C_cv'])} ({_f(r['C_cv_se'])}) | {_f(r['dC_om'])} ({_f(r['dC_om_se'])}), {int(r['dC_om_better'])}/{n} | "
                     f"{_f(r['MAE_cv_h'], 1)} ({_f(r['MAE_cv_se_h'], 1)}) | {_f(r['dMAE_om_h'], 1)} ({_f(r['dMAE_om_se'], 1)}), {int(r['dMAE_om_better'])}/{n} | {pb} | {pm} |")
    for m in ("ridge_so_alw_a100", OLD_MODEL):
        r = x.loc[m]
        if m in pv:
            v = pv[m]
            pb = f"{_f(v['dC'])} ({_f(v['dC_se'])}), {v['dC_better']}/{n} (vs {OLD_MODEL})"
            pm = f"{_f(v['dMAE'], 1)} ({_f(v['dMAE_se'], 1)}), {v['dMAE_better']}/{n}"
        else:
            pb = pm = "(base)"
        L.append(f"| {m} | {_f(r['C_cv'])} ({_f(r['C_cv_se'])}) | {_f(r['dC_om'])} ({_f(r['dC_om_se'])}), {int(r['dC_om_better'])}/{n} | "
                 f"{_f(r['MAE_cv_h'], 1)} ({_f(r['MAE_cv_se_h'], 1)}) | {_f(r['dMAE_om_h'], 1)} ({_f(r['dMAE_om_se'], 1)}), {int(r['dMAE_om_better'])}/{n} | {pb} | {pm} |")
    for m in ei.REFERENCES:
        r = x.loc[m]
        L.append(f"| {m} | {_f(r['C_cv'])} ({_f(r['C_cv_se'])}) | {_f(r['dC_om'])} ({_f(r['dC_om_se'])}), {int(r['dC_om_better'])}/{n} | "
                 f"{_f(r['MAE_cv_h'], 1)} ({_f(r['MAE_cv_se_h'], 1)}) | {_f(r['dMAE_om_h'], 1)} ({_f(r['dMAE_om_se'], 1)}), {int(r['dMAE_om_better'])}/{n} | | |")
    L += ["", "## 3. Ablation: alerts only, weather only, both (the `_al`, `_w`, `_alw` rows above; paired against the base)", "",
          "| family | suffix | dC vs base (SE) | folds better | pooled-bootstrap dC vs base [95% CI] | dMAE vs base h (SE) | folds better | pooled-bootstrap dMAE [95% CI] | verdict by the fixed rule |",
          "|---|---|---|---|---|---|---|---|---|"]
    for fam, (f, s, p) in BASES.items():
        for suf in ("_al", "_w", "_alw"):
            m = design_names(f, s + suf, p)
            v, b = pv[m], bt[m]
            L.append(f"| {fam} ({s}) | {suf} | {_f(v['dC'], 4)} ({_f(v['dC_se'], 4)}) | {v['dC_better']}/{n} | {b['base']['diff']:+.4f} [{b['base']['diff_lo']:+.4f}, {b['base']['diff_hi']:+.4f}] | "
                     f"{_f(v['dMAE'], 1)} ({_f(v['dMAE_se'], 1)}) | {v['dMAE_better']}/{n} | {b['mae_base']['diff']:+.1f} [{b['mae_base']['lo']:+.1f}, {b['mae_base']['hi']:+.1f}] | {_verdict(v, b)} |")
    L += ["", "Fixed rule (written before the run): an effect is called 'helps' only if the mean per-fold difference is beyond 2 SE, at least 6 of 8 folds agree and the pooled bootstrap "
          "CI (strikes resampled, paired) excludes 0; 'hurts' if the CI is entirely on the wrong side; otherwise 'no clear effect'.", "",
          "## 4. Pooled bootstrap over strikes against oblast_median (all scored dev strikes, %d resamples, seed 42)" % i["n_boot"], "",
          "| design | C pooled | C oblast_median | dC [95% CI] | MAE pooled h | dMAE vs oblast_median h [95% CI] |", "|---|---|---|---|---|---|"]
    obs = o["ev"][o["ev"]["observed"]]
    for m in list(pv):
        b = bt[m]["om"]
        mb = bt[m]["mae_om"]
        mp = float((obs[m] - obs["hours"]).abs().mean())
        L.append(f"| {m} | {b['C_model']:.3f} | {b['C_base']:.3f} | {b['diff']:+.4f} [{b['diff_lo']:+.4f}, {b['diff_hi']:+.4f}] | {mp:.1f} | {mb['diff']:+.1f} [{mb['lo']:+.1f}, {mb['hi']:+.1f}] |")
    L += ["", "## 5. Verdict", "", "VERDICT_PLACEHOLDER", "",
          "## 6. Caveats", "",
          "- All numbers are development numbers (the whole period is dev); nothing here is a blind result. With 12 designs the table is small, but the earlier ~67-design search chose the bases on the same CV.",
          "- Alerts are a union of oblast, raion and hromada alerts: a hromada alert in a large oblast counts as the oblast being under alert, and the share of raion/hromada records grows over time, so the alert hours are not homogeneous across the years.",
          "- Weather is taken at one point (the oblast centre) and the strike-day value is observed by the end of the day (assumption of README §6). The day of a strike is also the day of its attack, so alerts of D partly describe the strike itself.",
          "- The label is the first restoration report (maybe partial) in the Energy Map event records, day resolution; the cohort (blackout reported on D or D+1) is label-side.", ""]
    txt = "\r\n".join(L)
    txt = txt.replace("VERDICT_PLACEHOLDER", verdict_text(o))
    Path(path).write_text(txt, encoding="utf-8")


def verdict_text(o: dict) -> str:
    pv, bt = o["pv"], o["boot"]
    lines = []
    for m, v in pv.items():
        if not m.endswith(("_alw_a100", "_alw_d2n25")):
            continue
        lines.append(f"- `{m}` vs `{v['base']}`: dC {v['dC']:+.4f} (SE {v['dC_se']:.4f}), {v['dC_better']}/{v['n']} folds; dMAE {v['dMAE']:+.1f} h (SE {v['dMAE_se']:.1f}), "
                     f"{v['dMAE_better']}/{v['n']} folds; by the fixed rule: {_verdict(v, bt[m])}.")
    helps = [m for m, v in pv.items() if _verdict(v, bt[m]).startswith("C helps") or "MAE helps" in _verdict(v, bt[m])]
    lines.append("")
    lines.append("Conclusion: " + (f"by the fixed rule the new features help in: {', '.join(helps)}." if helps else
                 "alerts and weather (alone or together) do not improve C or MAE of any design by the fixed rule; the paired differences are within noise "
                 "(or slightly negative), so they are NOT added to the model. Not a statement that alerts or weather do not matter physically: at the "
                 "(strike, oblast) level, with ~2400 observed episodes, day resolution and a first-report label, they carry no usable extra signal beyond "
                 "oblast, strike and Energy Map history. Alerts are almost always on (alert on the strike day in ~96% of pairs), which limits their contrast."))
    return "\r\n".join(lines)
