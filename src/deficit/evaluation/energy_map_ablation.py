"""Which data are redundant or harmful for the Energy Map restoration forecast?  (DEV ONLY: the same CV as `energy_map_improve`: all strikes up to
`energy_map_test_end`, 8 expanding folds, the same scored pairs; the blind test of `docs/preregistration_energy_map_v2_draft.md` is not touched.)

1. Feature-group ablation: on the reference designs (`ipcw_so_all_a100` = best MAE, `ridge_so_em_a100`, `xgbaft_so_em_d2n25` = selected by C) one group
   is dropped at a time: strike-level Telegram features, oblast dummies, EM_F, HIST_F, GEO_F, CTX_F (only those the design contains); alerts and weather
   are ADDED (the reference designs do not contain them; same sets as `energy_map_alerts_weather`).  Plus a minimal model (oblast dummies + EM_F).
   Per design: C within strike, MAE, per-fold paired difference to the full design, pooled paired bootstrap over strikes.
2. Strike-subset ablation: the TRAINING set is (i) all strikes, (ii) Air Force strikes + derived strikes with n_channels >= 2, (iii) Air Force strikes only.
   Evaluation is always on the same pairs (all, and Air Force strikes only), so only the training set differs.  `n_channels` is a whole-day count:
   it is used ONLY to filter past training strikes (the filter is applied after `labels_as_of` to strikes j < i whose day is over before the issue time
   of strike i), never as a feature and never to select the evaluated strike.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..ingestion.alerts import AlertIndex, load_alerts
from ..ingestion.weather_oblast import load_weather_oblast
from . import energy_map_alerts_weather as eaw
from . import energy_map_improve as ei
from . import energy_map_labels as eml
from . import energy_map_test as emt
from . import region_dev as rd
from . import region_restoration as rr

log = logging.getLogger(__name__)

STRIKE_F, HIST_F, EM_F, GEO_F, CTX_F = list(rr.STRIKE_F), list(rr.HIST_F), list(ei.EM_F), list(ei.GEO_F), list(ei.CTX_F)
AL_F, W_F = list(eaw.AL_F), list(eaw.W_F)
DUM = rr.dummies()
GROUPS = {"telegram_strike": STRIKE_F, "oblast_dummies": DUM, "EM_F": EM_F, "HIST_F": HIST_F, "GEO_F": GEO_F, "CTX_F": CTX_F}
ADD_GROUPS = {"alerts": AL_F, "weather": W_F}
BASES = {"ipcw_so_all_a100": ("ipcw", "so_all", 100.0), "ridge_so_em_a100": ("ridge", "so_em", 100.0), "xgbaft_so_em_d2n25": ("xgbaft", "so_em", (2, 25))}
FAMRANK = {"ridge": 0, "ipcw": 1, "xgbaft": 2}
MIN_TRAIN = 5                      # fewer observed training episodes than this: no forecast (NaN), the strike is dropped from the subset comparison
REGIMES = {"all": "all strikes (current)", "af_ch2": "Air Force + derived with n_channels >= 2", "af": "Air Force only"}


def _ordered(cols_set: set) -> list[str]:
    full = STRIKE_F + DUM + HIST_F + EM_F + GEO_F + CTX_F + AL_F + W_F
    return [c for c in full if c in cols_set]


def design_set() -> dict[str, dict]:
    """name -> {fam, cols, p, base, dropped}"""
    out = {}
    for b, (fam, s, p) in BASES.items():
        full = list(ei.SETS[s])
        out[b] = {"fam": fam, "cols": full, "p": p, "base": b, "label": "full"}
        for g, gc in GROUPS.items():
            if set(gc) & set(full) == set(gc):
                out[f"{b}__no_{g}"] = {"fam": fam, "cols": [c for c in full if c not in gc], "p": p, "base": b, "label": f"- {g}"}
        for g, gc in ADD_GROUPS.items():
            out[f"{b}__add_{g}"] = {"fam": fam, "cols": full + gc, "p": p, "base": b, "label": f"+ {g}"}
        out[f"{b}__minimal"] = {"fam": fam, "cols": DUM + EM_F, "p": p, "base": b, "label": "minimal (dummies + EM_F)"}
    return out


# --------------------------------------------------------------------------------------------------
# table, walk-forward with a filtered training set
# --------------------------------------------------------------------------------------------------
def build_table(cfg: Config) -> pd.DataFrame:
    root = cfg._abs(cfg.paths.data_dir) / "raw"
    tab = ei.build_dev_table(cfg)
    al = load_alerts(root / "alerts" / "official_data_en.csv")
    wx = load_weather_oblast(root / "weather_oblast")
    tab = eaw.add_alert_weather_features(tab, AlertIndex(al), wx)
    sl = pd.read_csv(cfg.strike_list_path)
    time = sl["time_utc"].fillna("00:00") if "time_utc" in sl.columns else "00:00"
    sl["strike_at"] = pd.to_datetime(sl["date"].astype(str) + " " + time, utc=True)
    nch = sl.drop_duplicates("strike_at").set_index("strike_at")["n_channels"]
    tab["n_channels"] = pd.to_datetime(tab["strike_at"], utc=True).map(nch).fillna(0).astype(int)      # whole-day count: FILTER only, never a feature
    af = tab["is_af"].astype(bool)
    tab["tr_all"] = True
    tab["tr_af_ch2"] = af | (tab["n_channels"] >= 2)
    tab["tr_af"] = af
    return tab


def _one(i: int, train: pd.DataFrame, test: pd.DataFrame, designs: list, min_oblast: int) -> tuple[int, pd.DataFrame]:
    if int(train["observed"].sum()) < MIN_TRAIN:
        return i, pd.DataFrame(np.nan, index=test.index, columns=[d[0] for d in designs] + ei.REFERENCES)
    return i, ei.forecast_all(train, test, designs, min_oblast)


def walk_forward(tab: pd.DataFrame, designs: list, min_obs: int, min_oblast: int, regime: str, n_jobs: int) -> pd.DataFrame:
    """As `energy_map_improve.walk_forward`; the scored strikes are decided by the UNFILTERED training observations (same pairs for every regime);
    the filter `tr_<regime>` is applied only to the training rows (strikes j < i, labels as of the issue time of i)."""
    import os
    from joblib import Parallel, delayed

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    tab = tab.copy()
    tab["evaluated"] = False
    jobs = []
    for i in sorted(tab["idx"].unique()):
        test = tab[tab["idx"] == i]
        train = rr.labels_as_of(tab[tab["idx"] < i], test["issue"].iloc[0] - pd.Timedelta(microseconds=1))
        if int(train["observed"].sum()) < min_obs:
            continue
        jobs.append((i, train[train[f"tr_{regime}"]], test))
    res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one)(i, tr, te, designs, min_oblast) for i, tr, te in jobs)
    pred = pd.concat([p for _, p in res])
    tab.loc[pred.index, "evaluated"] = True
    return pd.concat([tab, pred.reindex(tab.index)], axis=1)


# --------------------------------------------------------------------------------------------------
# paired comparison
# --------------------------------------------------------------------------------------------------
def compare(sc: ei.Scorer, folds: list[list], m: str, base: str) -> dict:
    sc.prep([m, base])
    c, a = sc.fold_stats(m, folds)
    cb, ab = sc.fold_stats(base, folds)
    dc, da = c - cb, a - ab
    return {"C": float(np.nanmean(c)), "C_se": ei._se(c), "MAE": float(np.nanmean(a)), "MAE_se": ei._se(a), "dC": float(np.nanmean(dc)), "dC_se": ei._se(dc),
            "dC_better": int((dc > 0).sum()), "dMAE": float(np.nanmean(da)), "dMAE_se": ei._se(da), "dMAE_better": int((da < 0).sum()), "n": int(np.sum(~np.isnan(dc)))}


def boot_pair(ev: pd.DataFrame, m: str, base: str, n_boot: int, seed: int) -> dict:
    c = emt.paired_c_bootstrap(ev, m, base, n_boot, seed)
    a = ei.paired_mae_bootstrap(ev, m, base, n_boot, seed)
    return {"dC": c["diff"], "dC_lo": c["diff_lo"], "dC_hi": c["diff_hi"], "dM": a["diff"], "dM_lo": a["lo"], "dM_hi": a["hi"]}


def verdict(b: dict) -> str:
    """Removal rule fixed in the code: removing is 'safe' unless the pooled paired CI shows a worse C (upper bound < 0) or a worse MAE (lower bound > 0)."""
    worse_c, worse_m = b["dC_hi"] < 0, b["dM_lo"] > 0
    better_c, better_m = b["dC_lo"] > 0, b["dM_hi"] < 0
    if worse_c or worse_m:
        return "KEEP (" + ", ".join(x for x, f in (("C worse", worse_c), ("MAE worse", worse_m)) if f) + ")"
    if better_c or better_m:
        return "REMOVE (better: " + ", ".join(x for x, f in (("C", better_c), ("MAE", better_m)) if f) + ")"
    return "REMOVE (no difference)"


def one_se_simplest(rows: dict[str, dict], what: str) -> dict:
    key, se = ("C", "C_se") if what == "C" else ("MAE", "MAE_se")
    best = (max if what == "C" else min)(rows, key=lambda n: rows[n][key])
    s = rows[best][se]
    ok = [n for n in rows if (rows[n][key] >= rows[best][key] - s if what == "C" else rows[n][key] <= rows[best][key] + s)]
    pick = min(ok, key=lambda n: (len(rows[n]["cols"]), FAMRANK[rows[n]["fam"]], rows[n][key] * (-1 if what == "C" else 1)))
    return {"what": what, "best": best, "best_val": rows[best][key], "se": s, "n_within": len(ok), "selected": pick, "sel_val": rows[pick][key], "sel_se": rows[pick][se],
            "within": ok}


# --------------------------------------------------------------------------------------------------
def run_ablation(cfg: Config, k: int | None = None, n_boot: int = 2000, seed: int = 42, n_jobs: int = 6) -> dict:
    tc = cfg.telegram
    k = int(k or 8)
    tab = build_table(cfg)
    ds = design_set()
    des = [(n, d["fam"], d["cols"], d["p"]) for n, d in ds.items()]
    log.info("ablation: %d pairs, %d designs", len(tab), len(des))
    mo, mb = int(tc.region_dev_min_obs), int(tc.region_min_oblast)
    wf = walk_forward(tab, des, mo, mb, "all", n_jobs)
    ev = wf[wf["evaluated"]].copy()
    folds = rd.make_folds(list(ev["idx"].unique()), k)
    sc = ei.Scorer(ev)
    sc.prep(list(ds) + ["oblast_median"])
    res1 = {}
    for n, d in ds.items():
        r = compare(sc, folds, n, d["base"])
        r.update({"fam": d["fam"], "cols": d["cols"], "label": d["label"], "base": d["base"], "n_cols": len(d["cols"])})
        if n != d["base"]:
            r["boot"] = boot_pair(ev, n, d["base"], n_boot, seed)
            r["verdict"] = verdict(r["boot"])
        res1[n] = r
    # simplest within 1 SE of the best, over all designs of part 1 (own C / MAE)
    sel_c, sel_m = one_se_simplest(res1, "C"), one_se_simplest(res1, "MAE")

    # part 2: training-set ablation on the three full designs
    des2 = [(n, d["fam"], d["cols"], d["p"]) for n, d in ds.items() if n == d["base"]]
    preds = {}
    for rg in REGIMES:
        w = wf if rg == "all" else walk_forward(tab, des2, mo, mb, rg, n_jobs)
        preds[rg] = w[w["evaluated"]].copy()
    ok_idx = set(ev["idx"].unique())
    for rg in REGIMES:
        good = preds[rg].groupby("idx")[[n for n, *_ in des2]].apply(lambda g: g.notna().all().all())
        ok_idx &= set(good.index[good])
    common = ev[ev["idx"].isin(ok_idx)][["idx", "hours", "observed", "is_af"]].copy()
    cfolds = rd.make_folds(sorted(ok_idx), k)
    evs = {}
    for rg in REGIMES:
        e = preds[rg][preds[rg]["idx"].isin(ok_idx)].copy()
        evs[rg] = e
    comb = common.copy()
    for rg in REGIMES:
        for n, *_ in des2:
            comb[f"{n}@{rg}"] = evs[rg].loc[comb.index, n].to_numpy(float)
    sub = {}
    for scope in ("all pairs", "Air Force strikes only"):
        e = comb if scope == "all pairs" else comb[comb["is_af"].astype(bool)]
        s2 = ei.Scorer(e)
        for n, *_ in des2:
            for rg in REGIMES:
                m = f"{n}@{rg}"
                r = compare(s2, cfolds, m, f"{n}@all")
                if rg != "all":
                    r["boot"] = boot_pair(e, m, f"{n}@all", n_boot, seed)
                    r["verdict"] = verdict(r["boot"])
                r["n_strikes"] = int(e["idx"].nunique())
                r["n_obs"] = int(e["observed"].sum())
                sub[(scope, n, rg)] = r
    info = {"pairs": len(tab), "observed": int(tab["observed"].sum()), "strikes": int(tab["idx"].nunique()), "scored": int(ev["idx"].nunique()),
            "common": len(ok_idx), "first": ev["strike_at"].min(), "last": ev["strike_at"].max(), "k": k, "n_boot": n_boot, "min_obs": mo,
            "fold_sizes": [(len(ids), int(ev["idx"].isin(ids).sum()), int((ev["idx"].isin(ids) & ev["observed"]).sum())) for ids in folds],
            "n_af": int(tab.drop_duplicates("idx")["is_af"].sum()), "n_ch2": int(tab.drop_duplicates("idx")["tr_af_ch2"].sum()),
            "scored_af": int(ev.drop_duplicates("idx")["is_af"].sum())}
    out = {"res1": res1, "sel_c": sel_c, "sel_m": sel_m, "sub": sub, "info": info}
    d = cfg.reports_dir
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"design": n, **{k_: v for k_, v in r.items() if k_ not in ("cols", "boot")}, **(r.get("boot") or {})} for n, r in res1.items()]
    pd.DataFrame(rows).to_csv(d / "energy_map_ablation_dev_features.csv", index=False)
    pd.DataFrame([{"scope": s, "design": n, "train": rg, **{k_: v for k_, v in r.items() if k_ != "boot"}, **(r.get("boot") or {})}
                  for (s, n, rg), r in sub.items()]).to_csv(d / "energy_map_ablation_dev_subsets.csv", index=False)
    write_report(d / "energy_map_ablation_dev.md", out)
    return out


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
def _f(x, nd=3):
    return "" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def _feat_row(n: str, r: dict) -> str:
    if "boot" not in r:
        return (f"| {r['label']} | {r['n_cols']} | {_f(r['C'])} ({_f(r['C_se'])}) | {_f(r['MAE'], 1)} ({_f(r['MAE_se'], 1)}) | (reference) | | | | |")
    b = r["boot"]
    return (f"| {r['label']} | {r['n_cols']} | {_f(r['C'])} ({_f(r['C_se'])}) | {_f(r['MAE'], 1)} ({_f(r['MAE_se'], 1)}) | "
            f"{r['dC']:+.4f} ({_f(r['dC_se'], 4)}), {r['dC_better']}/{r['n']} | {r['dMAE']:+.1f} ({_f(r['dMAE_se'], 1)}), {r['dMAE_better']}/{r['n']} | "
            f"{b['dC']:+.4f} [{b['dC_lo']:+.4f}, {b['dC_hi']:+.4f}] | {b['dM']:+.1f} [{b['dM_lo']:+.1f}, {b['dM_hi']:+.1f}] | {r['verdict']} |")


def write_report(path, o: dict) -> None:
    i, r1, sub = o["info"], o["res1"], o["sub"]
    L = ["# Energy Map restoration forecast: ablation of features and training strikes (DEV; CV over ALL strikes up to 2026-06-26)", "",
         "Question: which data are redundant or harmful? Same label, cohort, walk-forward, 8 expanding folds, scored pairs and metrics as `energy_map_improve_dev.md` "
         "(the frozen test is spent, the whole period is development data; the blind test of `docs/preregistration_energy_map_v2_draft.md` is untouched). "
         f"Dev pairs {i['pairs']} ({i['observed']} observed) in {i['strikes']} strikes; scored {i['scored']} strikes ({i['first']:%Y-%m-%d} .. {i['last']:%Y-%m-%d}); "
         "folds (strikes / pairs / observed): " + "; ".join(f"{a} / {b} / {c}" for a, b, c in i["fold_sizes"]) + ".", "",
         "Cells: C = Harrell's C within strike, MAE h on observed episodes, mean over folds (SE over folds). Paired columns: difference (reduced design - full design) per "
         "fold, mean (SE), folds where the reduced design is BETTER (dC > 0, dMAE < 0); pooled paired bootstrap over strikes (%d resamples, seed 42) with 95%% CI. " % i["n_boot"] +
         "Rule fixed in the code: removal is 'KEEP' only if the CI shows a worse C (upper bound < 0) or a worse MAE (lower bound > 0); otherwise 'REMOVE' "
         "(an improvement is flagged when a CI is entirely on the better side). 'Safe to remove' is therefore 'not provably worse', not 'proven useless'.", "",
         "Notes: the columns of `so` contain only `log_through, gap_prev, n_strikes_30, heating, temp_24h` and the oblast dummies; no `sev_`/`act_`/`early_` feature is in any of the "
         "designs, so group (a) is exactly these five. For `ridge_so_em` / `xgbaft_so_em` the groups HIST_F, GEO_F, CTX_F are not part of the design (not ablated). Alerts and weather "
         "are not part of the reference designs: rows `+ alerts` / `+ weather` ADD them (the sets of `energy_map_alerts_weather_dev.md`), so the sign of the effect is that of adding. "
         "Features: strike-level `log_through, gap_prev, n_strikes_30, heating, temp_24h`; EM_F record density/history; HIST_F Telegram history; GEO_F front/tier/border; CTX_F `is_af`, sin/cos day of year.", ""]
    L += ["## 1. Feature-group ablation", ""]
    for b in BASES:
        L += [f"### {b}", "", "| design | n features | C_cv (SE) | MAE_cv h (SE) | dC vs full (SE), folds better | dMAE vs full h (SE), folds better | pooled dC [95% CI] | pooled dMAE h [95% CI] | verdict |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n, r in r1.items():
            if r["base"] == b:
                L.append(_feat_row(n, r))
        L.append("")
    sc_, sm = o["sel_c"], o["sel_m"]
    L += ["## 2. Strike-subset ablation (training set only changes; evaluation on the same pairs)", "",
          f"Regimes: (i) {REGIMES['all']}; (ii) {REGIMES['af_ch2']}; (iii) {REGIMES['af']}. Scored strikes {i['scored']} (Air Force strikes among them: {i['scored_af']}; "
          f"{i['n_af']} Air Force strikes and {i['n_ch2']} strikes with Air Force or n_channels >= 2 in the whole list). A strike is scored only if the UNFILTERED training set has "
          f">= {i['min_obs']} observed episodes (as before); a regime whose filtered training set has < {MIN_TRAIN} observed episodes gives no forecast, and such strikes are dropped for ALL "
          f"regimes: the comparison uses the {i['common']} strikes where every regime forecasts. **`n_channels` is a whole-day count (known only after the day):** it filters "
          "the PAST training strikes (their day is over before the issue time of the strike being forecast; labels evaluated later) and is never a feature nor used to select "
          "evaluated strikes. Paired against regime (i) with the same design.", ""]
    for scope in ("all pairs", "Air Force strikes only"):
        L += [f"### Evaluation on: {scope}", "",
              "| design | training set | strikes / observed evaluated | C_cv (SE) | MAE_cv h (SE) | dC vs (i) (SE), folds better | dMAE vs (i) h (SE), folds better | pooled dC [95% CI] | pooled dMAE h [95% CI] | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b in BASES:
            for rg in REGIMES:
                r = sub[(scope, b, rg)]
                if rg == "all":
                    L.append(f"| {b} | (i) all | {r['n_strikes']} / {r['n_obs']} | {_f(r['C'])} ({_f(r['C_se'])}) | {_f(r['MAE'], 1)} ({_f(r['MAE_se'], 1)}) | (reference) | | | | |")
                else:
                    bt = r["boot"]
                    L.append(f"| {b} | {'(ii) AF + ch>=2' if rg == 'af_ch2' else '(iii) AF only'} | {r['n_strikes']} / {r['n_obs']} | {_f(r['C'])} ({_f(r['C_se'])}) | {_f(r['MAE'], 1)} ({_f(r['MAE_se'], 1)}) | "
                             f"{r['dC']:+.4f} ({_f(r['dC_se'], 4)}), {r['dC_better']}/{r['n']} | {r['dMAE']:+.1f} ({_f(r['dMAE_se'], 1)}), {r['dMAE_better']}/{r['n']} | "
                             f"{bt['dC']:+.4f} [{bt['dC_lo']:+.4f}, {bt['dC_hi']:+.4f}] | {bt['dM']:+.1f} [{bt['dM_lo']:+.1f}, {bt['dM_hi']:+.1f}] | {r['verdict']} |")
        L.append("")
    L += ["## 3. Verdict", "", "VERDICT_PLACEHOLDER", "",
          "## 4. Caveats", "",
          "- Development numbers only (the whole period is dev); the groups were fixed before the run, but the three reference designs were chosen on the same CV. A pooled CI that includes 0 is "
          "weak evidence of redundancy with ~2400 observed episodes; the blind test decides.",
          "- Dropping one group at a time cannot reveal redundancy between two groups that carry the same signal (e.g. EM_F and the oblast dummies): each can be removed alone while removing both would hurt. "
          "The 'minimal' row is the joint check.",
          "- The simplest design within one SE (by number of features, then family) is chosen among the designs of section 1 only, with the SE of the best design's fold-mean.", ""]
    txt = "\r\n".join(L).replace("VERDICT_PLACEHOLDER", verdict_text(o))
    Path(path).write_text(txt, encoding="utf-8")


def verdict_text(o: dict) -> str:
    r1, sub, sc_, sm = o["res1"], o["sub"], o["sel_c"], o["sel_m"]
    L = ["**Feature groups** (verdict by the fixed rule, per reference design):", ""]
    for b in BASES:
        keep = [r["label"][2:] for n, r in r1.items() if r["base"] == b and r["label"].startswith("- ") and r["verdict"].startswith("KEEP")]
        rem = [r["label"][2:] for n, r in r1.items() if r["base"] == b and r["label"].startswith("- ") and r["verdict"].startswith("REMOVE")]
        add = [f"{r['label'][2:]}: {r['verdict']}" for n, r in r1.items() if r["base"] == b and r["label"].startswith("+ ")]
        mn = r1[f"{b}__minimal"]
        L.append(f"- `{b}`: keep {', '.join(keep) or 'nothing'}; removable alone (not provably worse): {', '.join(rem) or 'nothing'}; adding " + "; ".join(add) +
                 f"; minimal model ({mn['n_cols']} features): C {mn['C']:.3f} vs {r1[b]['C']:.3f}, MAE {mn['MAE']:.1f} vs {r1[b]['MAE']:.1f}, {mn['verdict']}.")
    L += ["", f"**Simplest model not worse than the best (1 SE, among all designs of section 1)**: by C the best is `{sc_['best']}` C = {sc_['best_val']:.4f} (SE {sc_['se']:.4f}), "
          f"{sc_['n_within']} designs within one SE; simplest `{sc_['selected']}` ({len(r1[sc_['selected']]['cols'])} features) C = {sc_['sel_val']:.4f}. "
          f"By MAE the best is `{sm['best']}` {sm['best_val']:.1f} h (SE {sm['se']:.1f}), {sm['n_within']} within one SE; simplest `{sm['selected']}` "
          f"({len(r1[sm['selected']]['cols'])} features) {sm['sel_val']:.1f} h.", "", "**Training strikes**:", ""]
    for scope in ("all pairs", "Air Force strikes only"):
        for rg in ("af_ch2", "af"):
            vs = [f"{b}: {sub[(scope, b, rg)]['verdict']} (dC {sub[(scope, b, rg)]['dC']:+.3f}, dMAE {sub[(scope, b, rg)]['dMAE']:+.1f} h)" for b in BASES]
            L.append(f"- evaluated on {scope}, trained on {REGIMES[rg]}: " + "; ".join(vs) + ".")
    return "\r\n".join(L)
