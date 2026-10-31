"""Markdown report `reports/telegram_region_results.md` of the oblast-level dataset and restoration forecast.
Every number is read from the tables of the run; the wording of the verdicts follows from those numbers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF = {"global_median", "oblast_median", "geo_mean", "km_global", "const_30d"}
TITLES = {"next_strike": "label window: until the next strike day (strict; most episodes censored)",
          "30d": "label window: 30 days (a lifting post may follow later strikes; training labels as known at the issue time)"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    if df.empty:
        return "(empty)\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else (f"{v:.2g}" if abs(v) < 0.01 and v != 0 else floatfmt.format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def _p(v) -> str:
    return "n/a" if pd.isna(v) else (f"{v:.3f}" if v >= 0.001 else "<0.001")


def episode_counts(eps: pd.DataFrame, res: dict) -> list[str]:
    imp = eps[eps["outage"]]
    obs = imp[~imp["censored"]]
    per = (imp.groupby("oblast").agg(episodes=("strike_at", "size"), observed=("censored", lambda s: int((~s).sum())),
                                     censored=("censored", "sum"), own_channel=("coverage", lambda s: int((s == "own_channel").sum())))
           .reset_index().sort_values("episodes", ascending=False))
    for v, r in res.items():
        t = r["table"].groupby("oblast").agg(n=("observed", "size"), o=("observed", "sum")).reset_index()
        per = per.merge(t.rename(columns={"n": f"cohort_{v}", "o": f"cohort_{v}_obs"}), on="oblast", how="left")
    per = per.fillna(0)
    num = [c for c in per.columns if c != "oblast"]
    per[num] = per[num].astype(int)
    per = pd.concat([per, pd.DataFrame([{"oblast": "TOTAL", **{c: int(per[c].sum()) for c in num}}])], ignore_index=True)
    lines = [f"- (strike, oblast) pairs: {len(eps)} ({eps['strike_at'].nunique()} strikes x {eps['oblast'].nunique()} oblasts). "
             f"Pairs with oblast impact reported inside the window (outages, emergency outages or schedules): **{len(imp)}**; "
             f"observed restoration **{len(obs)}**, censored **{int(imp['censored'].sum())}** (primary window: until the next strike day, "
             f"at most 30 days).",
             "- Observed resolutions by kind / source: " + ", ".join(f"{k[0]} / {k[1]}: {v}" for k, v in
                                                                 obs.groupby(["resolution_kind", "resolution_source"]).size().items()) + ".",
             "- Censoring reasons: " + ", ".join(f"{k}: {v}" for k, v in imp[imp["censored"]].groupby("window_end_reason").size().items())
             + " (`next_strike` = the next strike day began before any restoration post; `horizon` = 30 days passed).",
             f"- With the plain 30-day window: {int((imp['status_30d'] == 'observed').sum())} observed, "
             f"{int((imp['status_30d'] == 'censored').sum())} censored of {int((imp['status_30d'] != 'no_impact_reported').sum())} with impact.",
             "- Forecast cohort (an impact post for the oblast before the end of the strike day, not resolved by then): " +
             "; ".join(f"`{v}`: **{len(r['table'])}** episodes, {int(r['table']['observed'].sum())} observed, "
                       f"{int((~r['table']['observed']).sum())} censored" for v, r in res.items()) + ".", "",
             md_table(per), "",
             "Pairs with impact by strike year: " + ", ".join(f"{y}: {n}" for y, n in imp.groupby(imp["strike_at"].dt.year).size().items()) + ".", ""]
    return lines


def _geo_line(tests, best) -> list[str]:
    g = tests[(tests["split"] == "test") & (tests["method"] == best) & (tests["baseline"] == "geo_mean")]
    if g.empty:
        return []
    q = g.iloc[0]
    return [f"- Against the intercept-only reference `geo_mean` (exp of the mean log duration): difference {q['mean_diff_h']:+.1f} h "
            f"({q['rel_diff_pct']:+.1f}%), per-strike p = {_p(q['p_per_strike'])}; if this is not clearly negative the gain over the median "
            "baseline is the log-scale centre, not the features."]


def variant_section(cfg, r: dict) -> list[str]:
    tc = cfg.telegram
    tab, scores, chosen, tests, obl = r["table"], r["scores"], r["chosen"], r["tests"], r["oblast"]
    ev = tab[tab["evaluated"]]
    v = r["variant"]
    early = tab[(tab["strike_at"] <= pd.Timestamp("2023-12-31", tz="UTC")) & tab["observed"]]
    L = [f"### 4.{1 + list(TITLES).index(v)} {TITLES[v]}", "",
         "Evaluated episodes: " + ", ".join(
             f"{s}: {int((ev['split'] == s).sum())} episodes ({int(((ev['split'] == s) & ev['observed']).sum())} observed) in "
             f"{ev.loc[ev['split'] == s, 'idx'].nunique()} strikes" for s in ("validation", "test")) +
         f". (Observed cohort episodes up to 2023-12-31, the README validation period: {len(early)}.)", "",
         "Chosen on validation: " + ", ".join(f"{k} = `{x}`" for k, x in chosen.items()) + ".", ""]
    show = list(dict.fromkeys(["global_median", "oblast_median", "geo_mean", "km_global", "const_30d"] + [chosen[k] for k in ("ridge", "aft", "overall") if k in chosen]))
    for split in ("validation", "test"):
        t = scores[(scores["split"] == split) & scores["method"].isin(show)].copy()
        t["method"] = t["method"].map(lambda m: m + (" (chosen)" if m in chosen.values() and m not in REF else ""))
        t = t.drop(columns=["split", "RMSE_h"])
        L += [f"**{split.capitalize()}** (MAE on observed episodes; lower bound and within-strike C use all episodes)", "", md_table(t, "{:.2f}"), ""]
    best, base = chosen.get("overall"), chosen.get("baseline")
    if best and base:
        ts = tests[(tests["split"] == "test") & (tests["method"] == best) & (tests["baseline"] == base)]
        vs = scores[(scores["split"] == "validation") & scores["method"].isin([best, base])].set_index("method")["MAE_h"]
        tt = scores[(scores["split"] == "test") & scores["method"].isin([best, base])].set_index("method")
        if len(ts) and len(tt) == 2 and len(vs) == 2:
            q = ts.iloc[0]
            sig = bool(pd.notna(q["p_per_strike"]) and q["p_per_strike"] < 0.05 and q["mean_diff_h"] < 0)
            L += ["**Verdict (test)**", "",
                  f"- Best model on validation: `{best}` (validation MAE {vs[best]:.0f} h vs {vs[base]:.0f} h for the best baseline `{base}`).",
                  f"- Test MAE on observed episodes: `{best}` {tt.loc[best, 'MAE_h']:.0f} h ({tt.loc[best, 'MAE_days']:.1f} d) vs `{base}` "
                  f"{tt.loc[base, 'MAE_h']:.0f} h ({tt.loc[base, 'MAE_days']:.1f} d); n = {int(q['n_episodes'])} observed episodes in "
                  f"{int(q['n_strikes'])} strikes; difference {q['mean_diff_h']:+.1f} h ({q['rel_diff_pct']:+.1f}%), bootstrap-over-strikes 95% CI "
                  f"[{q['boot_ci95_lo_h']:+.1f}, {q['boot_ci95_hi_h']:+.1f}] h.",
                  f"- Wilcoxon p = {_p(q['p_episodes'])} per episode (dependent episodes: optimistic) and p = {_p(q['p_per_strike'])} on "
                  f"per-strike means (n = {int(q['n_strikes'])} strikes).",
                  *_geo_line(tests, best),
                  "- Conclusion: **" + ("the model beats the best baseline at 5% in the per-strike test." if sig else
                                       "no evidence that the model beats the best baseline in the per-strike test.") + "**", ""]
        L += ["**Paired tests** (difference of |error|, negative = model closer to the truth; `p_episodes` treats episodes as independent)", "",
              md_table(tests.drop(columns=["share_better_episodes"], errors="ignore"), "{:.2f}"), ""]
    if len(obl):
        L += ["**Test MAE (hours) per oblast**, observed episodes, oblasts with enough of them", "", md_table(obl, "{:.0f}"), ""]
    return L


def build_text(cfg, eps, res, comp, facts) -> str:
    tc = cfg.telegram
    L: list[str] = ["# Oblast-level strike impact and restoration duration (Telegram)", "",
                    "Generated by `deficit region-restoration` (inputs `data/processed/telegram_region_*.parquet` from `deficit extract`). "
                    "Definitions: docstrings of `ingestion/telegram_regional.py` and `evaluation/region_restoration.py`.", "",
                    "## 1. Dataset", "", *episode_counts(eps, res)]
    L += ["## 2. Data completeness", "",
          "Raw channel files at the time of the run (public t.me/s preview). Oblast channels whose stored history starts after a strike "
          "do not count as `own_channel` coverage for that strike. Several oblenergo channels are tiny because the public preview holds "
          "only a few posts (history ends or channel is nearly empty); they add almost nothing.", "", md_table(comp), ""]
    nocov = [o for o in sorted(eps["oblast"].unique()) if not (eps[eps["oblast"] == o]["coverage"] == "own_channel").any()]
    L += ["Oblasts without an own channel at any strike (only mentions in national channels, so a restoration is seen only when a "
          "national post names the oblast): " + ", ".join(nocov) + ".", ""]
    L += ["## 3. Extraction quality", "",
          "Hand-labelled regional posts (`tests/fixtures/telegram_region_gold.csv`, sampled by stratum, labelled before looking at the "
          "extractor output); item-level precision / recall per field, recomputed with this run's extraction. "
          "The same table is in the `Oblast-level dataset` section of `reports/telegram_extraction_quality.md`.", ""]
    gp = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "telegram_region_gold.csv"
    if gp.exists() and facts is not None:
        from ..ingestion.telegram_regional import score_region_gold
        g = pd.read_csv(gp, dtype=str, keep_default_na=False)
        g["post_id"] = g["post_id"].astype(int)
        L += [f"Gold posts: {len(g)} ({', '.join(f'{k} {n}' for k, n in g['stratum'].value_counts().items())}).", "",
              md_table(score_region_gold(g, facts).round(2), "{:.2f}"), ""]
    L += ["## 4. Forecast of the restoration duration per (strike, oblast)", "",
          f"Issue time = end of the strike day; features from posts published before it, plus strike strength and the temperature of the 24 h "
          f"before it. Walk-forward over strikes: strike i is forecast from the episodes of strikes before i; first forecast once "
          f"{tc.region_min_obs} observed episodes are known. **Split by strike date: validation up to {tc.region_validation_end}, test after** "
          f"(the README's 2022-23 / 2024-25 split would leave too few observed episodes in validation, counts below; the split was fixed "
          f"before any model result was seen). Designs (model, feature set, penalty) are chosen on validation only; test strikes were "
          f"never used for a choice. Baselines: global median and oblast median of past observed durations; `km_global` (Kaplan-Meier "
          f"median, uses the censored episodes) and `const_30d` (always 30 days) are references that show the bias of the metrics.", ""]
    for r in res.values():
        L += variant_section(cfg, r)
    L += ["## 5. Caveats", "",
          "- A label is a statement in a post. A censored episode means that no restoration post was found, not that power stayed out. "
          "Restoration posts are rarer than outage posts, so observed episodes are a biased, short-duration subsample; MAE on observed "
          "episodes understates the error for long outages.",
          "- Observed-only MAE favours short forecasts; the censoring-aware lower bound favours long ones (see `const_30d`). Neither is an "
          "unbiased error estimate under censoring; `km_global` and the AFT models that use censored episodes can look bad on the first "
          "metric and good on the second for that reason. The within-strike C ignores the level and only checks the ranking of oblasts.",
          "- When an oblast had a restriction regime, its end is often the national post 'restrictions are not applied' (`national_lift`): "
          "one post ends many oblasts, so their durations are almost copies. Episodes of one strike are therefore dependent; "
          "the per-strike test and the bootstrap over strikes account for it, the per-episode p-value does not.",
          "- Oblasts without an own channel (`text_only`) are seen only through national posts: 'no impact reported' there is weak evidence, "
          "and their restoration is often not announced per oblast.",
          "- Strike times in attacks.csv are placeholders (06:00 UTC): durations carry a few hours of noise.",
          "- One national temperature series (no oblast weather); observed temperature of the 24 h before the issue time.",
          "- Oblast dummies of rarely affected oblasts rest on very few episodes (ridge shrinks them).", ""]
    return "\n".join(L)


def write_region_report(cfg, eps, res, comp, path: Path) -> None:
    root = cfg._abs(cfg.paths.data_dir)
    fp = root / "processed" / "telegram_region_posts.parquet"
    facts = pd.read_parquet(fp) if fp.exists() else None
    Path(path).write_text(build_text(cfg, eps, res, comp, facts), encoding="utf-8")
