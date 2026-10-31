# Results on real data

Two real-data scenarios built from Ukrenergo files published on Energy Map (README, "Real data without ENTSO-E").
Reproduce with the commands in the README; all tables are in `reports/<scenario>/`, figures in `reports/<scenario>/figures/`.
Everything below was tuned on the validation period only. The test period was not used for any choice of
parameters, but the baselines `seasonal_naive` and `xgb_rolling` were added after a first look at the test results.

## Scenario `net_import` (MW, 2019-07 to 2026-06; train 2019-2021, validation 2022-2023, test 2024-2026)

Target: hourly net imports (imports minus exports), a measured deficit proxy that neighbouring systems covered.

| Model | Test MAE, MW | First 14 days after attacks | Test RMSE, MW |
|---|---|---|---|
| xgb_adaptive | **210.8** | **194.2** | **286.7** |
| xgb_rolling (ablation) | 219.6 | 200.0 | 296.9 |
| xgb_static | 264.2 | 251.7 | 357.8 |
| seasonal_naive | 315.7 | 317.0 | 429.6 |
| prophet_static | 5611.2 | 4354.3 | 5764.3 |

* Retraining on a recent window helps: adaptive vs static is -20% MAE.
* **Change-point detection adds little:** adaptive vs the rolling ablation is -4% MAE. Only 7 of 241 refits were triggered by an alarm; the rest were scheduled.
* Prophet's error is larger than the whole range of the data: its linear trend runs away to 8000 MW. It is a poor baseline here, not evidence about the method.
* Episodes (days with a daily mean above 800 MW): 23 in the test period; depth error 249 MW (adaptive) vs 421 MW (static).
* Detection: BOCPD found 6 breaks and matched only 1 of 18 listed attacks (precision 0.17, recall 0.06); PELT 12 breaks (0.25 / 0.17). On this series detection is weak, and 2024 regime changes were missed by BOCPD.

## Scenario `outages` (share of regions under planned outages, 2023-01 to 2026-06; test 2025-07 to 2026-06)

Target: hourly share of regions under Ukrenergo planned outages (0 to 1), from the outage dataset. Unknown hours are
NaN and 2025-12-10 to 2026-01-28 is excluded (aggregated from distribution operators).

| Model | Test MAE | Test RMSE | Validation MAE |
|---|---|---|---|
| xgb_rolling (ablation) | **0.063** | **0.174** | 0.123 |
| seasonal_naive | 0.074 (on 6893 of 7229 hours; last week unknown for the rest) | 0.213 | 0.128 |
| xgb_adaptive | 0.092 | 0.190 | 0.117 |
| xgb_static | 0.118 | 0.245 | **0.104** |
| prophet_static | 0.264 | 0.403 | 0.215 |

* **Plain rolling retraining is best on the test period** and beats the change-point-triggered model by about 46% (p < 0.001 on daily MAE).
* The ranking is not stable: on validation the static model has the lowest MAE.
* Adaptive vs static: -22% MAE on test, but the daily Wilcoxon test gives p = 0.054, not significant at 5%.
* The weekly-naive baseline (0.074) also beats the change-point-triggered adaptive model (0.092).
* Detection works better than on net imports: BOCPD raised 7 alarms, 6 of them near a listed attack (precision 0.86), but it matched only 5 of 12 attacks inside the data window (recall 0.42, mean lag 3.2 days). Many March to April 2024 attacks caused no visible outages until summer.
* Only 7 validation and 3 test deficit episodes, so every difference between models is indicative.
* Day classification (deficit day or not, test): xgb_adaptive finds 98% of deficit days but over-predicts (precision 0.64); seasonal_naive is balanced (0.92 / 0.91); xgb_rolling 0.84 / 0.92.
* **Defect found and fixed during the analysis:** during the excluded period the rolling window held no observed targets, the refit produced an empty model, and the model predicted a flat 0 for three weeks. Refits are now skipped when fewer than 168 targets are observed (previous model kept). This lowered the rolling model's test MAE from 0.083 to 0.063 and did not change the adaptive model or the net-import results.

## What this means for the thesis

* Supported: static models degrade after the regime changes (Prophet fails badly; static XGBoost is 20-22% worse than adaptive), and the target series do show regimes (see `changepoints.png` in each scenario).
* **Not supported by these data:** that retraining triggered by a detected break beats plain periodic retraining on a rolling window. The gain comes from recency, not from detection; on the outage index plain rolling retraining is clearly better. A fair way to present it: BOCPD is a useful early-warning diagnostic (high precision on the outage series), and the adaptive window is what improves the forecast.
* A simple weekly-naive baseline is competitive (it beats the adaptive model on the outage index), so claims should always be shown next to it.

## Limits to state when presenting

* The outage target is an index of how widely outages were applied, not MW; the mapping from the dataset's wording to shares is an approximation (`config/outages.yaml`).
* Net imports are capped by interconnector capacity (hourly maximum 2,233 MW) and in 2022-2023 the system was roughly balanced, so they track the later waves, not the first ones.
* `data/external/attacks.csv` ends in December 2024 and has no capacities; detection scores are limited to its period (`changepoints.eval_until`).
* The suspect period and 13 corrupted April 2026 days (328 rows) are unknown, not zero.
* Wilcoxon p-values use daily errors, which are still autocorrelated, so they are optimistic.
* ENTSO-E data was not used; whether it covers Ukraine in the war years is unverified.

## Telegram severity features and restriction-regime duration (details: `reports/telegram_impact_results.md`)

* Post-event direct models got Telegram severity features (`features/telegram.py`, national channel, posts before the end of the issue day) as extra sets `_g` / `_tg`. **Validation (2022-2023): they are worse than the same models without them in price, net import and outages; test (2024-2025): better in all three** (e.g. net import `xgb_direct_g` 333 vs 383 MW, p=0.001). Per the rules the choice is made on validation, so they are not adopted; the test gain is not independent across scenarios (same strikes).
* Depth beats persistence without Telegram (price validation p=0.009, test p<0.001; net import test p<0.001); outages: nothing beats persistence. **Duration never beats persistence**, with or without Telegram features.
* New target, duration of the restriction regime (77 strikes, 33 observed, 44 censored; 12 validation / 60 test strikes scored, 9 / 21 observed): no model beats the median of past durations (test MAE 6.45 d for the median vs 9.7 d ridge and 12.0 d AFT on observed strikes); the AFT lower-bound advantage is a censoring artefact (a constant 30-day forecast scores better); Telegram features add nothing over attack-only features (test p>=0.24). Censoring is partly informative (9 censored strikes had at most 5 days of restrictions).

## Oblast-level episodes and restoration duration (details: `reports/telegram_region_results.md`, extraction quality: `reports/telegram_extraction_quality.md`)

* `deficit extract` now also writes `data/processed/telegram_region_{posts,episodes}.parquet`: one row per (strike, oblast) for 77 strikes x 25 oblasts. Oblast impact (consumers cut off, emergency outages or schedules) is reported for 566 pairs; **201 have an observed restoration, 365 are censored** (episode watched until the next strike day or 30 days; 347 of the censored ones were cut by the next strike). With the plain 30-day window 376 are observed. Only 6 oblasts have a usable own channel (Sumy, Zhytomyr, Khmelnytskyi, Lviv, Dnipropetrovsk since 2024-11, Volyn to 2023-09); the other oblasts are seen through Ukrenergo / DTEK mentions only. Several oblenergo channels hold 1 to 10 posts in the public preview.
* Extraction on 112 hand-labelled regional posts (enriched strata, so not population rates): schedules precision 0.95 / recall 0.67, emergency 0.85 / 0.94, lifted 1.00 / 1.00, consumers 0.89 / 0.62, attack 0.70 / 0.86, restored_any 0.83 / 0.66, restored_full 0.46 / 0.60. Recall on ordinary posts is low; one structural fix (regional channels that repost national digests) was made after seeing the gold errors, so the table is not fully blind (the blind first pass is in the notes).
* Forecast of the restoration duration per (strike, oblast) at the end of the strike day, split by strike date (validation up to 2024-12-31, test after; the README split leaves too few observed episodes). Baselines: global and oblast median of past observed durations. **Strict label (until the next strike): only 22 observed test episodes in 13 strikes; no model beats the global median (test MAE 44 h ridge vs 39 h, per-strike p=0.22).** **30-day label (training labels as known at the issue time): 98 observed test episodes in 28 strikes; ridge on strike features is 5% better than the global median (MAE 200 h vs 212 h, per-strike Wilcoxon p=0.012, bootstrap CI of the difference -17 to -6 h) and 7% better than an intercept-only log-mean (p=0.023), but about 8 days of error remain, the within-strike ranking of oblasts is no better than chance and the AFT model is not better than the median.** Oblast information does not help (the oblast median is not better than the global median); episodes of one strike are strongly dependent, so the p-values are optimistic.
