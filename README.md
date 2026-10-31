# UA Deficit Forecast

**Прогнозування глибини та тривалості дефіциту електроенергії після атак на інфраструктуру на основі методів машинного навчання**

This is the implementation of the ML pipeline designed in the course work of K. S. Danylchenko (Lviv Polytechnic, PMID-11, 2026). It forecasts the hourly electricity deficit of Ukraine's power system (IPS of Ukraine) and the **depth** and **duration** of deficit episodes after attacks on energy infrastructure. It detects structural breaks automatically and then **retrains the forecasting model only on data from the new regime**.

---

## 1. Idea in one paragraph

Classical models (ARIMA, Prophet, a static neural net) assume the data's structure is stable or changes slowly. Each attack wave (Feb 2022 demand shock; Oct 2022 to Mar 2023 substation strikes; Mar to Aug 2024 destruction of more than 9 GW of generation) permanently changes the statistical properties of load and deficit. That is a **structural break**. The system therefore has three levels:

| Level | Role | Model | Why |
|---|---|---|---|
| 1. Baseline | Reference point | **Prophet** (trained on 2019-2021, never retrained) | Interpretable, industry standard; its error growth after each wave quantifies the breaks |
| 2. Break detection | Find regime changes | **PELT** (offline) / **BOCPD** (online) | PELT gives retrospective analysis; BOCPD raises an alarm in real time |
| 3. Adaptive forecast | Forecast after a break | **XGBoost**, **LSTM** | XGBoost is interpretable (for example the importance of `capacity_lost_MW`); LSTM captures sequential dependence |

**Automatic retraining:** after BOCPD detects a break at time τ, the adaptive model is refitted on `[τ − window, now]` only, so old patterns don't "dilute" the new regime.

---

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language | Python ≥ 3.11 (tested on 3.12) |
| Data | pandas, numpy, pyarrow (**Parquet** storage) |
| Ingestion | `entsoe-py` (ENTSO-E), `requests` (Open-Meteo, ACLED) |
| Change points | `ruptures` (PELT); BOCPD implemented in `changepoints/bocpd.py` (Adams & MacKay 2007, Normal-Gamma model) |
| Models | `prophet`, `xgboost`, `keras` 3 + TensorFlow (LSTM, optional extra) |
| Tuning / CV | `optuna`, `sklearn.model_selection.TimeSeriesSplit` |
| Evaluation | `scipy` (Wilcoxon), `matplotlib` |
| Config | YAML + pydantic (`config/config.yaml`) |
| Tests | pytest |

BOCPD is implemented in-house because the `bayesian_changepoint` package named in the thesis is unmaintained. The algorithm is the same.

---

## 3. Project layout

```
ua-deficit-forecast/
├── config/config.yaml          all parameters (periods, splits, models, thresholds)
├── config/outages.yaml         real-data scenario: planned-outage index (extends config.yaml)
├── config/net_import.yaml      real-data scenario: net imports in MW (extends config.yaml)
├── config/demand.yaml          real-data scenario: lost demand in MW, first war wave 2022 (extends config.yaml)
├── data/
│   ├── external/attacks.csv    curated attack chronology (YOU maintain this)
│   ├── raw/                    step 1 output (Parquet)           [git-ignored]
│   ├── interim/                gap / ACLED candidate lists        [git-ignored]
│   ├── processed/              master.parquet, features.parquet   [git-ignored]
│   └── synthetic/              same layout, generated demo data   [git-ignored]
├── artifacts/                  bocpd signal, change points, forecasts, tuned.json
├── reports/                    metrics.csv, wilcoxon.csv, events_*.csv, figures/
├── src/deficit/
│   ├── cli.py                  `deficit <step>` entry point
│   ├── config.py               typed config
│   ├── ingestion/              entsoe_client, weather, acled, attacks, synthetic
│   ├── preprocessing/          gaps (missing-value strategy), master (dataset + target)
│   ├── features/               calendar, build (lags / rolling / attack features)
│   ├── changepoints/           bocpd, detect (PELT + BOCPD + precision/recall/lag)
│   ├── models/                 base interface, prophet_model, xgb_model, lstm_model
│   ├── training/               walkforward (static + adaptive), run, tune (Optuna)
│   └── evaluation/             metrics, events (depth/duration), report (tables + plots)
└── tests/                      unit tests + leakage guards + end-to-end smoke test
```

---

## 4. Pipeline (thesis §3.3)

| # | Step | Command | Input → Output |
|---|---|---|---|
| 1 | Ingestion | `deficit ingest` | APIs → `data/raw/*.parquet` |
| 2 | Preprocessing | `deficit preprocess` | raw → `processed/master.parquet`, `processed/deficit_events.csv` |
| 3 | Feature engineering | `deficit features` | master → `processed/features.parquet` |
| 4 | Change point detection | `deficit detect` | daily deficit → `artifacts/changepoints.json`, `bocpd_signal.parquet` |
| (4b) | Tuning (optional) | `deficit tune` | validation period → `artifacts/tuned.json` |
| 5 | Adaptive training | `deficit train` | features + BOCPD → `artifacts/forecasts/<model>.parquet` |
| 6 | Evaluation & output | `deficit evaluate` | forecasts → `reports/` |

`deficit run` executes 1→6 (`--tune` adds 4b). Add `--synthetic` to any command to use generated demo data instead of real sources.

### Models trained

| Name | Training |
|---|---|
| `seasonal_naive` | no training: the value one week earlier (`lag_168h`). The simple baseline every model must beat |
| `prophet_static` | once on the training period, with UA holidays and temperature regressors. Its linear trend extrapolates badly (README §9, RESULTS.md) |
| `xgb_static` | once on 2019-2021 (the "traditional" approach) |
| `xgb_adaptive` | walk-forward: weekly refits + a refit on every BOCPD alarm, window `[τ − window, now]` |
| `xgb_rolling` | ablation: the same schedule as `xgb_adaptive` but ignoring alarms, always the last `window` days. Shows whether change-point detection adds anything beyond plain rolling retraining |
| `xgb_attack` | `xgb_rolling` + an extra refit the day after each strike in `attacks.csv`, on a short recent window (tuned on validation: `deficit tune-triggers`) |
| `xgb_drift` | `xgb_rolling` + an extra refit when a CUSUM on the model's own daily MAE (errors of hours < T only) drifts up; short recent window |
| `lstm_static` | once on 2019-2021 |
| `lstm_adaptive` | walk-forward: refits every 28 days + on every alarm |

### Target variables

* `deficit = actual_load_MW − (generation_total_MW + net_import_MW)`. A positive value means deficit (thesis §2.2). `net_import_MW` is the sum of physical flows from all neighbours (PL, SK, HU, RO, MD) minus the flows to them.
* Per major attack (`evaluation/events.py`):
  * `depth`: peak daily-mean deficit in the 14 days after the attack
  * `peak_hourly`
  * `cumulative`
  * `recovery_days`: days until the daily-mean deficit returns to ≤ `recovery_threshold`; `censored` if the next attack came first
* The same depth/duration numbers are computed from each model's forecasts, so the models are compared on the two quantities in the thesis title (`reports/events_depth_duration.csv`, `reports/events_summary.csv`).

### Features (all computed as of the issue time `t − horizon`)

* **Lag features:** `lag_24h`, `lag_48h`, `lag_168h`. `lag_1h` is used only when `horizon_hours: 1`.
* **Rolling features:** `rolling_mean_72h`, `rolling_std_72h`, lagged load and net import.
* **Calendar (Kyiv local time):** hour, day of week, month, weekend, UA holidays, Fourier terms.
* **Weather:** temperature, wind, heating degree.
* **Attack features:**
  * `capacity_lost_MW` (last 24 h)
  * `attack_impulse_MW`: exponentially decayed lost capacity, τ = 168 h
  * `attack_count_impulse`, `hours_since_attack` (capped at `features.attack_recency_cap_hours`, 720 h: a clock that only grows is never seen in training and made retrained models fail)
  * `acled_energy_events_7d`, if ACLED data is present

---

## 5. Rules

### 5.1 Methodological rules (from the thesis; code enforces them)

1. **Chronological split only.**
   * Train: 2019-2021 (peace).
   * Validation: 2022-2023 (breaks 1-2).
   * Test: 2024-2025 (break 3).
   * Rows are **never shuffled** across time.
   * Cross-validation uses `TimeSeriesSplit` (with a gap equal to the horizon), never K-Fold.
2. **No leakage ("витік майбутнього").** Every feature for target hour `t` uses only information available at the issue time `t − horizon_hours` (end of that hour). The only exceptions:
   * calendar features, which are known in advance;
   * weather, where observed weather stands in for a weather forecast. This "perfect weather forecast" assumption must be stated in the thesis.

   The tests `tests/test_no_leakage.py` guard this.
3. **Walk-forward timing.**
   * A model refitted at time T trains only on target hours `< T`.
   * It serves forecasts issued at `≥ T`, i.e. target hours `≥ T + horizon`.
   * A BOCPD value for day d is usable only from `d + 1 00:00 UTC`.
4. **The test period is sacred.**
   * All tuning (Optuna hyperparameters, adaptive `window_days`, the BOCPD threshold) uses the **validation** period only.
   * The final model choice is made on the test set, as the thesis states. Nothing is tuned on it.
5. **PELT is retrospective only.** PELT sees the whole series, so its breakpoints must never become model inputs. Only the causal BOCPD signal may drive retraining.
6. **Missing values** (thesis §2.2):
   * gap ≤ 3 h: linear interpolation;
   * 3-24 h: the same hour of the previous week;
   * > 24 h: left NaN and flagged `long_gap`. These gaps are listed in `interim/long_gap_candidates.csv` as possible attack consequences.
   * Load and generation filled from *different* weeks don't balance, so rows filled from the previous week are flagged `imputed_weekly`. They stay available as lag inputs but are never used as a training or evaluation target.
   * A generation total is NaN if any generation type is missing. A partial sum would look like a fake deficit.
   * An unreported flow direction counts as 0.
7. **Time zones.** All data is stored and joined in **UTC** at hourly resolution. Local time (Europe/Kyiv) is used only for calendar features. 15-min ENTSO-E data is averaged to hourly.
8. **Metrics.**
   * Report MAE, RMSE and MAPE per split, for all hours and separately for **crisis weeks** (the first 14 days after each major attack).
   * MAPE skips |y| < `mape_min_abs`, because the deficit crosses zero.
   * Model differences are tested with a **paired Wilcoxon test on daily MAE**. Daily aggregation reduces the autocorrelation of hourly errors.
9. **Change-point quality** is reported as precision, recall and mean detection lag (days), with a tolerance of 14 days after an attack.
10. **Attack data must be verified.**
    * ACLED output is a *candidate list* (`interim/acled_energy_candidates.csv`) and needs manual checking before it goes into `data/external/attacks.csv`.
    * `capacity_lost_MW` comes from Ministry of Energy reports.
    * The seed `attacks.csv` lists only dates of widely reported strikes. Every row is `verified=false` and the capacities are empty.
11. **Synthetic data is only for demos and tests.** It lives in `data/synthetic/`, `artifacts/synthetic/` and `reports/synthetic/`. Never quote synthetic numbers as results.

### 5.2 Engineering rules

1. Every pipeline step reads and writes files. Steps are independent and can be rerun one at a time.
2. All tunable values live in `config/config.yaml`. No magic numbers in code for things someone might want to change.
3. Secrets go only in `.env` (git-ignored). Never commit tokens or passwords.
4. Raw, processed and artifact data is not committed. Only `data/external/attacks.csv` (the curated input) is.
5. Every model implements `models/base.py::Forecaster` (`fit(frame, mask)` / `predict(frame, mask)`). A new model is one class plus one line in `training/run.py::_factory`.
6. Time series are Parquet files with a tz-aware UTC `DatetimeIndex` (`io.write_parquet` / `read_parquet`).
7. Run `pytest` before every change to the methodology. A new feature needs a matching case in the leakage test.
8. Seeds are fixed (`tuning.seed`) for reproducibility.

---

## 6. Setup

The project runs only in Docker (isolated; nothing is installed on Windows except Docker Desktop). For real data, also copy `.env.example` to `.env` and fill in your tokens.

```bash
docker compose build                                   # once; no TensorFlow by default (LSTM models skipped)
# for the LSTM models (large download):  $env:EXTRAS="lstm"; docker compose build   (PowerShell)
docker compose run --rm deficit-offline run --synthetic --models prophet_static xgb_static xgb_adaptive
docker compose run --rm deficit-offline run --synthetic        # all 5 models (needs the LSTM build; about 1 h)
docker compose run --rm --entrypoint python deficit -m pytest  # tests
```

* Results appear in `./data`, `./artifacts` and `./reports` on the host.
* Tokens are read from `.env` at run time and are never baked into the image.
* A run on synthetic data needs no network and no tokens. Use the service `deficit-offline` instead of `deficit` (for example `docker compose run --rm deficit-offline run --synthetic`): it has no network at all and never receives `.env`.
* Thread counts are limited to 4 (`OMP_NUM_THREADS` in `docker-compose.yml`). Without it, the tests ran more than 5 minutes at 2300% CPU on a 22-core machine instead of about 9 seconds. Raise it only if you have spare cores and big data.
* Code changes need a rebuild (`docker compose build`). `config/` is mounted read-only, so config edits take effect immediately.

### Real data

Each command is `docker compose run --rm deficit <step>`:

```bash
docker compose run --rm deficit ingest       # needs ENTSOE_API_KEY in .env; ACLED optional
docker compose run --rm deficit preprocess
docker compose run --rm deficit features
docker compose run --rm deficit detect       # inspect reports/figures/changepoints.png, adjust penalty/threshold
docker compose run --rm deficit tune         # optional, validation-only
docker compose run --rm deficit train
docker compose run --rm deficit evaluate
```

### Real data without ENTSO-E: two scenarios from Energy Map files

Energy Map (DiXi Group) republishes Ukrenergo data. The files need a free login and have download limits, so
download them by hand and keep the file names in `data/raw/`:

| File | Used for |
|---|---|
| `2026_07_01_electricity_outages.csv` | outage scenario: target |
| `2026_07_01_import_eksport_pohodynno.csv` | net-import scenario: target; covariate in the outage scenario |
| `2026_07_01_pohodynnyi_balans.csv` | demand scenario: hourly consumption in MW, 2019 to 2022-10-27 |

Not used: `2026_07_01_elektro_boiovi_dii_ukraina_minenergo.csv` (consumers without power, empty after 2022).

| Scenario | Config | Target `deficit` | Period / split |
|---|---|---|---|
| `outages` | `config/outages.yaml` | share of regions under planned outages (0 to 1), hourly | 2023-01 to 2026-06; train to 2024-04, validation to 2025-06, test after |
| `net_import` | `config/net_import.yaml` | net imports in MW (imports minus exports), hourly | 2019-07 to 2026-06; train 2019-2021, validation 2022-2023, test 2024-2026 |
| `demand` | `config/demand.yaml` | **lost demand in MW** = expected peacetime consumption minus real consumption, hourly | 2019-01 to 2022-10-27; train 2019-2021, validation Jan-Mar 2022, test Apr-Oct 2022 |

Each scenario writes to its own folders (`data/processed/<name>`, `artifacts/<name>`, `reports/<name>`).
Weather needs the network once; everything else can run offline:

```bash
docker compose run --rm deficit ingest --config config/outages.yaml --sources weather
docker compose run --rm deficit-offline ingest --config config/outages.yaml --sources outages trade
docker compose run --rm deficit-offline preprocess --config config/outages.yaml
docker compose run --rm deficit-offline features   --config config/outages.yaml
docker compose run --rm deficit-offline detect     --config config/outages.yaml
docker compose run --rm deficit-offline tune       --config config/outages.yaml   # validation only
docker compose run --rm deficit-offline train      --config config/outages.yaml
docker compose run --rm deficit-offline evaluate   --config config/outages.yaml
```

Use `config/net_import.yaml` for the other scenario (the `ingest` step needs no `--sources` then).

How the outage file is read (`ingestion/outages.py`):
* Hours are local Kyiv time; they are converted to UTC, including the repeated hour on the DST-end day.
* Rows without a valid date or hour (spreadsheet formulas such as `=B28442`) are dropped; those days stay unknown (NaN), never 0.
* The `number` column ("in all regions", "in most regions", ...) becomes a share of regions (`outages.extent_scores` in the config; the values are approximations).
* The queue amounts (`outage_amount`) are empty for most outage hours, so they are only a covariate (`outage_queues`), not the target.
* 2025-12-10 to 2026-01-28 is excluded from the target (`outages.suspect_periods`): the dataset says it was assembled from distribution operators, not Ukrenergo.

Useful flags:
* `--sources entsoe weather` ingests only some sources.
* `--models xgb_adaptive prophet_static` trains only some models.
* `-v` turns on debug logging.

---

## 7. Data sources

| Source | What | Access | Code |
|---|---|---|---|
| ENTSO-E Transparency Platform | Actual Load, Generation per Type, cross-border physical flows (hourly) | free token: register at transparency.entsoe.eu, then e-mail `transparency@entsoe.eu` with the subject "Restful API access" | `ingestion/entsoe_client.py` |
| ACLED | conflict events, filtered to strikes on energy infrastructure | free academic account at acleddata.com; either download a CSV export to `data/raw/acled_export.csv` or set `ACLED_EMAIL`/`ACLED_PASSWORD` | `ingestion/acled.py` |
| Ministry of Energy of Ukraine | lost generating capacity per attack (`capacity_lost_MW`) | unstructured reports, entered by hand into `data/external/attacks.csv` | `ingestion/attacks.py` |
| Open-Meteo | hourly temperature and wind, population-weighted over 6 cities | no key needed | `ingestion/weather.py` |
| Energy Map: [electricity consumption limitation measures](https://energy-map.info/en/resources/0f8f9882-1fb2-47c6-81dc-31fbad914f16) | hourly planned-outage records of Ukrenergo (extent, queues) | free login, limited downloads; license not stated, cite Ukrenergo / DiXi Group | `ingestion/outages.py` |
| Energy Map: [hourly electricity imports and exports](https://energy-map.info/en/resources/56df70b0-6bc1-4c7d-a82f-284cf723438d/) | hourly imports and exports by country (since 2022 from the regulator dashboard and ENTSO-E) | same | `ingestion/trade.py` |

`attacks.csv` columns: `date, time_utc, description, capacity_lost_MW, major, source, verified`. `major=true` rows define the crisis windows and the depth/duration events.

### Telegram strike-impact dataset (`deficit collect`, then `deficit extract`)

`collect` stores public channel posts (Ukrenergo, DTEK, oblenergos) in `data/raw/telegram/<channel>.parquet`. `extract` turns the text into
structured, audited fields with regular expressions (`ingestion/telegram_extract.py`, no external NLP; every value keeps its matched span):
strike reports and damaged object types, oblasts, consumers without power, schedules (hours, queues), emergency outages, MW values,
restoration and "restrictions lifted" statements, consumption change. Outputs in `data/processed/`: `telegram_posts.parquet` (one row per post,
with the post timestamp), `telegram_daily.parquet` (per UTC day and channel, usable from `available_from` = next 00:00 UTC) and
`telegram_episodes.parquet` (one row per strike in `attacks.csv` and scope: first reports, maxima, regions, observed restoration duration).
Precision/recall against 186 hand-labelled posts, coverage per year and channel, and the known failure cases are in
`reports/telegram_extraction_quality.md` (gold labels: `tests/fixtures/telegram_gold.csv`).

High-volume oblast administrations (100+ posts/day, 20 per page) are collected in **window mode**:
`deficit collect --channels kharkivoda kyivoda ... --windows-from-attacks [--window-days 7] [--strike-share 1.0] [--max-hours 6]`
reads only the posts from [strike day - 1, strike day + 7] (UTC) of the strikes in `data/external/attacks.csv` (post id of a date: interpolation
between known points + at most 6 probe requests, pages are read forwards with `?after=<id>`; chained strikes are merged; column
`window_strike`; finished windows are listed in `data/raw/telegram/_windows.csv`). The strikes are chosen without looking at any
outcome: all of them if the estimated time fits `--max-hours`, else descending `weapons_through` (rule and estimate:
`reports/telegram_windows_plan.md`; `--plan-only` writes it without requests). Request budget (code constants in
`ingestion/telegram_public.py`): one process, 3-5 s random delay, at most 800 requests/hour, 60 s between channels, 429 honoured with
growing backoff and a stop at the 3rd 429. Channels flagged `window_collected` in `config/sources.yaml` are observed only inside these
windows, so `extract` right-censors their oblast episodes at the window end (`window_end_reason = collection_window`).

Rules: the only model inputs built from these fields are in `features/telegram.py` (severity flags of the issue day and 7 / 30-day activity from
the national channel, `telegram.feature_channels`; row d reads only posts published before the end of day d) and are used by the `*_g` / `*_tg`
direct models of `deficit events` and by `deficit restoration` (forecast of the restriction-regime duration, `evaluation/restoration.py`;
leave-later-out, labels as known at the issue time, right-censoring handled). A new feature must restrict posts with
`posts_before(df, issue_time)` or `episode_as_of(posts, strike_at, as_of)` (later posts are ignored) and get a case in `tests/test_no_leakage.py`; the observed
durations (`days_to_resolution`) are ex-post labels, never features. Findings: `reports/telegram_impact_results.md`. `excluded` channels (`telegram.exclude_channels`, default the Air Force
channel) are skipped by `extract`.

---

## 8. Outputs

* `reports/metrics.csv`: MAE/RMSE/MAPE × model × split × {all, crisis_14d, non_crisis}
* `reports/wilcoxon.csv`: significance of each model vs `xgb_adaptive`
* `reports/events_depth_duration.csv`: actual vs predicted depth and recovery days per attack
* `reports/events_summary.csv`: depth MAE (MW) and recovery MAE (days) per model
* `reports/feature_importance_xgb_*.csv`: shows how much `capacity_lost_MW` and the other attack features matter
* `artifacts/changepoints.json`: PELT breaks, BOCPD alarms, precision/recall/lag
* `reports/figures/`:
  * `changepoints.png`: deficit, attacks, PELT breaks, BOCPD probability
  * `forecasts.png`
  * `crisis_*.png`: zoom on the first 3 test attacks
  * `mae_comparison.png`

---

## 9. Known caveats (important for the thesis)

1. **The balance formula can collapse to ≈ 0.** ENTSO-E "Actual Total Load" is the load that was actually *served*, which by construction is about generation + net import. So `load − (generation + net_import)` then measures only reporting gaps and noise, not unserved demand. Before relying on it, check `processed/master.parquet`. If `deficit` is ≈ 0 during known blackout periods, switch to:
   ```yaml
   target:
     definition: demand_gap   # deficit = counterfactual peacetime demand − served load
   ```
   In `demand_gap` mode, the counterfactual demand model (calendar + weather, trained on 2019-2021) is built in `preprocessing/master.py::counterfactual_demand`. Other possible ground truths are Ukrenergo's published outage-schedule volumes or emergency-assistance imports.
2. **ENTSO-E has no Ukrainian consumption after January 2022 (verified).** `deficit coverage` (one day per quarter, areas `UA` and `UA_IPS`, 2019-2025) found hourly actual load for 2019 to January 2022, no answer from April 2022 to the end of 2025, generation per type in only 2 of 28 sampled days (even in peacetime), and cross-border flows throughout. So the thesis balance formula cannot be computed for the war years; the real-data scenarios use Ukrenergo files from Energy Map instead. Not checked: 2026 and the exact cut-off date (between January and April 2022); results are in `artifacts/entsoe_coverage.csv`.
3. **Weather** uses observed values, which is a "perfect forecast" assumption.
4. **BOCPD** runs on the daily-mean deficit, so the detection lag is at least 1 day. Its signal is P(run length ≤ 3 days), because P(r = 0) is constant under a constant hazard.
5. **Horizon.** The default `horizon_hours: 24` (day-ahead) matches Ukraine's day-ahead market (bids close at 12:00 on D-1), so `lag_1h` is dropped. `horizon_hours: 1` reproduces the thesis' exact lag set (`lag_1h, lag_24h, lag_168h`).
6. **LSTM batches.** LSTM training shuffles *windows* between epochs inside the training range. That is standard mini-batching and does not break rule 5.1.1: all training windows still come before the forecast period.
7. **The outage target is an index, not MW.** It says how widely planned outages were applied, as a share of regions (an approximation from the dataset's wording). It is a real record of deficit severity, but it saturates at 1 and says nothing about depth beyond that.
8. **Attack list.** `data/external/attacks.csv` ends in December 2024, so crisis windows and attack-based scores say nothing about 2025-2026. Detection scores are limited by `changepoints.eval_until`. The winter 2025/26 episode is therefore not matched to attacks.
9. **Few episodes.** The outage scenario has 7 validation and 3 test episodes, so model differences are indicative only.
10. **Net imports are capped** by interconnector capacity (the hourly maximum in the data is 2,233 MW), so at the cap they understate the deficit. In 2022-2023 the system was roughly balanced, so a rise in imports reflects the later waves, not the first ones.

---

## 10. Mapping to the thesis

| Thesis | Code |
|---|---|
| §1.4 structural breaks, PELT, BOCPD | `changepoints/` |
| §1.5 Prophet / LSTM / XGBoost, lag features, `capacity_lost_MW` | `models/`, `features/build.py` |
| §1.5 MAE/RMSE/MAPE, crisis weeks, Wilcoxon | `evaluation/metrics.py`, `evaluation/report.py` |
| §2.1 data sources | `ingestion/` |
| §2.2 gaps, master dataset, `deficit`, `recovery_days`, split | `preprocessing/`, `evaluation/events.py`, `config.yaml: split` |
| §3.1 three-level architecture, automatic retraining | `training/walkforward.py` |
| §3.2 precision/recall/lag of detection, threshold P > 0.7 | `changepoints/detect.py`, `training/tune.py` |
| §3.3 six-step pipeline, adaptive window, Optuna, TimeSeriesSplit | `cli.py`, `training/` |

### Next research steps

* Fill in `capacity_lost_MW` and verify attacks.
* Resolve the target definition (caveat 1).
* Try `horizon_hours` ∈ {1, 24, 48}.
* Add a direct event-level model: predict depth/duration at attack time from `capacity_lost_MW`, season and pre-attack state.
