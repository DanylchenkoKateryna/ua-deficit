# UA Deficit Forecast

**Forecasting the depth and duration of electricity deficit after attacks on infrastructure using machine learning methods**

Software of the master's thesis of K. Danylchenko (Lviv Polytechnic National University, group PMID-11, 2026). The thesis
itself is in `docs/` (Ukrainian, Word and PDF).

The program works with open hourly data on Ukraine's power system (Energy Map, DiXi Group) and the list of massive strikes. It:

* forecasts, on the day of a strike, the **depth** (largest daily level in the next 30 days) and the **duration** of the deficit;
* forecasts the **restoration time by oblast** and ranks oblasts by how long restoration will take;
* compares **retraining strategies** of day-ahead forecasting models (by schedule, after a strike, on a BOCPD break signal,
  on growing model error measured by CUSUM) with static models and baselines;
* detects **structural breaks** (PELT for the history, BOCPD as an online signal).

The deficit is described by four open series (scenarios): lost demand (MW), net imports (MW), the planned-outage index
(share of oblasts) and the day-ahead market price relative to the price cap.

---

## 1. Requirements

* Docker Desktop (Windows, macOS or Linux). Nothing else is installed on the host: Python and all libraries live in the image.
* About 1.3 GB of disk for the image, 8 GB of RAM recommended, no GPU.
* Network access only for building the image and for downloading weather (Open-Meteo) during `ingest`.

## 2. Quick start

All commands are run from the project folder.

```bash
docker compose build                                           # build the image (once and after code changes)
docker compose run --rm --entrypoint python deficit -m pytest  # automated tests
docker compose run --rm deficit-offline run --synthetic        # whole pipeline on generated data, no network
```

The synthetic run needs no input files. Its results go to `data/synthetic/`, `artifacts/synthetic/` and
`reports/synthetic/`; they only show that the program works and are not research results.

## 3. Running on real data

1. Unpack the input data (paths inside the archive start at the project folder):

   ```powershell
   Expand-Archive data\raw.zip -DestinationPath . -Force
   ```

   For the regional part also unpack `data\raw_telegram_1.zip` … `data\raw_telegram_3.zip` (public Telegram posts).

2. Choose a scenario and run all steps:

   ```bash
   docker compose run --rm deficit run --config config/outages.yaml
   ```

   | Config | Target series | Period |
   |---|---|---|
   | `config/demand.yaml` | lost demand, MW (expected consumption without the war minus actual) | 2019 – 2022-10 |
   | `config/net_import.yaml` | net imports, MW | 2019-07 – 2026-06 |
   | `config/outages.yaml` | planned-outage index, share of oblasts (0–1) | 2023 – 2026-06 |
   | `config/price.yaml` | day-ahead price / price cap (0–1) | 2019-07 – 2026-06 |

   `run` executes `ingest → preprocess → features → detect → train → evaluate → events`. The first step converts the CSV files
   to Parquet and downloads weather, so it uses the `deficit` service (with network). Every step can also be run on its own,
   for example `docker compose run --rm deficit-offline evaluate --config config/outages.yaml`.

3. Results: `reports/<scenario>/` (tables and figures), `artifacts/<scenario>/` (forecasts, change points),
   `data/processed/<scenario>/` (prepared series and features).

### Regional restoration (oblast level)

```bash
docker compose run --rm deficit-offline extract --config config/telegram_extended.yaml             # facts from the Telegram posts
docker compose run --rm deficit-offline region-restoration --config config/telegram_extended.yaml  # restoration time by oblast
docker compose run --rm deficit-offline energy-map-select --config config/telegram_extended.yaml   # model choice on development data
docker compose run --rm deficit-offline energy-map-test --config config/telegram_extended.yaml     # single pre-registered test
```

The test protocol is fixed in `docs/preregistration_energy_map.md`; `energy-map-test` evaluates only the model written by
`energy-map-select`.

### Results of the thesis

The result files used in the thesis are kept in archives: `reports.zip`, `artifacts.zip` and `data/processed.zip`
(unpack them in the project folder to view them without recomputing).

## 4. Commands

| Command | Purpose |
|---|---|
| `ingest` | read the source CSV files and weather, write Parquet files to `data/raw/` |
| `preprocess` | target series, missing values, deficit episodes |
| `features` | lags, calendar, weather and strike features, using only data known at the forecast time |
| `detect` | structural breaks: PELT (history), BOCPD (online signal) |
| `tune`, `tune-triggers` | hyperparameters and retraining triggers, validation period only |
| `train` | day-ahead models and retraining strategies (walk-forward) |
| `evaluate` | MAE, RMSE, paired Wilcoxon test of daily errors against a reference model (`evaluation.wilcoxon_reference`), figures |
| `events` | depth and duration of the deficit after each strike |
| `restoration` | duration of the national restriction regime |
| `extract`, `derive-strikes` | facts from Telegram posts; extended list of strike days |
| `region-restoration` | restoration time by oblast |
| `energy-map-select`, `energy-map-test` | model choice and the single test on Energy Map labels |
| `collect`, `collect-black-marble`, `ntl-response` | collection of public Telegram pages and NASA night lights (network) |
| `run` | `ingest` … `events` in one call |

Options: `--config <file>` (scenario), `--synthetic` (generated data), `--models <names>` (train only these models),
`--dev` (development mode of `restoration` / `region-restoration`), `-v` (debug log).

## 5. Project layout

```
config/            scenario parameters (YAML): periods, splits, thresholds, models; config.yaml holds the defaults
data/external/     strike lists: attacks.csv (massive strikes), attacks_extended.csv (all strike days)
data/*.zip         input data (raw.zip, raw_telegram_*.zip) and prepared tables (processed.zip)
src/deficit/
  cli.py           entry point: deficit <step> [--config ...]
  ingestion/       Energy Map files, weather, strike lists, Telegram posts, night lights
  preprocessing/   target series, missing values, episodes
  features/        lag, calendar, weather and strike features
  changepoints/    PELT and BOCPD, detection quality
  models/          seasonal naive, Prophet, XGBoost (LSTM optional)
  training/        walk-forward training and retraining strategies, tuning
  evaluation/      metrics, Wilcoxon test, post-strike depth/duration, survival models, regional restoration
tests/             unit tests, leakage tests, end-to-end test on synthetic data
docs/              thesis (Word, PDF) and pre-registered test protocols
```

## 6. Methodological rules enforced by the code

* **Chronological split.** Train, validation and test periods follow each other in time (set per scenario in `config/*.yaml`);
  rows are never shuffled, cross-validation uses expanding time windows.
* **No data leakage.** A feature for target hour `t` uses only data available at the issue time `t − horizon`; strike
  features and Telegram facts use only posts published before that time. `tests/test_no_leakage.py` checks this by
  replacing future values with huge numbers and verifying that the features do not change.
* **Weather.** Observed weather stands in for a weather forecast (a "perfect forecast" assumption); weather of day D is
  used only once day D is over.
* **Walk-forward training.** A model retrained at time T learns only from target hours before T; a BOCPD value of day d
  is used from the next day.
* **The test period is not used for tuning.** Hyperparameters, windows and thresholds are chosen on validation data only;
  PELT breakpoints are never model inputs.
* **Baselines.** Every model is compared with the seasonal naive forecast (or persistence for post-strike forecasts) and
  with weekly retraining (`xgb_rolling`).
* **Time.** Data are stored in UTC at hourly resolution; Kyiv local time is used only for calendar features.

## 7. Data sources

| Source | Data |
|---|---|
| Energy Map (DiXi Group), energy-map.info | hourly balance, imports/exports, planned outages, day-ahead market results, massive strikes, outages and restoration by oblast |
| Open-Meteo | hourly temperature and wind |
| Public Telegram channels (t.me/s) of Ukrenergo, DTEK, oblenergos and oblast administrations | reports on strikes, outages and restoration |
| NASA Black Marble | night lights (optional check of large blackouts; about 10 GB, downloaded by `collect-black-marble`) |

ENTSO-E and ACLED clients are also included (`ingestion/entsoe_client.py`, `ingestion/acled.py`); they need personal
tokens in `.env` (template: `.env.example`) and are not used by the scenarios above.
