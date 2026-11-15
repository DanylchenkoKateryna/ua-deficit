# DRAFT blind evaluation protocol v2: restoration duration by oblast on Energy Map labels (strikes from 2026-06-27)

Status: **DRAFT, not frozen.** It is frozen (date and git state are recorded at the end) before the fresh Energy Map file is opened for the first time
(`data/raw/energy_map/new/`, downloaded separately, neither used nor looked at in development) and before any model metric is computed
on strikes from 2026-06-27. After freezing, the protocol is not changed; the result is reported whether it is good or not.
The earlier protocol `docs/preregistration_energy_map.md` is not changed.

## Background

- The first test (strikes 2025-01-01 .. 2026-06-26) was run once: within-strike C 0.699 against 0.652 for `oblast_median`, difference +0.047
  [0.023; 0.071] (`reports/telegram_extended/energy_map_test_results.md`). This period is **used up**: all strikes up to 2026-06-26 are now development data.
- Development v2 (`reports/telegram_extended/energy_map_improve_dev.md`): expanding window, 8 folds over all strikes up to 2026-06-26 (767 strikes, 2402 observed pairs).
  Over the whole period the old model `ridge_so_a100` is only slightly better than `oblast_median` (C 0.741 against 0.721). New features from the Energy Map records known at forecast
  time (records dated strictly before the strike day: size of the last outage, number of oblasts with records in the past week, age of the last restoration record)
  give C ≈ 0.81 against 0.72 (difference positive in 8/8 folds), and they also improve the MAE of the ridge/IPCW models (−6 h). Features of the oblast's own duration history, geography and seasonality
  give nothing on their own. This is development, not a blind result; about 67 variants were tried, so the winner's estimates are optimistic.
- Main risk: the gain comes from features of the source's "record density", so it may depend on how the file is curated. The fresh file is the check of this.

## Data (frozen)

- Label: Energy Map, definition `energy_map_labels.build_labels` unchanged (outage on the strike day or the next day; duration until the first later restoration record
  within 30 days, otherwise censored; time step = one day). Sources: `*_oblasti_osr.csv`, `*_oblasti_minenergo.csv` from the fresh download; old records are taken from the `2026_07_01_...` files
  where the fresh file does not contain them (after merging, duplicates are removed as in `load_energy_map`). For strikes ≤ 2026-06-26, labels and features stay as in development:
  **training data are not recomputed from the fresh file**, so that the model is the one chosen in development.
- Strikes: `data/external/attacks_extended.csv`, extended to the end of the fresh data by the same `derive-strikes` procedure. Test: strikes from 2026-06-27 to the last day
  for which the 30-day window fits fully into the fresh data (**the end date is recorded at freezing**: date of the last record minus 31 days).
- Eligibility: at least 50 strikes scored by the model and at least 100 comparable within-strike pairs. Otherwise the test is not run; we wait for new data,
  and the protocol stays in force (neither the model nor the criteria change).
- Training: walk-forward, as in `energy_map_test.walk_forward_chosen`: the forecast for strike i is trained on pairs of strikes j < i (development + test strikes already past) with labels known before
  the issue time of strike i (`labels_as_of`). Features are computed as in `energy_map_improve.build_dev_table` (only records dated strictly before the strike day; duration history only from episodes
  whose restoration day ended before the issue time).

## Models (fixed from development v2)

- **Primary (for C):** `xgbaft_so_em_d2n25`, chosen by the one-SE rule on mean C in development (C 0.8135 over 8 folds; best `ens_top3_C` 0.8244, SE 0.0132; the rule
  picks the simplest within one SE). Boosted AFT, depth 2, 25 rounds, eta 0.1, lambda 10, seed 42; features `strike_oblast` + `em_*`.
- **Secondary (for hours):** `ipcw_so_all_a100`, the best MAE in development (68.1 h against 74.3 for `oblast_median`; difference over folds −6.2, SE 2.0, 8/8 folds). The one-SE rule on MAE
  selects nothing meaningful (45 variants within one SE), so the best MAE is taken as the secondary model; this deviation from the rule is stated openly.
- For reference (no conclusions): the old `ridge_so_a100` (model of the first test), `ridge_so_em_a100`, `oblast_median`, `global_median`.
- Hyperparameters, feature sets, seed and code (git state) are fixed at the moment of freezing.

## Main metric and success criterion

- Cohort: **all oblasts**; metric: **within-strike C** (`harrell_c_within`).
- Comparison: the primary model against `oblast_median` on the same test pairs.
- **Success criterion:** the lower bound of the 95 % paired bootstrap interval over strikes (2000 resamples, seed 42, `energy_map_test.paired_c_bootstrap`) for the difference
  C(`xgbaft_so_em_d2n25`) − C(`oblast_median`) > 0.
- Power guide (not a criterion): in development the difference is ≈ +0.06 (bootstrap SE ≈ 0.007 on 767 strikes); on ~60 strikes SE ≈ 0.024, i.e. the bound is 0.06 − 1.96·0.024 ≈ +0.013:
  a short test can barely confirm a gain of this size.

## Secondary metric (no conclusion about success; reported for any result)

- MAE on observed episodes: `ipcw_so_all_a100` against `oblast_median`; paired bootstrap over strikes (2000, seed 42) of the difference in total MAE (`energy_map_improve.paired_mae_bootstrap`) and
  Wilcoxon on the mean errors of strikes; the same for the primary model.
- Additionally: C without the frontline oblasts (Donetsk, Luhansk, Zaporizhzhia, Kherson), C against `global_median` and against the old `ridge_so_a100`, C of the primary model against 0.5, share of strikes where the model is better.

## Limitations stated in the thesis

- The label is the first restoration report, not full restoration; the records are events, not a continuous series; the time step is one day.
- The cohort is defined by the label (outage on the strike day or the next day), so the results are conditional on it.
- The gain in development comes from Energy Map record-density features; transfer to another file or curation regime is not guaranteed.
- Episodes of the same strike are dependent; intervals are computed over strikes. There is one main test; secondary and additional metrics are descriptive, and no multiplicity correction is applied.

## Freezing

Date and git state of freezing: _(to fill in)_. End date of the test: _(to fill in)_. Date of the last record of the fresh file: _(to fill in)_.
