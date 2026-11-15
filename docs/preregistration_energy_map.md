# Final evaluation protocol: restoration duration by oblast on Energy Map labels

Fixed on 2026-11-09, before model selection under this protocol and before any model metric was computed on the test period
with the Energy Map label. After fixing, the protocol is not changed; the result is reported whether it is good or not.

## Background

- Duration labels from Telegram (regular expressions over texts) gave 385 observed "strike × oblast" pairs out of 3 177 (88 % censored);
  the pre-registered test on them was not passed (within-strike C 0.501, `docs/preregistration_region.md`).
- The new label is the Energy Map records (DiXi Group, https://energy-map.info; data of DSOs / oblast military administrations and the Ministry of Energy, collected and checked by people):
  `data/raw/energy_map/2026_07_01_elektro_boiovi_dii_oblasti_{osr,minenergo}.csv`. Label definition: `evaluation/energy_map_labels.py`.
  A "strike × oblast" pair has a label if an outage is recorded on the strike day or the next day; the duration runs until the first
  later restoration record within 30 days (otherwise censored). This is the **time to the first restoration report**, not to full restoration.
- The development mode (strikes up to 2024-12-31) has already been looked at: `reports/telegram_extended/energy_map_labels_dev.md` (within-strike C
  of the best models ≈ 0.81 against 0.77 for `oblast_median`). The test period (strikes from 2025-01-01 to 2026-06-26) with this label
  has not been looked at: only the pair counts are known (641, 423 of them observed).

## Data (frozen)

- Strikes: `data/external/attacks_extended.csv`; features are the same as in `region-restoration` (`config/telegram_extended.yaml`).
- Label: Energy Map, 30-day window. Test: strikes from 2025-01-01 to the last day for which the 30-day window fits fully into the data
  (2026-06-26); pairs with an incomplete window are censored at the end of the data, as in the development mode.

## Main metric and success criterion

- Cohort: **all oblasts**.
- Metric: **within-strike C** (Harrell's C over pairs of episodes of the same strike, `harrell_c_within`).
- Comparison: the chosen model against `oblast_median` (median of the oblast's past durations) on the same test pairs.
- **Success criterion:** the lower bound of the 95 % paired bootstrap interval over strikes (2000 resamples, seed 42) for the difference
  C(model) − C(`oblast_median`) > 0.
- `global_median` (C = 0.5) and `oblast_median` are always shown alongside, as well as the model's C with a 95 % interval and p for C > 0.5.

## Model selection (development mode only)

- Candidates: regression variants of the development mode (ridge, ipcw, aft, xgbaft, ensembles); baselines are not selected.
- Criterion: mean within-strike C over 5 consecutive folds of the development mode, cohort "all oblasts", 30-day window.
  One-standard-error rule: among the variants no worse than the best by more than one SE (SE over folds),
  the simplest is taken (`complexity_key` in `region_dev.py`).
- The choice is written to `reports/telegram_extended/energy_map_chosen_for_test.md` before the test is run; exactly one model is evaluated on the test.
- Training for the test: the same walk-forward procedure as in the normal `region-restoration` mode (training labels are taken
  as they were known at forecast time).

## Additional metrics (reported for any result; no conclusions about success are drawn from them)

1. The same model and metric without the frontline oblasts (Donetsk, Luhansk, Zaporizhzhia, Kherson).
2. MAE on observed episodes against `global_median` (Wilcoxon over strikes).
3. Within-strike C against 0.5.

## Limitations stated in the thesis

- The label is the first restoration report, not full restoration; the records are events, not a continuous series (no record ≠ no outage).
- The Energy Map label hardly correlates with the Telegram labels (Spearman 0.01 on common pairs): they measure different things.
- A large part of C is explained by differences between oblasts (`oblast_median` ≈ 0.77 in development), so the criterion is an advantage over `oblast_median`.
- Strikes overlap and episodes of the same strike are dependent; intervals are computed over strikes.
