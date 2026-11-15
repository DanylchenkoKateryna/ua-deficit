# Protocol (DRAFT): national forecast of the duration of the restriction regime

Status: draft for the author. It is fixed (date, git commit) before the first look at the test on the extended strike list.
After fixing, it is not changed; all results are reported whether they are good or not.

## Background

This target (`deficit restoration`, `days_to_resolution` of the national episode) has already been evaluated on the test: 77 Air Force
strikes, 60 test strikes, 21 of them with a completed episode. No model beat the median of past durations (MAE 6.45 d against 9.7 d for ridge
and 12.0 d for AFT). This test **has already been looked at**, so it cannot serve as independent evidence on the same list. The extended list
(`data/external/attacks_extended.csv`, 890 strikes, 517 of them before 2025-01-01) is a new sample: model selection is done only in the mode
`deficit --config config/telegram_extended.yaml restoration --dev`; the test (strikes from 2025-01-01) on the extended list has not been computed yet,
but 60 of the 77 test strikes are also in it, so the independence is partial. This is stated in the thesis.

## Target and data

- Target: `days_to_resolution` (from the first post about the strike to Ukrenergo's post cancelling the restrictions; otherwise "all consumers have power"),
  right-censored at 30 days. Extended list: attacks.csv + days with attack reports in the continuous channels.
- The forecast is issued at the end of the strike day. Features: `weapons_through` (Air Force strikes only), gap to the previous strike,
  strikes in the last 30 days, heating season, temperature (observed = perfect weather forecast) and the history of earlier labels known before 00:00 UTC
  of the strike day. `n_channels`, `n_attack_posts` (counts over the whole day) are not features (test `tests/test_no_leakage.py`).
- Training labels of earlier strikes are taken as they were known at the time the forecast is issued.

## Main metric

- **MAE in days on strikes with a completed episode** (observed duration), model against `median_obs` (median of the observed durations
  known at forecast time), all test strikes with enough history (>= `duration_dev_min_obs`).
- Auxiliary (always reported): censoring-aware lower-bound MAE (neutralises the "short forecasts win" bias;
  the constant 30 d is shown as a control), Harrell's C over strikes, Wilcoxon and block bootstrap.

## Model selection (development mode only)

- Candidates: baselines (`median_obs`, `last_episode`, `prev_strike`, `km_global`), ridge, AFT, IPCW-ridge, XGBoost-AFT on the feature sets
  `strike`, `hist`, `strike_hist`; penalties {1, 10, 100}.
- Criterion: mean MAE over 5 consecutive folds (expanding window), one-standard-error rule; among the variants within 1 SE
  the simplest is taken (complexity order: baselines < ridge < AFT by feature set < IPCW < XGB-AFT; a larger penalty = simpler).
- The choice is written to a file before the test is run. Exactly one model (or a baseline, if one is chosen) is evaluated on the test.

## Success criterion

The model chosen in the development mode is better than `median_obs` if all of the following hold:
1. the mean difference of absolute errors over strikes is < 0, its 95 % block bootstrap interval (blocks of 10 consecutive strikes, 2000 resamples,
   seed 42) lies entirely below 0, and the one-sided Wilcoxon p < 0.05;
2. the lower-bound MAE is not worse than that of `median_obs`;
3. Harrell's C over strikes is > 0.5.

If a baseline is chosen, there is no success by construction, and the result is stated as such ("no model is better than the median").

## Limitations stated in the thesis

- Strikes in a chain share one cancellation post; episodes are dependent and p-values are optimistic (the block bootstrap reduces this but does not remove it).
- Labels are approximations from Telegram texts; the measured duration is the time until the national regime is cancelled, not restoration in an oblast.
- The observed MAE is biased in favour of short forecasts (long episodes are censored more often), so the lower bound and C are always shown alongside.
- Out-of-process risks: the feature set and the grid were chosen in the development mode on the same strikes on which the CV folds are defined.
