# Final evaluation protocol: regional forecast of restoration duration

Fixed on 2026-10-31, before the collection of 11 new channels was finished, before the data were rebuilt with the new
extraction rules and before any look at the test period on these data. After fixing, the protocol is not changed; all
results are reported whether they are good or not.

## Background

The test period (strikes from 2025-01-01, `telegram.region_validation_end = 2024-12-31`) was already looked at four times
during development: 60 strikes × 7 days, 76 strikes × 7 days, oblast history features added, 14-day windows. After that
the test was closed: development continued only in `region-restoration --dev` mode (rolling validation on strikes before
2025, test rows are not computed). This protocol responds to the instability of model selection shown by the development mode.

## Amendment 1 (2026-10-31, before looking at any result on the new data): extended strike list

The Air Force list (77 massive strikes) covers only the largest waves, while attacks on the energy sector happen on
15–20 days a month. Therefore, the regional part (window collection, extraction, `region-restoration`) gets the extended
list `data/external/attacks_extended.csv`; the main forecasting pipeline and `attacks.csv` are unchanged.

- Event = a UTC day with at least one attack report (`is_attack_report`) in the continuously collected channels
  (Ukrenergo, DTEK, Lvivoblenergo, Sumyoblenergo, Khmelnytskoblenergo, Zhytomyr ztoe). Strike time = time of the first such report.
- Event selection does not depend on outcomes (restoration, outages): rank = number of channels, then number of reports.
- The unit of analysis stays the pair "strike × oblast"; because events overlap, the main window stays 30 days,
  and the "until the next strike" window is reported as an additional one.
- The test boundary, success criterion, model selection rule and additional metrics are unchanged. The set of test
  strikes (from 2025-01-01) changes because events are added; the test result on the new data has not been looked at yet.
- The quality of day detection is assessed on a random sample of 60 days (automatic labelling, spot-checked by the author).
- An independent check against the ACLED database (keyword filter for the energy sector) is planned in parallel; it is used
  only for comparison and does not change the protocol.

## Data (frozen before model selection)

- Telegram: 6 oblast military administrations with windows [strike − 1 d, strike + 14 d] for all 76 strikes + 11 channels
  collected the same way (`config/sources.yaml`, `window_collected: true`); national channels and oblenergos unchanged.
- Extraction: rules of `telegram_extract.py` / `telegram_regional.py` as of 2026-10-31 (the version with the new labels
  `tests/fixtures/telegram_region_gold_v2_*`).
- Sequence: `extract` → `region-restoration --dev` → model selection → one run of `region-restoration` (normal mode).

## Main metric

- **Within-strike C** (Harrell's C over pairs of episodes of the same strike, `harrell_c_within`), label window **30 days**,
  all test strikes with at least one comparable pair.
- **Success criterion:** the lower bound of the 95 % bootstrap interval over strikes (2000 resamples, seed 42, `ranking_bootstrap`)
  > 0.5, i.e. one-sided p < 0.05 for C > 0.5.
- The within-strike C of `oblast_median` and `global_median` (= 0.5) is always shown alongside.

## Model selection (development mode only)

- Candidates: all regression variants of the development mode (ridge, ipcw, aft, xgbaft, ensembles); baselines are shown
  but are not selected as a "model with ranking skill".
- Criterion: mean within-strike C over 5 consecutive folds (`region_cv_folds`, warm-up `region_dev_min_obs = 15`),
  30-day window. One-standard-error rule: among the variants no worse than the best by more than one SE,
  the simplest is taken (`complexity_key` in `region_dev.py`).
- The choice is written to a file before the test is run; exactly one model is evaluated on the test.

## Additional metrics (reported for any result; no conclusions about success are drawn from them)

1. The same model and metric in the strict "until the next strike" window.
2. MAE on completed episodes against `global_median` (Wilcoxon over strikes), for the model chosen by the
   one-standard-error rule on MAE.
3. Binary task "restoration longer than T h", T = 72 (main threshold) and 168: Brier skill against the base rate, and AUC.

## Limitations stated in the thesis

- Duration labels are approximations from Telegram texts; the `gold_v2` labels were made automatically and are spot-checked by the author.
- Strikes overlap and episodes of the same strike are dependent; therefore, intervals are computed over strikes.
- The earlier looks at the test (see above) are part of an honest description.
