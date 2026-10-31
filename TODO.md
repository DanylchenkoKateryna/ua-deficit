# TODO

Results so far: see RESULTS.md.

Status: the pipeline runs end to end in Docker on synthetic data and on three real-data scenarios built from Energy Map
files (README "Real data without ENTSO-E"): `demand` (lost demand in MW, Jan-Oct 2022), `outages` (planned-outage index,
2023-2026) and `net_import` (MW, 2019-2026).

## ENTSO-E (token received, coverage checked)
- [x] `deficit coverage` run (one day per quarter, areas UA and UA_IPS, 2019-2025): hourly load only until January 2022,
      no answer from April 2022, generation per type almost never, cross-border flows throughout. See README §9.2.
- [ ] Optional: refine the cut-off date (daily check Jan-Apr 2022) and check 2026; try areas `UA_BEI`.
- [ ] ENTSO-E consumption 2019-01 .. 2022-01 could replace/cross-check the balance file's consumption for the same period.

## Data work
- [ ] **Look for hourly consumption and generation after 2022-10-27** (the balance file stops there). Candidates to check:
      Ukrenergo's own reports, the regulator dashboard, day-ahead market volumes and prices (Energy Map "DAM indexes and
      weighted average prices"; prices are known before the delivery day, so they are a legitimate feature for outages).
- [ ] `data/external/attacks.csv`: fill `capacity_lost_MW`, mark `verified`, and **add the attacks after 2024-12-25**
      (until then crisis windows and detection scores say nothing about 2025-2026; `changepoints.eval_until` hides them).
      `data/raw/2026_07_01_elektro_boiovi_dii_ukraina_minenergo.csv` has source links per date.
- [ ] Re-download `2026_07_01_electricity_outages.csv`: 328 rows (13 days in April 2026) are corrupted spreadsheet formulas.
- [ ] Confirm what `outage_amount` means ("Fields description" tab of the dataset page); the code treats it as outage queues.
- [ ] ACLED: free account, then `data/raw/acled_export.csv` or credentials in `.env` (optional).

## Optional
- [ ] LSTM models in Docker: `$env:EXTRAS="lstm"; docker compose build` (large download, adaptive LSTM run is about 1 h).
- [ ] Try `horizon_hours` 1 / 24 / 48.
- [ ] Initial git commit (nothing is committed yet).
