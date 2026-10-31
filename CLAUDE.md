# CLAUDE.md

Project description, pipeline, and all rules live in README.md — read §5 (Rules) before changing anything.

Key points:
- Python package in `src/deficit`, CLI `deficit <step> [--synthetic]`, config in `config/config.yaml`.
- Never introduce look-ahead: features for hour t may only use data known at t - horizon (README §5.1.2-3).
  Extend `tests/test_no_leakage.py` when adding a feature.
- Never tune on the test period (2024-2025); PELT output must never become a model input.
- No local venv: everything runs in Docker (README "Docker"). After code changes run `docker compose build`, then
  `docker compose run --rm --entrypoint python deficit -m pytest` and
  `docker compose run --rm deficit-offline run --synthetic --models prophet_static xgb_static xgb_adaptive`.
- The image is built without TensorFlow by default (EXTRAS empty), so LSTM models/tests are skipped there.
- Real-data scenarios: `config/outages.yaml` (outage index) and `config/net_import.yaml` (MW), run with
  `--config config/<name>.yaml`; outputs in `data/processed/<name>`, `artifacts/<name>`, `reports/<name>`.
  Findings and caveats are in RESULTS.md; keep it in sync when results change.
- Always show results next to `seasonal_naive` and the `xgb_rolling` ablation; do not claim that change-point-triggered
  retraining helps unless it beats `xgb_rolling`.
