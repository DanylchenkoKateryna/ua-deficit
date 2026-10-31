import pytest

from deficit.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """Config pointed at a temp dir, with a short period so tests stay fast."""
    c = load_config(synthetic=True)
    c = c.model_copy(deep=True)
    c.paths.data_dir = tmp_path / "data"
    c.paths.synthetic_dir = tmp_path / "synthetic"
    c.paths.artifacts_dir = tmp_path / "artifacts"
    c.paths.reports_dir = tmp_path / "reports"
    c.period.start = "2021-06-01"
    c.period.end = "2023-02-28"
    c.split.train_end = "2022-01-31 23:00"
    c.split.validation_end = "2022-09-30 23:00"
    c.models.xgboost.n_estimators = 60
    c.adaptive.retrain_every_days = 30
    c.ensure_dirs()
    return c
