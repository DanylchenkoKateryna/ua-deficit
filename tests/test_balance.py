"""Hourly balance file (Energy Map) -> consumption in MW, and the demand-loss scenario end to end."""
import numpy as np
import pandas as pd
import pytest

from deficit.changepoints.detect import detect
from deficit.config import load_config
from deficit.evaluation.report import evaluate
from deficit.features.build import build_features
from deficit.ingestion.attacks import load_attacks
from deficit.ingestion.balance import fetch_balance, hourly_balance
from deficit.io import read_parquet
from deficit.preprocessing.master import build_master
from deficit.training.run import train_all

GEN = {"АЕС": 9000.0, "ТЕС": 4000.0, "ТЕЦ": 1000.0, "ВДЕ": 300.0, "ГАЕС": 150.0, "ГЕС": 700.0}


def _rows(day, hour, scale=1.0, island=False, repeat=1):
    label = f"{hour:02d}:00-{hour + 1:02d}:00"
    out = []
    for _ in range(repeat):
        for k, v in GEN.items():
            out.append((day, label, "генерація", k, v * scale))
        if island:
            out.append((day, label, "генерація", "Бурштинський енергоострів", 800.0))
        out += [(day, label, "міждержавні перетоки", "Україна - ЄС", -300.0),
                (day, label, "міждержавні перетоки", "Україна - Білорусь, РФ", 0.0),
                (day, label, "міждержавні перетоки", "Україна - Молдова", -30.0),
                (day, label, "споживання", "закачка ГАЕС", -200.0)]
    return out


COLS = ["date", "time", "balance_component", "balance_subcomponent", "amount"]


def test_consumption_formula_island_and_dst():
    rows = []
    for h in range(24):
        rows += _rows("2023-10-29", h, island=True, repeat=2 if h == 3 else 1)    # DST-end day: hour 03 occurs twice
    out, stats = hourly_balance(pd.DataFrame(rows, columns=COLS), "Europe/Kyiv")
    assert len(out) == 25 and out.index.is_unique and stats["island_rows_ignored"] == 25
    # 15150 generation - 330 flows (exports are negative) - 200 pumping; the island's 800 MW is not counted
    assert (out["consumption_MW"] == 15150 - 330 - 200).all()


@pytest.fixture
def demand_cfg(tmp_path):
    c = load_config("config/demand.yaml").model_copy(deep=True)
    c.paths.data_dir = tmp_path / "data"
    c.paths.artifacts_dir = tmp_path / "artifacts"
    c.paths.reports_dir = tmp_path / "reports"
    c.period.start, c.period.end = "2021-03-01", "2022-05-31"
    c.split.train_end, c.split.validation_end = "2021-12-31 23:00", "2022-03-31 23:00"
    c.weather.start, c.weather.end = None, None
    c.changepoints.eval_until = None
    c.models.xgboost.n_estimators = 40
    c.ensure_dirs()
    return c


def _write_inputs(cfg):
    rng = np.random.default_rng(1)
    rows = []
    for day in pd.date_range("2021-03-01", "2022-05-31"):
        scale = 0.65 if day >= pd.Timestamp("2022-02-24") else 1.0           # the break
        for h in range(24):
            season = 1 + 0.1 * np.cos(2 * np.pi * (day.dayofyear - 20) / 365) + 0.05 * np.sin(2 * np.pi * (h - 8) / 24)
            rows += _rows(day.strftime("%Y-%m-%d"), h, scale=scale * season * (1 + rng.normal(0, 0.01)))
    pd.DataFrame(rows, columns=COLS).to_csv(cfg.raw_dir / cfg.balance.file, index=False)
    idx = pd.date_range(cfg.start_ts, cfg.end_ts, freq="h", name="datetime")
    pd.DataFrame({"temperature_C": 8 - 12 * np.cos(2 * np.pi * (idx.dayofyear - 15) / 365.25), "wind_speed_kmh": 12.0},
                 index=idx).to_parquet(cfg.raw_dir / "weather.parquet")
    pd.DataFrame({"date": ["2022-02-24"], "time_utc": ["03:00"], "description": ["invasion"], "capacity_lost_MW": [0.0],
                  "major": [True], "source": ["test"], "verified": [True]}).to_csv(cfg.external_dir / "attacks.csv", index=False)



def test_demand_loss_pipeline_end_to_end(demand_cfg):
    cfg = demand_cfg
    _write_inputs(cfg)
    fetch_balance(cfg)
    master = build_master(cfg)
    before, after = master.loc[:"2022-02-20", "deficit"], master.loc["2022-03-05":, "deficit"]
    assert abs(before.mean()) < 600 and after.mean() > 3000                  # lost demand appears after the break
    build_features(master, cfg)
    cp = detect(master, load_attacks(cfg), cfg)
    assert cp["bocpd"]["scores"]["recall"] == 1.0                            # the known break is found
    names = ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_adaptive"]
    assert set(train_all(cfg, names)) == set(names)
    res = evaluate(cfg)
    m = res["metrics"].set_index(["model", "split", "subset"])["MAE"]
    assert m[("xgb_static", "test", "all")] > 2 * m[("xgb_rolling", "test", "all")]   # a static model misses the new regime
    assert read_parquet(cfg.processed_dir / "master.parquet").shape[0] == len(pd.date_range(cfg.start_ts, cfg.end_ts, freq="h"))


def test_post_event_forecast_depth_and_duration(demand_cfg):
    from deficit.evaluation.post_event import depth_duration, run_post_event

    # rule check: duration = days until 3 consecutive days at or below the threshold
    d, dur, cens = depth_duration(np.array([5, 6, 4, 1, 1, 1, 9.0]), threshold=2)
    assert (d, dur, cens) == (9.0, 3, False)
    assert depth_duration(np.array([5, 6, 7.0]), 2)[2] is True            # never recovers: censored

    cfg = demand_cfg
    _write_inputs(cfg)
    fetch_balance(cfg)
    master = build_master(cfg)
    build_features(master, cfg)
    detect(master, load_attacks(cfg), cfg)
    res = run_post_event(cfg)
    ev = res["events"]
    inv = ev[(ev["trigger"] == "attack") & (ev["issue_day"] == "2022-02-24")].iloc[0]
    # nothing in the peacetime training data shows deficits, so a static model misses the new regime; persistence does not
    assert abs(inv["persistence_depth"] - inv["actual_depth"]) < abs(inv["xgb_static_depth"] - inv["actual_depth"])
    assert {"depth_MAE", "duration_MAE_days"} <= set(res["summary"].columns)
