"""Outage-based target (config/outages.yaml): parsing, time handling, episodes, end-to-end run."""
import numpy as np
import pandas as pd
import pytest

from deficit.changepoints.detect import detect
from deficit.config import load_config
from deficit.evaluation.events import episodes, match_episodes
from deficit.evaluation.report import evaluate
from deficit.features.build import build_features, feature_columns
from deficit.ingestion.attacks import load_attacks
from deficit.ingestion.outages import fetch_outages, load_outages, parse_amount, to_utc_hourly
from deficit.io import read_parquet
from deficit.preprocessing.master import build_master
from deficit.training.run import train_all


def test_parse_amount():
    assert parse_amount("2") == 2.0
    assert parse_amount("від 0,5 до 3") == 1.75
    assert np.isnan(parse_amount(np.nan)) and np.isnan(parse_amount(""))


def test_dst_duplicates_and_formula_rows():
    rows = [("2023-10-29", f"{h:02d}:00-{h + 1:02d}:00") for h in range(24)]
    rows.insert(4, ("2023-10-29", "03:00-04:00"))             # the repeated hour on the DST-end day
    df = pd.DataFrame(rows, columns=["date", "hour"])
    df.loc[len(df)] = [np.nan, "=B28442"]                     # spreadsheet artefact
    out, stats = to_utc_hourly(df, "Europe/Kyiv")
    assert stats["dropped_unparseable"] == 1
    assert len(out) == 25 and out.index.is_unique             # 25 distinct UTC hours, none lost
    assert out.index.min() == pd.Timestamp("2023-10-28 21:00", tz="UTC")   # local midnight = UTC+3


def test_episodes_merge_short_gaps_and_threshold():
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h", tz="UTC")
    s = pd.Series(0.0, index=idx)
    s["2024-01-03":"2024-01-05 23:00"] = 0.5     # episode 1 ...
    s["2024-01-08":"2024-01-09 23:00"] = 0.8     # ... merged (2 calm days < gap 3) with this one
    s["2024-01-20":"2024-01-20 23:00"] = 0.3     # separate one-day episode
    eps = episodes(s, threshold=0.02, gap_days=3)
    assert len(eps) == 2
    assert eps.loc[0, "duration_days"] == 7 and eps.loc[0, "depth"] == pytest.approx(0.8)
    assert eps.loc[1, "duration_days"] == 1
    pred = s * 0.5
    m = match_episodes(eps, pred, threshold=0.02)
    assert m.loc[0, "pred_depth"] == pytest.approx(0.4) and m.loc[0, "pred_duration_days"] == 5   # calm gap days not counted


@pytest.fixture
def outage_cfg(tmp_path):
    c = load_config("config/outages.yaml").model_copy(deep=True)
    c.paths.data_dir = tmp_path / "data"
    c.paths.artifacts_dir = tmp_path / "artifacts"
    c.paths.reports_dir = tmp_path / "reports"
    c.period.start, c.period.end = "2023-01-01", "2023-05-31"
    c.split.train_end, c.split.validation_end = "2023-03-15 23:00", "2023-04-20 23:00"
    c.weather.start, c.weather.end = None, None
    c.changepoints.eval_until = None
    c.models.xgboost.n_estimators = 40
    c.ensure_dirs()
    return c


def _write_inputs(cfg):
    days = pd.date_range("2023-01-01", "2023-05-31")
    on_days = {*pd.date_range("2023-01-10", "2023-01-14"), *pd.date_range("2023-03-20", "2023-03-25"),
               *pd.date_range("2023-05-01", "2023-05-05")}
    rows = []
    for d in days:
        for h in range(24):
            if d == pd.Timestamp("2023-03-26") and h == 3:    # skipped hour on the DST-start day
                continue
            on = d in on_days and 18 <= h <= 22
            rows.append((d.strftime("%Y-%m-%d"), f"{h:02d}:00-{h + 1:02d}:00", "так" if on else "ні",
                         "від 0,5 до 3" if on and d.day % 2 else "", "в усіх областях" if on else ""))
    rows.append(("", "=B28442", "ні", "", ""))                 # corrupted row
    pd.DataFrame(rows, columns=["date", "hour", "outage_use", "outage_amount", "number"]).to_csv(
        cfg.raw_dir / cfg.outages.file, index=False)
    idx = pd.date_range(cfg.start_ts, cfg.end_ts, freq="h", name="datetime")
    rng = np.random.default_rng(0)
    pd.DataFrame({"temperature_C": rng.normal(0, 3, len(idx)), "wind_speed_kmh": rng.normal(12, 2, len(idx))},
                 index=idx).to_parquet(cfg.raw_dir / "weather.parquet")
    pd.DataFrame({"date": ["2023-03-20"], "time_utc": ["03:00"], "description": ["x"], "capacity_lost_MW": [0.0],
                  "major": [True], "source": ["test"], "verified": [True]}).to_csv(
        cfg.external_dir / "attacks.csv", index=False)


def test_outage_pipeline_end_to_end(outage_cfg):
    cfg = outage_cfg
    _write_inputs(cfg)
    out, stats = load_outages(cfg)
    assert stats["dropped_unparseable"] == 1
    assert (out["outage_on"] == 1).sum() == 16 * 5      # 16 outage days x 5 evening hours
    assert out["outage_extent"].dropna().isin([0.0, 1.0]).all()
    fetch_outages(cfg)

    master = build_master(cfg)
    assert master["deficit"].max() == 1.0 and master["deficit"].min() == 0.0
    # no hole inside the file: the DST-skipped local hour is not a missing UTC hour; only the last 3 UTC
    # hours of the period (local 1 June, not in the file) are unknown
    assert master["deficit"].iloc[:-3].notna().all() and master["deficit"].iloc[-3:].isna().all()
    f = build_features(master, cfg)
    assert "outage_queues_lag24h" in feature_columns(f)

    cp = detect(master, load_attacks(cfg), cfg)
    assert cp["bocpd"]["scores"]["n_attacks"] == 1
    names = ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_adaptive"]
    fcs = train_all(cfg, names)
    assert set(fcs) == set(names)
    refits = pd.read_csv(cfg.artifacts_dir / "forecasts" / "xgb_rolling_refits.csv")
    assert set(refits["reason"]) == {"scheduled"}      # the ablation never retrains on an alarm
    res = evaluate(cfg)
    assert not res["episodes"].empty and {"depth_MAE", "duration_MAE_days"} <= set(res["episodes_summary"].columns)


def test_suspect_period_is_excluded_from_target(outage_cfg):
    cfg = outage_cfg
    _write_inputs(cfg)
    cfg.outages.suspect_periods = [["2023-02-01", "2023-02-10"]]
    fetch_outages(cfg)
    master = build_master(cfg)
    local = master.index.tz_convert("Europe/Kyiv").tz_localize(None).normalize()
    inside = (local >= "2023-02-01") & (local <= "2023-02-10")
    assert master.loc[inside, "deficit"].isna().all()
    assert master.loc[~inside, "deficit"].iloc[:-3].notna().all()
    assert (read_parquet(cfg.raw_dir / "outages_hourly.parquet")["suspect"].sum()) == 10 * 24


def test_trade_long_format_to_hourly_utc():
    from deficit.ingestion.trade import hourly_trade

    rows = []
    for h in range(24):
        reps = 2 if h == 3 else 1                                   # DST-end day: hour 03:00-04:00 occurs twice
        for _ in range(reps):
            rows.append(("2023-10-29", f"{h:02d}:00-{h + 1:02d}:00", "ОЕС України", "Польща", "імпорт", 100.0))
            rows.append(("2023-10-29", f"{h:02d}:00-{h + 1:02d}:00", "ОЕС України", "Польща", "експорт", 40.0))
    raw = pd.DataFrame(rows, columns=["date", "hour", "energy_system", "country", "customs_regime", "amount"])
    out, stats = hourly_trade(raw, "Europe/Kyiv")
    assert len(out) == 25 and out.index.is_unique and stats["duplicate_utc_hours"] == 0
    assert (out["net_import_MW"] == 60.0).all() and (out["import_MW"] == 100.0).all()
