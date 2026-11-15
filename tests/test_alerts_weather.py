from pathlib import Path

import numpy as np
import pandas as pd

from deficit.evaluation import energy_map_alerts_weather as aw
from deficit.ingestion import alerts as al
from deficit.ingestion import weather_oblast as wo
from deficit.ingestion.telegram_regional import OBLASTS

FIX = Path(__file__).parent / "fixtures"


def T(s):
    return pd.Timestamp(s, tz="UTC")


def test_alert_name_mapping_covers_oblasts():
    assert set(al.NAME_TO_OBLAST.values()) == set(OBLASTS)            # all 25 oblasts, one name each, no Crimea
    assert len(al.NAME_TO_OBLAST) == len(OBLASTS)
    assert al.map_oblast("Kyiv City") == "Kyiv_city" and al.map_oblast("Kyivska oblast") == "Kyiv"
    assert al.map_oblast("Zaporizka oblast") == "Zaporizhzhia" and al.map_oblast("Crimea") is None


def test_union_intervals_merges_overlaps_and_touching():
    s = np.array(["2025-01-01T10", "2025-01-01T11", "2025-01-01T12", "2025-01-01T20"], dtype="datetime64[ns]")
    e = np.array(["2025-01-01T12", "2025-01-01T11:30", "2025-01-01T13", "2025-01-01T21"], dtype="datetime64[ns]")
    ms, me = al.union_intervals(s, e)
    assert len(ms) == 2 and (me - ms)[0] == np.timedelta64(3, "h") and (me - ms)[1] == np.timedelta64(1, "h")   # [10,13] (touching merged) and [20,21]


def test_load_alerts_and_union_over_levels():
    df = al.load_alerts(FIX / "alerts_tiny.csv")
    assert set(df["oblast"]) == {"Kharkiv", "Kyiv_city", "Kyiv"}      # Crimea unknown, Sumy has end == start
    idx = al.AlertIndex(df)
    # oblast 10-12, raion 11-14, hromada 11:30-12:30 -> 4 h (no double counting); 20:00-02:00 -> 4 h on Dec 1, 2 h on Dec 2
    assert idx.hours("Kharkiv", T("2025-12-01"), T("2025-12-02")) == 8.0
    assert idx.hours("Kharkiv", T("2025-12-02"), T("2025-12-03")) == 2.0
    assert idx.count("Kharkiv", T("2025-12-01"), T("2025-12-02")) == 2
    assert idx.hours("Kharkiv", T("2025-12-01 13:00"), T("2025-12-01 20:30")) == 1.5      # 13-14 and 20-20:30
    assert idx.hours("Kyiv_city", T("2025-12-01"), T("2025-12-02")) == 1.0 and idx.hours("Kyiv", T("2025-12-01"), T("2025-12-02")) == 1.0
    assert idx.hours("Lviv", T("2025-12-01"), T("2025-12-02")) == 0.0 and idx.count("Lviv", T("2025-12-01"), T("2025-12-02")) == 0


def test_alert_window_is_clipped_to_the_upper_bound():
    idx = al.AlertIndex(al.load_alerts(FIX / "alerts_tiny.csv"))
    assert idx.hours("Kharkiv", T("2025-12-01"), T("2025-12-01 23:00")) == 7.0                 # 4 h + 20:00-23:00 of the running alert
    assert idx.hours("Kharkiv", T("2025-12-01"), T("2025-12-01 11:00")) == 1.0


def test_load_weather_oblast_fixture():
    w = wo.load_weather_oblast(FIX / "weather_oblast_tiny")
    assert set(w["oblast"]) == {"Kyiv", "Kharkiv"}                    # Atlantis is not an oblast code
    k = w[w["oblast"] == "Kyiv"].set_index("date")
    assert len(k) == 3 and k.loc["2025-12-02", "temperature_2m_min"] == -6.5 and k.loc["2025-12-03", "snowfall_sum"] == 3.0
    h = w[w["oblast"] == "Kharkiv"]
    assert h["temperature_2m_min"].isna().sum() == 1 and h["wind_speed_10m_max"].isna().all()   # null and absent variables are NaN
    assert list(w.columns) == wo.COLS


def _tab(rows):
    return pd.DataFrame({"oblast": [r[0] for r in rows], "issue": [T(r[1]) for r in rows]})


def test_alert_and_weather_features_on_fixture():
    idx = al.AlertIndex(al.load_alerts(FIX / "alerts_tiny.csv"))
    w = wo.load_weather_oblast(FIX / "weather_oblast_tiny")
    tab = _tab([("Kharkiv", "2025-12-02"), ("Kharkiv", "2025-12-03"), ("Kyiv", "2025-12-04")])      # issue = end of the strike day
    a = aw.alert_features(tab, idx)
    assert list(a.loc[0]) == [8.0, 0.0, 0.0, 2.0]                     # strike day Dec 1: 8 h; Nov 30 none; previous 7 days none
    assert list(a.loc[1]) == [2.0, 8.0, 8.0, 0.0]                     # strike day Dec 2: 2 h (tail of the evening alert); Dec 1: 8 h
    f = aw.weather_features(tab, w)
    assert list(f.loc[2]) == [-12.0, -12.0, 1.0, 1.0, 60.0, 60.0, 3.0, 4.5]                         # Kyiv, strike day Dec 3: frost, gusts, snow
    g = aw.weather_features(_tab([("Kyiv", "2025-12-03")]), w).loc[0]                              # strike day Dec 2, window Nov 30..Dec 2
    assert g["w_tmin_d"] == -6.5 and g["w_tmin_3d"] == -6.5 and g["w_frost_d"] == 1.0 and g["w_gust_3d"] == 45.0 and g["w_snow_3d"] == 1.5
    h = aw.weather_features(_tab([("Kharkiv", "2025-12-03")]), w).loc[0]                           # Dec 2 tmin is null -> NaN flag, 3-day min uses Dec 1
    assert np.isnan(h["w_tmin_d"]) and h["w_tmin_3d"] == 0.0 and np.isnan(h["w_frost_d"]) and h["w_frost_3d"] == 0.0
    assert aw.weather_features(_tab([("Lviv", "2025-12-03")]), w).isna().all().all()                # no file -> NaN (imputed by the training median)


def test_design_sets_exist_and_are_supersets():
    from deficit.evaluation import energy_map_improve as ei
    s = aw.new_sets()
    for _, _, cols, _ in aw.designs():
        assert isinstance(cols, list) and len(cols) == len(set(cols))      # each design carries its own column list
    assert set(ei.SETS["so_em"]) < set(s["so_em_alw"]) and set(s["so_em_al"]) | set(s["so_em_w"]) == set(s["so_em_alw"])
    assert [d[0] for d in aw.designs()].count("ipcw_so_all_a100") == 1
