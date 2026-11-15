from pathlib import Path

import numpy as np
import pandas as pd

from deficit.ingestion import energy_map as em
from deficit.ingestion.telegram_regional import OBLASTS

FIX = Path(__file__).parent / "fixtures"


def test_region_mapping_covers_oblasts():
    assert set(em.REGION_TO_OBLAST.values()) <= set(OBLASTS)
    assert em.map_region("Київ") == "Kyiv_city" and em.map_region("Київська") == "Kyiv"
    assert em.map_region("Невідома") is None


def test_parse_number():
    assert em.parse_number("230 000") == 230000 and em.parse_number("1,5") == 1.5
    assert np.isnan(em.parse_number("")) and np.isnan(em.parse_number("abc")) and np.isnan(em.parse_number("-3"))


def test_load_osr_fixture():
    df = em.load_osr(FIX / "energy_map_osr_tiny.csv")
    assert len(df) == 3                                                     # unknown region dropped
    a, b, c = df.iloc[0], df.iloc[1], df.iloc[2]
    assert a.oblast == "Kharkiv" and a.blackout_consumers == 230000 and a.has_blackout and not a.has_restored
    assert b.has_restored and not b.has_blackout and b.link == "https://example.org/b"
    assert c.oblast == "Kyiv_city" and c.is_period and c.n_days == 7 and c.blackout_settlements == 12


def test_load_minenergo_fixture():
    df = em.load_minenergo(FIX / "energy_map_minenergo_tiny.csv")
    assert list(df["oblast"]) == ["Kharkiv", "Lviv"]
    assert df.iloc[0].blackout_consumers == 150000 and df.iloc[0].has_blackout and df.iloc[1].has_restored
    assert df["source"].eq("minenergo").all() and not df["is_period"].any()


def test_load_energy_map_finds_files(tmp_path):
    for src, name in ((FIX / "energy_map_osr_tiny.csv", "2026_01_01_x_oblasti_osr.csv"),
                      (FIX / "energy_map_minenergo_tiny.csv", "2026_01_01_x_oblasti_minenergo.csv")):
        (tmp_path / name).write_bytes(src.read_bytes())
    df = em.load_energy_map(tmp_path)
    assert len(df) == 5 and set(df["source"]) == {"osr", "minenergo"}
    assert isinstance(df["date"].iloc[0], pd.Timestamp)


def _rec(rows):
    cols = ["date", "oblast", "source", "n_days", "is_period", "blackout_consumers", "blackout_settlements", "restored_consumers",
            "restored_settlements", "has_blackout", "has_restored", "notes", "link"]
    out = []
    for d, ob, kind, nd in rows:
        out.append({"date": pd.Timestamp(d), "oblast": ob, "source": "osr", "n_days": nd, "is_period": nd > 1,
                    "blackout_consumers": 100.0 if kind == "b" else np.nan, "blackout_settlements": np.nan,
                    "restored_consumers": 50.0 if kind == "r" else np.nan, "restored_settlements": np.nan,
                    "has_blackout": kind == "b", "has_restored": kind == "r", "notes": None, "link": "u"})
    return pd.DataFrame(out, columns=cols)


def test_build_labels_observed_censored_and_causal_known_time():
    from deficit.evaluation import energy_map_labels as L
    st = pd.DataFrame({"strike_at": pd.to_datetime(["2024-03-22 03:00", "2024-03-25 03:00"], utc=True), "major": [True, False]})
    rec = _rec([("2024-03-22", "Kharkiv", "b", 1), ("2024-03-22", "Kharkiv", "r", 1),       # same-day restoration is not used
                ("2024-03-24", "Kharkiv", "r", 1),                                         # restoration, 2 days later
                ("2024-03-26", "Sumy", "b", 1),                                            # blackout on D+1 of strike 2, never restored
                ("2024-03-28", "Sumy", "r", 7),                                            # period record is ignored
                ("2024-03-23", "Lviv", "r", 1)])                                           # restoration without blackout: no label
    lab = L.build_labels(st, rec)
    assert set(zip(lab["idx_all"], lab["oblast"])) == {(0, "Kharkiv"), (1, "Sumy")}
    k = lab[lab["oblast"] == "Kharkiv"].iloc[0]
    assert k.status_30d == "observed" and k.restored_same_day
    assert k.resolution_at_30d == pd.Timestamp("2024-03-25", tz="UTC")                    # known at the END of the record day
    assert abs(k.hours_to_resolution_30d - 57.0) < 1e-9                                   # 2024-03-24 12:00 - 2024-03-22 03:00
    s = lab[lab["oblast"] == "Sumy"].iloc[0]
    assert s.status_30d == "censored" and pd.isna(s.resolution_at_30d) and pd.isna(s.hours_to_resolution_30d)
    assert s.window_end_30d == pd.Timestamp("2024-04-25", tz="UTC")


def test_labels_as_of_never_sees_restoration_before_record_day_ended():
    from deficit.evaluation import energy_map_labels as L
    from deficit.evaluation import region_restoration as rr
    st = pd.DataFrame({"strike_at": pd.to_datetime(["2024-03-22 03:00"], utc=True), "major": [True]})
    lab = L.build_labels(st, _rec([("2024-03-22", "Kharkiv", "b", 1), ("2024-03-24", "Kharkiv", "r", 1)]))
    tab = lab.rename(columns={"resolution_at_30d": "res_at", "hours_to_resolution_30d": "res_hours"}).assign(
        end_hours=lambda d: (d["window_end_30d"] - d["strike_at"]).dt.total_seconds() / 3600)
    during = rr.labels_as_of(tab, pd.Timestamp("2024-03-24 23:59", tz="UTC"))
    after = rr.labels_as_of(tab, pd.Timestamp("2024-03-25 00:00", tz="UTC"))
    assert not during["observed"].iloc[0] and after["observed"].iloc[0]
