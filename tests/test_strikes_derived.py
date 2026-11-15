"""Extended strike list: builder (dedupe, first-post time, near_major, determinism) and the strike_list switch."""
import pandas as pd

from deficit.config import PROJECT_ROOT, load_config
from deficit.ingestion.strikes_derived import build_extended

CH = ("ukrenergo", "dtek_ua")


def _attacks():
    return pd.DataFrame({"date": ["2024-03-22", "2024-04-11"], "time_utc": ["06:00", "06:00"],
                         "description": ["a", "b"], "capacity_lost_MW": ["", ""], "major": ["True", "True"],
                         "source": ["s1", "s2"], "verified": ["True", "True"], "weapons_fired": ["100", "50"],
                         "weapons_through": ["40", "20"], "missiles_fired": ["30", "10"]})


def _posts():
    def r(ch, ts, att, pid):
        return {"channel": ch, "post_id": pid, "datetime": pd.Timestamp(ts, tz="UTC"), "is_attack_report": att}
    return pd.DataFrame([
        r("dtek_ua", "2024-03-23 09:30", True, 2),       # day after a major strike -> near_major
        r("ukrenergo", "2024-03-23 05:10", True, 1),     # first post of the day
        r("ukrenergo", "2024-03-23 20:00", True, 3),
        r("ukrenergo", "2024-03-23 06:00", False, 9),    # not an attack report
        r("ukrenergo", "2024-04-11 03:00", True, 4),     # already in attacks.csv -> no duplicate
        r("dtek_ua", "2024-05-02 23:59", True, 5),       # far from majors, single channel
        r("Lvivoblenergo", "2024-05-03 10:00", True, 6),  # not a continuous channel (here) -> ignored
    ])


def test_derived_rows():
    out = build_extended(_attacks(), _posts(), CH)
    assert list(out["date"]) == ["2024-03-22", "2024-03-23", "2024-04-11", "2024-05-02"]
    assert out["date"].is_unique
    d = out.set_index("date")
    assert list(out["derived"]) == [False, True, False, True]
    r = d.loc["2024-03-23"]
    assert r["time_utc"] == "05:10" and r["n_channels"] == 2 and r["n_attack_posts"] == 3
    assert r["major"] == "False" and r["source"] == "telegram_attack_reports" and r["verified"] == "False"
    assert bool(r["near_major"]) and not bool(d.loc["2024-05-02", "near_major"])
    assert d.loc["2024-05-02", "n_channels"] == 1
    # existing rows are kept verbatim
    assert d.loc["2024-04-11", "weapons_through"] == "20" and d.loc["2024-04-11", "major"] == "True"
    assert not bool(d.loc["2024-04-11", "near_major"])


def test_optional_filters_default_off():
    assert len(build_extended(_attacks(), _posts(), CH, min_channels=2)) == 3
    assert len(build_extended(_attacks(), _posts(), CH, min_posts=1)) == 4


def test_deterministic():
    a = build_extended(_attacks(), _posts(), CH)
    b = build_extended(_attacks(), _posts().sample(frac=1.0, random_state=3), CH)
    pd.testing.assert_frame_equal(a, b)


def test_default_config_reads_attacks_csv():
    cfg = load_config()
    assert cfg.telegram.strike_list == "data/external/attacks.csv"
    assert cfg.strike_list_path == PROJECT_ROOT / "data" / "external" / "attacks.csv"
    assert cfg.telegram_processed_dir == PROJECT_ROOT / "data" / "processed"


def test_extended_scenario_switches_only_telegram_paths():
    cfg = load_config(PROJECT_ROOT / "config" / "telegram_extended.yaml")
    assert cfg.strike_list_path.name == "attacks_extended.csv"
    assert cfg.telegram_processed_dir.name == "telegram_extended" and cfg.reports_dir.name == "telegram_extended"
    from deficit.ingestion.attacks import load_attacks
    assert cfg.external_dir == load_config().external_dir          # attacks.csv of the main pipeline is not redirected
    assert load_attacks.__module__.endswith("attacks")
