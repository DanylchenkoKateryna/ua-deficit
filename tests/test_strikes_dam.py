"""Strike list -> attacks.csv rows, and the day-ahead market file -> hourly UTC price ratio."""
import pandas as pd

from deficit.ingestion.dam import hourly_dam
from deficit.ingestion.strikes import daily_strikes


def test_daily_strikes_aggregate_weapons_that_reached_targets():
    raw = pd.DataFrame({"date": ["2024-03-22", "2024-03-22", "2024-04-11"],
                        "weapon_type": ["ракета", "БПЛА", "ракета"],
                        "weapon_fired": [88, 63, 40], "weapon_shot_down": [37, 55, 20],
                        "weapon_failed": [None, None, 5], "link": ["a", "b", "c"]})
    g = daily_strikes(raw).set_index("date")
    d = g.loc[pd.Timestamp("2024-03-22")]
    assert d["weapons_fired"] == 151 and d["missiles_fired"] == 88
    assert d["weapons_through"] == (88 - 37) + (63 - 55)
    assert g.loc[pd.Timestamp("2024-04-11"), "weapons_through"] == 40 - 20 - 5          # failed launches do not count
    assert len(g) == 2


def _dam_rows(date, hours, system="ОЕС України", price=1000.0, cap=2000.0):
    return [(date, h, system, price, 10.0, cap) for h in hours]


COLS = ["date", "hour", "energy_system", "price", "amount_purchase", "price_cap"]


def test_dam_hours_follow_elapsed_time_on_dst_days():
    rows = (_dam_rows("2023-03-26", range(1, 24)) +        # 23-hour day (spring forward)
            _dam_rows("2023-10-29", range(1, 26)) +        # 25-hour day (fall back)
            _dam_rows("2023-10-29", range(1, 26), system="Бурштинський п-в", price=1.0))   # the island is not used
    out, stats = hourly_dam(pd.DataFrame(rows, columns=COLS), "Europe/Kyiv")
    assert len(out) == 23 + 25 and out.index.is_unique and stats["duplicate_hours"] == 0
    assert out["dam_price"].eq(1000.0).all()                                       # no island rows leaked in
    day = out.loc["2023-10-28 21:00":"2023-10-29 21:59"]                           # local 2023-10-29 in UTC
    assert len(day) == 25 and (day.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert out.index[0] == pd.Timestamp("2023-03-25 22:00", tz="UTC")              # local midnight of the spring-forward day


def test_dam_ratio_is_capped_at_one_and_new_series_replaces_old():
    rows = _dam_rows("2023-06-01", [1], price=2500.0, cap=2000.0) + \
        _dam_rows("2023-06-01", [1], system="ОЕС України (синхронізована з ENTSO-E)", price=1500.0, cap=2000.0)
    out, stats = hourly_dam(pd.DataFrame(rows, columns=COLS), "Europe/Kyiv")
    assert stats["duplicate_hours"] == 1 and len(out) == 1
    assert out["dam_price_ratio"].iloc[0] == 0.75
