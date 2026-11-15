"""Energy Map / DiXi Group oblast-level records of combat-related power outages (https://energy-map.info, free account).

Two hand-curated files (human-curated EVENT records, not a continuous series):
  *_oblasti_osr.csv        regional military/civil administration and DSO reports:
                           date, number_days, region, number_blackout_consumers, number_blackout_settlements,
                           number_restored_consumers, number_restored_settlements, notes, link
  *_oblasti_minenergo.csv  Ministry of Energy / Cabinet reports:
                           date, region, number_blackout, number_blackout_day, number_restored, notes, link
`number_days` > 1 marks a PERIOD record (the `date` is the last day of the period and `notes` holds the period like "02.04-16.05"),
so it is not an observation of one day; `is_period` marks it and the label builder ignores such rows.
Region names (Ukrainian adjectives, "Київ" = the city) are mapped to the project's oblast codes (`telegram_regional.OBLASTS`);
an unknown name gets oblast NaN and is dropped with a warning (Crimea does not occur in the files).
Output of `load_energy_map`: one row per source record with columns
  date (naive UTC day), oblast, source (osr | minenergo), n_days, is_period, blackout_consumers, blackout_settlements,
  restored_consumers, restored_settlements, has_blackout (any blackout number > 0), has_restored (any restored number > 0), notes, link.
For the Ministry file `number_blackout` / `number_blackout_day` feed `blackout_consumers` (max of the two) and `number_restored`
feeds `restored_consumers`; it has no settlement counts.
`has_restored` means "restoration reported" (maybe partial), NOT "all consumers have power": the notes never say that.
Nothing here is a model input; the records are used as labels only.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .telegram_regional import OBLASTS

log = logging.getLogger(__name__)

REGION_TO_OBLAST: dict[str, str] = {
    "Вінницька": "Vinnytsia", "Волинська": "Volyn", "Дніпропетровська": "Dnipropetrovsk", "Донецька": "Donetsk",
    "Житомирська": "Zhytomyr", "Закарпатська": "Zakarpattia", "Запорізька": "Zaporizhzhia", "Івано-Франківська": "Ivano-Frankivsk",
    "Київська": "Kyiv", "Київ": "Kyiv_city", "Кіровоградська": "Kirovohrad", "Луганська": "Luhansk", "Львівська": "Lviv",
    "Миколаївська": "Mykolaiv", "Одеська": "Odesa", "Полтавська": "Poltava", "Рівненська": "Rivne", "Сумська": "Sumy",
    "Тернопільська": "Ternopil", "Харківська": "Kharkiv", "Херсонська": "Kherson", "Хмельницька": "Khmelnytskyi",
    "Черкаська": "Cherkasy", "Чернівецька": "Chernivtsi", "Чернігівська": "Chernihiv",
}
FRONTLINE = ("Donetsk", "Luhansk", "Zaporizhzhia", "Kherson")     # fixed before any model result (see the report)
OSR_NAME = "oblasti_osr"
MINENERGO_NAME = "oblasti_minenergo"
COLS = ["date", "oblast", "source", "n_days", "is_period", "blackout_consumers", "blackout_settlements", "restored_consumers",
        "restored_settlements", "has_blackout", "has_restored", "notes", "link"]


def map_region(name) -> str | None:
    """Region name of the files -> oblast code (None if unknown)."""
    code = REGION_TO_OBLAST.get(str(name).strip())
    return code if code in OBLASTS else None


def parse_number(x) -> float:
    """Number with optional thin/regular spaces or a decimal comma; empty / junk -> NaN. Negative values are junk -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).replace(" ", "").replace(" ", "").replace(",", ".").strip()
    try:
        v = float(s)
    except ValueError:
        return np.nan
    return v if v >= 0 else np.nan


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].map(parse_number) if col in df else pd.Series(np.nan, index=df.index)


def _finish(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source
    df["is_period"] = df["n_days"] > 1
    num = ["blackout_consumers", "blackout_settlements", "restored_consumers", "restored_settlements"]
    for c in num:
        df[c] = df[c].astype(float)
    df["has_blackout"] = (df[["blackout_consumers", "blackout_settlements"]].fillna(0) > 0).any(axis=1)
    df["has_restored"] = (df[["restored_consumers", "restored_settlements"]].fillna(0) > 0).any(axis=1)
    bad = df["oblast"].isna()
    if bad.any():
        log.warning("energy_map %s: %d record(s) with unknown region dropped (%s)", source, int(bad.sum()),
                    sorted(df.loc[bad, "region"].astype(str).unique())[:5])
    return df[~bad][COLS]


def load_osr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(), "region": df["region"],
        "n_days": pd.to_numeric(df.get("number_days"), errors="coerce").fillna(1).astype(int),
        "blackout_consumers": _numeric(df, "number_blackout_consumers"), "blackout_settlements": _numeric(df, "number_blackout_settlements"),
        "restored_consumers": _numeric(df, "number_restored_consumers"), "restored_settlements": _numeric(df, "number_restored_settlements"),
        "notes": df.get("notes"), "link": df.get("link")})
    out["oblast"] = out["region"].map(map_region)
    return _finish(out[out["date"].notna()], "osr")


def load_minenergo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(), "region": df["region"], "n_days": 1,
        "blackout_consumers": pd.concat([_numeric(df, "number_blackout"), _numeric(df, "number_blackout_day")], axis=1).max(axis=1),
        "blackout_settlements": np.nan, "restored_consumers": _numeric(df, "number_restored"), "restored_settlements": np.nan,
        "notes": df.get("notes"), "link": df.get("link")})
    out["oblast"] = out["region"].map(map_region)
    return _finish(out[out["date"].notna()], "minenergo")


def find_files(raw_dir: Path) -> dict[str, Path]:
    """The two oblast files of `raw_dir` (the newest if a name matches several dated downloads)."""
    out = {}
    for key, tag in (("osr", OSR_NAME), ("minenergo", MINENERGO_NAME)):
        hits = sorted(Path(raw_dir).glob(f"*{tag}.csv"))
        if not hits:
            raise FileNotFoundError(f"no *{tag}.csv in {raw_dir}")
        out[key] = hits[-1]
    return out


def load_energy_map(raw_dir: Path) -> pd.DataFrame:
    """Both files, concatenated and sorted by (oblast, date); exact repeats of one source record are dropped."""
    f = find_files(raw_dir)
    df = pd.concat([load_osr(f["osr"]), load_minenergo(f["minenergo"])], ignore_index=True)
    df = df.drop_duplicates(["date", "oblast", "source", "n_days", "blackout_consumers", "blackout_settlements", "restored_consumers",
                             "restored_settlements"])
    return df.sort_values(["oblast", "date", "source"]).reset_index(drop=True)
