"""WHAT was damaged: strike-level damage features from the posts of the national / DTEK channels (dev-mode duration models).

Per post (`tag_text`): which object classes the attack sentences name. A sentence counts (`_gate`) when it has an attack or damage cue,
an EVENT cue (hit / struck / damaged by ...), and no weather, "earlier attack", recap, repair or renewables cue; the post itself must be
an attack report (`is_attack_report`, set by the extractor). Classes (flags):
  tpp (ТЕС/ТЕЦ, thermal / CHP), hpp (ГЕС/ГАЕС), npp (АЕС; not a power-feed / line / repair sentence), gen_other (power plant / generation,
  unspecified), sub_hv (substation of the transmission grid: 110-750 kV, "magistral", or Ukrenergo wording; not a ТП),
  sub_other (other substations), line_hv (ПЛ / ЛЕП / 110-750 kV lines), distribution (oblenergo / 0.4-35 kV / ТП / city networks).
Per strike (`damage_features`): ONLY posts with  00:00 UTC of the strike day <= datetime < issue time (end of the strike day);
posts published later never enter (guarded by tests/test_no_leakage.py). The strike hour of attacks.csv is a placeholder, so the whole
strike day counts, exactly as in `restoration.issue_time`. History of the same object class is read from posts strictly BEFORE the strike day.
Channels: Ukrenergo and the DTEK channels (the Ministry of Energy is not among the collected channels).
Validation: checked automatically (not by a human) on a hand sample, tests/fixtures/telegram_damage_check.csv; precision/recall in the DEV report.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .telegram_extract import ATTACK, DAMAGE_STRICT, EVENT_VERB, HISTORICAL, OLDER, WEATHER, _rx, normalize, segments

DAMAGE_CHANNELS = ("ukrenergo", "dtek_ua", "dtek_ua_kyiv", "dtek_odesa")
TAGS = ["tpp", "hpp", "npp", "gen_other", "sub_hv", "sub_other", "line_hv", "distribution"]
DMG_F = ["dmg_tpp", "dmg_hpp", "dmg_npp", "dmg_gen", "dmg_sub_hv", "dmg_line_hv", "dmg_distr", "dmg_n_types", "dmg_n_oblasts",
         "dmg_repeat", "dmg_queue_next", "dmg_cap_lost"]
# columns of the post table the features read; anything else (whole-day counts etc.) can never enter
POST_COLS = ["datetime", "is_attack_report", "regions_hit", "capacity_lost_mw", "queue_next"] + TAGS

_TPP = [re.compile(r"\b[ЗЧХР]?(?:ТЕС|ТЕЦ)(?:-\d+)?\b"), _rx(r"теплов\w+\s+(?:електро)?станц|теплоелектро|вугільн\w+\s+станц")]
_HPP = [re.compile(r"\b[ЗЧХР]?(?:ГЕС|ГАЕС)\b"), _rx(r"гідроелектро|гідрогенер|гідроагрегат|гідростанц")]
_NPP = [re.compile(r"\b[ЗЧХР]?АЕС\b"), _rx(r"атомн\w+\s+електростанц")]
_GEN = [_rx(r"електростанц|генеруюч|об'єкт\w*\s+генерації|енергоблок|генераці[яїює]")]
_SUB = _rx(r"підстанц")
_SUB_TP = _rx(r"трансформаторн\w+\s+підстанц|\bТП\b|\bКТП\b|розподільч\w+\s+підстанц")
_HV = _rx(r"(?:110|150|220|330|400|750)\s*кВ|магістральн|системоутворюч|Укренерго|НЕК")
_LINE = [re.compile(r"\bПЛ\b|\bЛЕП\b"), _rx(r"лін(?:ія|ії|ій|ію|іям|іями)\s+електропередач|мереж\w*\s+передач|високовольтн\w+\s+(?:лін|мереж)|"
                                          r"лін\w*\s+(?:\w+\s+){0,2}(?:110|150|220|330|400|750)\s*кВ|(?:110|150|220|330|400|750)\s*кВ\s+(?:\w+\s+)?лін")]
_DISTR = [_rx(r"мереж\w*\s+розподіл|розподільч\w+\s+(?:електро)?мереж|мереж\w*\s+обленерго|об'єкт\w*\s+обленерго|обладнання\s+обленерго|"
              r"\bТП\b|трансформаторн\w+\s+підстанц|кабельн\w+\s+лін|(?:0[,.]4|6|10|35)\s*кВ|міськ\w+\s+(?:електро)?мереж|мереж\w*\s+міста|"
              r"повітрян\w+\s+лін(?:ії|ій|ія)(?!\s+\d{3})|розподільн\w+\s+пристро")]
_QUEUE_NEXT = _rx(r"завтра|наступн\w+\s+доб|на\s+наступн")
# a sentence names NEW damage only with an event cue; recaps, reasons, repairs, renewables and weather are dropped
_OBJ_W = r"(?:об'єкт|електростанц|підстанц|ТЕС|ТЕЦ|ГЕС|АЕС|лін|мереж|енерго|генерац)"
_EVENT = _rx(r"удар\w*\s+по|атак\w*\s+(?:на|по)|влуч\w+|завдан\w+|завдав\w*|уражен\w+|прилі?т\w*|знищен\w+|зруйнован\w+|"
             rf"пошкоджен\w+\s+(?:\w+\s+){{0,3}}(?:внаслідок|під час)|обстр[іи]л\w*\s+(?:\w+\s+){{0,3}}{_OBJ_W}|(?:атакува|обстріля|пошкодж)\w+\s+(?:\w+\s+){{0,3}}{_OBJ_W}")
_RECAP = _rx(r"нагадаємо|причин[аи]\b|з\s+початку|від\s+початку|після\s+ремонту|вихід\s+з\s+ремонту|вийшов\s+з\s+ремонту|"
             r"включен\w+\s+в\s+мережу|введен\w+\s+в\s+роботу|підключен\w+\s+до\s+мережі|ремонтн\w+\s+бригад|аварійно-відновлю|"
             r"відновлю\w+\s+(?:пошкоджен|лін)|сонячн\w+\s+електростанц|вітров\w+\s+електростанц|\bВДЕ\b|\bСЕС\b|за\s+(?:останні|минулі)\s|"
             r"остання\s+атака|\bу?\s*[а-яіїє]+\s+20\d\d\b|\b20\d\d\s*(?:році|року|р\b)|вийшов\s+з\s+ладу|ремонт\w*|відновлю\w+|відновлення|повернул\w*|повернуто|повернут\w+|розбір\s+завалів|"
             r"всі\s+(?:великі\s+)?теплов|\d+\s+раз(?:ів|и)?\b|майже\s+\d+\s*%|з\s+\d{1,2}\s+[а-яіїє]+\s+по\s")
_AES_FEED = _rx(r"живл|заживл|електропостач|енергопостач|власних\s+потреб|лінія|лінію|лінії|ліній|\bЛЕП\b|\bПЛ\b|підстанц|споживач|відключен|блок\w*\s+АЕС")


def _any(pats, s: str) -> bool:
    return any(p.search(s) for p in pats)


def _gate(seg: str) -> bool:
    """True when the sentence reports new damage (see module docstring)."""
    if not (ATTACK.search(seg) or DAMAGE_STRICT.search(seg)) or WEATHER.search(seg) or OLDER.search(seg) or _RECAP.search(seg):
        return False
    return bool(_EVENT.search(seg) or EVENT_VERB.search(seg)) and not (HISTORICAL.search(seg) and not _EVENT.search(seg))


def tag_text(text: str, channel: str = "", evidence: list | None = None) -> dict[str, bool]:
    """Object-class flags of one attack-report post (sentence level). `evidence` collects the sentences that were used."""
    t = normalize(text if isinstance(text, str) else "")
    out = dict.fromkeys(TAGS, False)
    hv_channel = channel == "ukrenergo"
    for s, e in segments(t):
        seg = t[s:e]
        if not _gate(seg):
            continue
        if evidence is not None:
            evidence.append(seg.replace("\n", " "))
        out["tpp"] |= _any(_TPP, seg)
        out["hpp"] |= _any(_HPP, seg)
        out["npp"] |= (not _AES_FEED.search(seg)) and _any(_NPP, seg)
        out["gen_other"] |= _any(_GEN, seg)
        if _SUB.search(seg):
            plain = _SUB.search(_SUB_TP.sub(" ", seg))
            if plain and (_HV.search(seg) or hv_channel):
                out["sub_hv"] = True
            else:
                out["sub_other"] = True
                if _SUB_TP.search(seg):
                    out["distribution"] = True
        out["line_hv"] |= _any(_LINE, seg)
        out["distribution"] |= _any(_DISTR, seg)
    return out


def load_damage_posts(raw_dir: Path, posts: pd.DataFrame, channels=DAMAGE_CHANNELS) -> pd.DataFrame:
    """Posts of the damage channels with tag flags (needs the raw texts) and the processed columns the features read."""
    keep = posts[posts["channel"].isin(channels)]
    frames = [pd.read_parquet(Path(raw_dir) / f"{ch}.parquet", columns=["channel", "post_id", "text"])
              for ch in channels if (Path(raw_dir) / f"{ch}.parquet").exists()]
    raw = pd.concat(frames, ignore_index=True)
    m = keep.merge(raw, on=["channel", "post_id"], how="left").reset_index(drop=True)
    tags = [tag_text(tx, ch) if att else dict.fromkeys(TAGS, False) for tx, ch, att in zip(m["text"], m["channel"], m["is_attack_report"])]
    out = pd.concat([m.drop(columns=["text"]), pd.DataFrame(tags)], axis=1)
    out["queue_next"] = [float(q) if (isinstance(tx, str) and pd.notna(q) and _QUEUE_NEXT.search(normalize(tx))) else np.nan
                         for tx, q in zip(m["text"], m["schedule_queues_max"])]
    return out[["channel", "post_id"] + POST_COLS]


# ---------------------------------------------------------------------------------------- strike-level features
def _types(df: pd.DataFrame) -> dict[str, bool]:
    f = {k: bool(df[k].any()) if len(df) else False for k in TAGS}
    return {"gen": f["tpp"] or f["hpp"] or f["npp"] or f["gen_other"], "trans": f["sub_hv"] or f["line_hv"], "distr": f["distribution"]}


def damage_features(strikes: pd.DataFrame, dposts: pd.DataFrame, repeat_days: int = 30) -> pd.DataFrame:
    """One row per strike (same order), columns `DMG_F`. Window of strike i: [00:00 UTC of the strike day, issue time = next 00:00 UTC).
    `dmg_repeat` = number of object types (generation / transmission / distribution) named in the window that were also named in
    posts of the preceding `repeat_days` days (strictly before the strike day). Posts at or after the issue time are never read."""
    p = dposts[POST_COLS].sort_values("datetime").reset_index(drop=True)
    tp = p["datetime"].to_numpy(dtype="datetime64[ns]")
    rows = []
    for s in strikes["strike_at"]:
        s = pd.Timestamp(s)
        day = (s.tz_convert("UTC") if s.tzinfo else s.tz_localize("UTC")).floor("D")
        issue = day + pd.Timedelta(days=1)
        a, b = np.searchsorted(tp, day.to_datetime64(), "left"), np.searchsorted(tp, issue.to_datetime64(), "left")
        w = p.iloc[a:b]
        w = w[w["is_attack_report"]]
        c = np.searchsorted(tp, (day - pd.Timedelta(days=repeat_days)).to_datetime64(), "left")
        prev = p.iloc[c:a]
        prev = prev[prev["is_attack_report"]]
        now, before = _types(w), _types(prev)
        regs = {r for lst in w["regions_hit"] for r in lst}
        fl = {k: float(w[k].any()) if len(w) else 0.0 for k in TAGS}
        qn, cl = w["queue_next"].dropna(), w["capacity_lost_mw"].dropna()
        rows.append({"dmg_tpp": fl["tpp"], "dmg_hpp": fl["hpp"], "dmg_npp": fl["npp"], "dmg_gen": float(now["gen"]),
                     "dmg_sub_hv": fl["sub_hv"], "dmg_line_hv": fl["line_hv"], "dmg_distr": fl["distribution"],
                     "dmg_n_types": float(sum(now.values())), "dmg_n_oblasts": float(len(regs)),
                     "dmg_repeat": float(sum(now[k] and before[k] for k in now)),
                     "dmg_queue_next": float(qn.max()) if len(qn) else 0.0, "dmg_cap_lost": float(np.log1p(cl.max())) if len(cl) else 0.0})
    return pd.DataFrame(rows, index=strikes.index)[DMG_F]
