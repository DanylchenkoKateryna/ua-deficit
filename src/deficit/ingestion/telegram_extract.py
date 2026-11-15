"""Rule-based extraction of structured strike-impact facts from Ukrainian Telegram posts.

Input : data/raw/telegram/<channel>.parquet written by `deficit collect`
        (columns channel, post_id, datetime [UTC], text, views, has_media, forwarded_from, reply_to, url).
Output: data/processed/telegram_posts.parquet     one row per post, extracted fields + matched spans
        data/processed/telegram_daily.parquet     one row per (UTC date, channel) plus channel "ALL"
        data/processed/telegram_episodes.parquet  one row per (strike in attacks.csv, scope)

Everything is regular expressions and keyword lists (no external NLP). Every extracted value keeps the matched
text span (`spans` column, JSON: field -> [{start, end, text, value}]) so a human can audit it.

Fields per post (None / False / empty list when absent):
  is_attack_report          a NEW strike or damage on energy objects is reported (recency marker + attack word +
                            damage/outage word in one sentence); `attack_ref` marks mere references ("consequences of
                            attacks" as a reason for restrictions, retrospectives, interviews)
  damaged_objects           subset of generation / substation / transmission / distribution / unspecified
  regions, regions_affected normalized oblast names (English); `regions_affected` only where the sentence also
                            speaks of damage, an outage, an attack or restrictions; `nationwide` flag
  consumers_without_power   max count in an outage context (non-weather); `consumers_weather` for weather outages;
                            `consumers_restored` for "returned power to N" statements; `settlements_without_power`
  outage_schedules_announced + schedule_kinds (gpv/gop/gav), schedule_hours ("HH:MM-HH:MM"), schedule_queues_min/max
  schedules_lifted          "restrictions are not applied / not forecast / cancelled" (only when nothing is announced
                            in the same post); `restrictions_possible` for hedged mentions ("possible", "if")
  emergency_outages         emergency (аварійні / екстрені) consumer outages in force
  deficit_mw, import_mw, export_mw, limit_mw, capacity_lost_mw   explicit MW values (GW converted), `mw_mentions`
  deficit_state             "present" / "absent" if the post says so
  restoration_power         "full" (all consumers) / "partial" (counts, "most", "partly"); restoration_repairs (bool)
  eta_hours(_min)           estimated restoration time stated as a duration ("within 2 hours", "2-3 days")
  consumption_change_pct    signed change of consumption versus the previous day/week when stated

Leakage: each row carries the post timestamp `datetime`; nothing here looks at later posts. Use
`posts_before(df, issue_time)` and `episode_as_of(...)` when building features.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# text normalisation and segmentation
# --------------------------------------------------------------------------------------------------
_ZW = dict.fromkeys(map(ord, "​‌‍⁠﻿"), " ")      # zero-width characters -> space (1:1)
_SPACES = dict.fromkeys(map(ord, "    "), " ")          # (non-)breaking thin spaces -> space
_APOS = dict.fromkeys(map(ord, "’ʼ‘`´"), "'")           # typographic apostrophes -> '
_DASH = dict.fromkeys(map(ord, "–—−"), "-")


def normalize(text: str) -> str:
    """Length-preserving normalisation: offsets of the result are valid in the original text."""
    return (text or "").translate(_ZW).translate(_SPACES).translate(_APOS).translate(_DASH)


_SEG_RE = re.compile(r"\n+(?![ \t]*[а-яіїєґa-z])|(?<!\bм\.)(?<!\bобл\.)(?<!\bтис\.)(?<!\bн\.п\.)(?<!\bім\.)(?<!\bвул\.)(?<!\bр\.)(?<!\bрр\.)(?<!\bмлн\.)(?<=[.!?…])\s+(?=[«\"“(\[A-ZА-ЯІЇЄҐ0-9🔹•▪‼❗⚠⚡💡🔸👉])")


def segments(t: str) -> list[tuple[int, int]]:
    """Sentence-like segments (start, end) split on line breaks and sentence ends."""
    out, pos = [], 0
    for m in _SEG_RE.finditer(t):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(t):
        out.append((pos, len(t)))
    return [(s, e) for s, e in out if t[s:e].strip()]


def paragraph_of(t: str, pos: int) -> tuple[int, int]:
    """Paragraph (blank-line separated block) containing position `pos`."""
    s = t.rfind("\n\n", 0, pos)
    e = t.find("\n\n", pos)
    return (0 if s < 0 else s + 2), (len(t) if e < 0 else e)


I = re.IGNORECASE
UA = r"[а-яіїєґ'\-]"                    # Ukrainian letter class for word tails


def _rx(p: str, flags: int = I) -> re.Pattern:
    return re.compile(p, flags)


# --------------------------------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------------------------------
ATTACK = _rx(r"обстр[іи]л|обстрілю|атак|\bудар(?:и|ів|ом|у|ах|ами)?\b|ракет|дрон|безпілотн|шахед|прилі?т|приліт"
             r"|масован|балістик|\bбпла\b|уламк|\bворог|окупант|рашист|агресор|ворожі|ворожих|ворожого|росіян\w*|\bрф\b")
DAMAGE = _rx(r"пошкод|знеструм|влуч|руйн|без\s+живлення|залиши\w+\s+без|прилі?т|прилетіл|уражен|зруйнов|знищен|вибух|пожеж|вивед\w+ з (?:ладу|роботи)"
             r"|аварійно-відновлю|аварійн\w+ відключ|відключен|без світла|без електро|без енергопостач|залишаються без"
             r"|жертв|загин|постраждал|ліквідац\w+ наслідк|наслідк\w+ (?:атак|обстріл|удар)")
RESTORE_WORD = _rx(r"повернув\b|повернул(?:и|а|о)\b|повернут(?:о|а|і|е)\b|відновил|відновлен(?:о|а|е|і)\b|заживил|заживлен|подал(?:и|а|о)\b|подано|"
                   r"зі світлом|з['’]явил\w* світло|завершил\w* (?:основн\w+ )?робот")
ENERGY_OBJ = _rx(r"об'єкт\w*\s+(?:критичн\w+\s+)?інфраструктур|об'єкт\w*\s+(?:\w+\s+)?(?:Укренерго|ДТЕК|обленерго)|енергооб|енергоб'?єкт|енергетичн|енергосистем|енергомереж|підстанц|електростанц|електромереж|мереж|лінії|лінія|ліній"
                 r"|енергоблок|генераці|трансформатор|обленерго|енергопостач|електропостач|споживач|світл|електроенерг"
                 r"|родин|осель|домівк|абонент|енергетичн\w+\s+підприємств")
ENERGY_ABBR = re.compile(r"\b[ЗЧ]?(?:ТЕС|ТЕЦ|ГЕС|ГАЕС|АЕС)\b|\bПЛ\b|\bЛЕП\b")
RECENT = _rx(r"вночі|уночі|вранці|ввечері|сьогодні|вчора|минулої (?:доби|ночі)|цієї ночі|протягом доби|на ранок|"
             r"напередодні|у ніч|в ніч|цього ранку|минулу добу|впродовж доби|протягом ночі|\bнов\w+ знеструмлен|"
             r"продовжу\w+ (?:атак|обстр)|триває ліквідація|тривають аварійно|тривають (?:відновлю|ремонт)|"
             r"вже розпочат|триває поступов|протягом\s+дня|за\s+день|упродовж\s+дня|впродовж\s+дня|усю\s+ніч|всю\s+ніч|за\s+ніч|"
             r"повітряної\s+тривоги|під\s+час\s+(?:\w+\s+)?(?:масованої|нічної|ранкової|атаки|обстрілу)|"
             r"після\s+(?:\w+\s+)?(?:нічн|ранков|масован|вчорашн|ворожого|російськ)\w*\s+(?:обстріл|атак|удар)|"
             r"\d{1,2}\.\d{1,2}\.\d{4}")
STRONG_RECENT = _rx(r"вночі|уночі|вранці|ввечері|сьогодні|вчора|минулої\s+(?:доби|ночі)|цієї\s+ночі|на\s+ранок|у\s+ніч|в\s+ніч|"
                    r"цього\s+ранку|минулу\s+добу|протягом\s+ночі|напередодні|\d{1,2}\.\d{1,2}\.\d{4}")
EVENT_VERB = _rx(r"пошкодив|пошкодил\w*|пошкоджено\b|влучив|влучил\w*|атакував|атакувал\w*|обстріляв|обстрілял\w*|"
                 r"завдав|завдал\w*|зруйнував|зруйновал\w*|знищив|знищил\w*|знищено|уражено|вразив|вразил\w*|знеструмлено|"
                 r"знеструмил\w*|припинил\w*")
COMBAT_NEW = _rx(r"бойов\w+\s+дії[^.\n]{0,60}?(?:нов\w+\s+(?:знеструмлен|пошкодж)|\bє\s+(?:нов\w+\s+)?пошкодж)")
ATTACK_TIME = _rx(r"вночі|уночі|вранці|ввечері|минулої\s+(?:доби|ночі)|цієї\s+ночі|сьогоднішн\w+|вчорашн\w+")
REASON_ONLY = _rx(r"(?:зумовлен|спричинен|викликан)\w*\s+(?:\w+\s+)?(?:\w+\s+)?(?:обстріл|атак|удар)|причин[аи][\s-]+(?:\w+\s+){0,5}?(?:наслідк|пошкодж)")
HISTORICAL = _rx(r"у\s+20\d\d\s+році|в\s+20\d\d\s+році|20\d\d\s+р\b|торік|з\s+початку|від\s+початку|протягом\s+(?:\d|\w+\s+(?:рок|місяц|тиж|зим))|"
                 r"за\s+(?:останні|минулі|час)|кожн\w+|щоразу|неодноразово|\d+\s+(?:ракет|дрон)|більше\s+\d+|понад\s+\d+\s+(?:ракет|дрон)")
OLDER = _rx(r"попередн\w+\s+(?:\w+\s+){0,2}?(?:атак|обстріл|удар)|наслідк\w+\s+(?:\w+\s+){0,2}?(?:попередн|масован)|"
            r"(?:після|через|внаслідок)\s+(?:\w+\s+){0,2}?попередн")
PRESENT = _rx(r"^\W*причин[аи]\b|\bє\b|\bзнеструмлен(?:о|а|і|ий|их)\b|залишаються|залишається|тривають|триває|вимушено|застосовуються|застосовано|"
              r"зафіксован|відбувається|введен\w+|застосован\w+")
CAUSAL = _rx(r"(?:через|внаслідок|унаслідок|в\s+результаті|у\s+зв'язку\s+з)\s+(?:\w+\s+){0,3}?(?:обстріл|атак|удар|прилі?т)")
CAUSE_REF = _rx(r"(?:наслідк\w*|зумовлен\w+|спричинен\w+|викликан\w+|через|внаслідок)\s+(?:\w+\s+){0,4}?(?:російськ\w+\s+)?(?:атак|обстріл|удар)")
DAMAGE_STRICT = _rx(r"пошкод|знищен|зруйнов|уражен|влучан|знеструм|прилі?т")
# daily boilerplate headers in Ukrenergo digests that always refer to the last 24 h
HEADER_ATTACK = _rx(r"^\W*(?:НАСЛІДКИ\s+(?:ОБСТРІЛІВ|АТАК\w*)|ОБСТРІЛИ|АТАК[АИ])\b", re.M)

WEATHER = _rx(r"негод\w*|погодн\w+ умов|несприятлив\w+ погод|снігопад|вітер|буревій|хуртовин|ожеледиц|підтопл|"
              r"паводк|ураган|шторм|град\b|злив\w*|гроз[аиу]|ожеледь|ожеледі")
NATIONWIDE = _rx(r"по\s+всій\s+(?:території\s+)?україн\w*|по\s+всій\s+території|у всіх регіонах|в усіх регіонах|по всій країні|у всіх областях|"
                 r"в усіх областях|в усіх регіонах|на всій території")

# ---- damaged objects -------------------------------------------------------------------------------
OBJECT_PATTERNS: dict[str, list[re.Pattern]] = {
    "generation": [
        re.compile(r"\b[ЗЧХР]?(?:ТЕС|ТЕЦ|ГЕС|ГАЕС|АЕС)(?:-\d+)?\b"),
        _rx(r"електростанц|теплов\w+ станц|генеруюч|об'єкт\w* генерації|енергоблок|об'єкти генерації|генераці[яї]"
            r"|теплоелектро|гідроелектро|блок\w* (?:на )?ТЕС|вугільн\w+ станц"),
    ],
    "substation": [_rx(r"підстанц")],
    "transmission": [
        re.compile(r"\bПЛ\b|\bЛЕП\b|\bМЕП\b"),
        _rx(r"мереж\w*\s+передач\w+|лін(?:ія|ії|ій|ію|іям|іями)\s+електропередач|магістральн\w+\s+(?:електро)?(?:мереж|лін)|високовольтн\w+\s+(?:лін|мереж)"
            r"|об'єкт\w*\s+магістральн|лін\w+\s+(?:\d{2,3}[-/]?)*(?:110|150|220|330|400|750)\s*кВ"
            r"|(?:110|150|220|330|400|750)\s*кВ"),
    ],
    "distribution": [
        _rx(r"передачі\s+та\s+розподіл\w+|мереж\w*\s+розподіл\w+|розподільч\w+\s+(?:електро)?мереж|мереж\w*\s+(?:операторів|оператора)\s+систем\w*\s+розподіл"
            r"|мереж\w*\s+обленерго|об'єкт\w*\s+обленерго|обладнання\s+обленерго|обленерго.{0,60}?пошкодж"
            r"|пошкодж.{0,60}?обленерго|трансформаторн\w+\s+підстанц|\bТП\b|кабельн\w+\s+лін"
            r"|повітрян\w+\s+лін(?:ії|ій|ія)(?!\s+\d{3})|розподільн\w+\s+(?:пристро|мереж)"
            r"|(?:0[,.]4|6|10|35)\s*кВ|міськ\w+\s+(?:електро)?мереж|мереж\w*\s+міста"),
    ],
    "unspecified": [_rx(r"енергооб['\w]*|енергоб'?єкт\w*|об'єкт\w*\s+(?:критичн\w+\s+|цивільн\w+\s+|енергетичн\w+\s+)*"
                        r"(?:інфраструктур|енергетик)|енергетичн\w+\s+інфраструктур|енергосистем|енергомереж|електромереж")],
}
# a line feeding a nuclear plant is a line, not damage to the plant itself
_AES_FEED = re.compile(r"(?:живл\w*|електропостачання|енергопостачання|надійність\w*)\s+(?:[^\W\d_]+\s+){0,3}[ЗЧХР]?АЕС")

# ---- regions ---------------------------------------------------------------------------------------
REGIONS: dict[str, list[re.Pattern]] = {
    "Vinnytsia": [_rx(r"Вінниччин|Вінницьк\w*\s+област"), re.compile(r"\bВінниц(?:я|і|ю|ею)\b")],
    "Volyn": [_rx(r"Волин|Волинськ"), re.compile(r"\bЛуцьк\w*")],
    "Dnipropetrovsk": [_rx(r"Дніпропетровськ|Дніпропетровщин"), re.compile(r"\bДніпр(?:о|і|а|у|ом)\b(?!\s+(?:річк|водосх))"),
                       re.compile(r"\bКривий Ріг|\bКриворіж\w*|\bКривого Рогу|\bНікопол\w*|\bПавлоград\w*|\bКам'янськ\w*|\bСамар\b")],
    "Donetsk": [_rx(r"Донеччин|Донецьк|Донеччині"), re.compile(r"\bКраматорськ\w*|\bСлов'янськ\w*|\bПокровськ\w*\b(?!\s+(?:(?:сільськ|селищн)\w*\s+)?громад)")],
    "Zhytomyr": [_rx(r"Житомирщин|Житомирськ|Житомир")],
    "Zakarpattia": [_rx(r"Закарпат|Ужгород")],
    "Zaporizhzhia": [_rx(r"Запоріжж|Запорізьк|Запоріжжя"), re.compile(r"\bЕнергодар\w*")],
    "Ivano-Frankivsk": [_rx(r"Івано-Франківськ|Прикарпат|Франківщин")],
    "Kyiv": [_rx(r"Київщин|Київськ(?:ій|ої|а|у|е|ою)\s+област"),
             _rx(r"Київськ(?:ій|ої|у|ою|а)(?!\w)(?!\s+(?:ТЕЦ|ТЕС|АЕС|район|міськ|водосх|гідро|обленерго|електро|об'єдна|громад))")],
    "Kyiv_city": [re.compile(r"\bКиїв\b|\bКиєв(?:а|у|і|ом)\b|\bКиїв(?:ськ\w+ міськ)")],
    "Kirovohrad": [_rx(r"Кіровоградщин|Кіровоградськ|Кіровоградо"), re.compile(r"\bКропивницьк\w*")],
    "Luhansk": [_rx(r"Луганщин|Луганськ")],
    "Lviv": [_rx(r"Львівщин|Львівськ(?!\w*обленерго)"), re.compile(r"\bЛьвів\b|\bЛьвов(?:а|у|і|ом)\b")],
    "Mykolaiv": [_rx(r"Миколаївщин|Миколаївськ"), re.compile(r"\bМиколаїв\b|\bМиколаєв(?:а|у|і|ом)\b")],
    "Odesa": [_rx(r"Одещин|Одеськ|Одещині"), re.compile(r"\bОдес(?:а|і|у|и|ою)\b")],
    "Poltava": [_rx(r"Полтавщин|Полтавськ"), re.compile(r"\bПолтав(?:а|і|у|и|ою)\b")],
    "Rivne": [_rx(r"Рівненщин|Рівненськ"), re.compile(r"\bРівне\b|\bРівному\b")],
    "Sumy": [_rx(r"Сумщин|Сумськ"), re.compile(r"\bСум(?:и|ах|ами|ам)\b")],
    "Ternopil": [_rx(r"Тернопільщин|Тернопільськ|Тернопіл")],
    "Kharkiv": [_rx(r"Харківщин|Харківськ"), re.compile(r"\bХарків\b|\bХаркова\b|\bХарков(?:і|у|ом)\b")],
    "Kherson": [_rx(r"Херсонщин|Херсонськ"), re.compile(r"\bХерсон\b|\bХерсон(?:а|у|і|ом)\b")],
    "Khmelnytskyi": [_rx(r"Хмельниччин|Хмельницьк")],
    "Cherkasy": [_rx(r"Черкащин|Черкаськ"), re.compile(r"\bЧеркас(?:и|ах|ами|ам)\b")],
    "Chernivtsi": [_rx(r"Буковин|Чернівецьк|Чернівц")],
    "Chernihiv": [_rx(r"Чернігівщин|Чернігівськ"), re.compile(r"\bЧернігів\b|\bЧернігов(?:а|у|і|ом)\b")],
    "Crimea": [re.compile(r"\bКрим\b|\bКримськ\w*|\bКриму\b")],
}

# ---- numbers ----------------------------------------------------------------------------------------
NUM = r"(?:\d{1,3}(?: \d{3})+|\d+)(?:[.,]\d+)?"
MULT = r"(?:тисяч\w*|тис\.?|млн\.?|мільйон\w*)"
CONS_NOUN = (r"(?:споживач\w*|абонент\w*|родин\w*|сім'ї\w*|сімей|домогосподар\w+|осел\w*|домівк\w*|домівок|будинк\w+|помешкан\w+|"
             r"мешканц\w+|клієнт\w+|киян|одеситів|харків['’]ян|дніпрян|запорожц\w+|сумчан\w*)")
CONS_RE = re.compile(rf"(?P<num>{NUM})(?:\s*(?P<mult>{MULT}))?(?:\s+(?!{CONS_NOUN}|черг)[^\W\d_]+){{0,2}}?\s+(?P<noun>{CONS_NOUN})", I)
CONS_ELLIPSIS = re.compile(rf"[^.\n]{{0,40}}?\s(?:та|і|й)\s+(?:понад\s+|більш\s+ніж\s+|майже\s+|близько\s+)?(?P<num>{NUM})\s*(?P<mult>{MULT})\s+(?:у|в|на)\b", I)
HEAT_OUT = _rx(r"без\s+(?:тепла|води|газу|опалення|теплопостачання|водопостачання|газопостачання)")
ELECTRIC = _rx(r"світл|електр|енергопостач|живлен")
HYPOTHETICAL = _rx(r"(?:можуть|може|загроз\w+|якщо|ризик\w*)\s+(?:[^\W\d_]+\s+){0,3}$")
SETTLE_RE = re.compile(rf"(?P<num>{NUM})(?:\s*(?P<mult>{MULT}))?\s+населен\w+\s+пункт\w*", I)
OUT_CUE = _rx(r"знеструм\w*|без\s+розподілу|без\s+(?:світла|електро\w*|енергопостач\w*|живлення|тепла)|відключен\w*|вимкнен\w*|"
              r"залишаються|залишається|залишилис\w*|залишилос\w*|позбавлен\w*|залишаються\s+без")
RESTORE_CUE = _rx(r"повернув\b|повернути|відновити|заживити|повертає|повертаємо|повернул(?:и|а|о|ися)\b|повернут(?:о|а|і|е)\b|повернен(?:о|і|а|е)\b|відновил\w*|відновлен(?:о|а|е|і|ий|ого)\b|"
                  r"заживил\w*|заживлен\w*|\bподал(?:и|а|о)\b|\bподан(?:о|а|і|е)\b|з'явил\w*|зі\s+світлом")
AGGREGATE = _rx(r"підсумк|за\s+(?:тиждень|місяць|рік|квартал|півріччя|перш\w+\s+(?:місяць|півріччя)|минул\w+\s+(?:тиждень|місяць))|"
                r"з\s+початку|минулого\s+тижня|за\s+\d+\s+(?:днів|діб|тижн)|з\s+\d{1,2}\s+по\s+\d{1,2}\s+[а-яіїє]+|"
                r"за\s+(?:травень|червень|липень|серпень|вересень|жовтень|листопад|грудень|січень|лютий|березень|квітень)")

MW_RE = re.compile(rf"(?P<a>{NUM})(?:\s*(?:-|до|–)\s*(?P<b>{NUM}))?\s*(?P<unit>МВт|ГВт|MW|GW|мегават\w*|гігават\w*)(?!\s*[·∙⋅•*.]?\s*год)(?![а-яіїєa-z])", I)
MW_CLASSES = [
    ("capacity_added", _rx(r"додасть|додалос\w*|додаєть\w*|додан\w+")),
    ("deficit", _rx(r"дефіцит|нестач|недостат|нестача")),
    ("import", _rx(r"імпорт|аварійн\w+ допомог|перетин")),
    ("export", _rx(r"експорт")),
    ("capacity_lost", _rx(r"пошкодж|втрат|знищен|вивед\w+ з|зруйнов|втратил|недовироб|недоотрим")),
    ("limit", _rx(r"\bГАВ\b|\bГОП\b|\bГПВ\b|ліміт|обсяг\w* (?:обмеж|відключ)|обмеження|обмежен")),
]

_WORDNUM = {"одн": 1, "один": 1, "одна": 1, "одну": 1, "одної": 1, "двох": 2, "дві": 2, "два": 2, "три": 3, "трьох": 3,
            "чотири": 4, "чотирьох": 4, "п'ять": 5, "п'яти": 5, "шість": 6, "шести": 6, "сім": 7, "семи": 7,
            "вісім": 8, "восьми": 8, "дев'ять": 9, "дев'яти": 9, "десять": 10, "десяти": 10, "півтори": 1.5,
            "дванадцять": 12, "дванадцяти": 12, "двадцять": 20, "двадцяти": 20}
_WN = "|".join(sorted(map(re.escape, _WORDNUM), key=len, reverse=True))
_QTY = rf"(?:\d+(?:[.,]\d+)?|{_WN})"
ETA_RE = re.compile(
    rf"(?:протягом|упродовж|впродовж|за|через|орієнтовно|приблизно|близько|до)\s+(?:найближч\w+\s+|наступн\w+\s+|перш\w+\s+)?(?P<a>{_QTY})(?:\s*(?:-|до|або)\s*(?P<b>{_QTY}))?"
    rf"\s+(?P<unit>годин\w*|хвилин\w*|діб\w*|дн(?:ів|я|і|ям|ями)\b|доб\w*|тижн\w+)", I)
ETA_CONTEXT = _rx(r"очікуєть\w*|планує\w*|прогноз\w*|орієнтовн\w*|\bбуде\s+відновлен|\bбудуть\s+(?:заживлен|відновлен)|відбудеться|"
                  r"маємо\s+(?:повернути|відновити|заживити)")
ETA_SCHEDULE = _rx(r"відсутн\w+\s+електроенергії|\bдалі\b|кожні\s+\d|складатиме|\bчерг|після\s+(?:\w+\s+){0,2}відновлення")
ETA_TOPIC = _rx(r"відновлен|заживл|повернен|повернути|подач|ліквідац|усунен|ремонт|електропостач|енергопостач|"
                r"живлення|світл")
ETA_PAST = _rx(r"протягом\s+(?:останніх|минулих)|за\s+(?:останні|минулі|останню|минулу)|за\s+(?:першу|другу)")

CONS_PCT = re.compile(rf"(?:на|більш(?:е|им|ий)?\s+на)\s+(?P<pct>\d+(?:[.,]\d+)?)\s*%")
UP = _rx(r"вище|вищ(?:им|ий|ого|ою)|більш(?:е|им|ий)|зрос\w*|збільш\w*|перевищ\w*|підвищ\w*|зріс|зростан\w*|збільшен\w*")
DOWN = _rx(r"нижче|нижч(?:им|ий|ого|ою)|менш(?:е|им|ий)|змен\w*|знизил\w*|знизив\w*|знижен\w*|скороч\w*|зниж\w*|спад\w*")
CONS_WORD = _rx(r"споживанн\w*|максимум\w*|навантаженн\w*|енергоспоживанн\w*")
SAME_LEVEL = _rx(r"(?:на\s+тому\s+(?:ж|самому)\s+рівні|таким\s+(?:же|самим)|такий\s+же|таке\s+ж|на\s+рівні\s+(?:вчорашн|попередн|того)\w*|"
                 r"без\s+змін)")

# ---- schedules, emergency, lifting -----------------------------------------------------------------
SCHED_TERM = re.compile(
    r"графік\w*\s+(?:погодинн\w+\s+)?(?:відключен|знеструмлен|обмеж|аварійн|споживання)\w*|"
    r"графік\w*\s+(?:ГПВ|ГОП|ГАВ)|погодинн\w+\s+(?:графік|відключен|обмеж)\w*|\bГПВ\b|\bГОП\b|\bГАВ\b|"
    r"заходи?\s+(?:примусового\s+)?(?:обмеження|зниження)|обмежен\w*\s+(?:споживання|потужності|електропостачання|енергоспоживання|енергопостачання|енергоживлення|електроживлення)|"
    r"відключен\w+\s+за\s+графіком|ліміт\w*\s+споживання|черг\w+\s+(?:аварійн\w+\s+)?(?:відключен|обмежен)\w*|планов\w+\s+(?:погодинн\w+\s+)?відключен\w*|"
    r"відключен\w+\s+на\s+(?:завтра|\d{1,2}\s+[а-яіїє]+)|графік\w*\s+відключен\w*\s+на|"
    r"(?:споживання|потужність)\s+(?:\w+\s+){0,2}буде\s+обмежуват\w+|обмежуватимуть\w*|"
    r"погодинн\w+\s+графіки|обсяг\w*\s+(?:для\s+)?обмеження|стабілізаційн\w+\s+(?:графік|відключен)\w*|"
    r"графік\w*\s+стабілізаційн\w+|планов\w+\s+(?:погодинн\w+\s+)?вимкнен\w*|"
    r"(?:діє|діють)\s+(?:\w+\s+){0,2}черг(?:а|и|у)\b|"
    r"(?:повертає\w*|поверну\w*)\s+до\s+(?:погодинн\w+\s+)?графік\w*|"
    r"знеструмлюват\w+\s+(?:\w+\s+){0,2}черг\w*|"
    r"оновлен\w+\s+графік\w*\b(?!\s+(?:роботи|прийому|візит|зустріч|занять|уроків|руху))", I)
SCHED_ABBR = re.compile(r"\bГПВ\b|\bГОП\b|\bГАВ\b")
META_SCHED = _rx(r"адресн\w+\s+повідомленн|формат|інформуван|чат-бот|застосунк|персональн\w+\s+кабінет|поширен\w+\s+запитання|часті\s+питання")
PARTIAL_CANCEL = _rx(r"скасован\w*\s+(?:одну|дві|кілька|частину|\d+)\s+(?:\w+\s+){0,2}черг")
APPLY = _rx(r"змінен\w+|продовжен\w+|\bдіє\b|діють|діят\w*|діяв\w*|застосову\w*|застосова\w*|застосуван\w*|застосує\w*|введен\w*|вводят\w*|вводить\w*|"
            r"вводиться|запровадж\w*|\bбудуть\b|триває|продовжу\w*|оновлен\w*|змін[иаі]\w*|збільшен\w*|зменшен\w*|"
            r"скороч\w*|на\s+(?:завтра|\d{1,2}\s+[а-яіїє]+)|\bз\s+\d{1,2}[:.]\d{2}|протягом\s+дня|до\s+\d{1,2}[:.]\d{2}|"
            r"вимушено|вимушен\w+|зсун\w+|повертає\w*|повинн\w+")
EMERG = re.compile(
    r"(?:аварійн\w+|екстрен\w+)\s+(?:графік\w*\s+)?(?:відключен\w*|вимкнен\w*|знеструмлен\w*|обмеження|графіки)|аварійн\w+\s+відключ|аварійн(?:і|их|ого|ому|ими)\s+вимкн|"
    r"графік\w+\s+аварійн\w+|\bГАВ\b|вимушен\w+\s+знеструмлен\w+\s+поза\s+графіками|аварійн\w+\s+обмежен\w+\s+споживан", I)
EMERG_EQUIP = _rx(r"^[^.\n]{0,25}?(?:ПЛ\b|лінії|лінія|ліній|блок|енергоблок|трансформатор|автотрансформатор|обладнан|АЕС|"
                  r"підстанц|ТЕС|ГЕС|генератор)")
NEG = re.compile(
    r"\bне\s+(?:\w+\s+){0,3}?(?:застосову\w*|застосов\w*|прогнозу\w*|плану\w*|діят\w*|\bдіє\b|діють|вводит\w*|вводил\w*|"
    r"запроваджу\w*|надава\w*|надав\w*|надходил\w*|доводил\w*|доводя\w*|відбува\w*|потреб\w*|передбача\w*|планується|застосуєть\w*|анонсу\w*)|"
    r"(?:застосову\w*|застосов\w*|діят\w*|діє|діють|вводит\w*)\s+не\s+буд(?:е|уть)\b|"
    r"\bне\s+буд(?:е|уть)\s+(?:\w+\s+){0,2}?(?:застосову\w*|застосов\w*|діят\w*|вводит\w*|запроваджу\w*)|"
    r"скасов\w+|скасуван\w+|зняті|знято|знят\w+|\bвідсутн(?:і|ій|є|я|ні)\b|жодн\w+|було\s+необхідності|без\s+(?:відключен|обмежен)|"
    r"(?:не\s+має|немає|не\s+мають)\s+(?:\w+\s+){0,3}?(?:підстав|потреби|необхідності)|не\s+йдеться", I)
FUTURE_CANCEL = _rx(r"(?:буд\w+|може\w*|можуть\s+бути)\s+(?:\w+\s+)?скасован\w+")
HEDGE = _rx(r"бувають|буває|аби\s+(?:уникнути|запобігти)|\bпри\s+(?:\w+\s+){0,2}?(?:екстрен|аварійн)\w*\s+відключ|можлив\w*|ймовірн\w*|\bможе\b|можуть|не\s+виключ\w*|за\s+умови|\bу\s+разі|\bв\s+разі|\bякщо\b|для\s+(?:уникнення|запобігання)|"
            r"щоб\s+(?:уникнути|запобігти)|\bпри\s+(?:зростанні|збільшенні|перевищенні)|загроз\w+|потенційн\w+|не\s+вдасться")
HEDGE_MASK = _rx(r"всі\s+можливі|усі\s+можливі|все\s+можлив\w*|усе\s+можлив\w*|максимально\s+можлив\w*|"
                 r"можливості|можливіст\w*|якнайшвидше\s+повернут\w*")
LIFT_FWD = re.compile(
    r"(?:відключен\w*|заходи?\s+обмеження|обмеження|графік\w*|ГПВ|ГОП|ГАВ|знеструмлен\w+)[^.\n!?]{0,70}?"
    r"(?:\bне\s+(?:\w+\s+){0,3}?(?:застосову\w*|застосов\w*|прогнозу\w*|плану\w*|діят\w*|\bдіє\b|діють|вводит\w*|вводил\w*|"
    r"запроваджу\w*|відбувал\w*|доводил\w*|надавал\w*|надав\w*|надходил\w*|застосуєть\w*)|скасован\w*|скасовано|знято|зняті|\bвідсутн(?:і|ій|є|я|ні)\b"
    r"|(?:застосову\w*|застосов\w*|діят\w*|вводит\w*)\s+не\s+буд(?:е|уть)\b"
    r"|\bне\s+буд(?:е|уть)\s+(?:\w+\s+){0,2}?(?:застосову\w*|застосов\w*|діят\w*|вводит\w*|запроваджу\w*))", I)
# exemption ("restrictions will not be applied to industrial consumers that ...") and non-grid cancellations are not a lifting
LIFT_EXEMPT = _rx(r"^\s+(?:до|для|щодо|стосовно)\s+(?:[^\W\d_]+\s+){0,2}?(?:промисл|комерц|підприємств|споживач|заклад|об'єкт|категорі|бізнес|домогосп|виробник)")
LIFT_NONGRID = _rx(r"водоканал|за\s+борг|заборгован|боржник|несплат|\bКП\b")
# explainer / FAQ posts talk about the regimes in general; "inform about planned works and emergency outages" is a service description
DATE_ANCHOR = _rx(r"\d{1,2}\s+(?:січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)|зранку|вранці")
OUTAGE_CTX = _rx(r"відключ|знеструм|вимкнен|ГПВ|електропостач|електроенерг|світл")
FAQ_LINE = re.compile(r"\?\s*$|^\W*❓", re.M)
INFO_BEFORE = _rx(r"(?:інформаці\w+|дізнат\w+|дізнавайтес\w*|отримув\w+|сповіщен\w+|повідомлен\w+|переглян\w+)\s+(?:[^\W\d_]+\s+){0,4}?(?:про|щодо)\s+(?:[^\W\d_]+\s+){0,4}$|розділ\w*\s+[^.\n]{0,90}$|(?:попередження|запобігання|уникнення)\s+$"
              r"|(?:призводить|призводять|створює|створюють|заявк\w+)\s+(?:[^\W\d_]+\s+){0,5}$")
PURPOSE = _rx(r"(?:з\s+метою|для|щоб|аби)\s+(?:уникнення|запобігання|уникнути|запобігти)[^,.;]*")
RETURN_TO_SCHED = _rx(r"(?:повертає\w*|поверну\w*)\s+до\s+(?:погодинн\w+\s+)?графік")
REINTRO = _rx(r"введен\w+|запроваджен\w+|застосован\w+|ввімкнен\w+")
LIFT_NOBASIS = _rx(r"(?:не\s+має|немає|не\s+мають)\s+(?:\w+\s+){0,3}?(?:підстав|потреби|необхідності)[^.\n]{0,60}?(?:графік|відключ|обмеж)")
LIFT_REV = re.compile(r"(?<!до\s)(?<!після\s)(?<!щодо\s)(?<!прискорити\s)(?<!наблизити\s)(?:скасуван\w+|скасовано|скасовані|зняття|зняти|скасували|скасуєть\w*)\s+(?:\w+\s+){0,3}"
               r"(?:графік\w*|відключен\w*|обмежен\w*|ГПВ|аварійн\w+)|жодних\s+відключен|відключен\w+\s+не\s+буде|"
               r"^\W*(?:понеділок|вівторок|середа|четвер|п'ятниця|субота|неділя|день|завтра|сьогодні)\s+без\s+(?:обмежень|відключень)|обмеження\s+не\s+діятим", I | re.M)
COUNTERFACTUAL = _rx(r"\bякби\b|\bпройшли\s+б\b|\bбуло\s+б\b|такого\s+споживача")      # not a statement about the system
DEFICIT_ABSENT = _rx(r"(?:немає|не\s+має|не\s+буде)\s+(?:\w+\s+){0,2}?дефіцит\w*|дефіцит\w*\s+(?:\w+\s+){0,4}?(?:відсутн\w+|немає|не\s+прогнозуєть\w*)|відсутн\w+\s+дефіцит|без\s+дефіциту|профіцит")
DEFICIT_PRESENT = _rx(r"(?:дефіцит\w*\s+(?:\w+\s+){0,4}?(?:зберіга\w*|значн\w+|наявн\w+|є\b|присутн\w+)|"
                      r"(?:є|наявн\w+|значн\w+|зберіга\w+)\s+(?:\w+\s+){0,2}?дефіцит)")
TIME_RANGE = re.compile(r"(?<![\d:.])(?P<h1>\d{1,2})[:.](?P<m1>\d{2})\s*(?:до|-|по)\s*(?P<h2>\d{1,2})[:.](?P<m2>\d{2})(?![\d])", I)
PAST_DAY = _rx(r"\bвчора\b|\bучора\b|минулої\s+доби|минулу\s+добу|напередодні|минулого\s+дня")
NOW_DAY = _rx(r"сьогодні|завтра|зараз|наразі|станом\s+на\s+ранок|на\s+ранок")
ADVICE = _rx(r"електроприлад|споживайте|ощадлив|користуйтесь|користування|перенест|потужн\w+\s+прилад|імпорт|експорт|"
             r"години\s+максимальн|ефективн\w+\s+робот|СЕС|сонячн|пік\w*\s+споживання|заряд")
QUEUE_RANGE = re.compile(r"(?<![\d:.,])(\d{1,2})\s*-\s*(\d{1,2})(?![\d:.,])")
QUEUE_NUM = re.compile(
    rf"(?:від\s+)?(?<![\d:.,])(?P<a>\d+(?:[.,]\d+)?|{_WN})(?:\s*(?:-|до)\s*(?P<b>\d+(?:[.,]\d+)?|{_WN}))?\s+черг(?:а|и|у|і|ах|ами|ою|)\b", I)

# ---- restoration ----------------------------------------------------------------------------------
_R_OBJ = (r"(?:світл\w*|електропостач\w*|енергопостач\w*|електроенерг\w*|живлен\w*|споживач\w*|абонент\w*|родин\w*|"
          r"осел\w*|домівк\w*|оселях|будинк\w+|населен\w+\s+пункт\w*|мешканц\w+|електрик[аиіу]\b)")
_R_VERB = (r"(?:повернув\b|повернул(?:и|а|о|ися)\b|повернут(?:о|а|і|е)\b|повернен(?:о|і|а|е)\b|відновил\w*|відновлен(?:о|а|е|і|ий|ого|у)\b|"
           r"заживил\w*|заживлен(?:о|і|а|е)\b|\bподал(?:и|а|о)\b|\bподан(?:о|а|і|е)\b|з'явил\w*)")
RESTORED_RES = [
    re.compile(rf"{_R_VERB}\s+(?:[^\W\d_]+\s+){{0,4}}?{_R_OBJ}", I),
    re.compile(rf"{_R_OBJ}\s+(?:[^\W\d_]+\s+){{0,3}}?{_R_VERB}", I),
    re.compile(r"(?:вс[іі]|усі|всіх|усіх)\b[^.\n]{0,40}?(?:зі\s+світлом|заживлен\w*|заживил\w*)", I),
    re.compile(r"(?:знову|вже)\s+(?:зі\s+світлом|з\s+електропостачанням|з\s+енергопостачанням)", I),
    re.compile(r"вдал\w+ся?\s+(?:\w+\s+){0,2}?(?:заживити|(?:повернути|подати)\s+(?:світло|електро)|відновити\s+(?:електро|енерго|живлення))", I),
    re.compile(r"знеструмлен\w+\s+(?:[^\W\d_]+,?\s+){0,5}?(?:яке|які|що)\s+(?:[^\W\d_]+\s+){0,2}?(?:відновил\w+|ліквідувал\w+)", I),
    re.compile(r"знеструмлених\s+(?:споживачів|абонентів|населених\s+пунктів)\s+(?:\w+\s+)?(?:немає|не\s+залишилось)", I),
]
# restoration of the critical objects / key substations only: partial by definition
RESTORED_PARTIAL_RES = [
    re.compile(rf"{_R_VERB}\s+(?:[^\W\d_]+\s+){{0,2}}?напруг\w*", I),
    re.compile(rf"{_R_VERB}\s+(?:[^\W\d_]+\s+){{0,3}}?(?:ключов\w+|критичн\w+|соціальн\w+)\s+(?:[^\W\d_]+\s+)?(?:об'єкт|інфраструктур|заклад)\w*", I),
    re.compile(r"(?<!які\s)(?<!що\s)(?<!яких\s)(?<!від\s)(?:заживлен(?:о|і|а|е)\b|заживил\w+)\s+(?:[^\W\d_]+\s+){0,3}?(?:інфраструктур|об'єкт|котельн|водогон|лікарн)\w*", I),
    re.compile(r"вдал\w+ся?\s+(?:[^\W\d_]+\s+){0,2}?перепідключ\w+", I),
]
CRIT_OBJ = _rx(r"(?:критичн\w+|ключов\w+|соціальн\w+)\s+(?:\w+\s+)?(?:об'єкт|інфраструктур|заклад)")
NOT_ELECTRIC = _rx(r"тепл\w*|опален\w*|опалюв\w*|газ\w*|вод[аиіу]\b|водо\w+")
RESTORE_NEG = re.compile(r"(?:\bне|ще\s+не|поки\s+що\s+не|не\s+вдалося|потребу\w+|щоб|\bдо|для|очікуєть\w*)\s+(?:\w+\s+){0,2}$", I)
RESTORE_ALL = _rx(r"(?:вс[іі]|усі|всіх|усіх)\s+(?:\w+\s+){0,2}?(?:споживач|абонент|родин|осел|домівк|населен)|повністю|кожн|"
                  r"вс[іі]\s+зі\s+світлом|знеструмлених[^.]{0,30}немає|вс[іі]\s+вони\s+(?:\w+\s+){0,2}?(?:заживлен|відновлен|повернут)")
RESTORE_PARTIAL = _rx(r"частков|частин|майже|більш(?:у|ість)|окрем|поступов|деяк|не\s+всіх|не\s+всім|залишаю\w+|локальн|"
                      r"основн\w+\s+робот|точков|поетапн|етап|за\s+графік\w*|за\s+чергам\w*|\d+\s+населен\w+\s+пункт|більшост\w+|однак|проте|нові\s+відключен\w+")
REPAIR_RES = [
    re.compile(r"відновлен(?:а|о|і|е)\s+(?:\w+\s+){0,3}?(?:обладнанн|лін|підстанц|енергоблок|об'єкт|енергооб)", I),
    re.compile(r"закінчен(?:і|о|а)\s+(?:\w+\s+){0,3}?робот|відремонтува\w+\s+\d", I),
    re.compile(r"(?:ремонт\w*|відновлювальн\w+\s+робот\w*|аварійно-відновлювальн\w+\s+робот\w*)\s+(?:\w+\s+){0,3}?"
               r"(?:заверш(?:ен\w+|ил\w+|ено)|виконан\w+)", I),
    re.compile(r"заверш(?:ил\w+|ено|ені)\s+(?:\w+\s+){0,3}?(?:ремонт\w*|робот\w*|відновлен\w+)", I),
    re.compile(r"(?:включен(?:о|а|і|е|ий)|введен(?:о|а|і|е|ий)|повернен(?:о|а|і|е|ий)|повернул(?:и|а|о|ися)|повернут(?:о|а|і|е|ий))\b\s+(?:\w+\s+){0,2}?(?:в|до)\s+(?:роботу|експлуатац\w+|ладу)", I),
    re.compile(r"відновил\w+\s+(?:\w+\s+){0,3}?(?:надійність|роботу|лінію|підстанц\w+|енергоблок|обладнанн\w+|зв'язок)", I),
    re.compile(r"(?:лін\w+|підстанц\w+|блок\w*)\s+(?:\w+\s+){0,5}?(?:включен(?:о|а|і|е|ий)|відновлен(?:о|а|і|е|ий))\b", I),
]

# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


@dataclass
class Hit:
    field: str
    start: int
    end: int
    text: str
    value: object = None

    def as_dict(self) -> dict:
        v = self.value
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        return {"start": self.start, "end": self.end, "text": self.text, "value": v}


def parse_number(s: str) -> float:
    return float(s.replace(" ", "").replace(",", "."))


def _qty(s: str) -> float:
    s = s.lower()
    return _WORDNUM[s] if s in _WORDNUM else parse_number(s)


def _scale(mult: str | None) -> float:
    if not mult:
        return 1.0
    m = mult.lower()
    return 1e6 if m.startswith(("млн", "мільйон")) else 1e3


def _has_neg(seg: str) -> bool:
    return bool(NEG.search(FUTURE_CANCEL.sub(lambda m: " " * len(m.group()), seg)))


def _has_hedge(seg: str) -> bool:
    return bool(HEDGE.search(HEDGE_MASK.sub(lambda m: " " * len(m.group()), seg)))


def _cue_type(t: str, seg_s: int, seg_e: int, ms: int, me: int, prev_seg_start: int = 0,
              prev_seg_end: int | None = None) -> str | None:
    """'out' / 'restore' / None for a number at [ms, me) inside the segment [seg_s, seg_e)."""
    pre, post = t[seg_s:ms], t[me:seg_e]
    last = None
    for rx, kind in ((OUT_CUE, "out"), (RESTORE_CUE, "restore")):
        for m in rx.finditer(pre):
            if last is None or m.end() > last[0]:
                last = (m.end(), kind)
    if last:
        return last[1]
    if prev_seg_end is not None and t[prev_seg_start:prev_seg_end].rstrip().endswith(":"):   # list items under "returned power to:"
        prev = t[prev_seg_start:prev_seg_end]
        for rx, kind in ((OUT_CUE, "out"), (RESTORE_CUE, "restore")):
            for m in rx.finditer(prev):
                if last is None or m.end() > last[0]:
                    last = (m.end(), kind)
        if last:
            return last[1]
    first = None
    for rx, kind in ((OUT_CUE, "out"), (RESTORE_CUE, "restore")):
        m = rx.search(post)
        if m and (first is None or m.start() < first[0]):
            first = (m.start(), kind)
    return first[1] if first else None


# --------------------------------------------------------------------------------------------------
# the extractor
# --------------------------------------------------------------------------------------------------
EMPTY: dict = {
    "is_attack_report": False, "attack_ref": False, "damaged_objects": [], "regions": [], "regions_affected": [],
    "nationwide": False, "consumers_without_power": None, "consumers_weather": None, "consumers_restored": None,
    "restored_is_aggregate": False, "settlements_without_power": None, "settlements_weather": None,
    "outage_schedules_announced": False, "schedule_kinds": [], "schedule_hours": [], "schedule_queues_min": None,
    "schedule_queues_max": None, "schedules_lifted": False, "restrictions_possible": False, "emergency_outages": False,
    "deficit_mw": None, "import_mw": None, "export_mw": None, "limit_mw": None, "capacity_lost_mw": None,
    "mw_mentions": "[]", "deficit_state": None, "restoration_power": None, "restoration_repairs": False,
    "restoration_any": False, "schedules_past": False, "emergency_past": False, "eta_hours": None, "eta_hours_min": None, "consumption_change_pct": None,
    "weather_related": False,
}


def extract_post(text: str) -> dict:
    """Extract all fields from one post text. Returns a flat dict (see module docstring) + `spans` (JSON string)."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in EMPTY.items()}
    raw = text if isinstance(text, str) else ""
    t = normalize(raw)
    hits: list[Hit] = []
    if not t.strip():
        out["spans"] = "{}"
        return out
    segs = segments(t)

    def add(field: str, s: int, e: int, value=None) -> None:
        hits.append(Hit(field, s, e, raw[s:e], value))

    # ---- attack report / reference --------------------------------------------------------------
    attack_segs: list[tuple[int, int]] = []
    header_pos = [m.end() for m in HEADER_ATTACK.finditer(t)]
    aggregate_post = bool(AGGREGATE.search(t[:400]))
    for k, (s, e) in enumerate(segs):
        seg = t[s:e]
        combat = COMBAT_NEW.search(seg)
        if not (ATTACK.search(seg) or combat):
            continue
        win = " ".join(t[a:b] for a, b in segs[max(0, k - 1):k + 3])       # the sentence and its neighbours
        event_verb = bool(EVENT_VERB.search(seg))
        hard = bool(DAMAGE.search(win)) or event_verb or bool(combat)
        soft = bool(RESTORE_WORD.search(seg))                              # "returned power after the attack"
        if not (hard or soft):
            continue
        causal_now = bool(CAUSAL.search(seg)) and bool(PRESENT.search(seg) or EMERG.search(seg))
        obj = ENERGY_OBJ.search(win) or ENERGY_ABBR.search(win) or EMERG.search(seg)
        if not obj:
            continue
        in_header = any(0 <= s - h < 400 for h in header_pos)
        older = bool(OLDER.search(seg))
        event_now = event_verb and not HISTORICAL.search(seg) and bool(ENERGY_OBJ.search(seg) or ENERGY_ABBR.search(seg))
        recent = (not older) and (bool(STRONG_RECENT.search(seg)) or bool(RECENT.search(seg)) or in_header or causal_now
                                  or event_now or bool(combat))
        if not hard and aggregate_post and not STRONG_RECENT.search(seg):
            recent = False                  # weekly / monthly restoration totals are not a strike report
        no_impact = re.search(r"не\s+мал\w+\s+наслідк|без\s+наслідк|не\s+призвел|не\s+спричинил", seg, I)
        if REASON_ONLY.search(seg) and not ATTACK_TIME.search(seg):
            recent = False                  # "restrictions because of shelling-induced damage": a reason, not a report
        if recent and not no_impact:
            out["is_attack_report"] = True
            attack_segs.append((s, e))
            add("is_attack_report", s, e, True)
        else:
            out["attack_ref"] = True
    if out["is_attack_report"]:
        out["attack_ref"] = False
    # "reasons" boilerplate: restrictions caused by consequences of attacks
    elif CAUSE_REF.search(t):
        out["attack_ref"] = True

    # ---- damaged objects (only in posts that report an attack) -------------------------------------
    objs: dict[str, list[tuple[int, int]]] = {}
    if out["is_attack_report"]:
        for s, e in segs:
            seg = t[s:e]
            if not (ATTACK.search(seg) or (DAMAGE_STRICT.search(seg) and not WEATHER.search(seg))) or OLDER.search(seg):
                continue
            for kind, pats in OBJECT_PATTERNS.items():
                for p in pats:
                    for m in p.finditer(seg):
                        if kind == "generation" and re.search(r"[ЗЧХР]?АЕС", m.group()) and _AES_FEED.search(seg):
                            continue
                        objs.setdefault(kind, []).append((s + m.start(), s + m.end()))
            m = re.search(r"мереж", seg, I)
            if m and re.search(r"МЕРЕЖІ\s+ОБЛЕНЕРГО", t[max(0, s - 500):s]):      # digest section "МЕРЕЖІ ОБЛЕНЕРГО"
                objs.setdefault("distribution", []).append((s + m.start(), s + m.end()))
    specific = [k for k in objs if k != "unspecified"]
    keep = specific if specific else (["unspecified"] if "unspecified" in objs else [])
    order = ["generation", "substation", "transmission", "distribution", "unspecified"]
    out["damaged_objects"] = [k for k in order if k in keep]
    for k in keep:
        for s, e in objs[k][:3]:
            add("damaged_objects", s, e, k)

    # ---- regions -------------------------------------------------------------------------------
    reg_hits: dict[str, list[tuple[int, int]]] = {}
    for name, pats in REGIONS.items():
        for p in pats:
            for m in p.finditer(t):
                reg_hits.setdefault(name, []).append((m.start(), m.end()))
    out["regions"] = sorted(reg_hits)
    for name, pos in reg_hits.items():
        s, e = pos[0]
        add("regions", s, e, name)
    affected = set()

    def impact_cue(sg: str) -> bool:
        return bool(DAMAGE.search(sg) or ATTACK.search(sg) or SCHED_TERM.search(sg) or EMERG.search(sg)
                    or re.search(r"обмеж|відключ|знеструм", sg, I))

    for name, pos in reg_hits.items():
        for s, e in pos:
            for k, (ss, se) in enumerate(segs):
                if not ss <= s < se:
                    continue
                seg = t[ss:se]
                ok = impact_cue(seg) and not (WEATHER.search(seg) and not ATTACK.search(seg))
                if not ok and k:                 # list continuation: "... у кількох областях. Зокрема - на Харківщині" / "повернули світло для:\n- 58 тис. Донеччини"
                    prev = t[slice(*segs[k - 1])]
                    cont = re.match(r"\W*(?:зокрема|а\s+також|також|найбільше|переважно|у\s+тому\s+числі)", seg, I) or prev.rstrip().endswith(":")
                    ok = bool(cont and (impact_cue(prev) or RESTORE_CUE.search(prev)) and not WEATHER.search(prev))
                if ok:
                    affected.add(name)
                    break
    out["regions_affected"] = sorted(affected)
    for ss, se in segs:
        m = NATIONWIDE.search(t, ss, se)
        if m and (DAMAGE.search(t[ss:se]) or ATTACK.search(t[ss:se]) or SCHED_TERM.search(t[ss:se]) or EMERG.search(t[ss:se])
                  or re.search(r"обмеж|відключ|знеструм", t[ss:se], I)):
            out["nationwide"] = True
            add("nationwide", m.start(), m.end(), True)
            break

    # ---- consumers / settlements ------------------------------------------------------------------
    cons_out, cons_weather, cons_rest = [], [], []
    for k, (s, e) in enumerate(segs):
        seg_has_weather = _weather_seg(t, segs, k)
        ps, pe = segs[k - 1] if k else (0, None)
        for m in CONS_RE.finditer(t, s, e):
            kind = _cue_type(t, s, e, m.start(), m.end(), ps, pe)
            if kind is None:
                continue
            val = parse_number(m.group("num")) * _scale(m.group("mult"))
            if m.group("noun").lower().startswith(("мешканц",)) and kind != "restore" and not re.search(r"знеструм|без\s", t[s:e], I):
                continue
            if val < 1 or (not m.group("mult") and val > 5_000_000):
                continue
            if re.search(r"(?:тис\.?|тисяч\w*|млн)\s+(?:грн|гривень|євро|тонн|кВт|МВт)", t[m.start():m.end() + 12], I):
                continue
            pre = t[s:m.start()]
            if kind == "out" and (HEAT_OUT.search(pre) and not ELECTRIC.search(pre)):
                continue                    # "без тепла залишилися 6 лікарень": heat / water / gas, not electricity
            if kind == "out" and HYPOTHETICAL.search(pre[-40:]):
                continue                    # "взимку можуть залишитися без світла 500 будинків"
            target = cons_rest if kind == "restore" else cons_weather if seg_has_weather else cons_out
            target.append((val, m))
            c = CONS_ELLIPSIS.match(t, m.end(), e)      # "3 тис абонентів у Нікополі та більш ніж 6,4 тис у районі"
            if c:
                target.append((parse_number(c.group("num")) * _scale(c.group("mult")), c))
    if cons_out:
        val, m = max(cons_out, key=lambda x: x[0])
        out["consumers_without_power"] = val
        add("consumers_without_power", m.start(), m.end(), val)
    if cons_weather:
        val, m = max(cons_weather, key=lambda x: x[0])
        out["consumers_weather"] = val
        add("consumers_weather", m.start(), m.end(), val)
    if cons_rest:
        val, m = max(cons_rest, key=lambda x: x[0])
        out["consumers_restored"] = val
        out["restored_is_aggregate"] = bool(AGGREGATE.search(t))
        add("consumers_restored", m.start(), m.end(), val)
    set_out, set_weather = [], []
    for s, e in segs:
        for m in SETTLE_RE.finditer(t, s, e):
            if _cue_type(t, s, e, m.start(), m.end()) != "out":
                continue
            val = parse_number(m.group("num")) * _scale(m.group("mult"))
            (set_weather if _weather_seg(t, segs, segs.index((s, e))) else set_out).append((val, m))
    if set_out:
        val, m = max(set_out, key=lambda x: x[0])
        out["settlements_without_power"] = val
        add("settlements_without_power", m.start(), m.end(), val)
    if set_weather:
        val, m = max(set_weather, key=lambda x: x[0])
        out["settlements_weather"] = val
        add("settlements_weather", m.start(), m.end(), val)
    out["weather_related"] = bool(WEATHER.search(t)) and bool(
        out["consumers_weather"] or out["settlements_weather"] or re.search(r"через\s+(?:\w+\s+)?(?:негод|погод)", t, I))

    # ---- schedules / emergency / lifting ---------------------------------------------------------
    announced_segs: list[tuple[int, int]] = []
    kinds: set[str] = set()
    lifted_hits: list[tuple[int, int]] = []
    faq_post = len(FAQ_LINE.findall(t)) >= 3
    returning_any = False
    for idx, (s, e) in enumerate(segs):
        seg = t[s:e]
        if seg.rstrip().endswith("?") or (faq_post and not (NOW_DAY.search(seg) or TIME_RANGE.search(seg) or DATE_ANCHOR.search(seg))):
            continue
        if PURPOSE.search(seg):
            # "з метою уникнення застосування аварійних відключень, діє графік": the purpose clause is not a statement
            seg = PURPOSE.sub(lambda m: " " * len(m.group()), seg)
        if "(" in seg:
            # "ГАВ введено повторно (о 14:18 були скасовані ГАВ для 6-10 черг)": the parenthesis is a side note on an earlier state
            outer = re.sub(r"\([^)]*\)", lambda m: " " * len(m.group()), seg)
            if outer != seg and REINTRO.search(outer) and _has_neg(seg) and not _has_neg(outer):
                seg = outer
        term = SCHED_TERM.search(seg)
        if term and term.group().lower().startswith("оновлен") and not OUTAGE_CTX.search(t[s:e + 200]):
            term = None                     # "оновлений графік вакцинації"
        lift = None if (PARTIAL_CANCEL.search(seg) or COUNTERFACTUAL.search(seg)) else (
            LIFT_FWD.search(seg) or LIFT_REV.search(seg) or LIFT_NOBASIS.search(seg))
        if lift and (LIFT_EXEMPT.match(seg[lift.end():]) or LIFT_NONGRID.search(seg)):
            lift = None
        if lift:
            lifted_hits.append((s + lift.start(), s + lift.end()))
        em = EMERG.search(seg)
        if em and INFO_BEFORE.search(seg[max(0, em.start() - 120):em.start()]):
            em = None
        past = bool(PAST_DAY.search(seg)) and not NOW_DAY.search(seg)
        if em and not EMERG_EQUIP.match(seg[em.end():]) and past and not _has_neg(seg):
            out["emergency_past"] = True
            em = None
        if em and not EMERG_EQUIP.match(seg[em.end():]):
            if not _has_neg(seg) and not _has_hedge(seg) and not lift:
                out["emergency_outages"] = True
                add("emergency_outages", s + em.start(), s + em.end(), True)
            elif _has_hedge(seg) and not _has_neg(seg):
                out["restrictions_possible"] = True
        if not term or META_SCHED.search(seg):
            continue
        returning = bool(RETURN_TO_SCHED.search(seg))
        returning_any = returning_any or returning        # "повертаємось до графіків, екстрені відключення скасовано"
        if (lift or _has_neg(seg)) and not PARTIAL_CANCEL.search(seg) and not returning:
            continue
        if past:
            out["schedules_past"] = True
            continue
        if _has_hedge(seg):
            out["restrictions_possible"] = True
            continue
        if em and term.start() >= em.start() and term.end() <= em.end() and not re.search(r"графік|ГАВ|черг", term.group(), I):
            continue          # "аварійні обмеження споживання": emergency outages, not a schedule
        applies = bool(APPLY.search(seg))
        if not applies and idx and t[slice(*segs[idx - 1])].rstrip().endswith(":"):
            # list item under "31 грудня на Хмельниччині діятимуть:" / "У Бучанському районі введено:"
            applies = bool(APPLY.search(t[slice(*segs[idx - 1])]))
        if not applies and len(seg) < 90:       # header line ("ГРАФІКИ ПОГОДИННИХ ВІДКЛЮЧЕНЬ") followed by hours/queues
            nxt = t[e:e + 300]
            applies = bool(TIME_RANGE.search(nxt) or QUEUE_NUM.search(nxt))
        if applies:
            announced_segs.append((s, e))
            add("outage_schedules_announced", s + term.start(), s + term.end(), True)
            kinds.update(_sched_kinds(seg))
    out["outage_schedules_announced"] = bool(announced_segs)
    out["schedule_kinds"] = sorted(kinds - {"other"} if kinds - {"other"} else kinds)
    if lifted_hits and (not announced_segs or returning_any) and not out["emergency_outages"]:
        out["schedules_lifted"] = True
        s, e = lifted_hits[0]
        add("schedules_lifted", s, e, True)
    elif lifted_hits and announced_segs and not returning_any:
        out["schedules_lifted"] = False     # partial: restrictions for some consumers remain
    if announced_segs:
        hours, qmin, qmax = [], [], []
        lines = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", t)]
        for li, (ls, le) in enumerate(lines):
            line = t[ls:le]
            prev = ""
            for pj in range(li - 1, -1, -1):
                if t[lines[pj][0]:lines[pj][1]].strip():
                    prev = t[lines[pj][0]:lines[pj][1]]
                    break
            ctx_ok = bool(SCHED_TERM.search(line) or SCHED_TERM.search(prev) or re.search(r"обмеж|відключ|графік|черг|час застосування", line + " " + prev, I))
            if not ctx_ok:
                continue
            for m in TIME_RANGE.finditer(line):
                h1, m1, h2, m2 = (int(m.group(g)) for g in ("h1", "m1", "h2", "m2"))
                if h1 > 24 or h2 > 24 or m1 > 59 or m2 > 59:
                    continue
                pos = ls + m.start()
                sg = next((t[x:y] for x, y in segs if x <= pos < y), line)
                if _has_neg(sg) or (PAST_DAY.search(sg) and not NOW_DAY.search(sg)):
                    continue
                if ADVICE.search(sg) and not SCHED_TERM.search(sg):
                    continue
                val = f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}"
                if val not in hours:
                    hours.append(val)
                    add("schedule_hours", ls + m.start(), ls + m.end(), val)
            for m in QUEUE_NUM.finditer(line):
                a = _qty(m.group("a"))
                b = _qty(m.group("b")) if m.group("b") else a
                if a > 12 or b > 12:
                    continue
                qmin.append(min(a, b))
                qmax.append(max(a, b))
                add("schedule_queues", ls + m.start(), ls + m.end(), [min(a, b), max(a, b)])
            if re.search(r"черг", line, I) and not _has_neg(line):      # "1-4 черги ввімкнені, 5-8 відключені"
                for m in QUEUE_RANGE.finditer(line):
                    a, b = int(m.group(1)), int(m.group(2))
                    if a < b <= 12:
                        qmin.append(a)
                        qmax.append(b)
                        add("schedule_queues", ls + m.start(), ls + m.end(), [a, b])
        out["schedule_hours"] = hours
        out["schedule_queues_min"] = min(qmin) if qmin else None
        out["schedule_queues_max"] = max(qmax) if qmax else None
    m = DEFICIT_ABSENT.search(t)
    m2 = DEFICIT_PRESENT.search(t)
    if m and not m2:
        out["deficit_state"] = "absent"
        add("deficit_state", m.start(), m.end(), "absent")
    elif m2 and not m:
        out["deficit_state"] = "present"
        add("deficit_state", m2.start(), m2.end(), "present")
    elif m and m2:
        first = m if m.start() < m2.start() else m2
        out["deficit_state"] = "absent" if first is m else "present"
        add("deficit_state", first.start(), first.end(), out["deficit_state"])

    # ---- megawatts ---------------------------------------------------------------------------------
    mw_list = []
    for s, e in segs:
        for m in MW_RE.finditer(t, s, e):
            mult = 1000.0 if m.group("unit").upper().startswith(("ГВТ", "GW", "ГІГАВАТ")) else 1.0
            a = parse_number(m.group("a")) * mult
            b = parse_number(m.group("b")) * mult if m.group("b") else a
            val = max(a, b)
            cls = _mw_class(t, s, e, m.start(), m.end())
            mw_list.append({"cls": cls, "value": val, "start": m.start(), "end": m.end(), "text": raw[m.start():m.end()]})
    for cls in ("deficit", "import", "export", "limit", "capacity_lost"):
        vals = [x for x in mw_list if x["cls"] == cls]
        if vals:
            best = max(vals, key=lambda x: x["value"])
            out[f"{cls}_mw"] = best["value"]
            add(f"{cls}_mw", best["start"], best["end"], best["value"])
    out["mw_mentions"] = json.dumps([{k: v for k, v in x.items() if k != "text"} | {"text": x["text"]} for x in mw_list],
                                    ensure_ascii=False)

    # ---- restoration -----------------------------------------------------------------------------
    rest_hits = []
    for s, e in segs:
        seg = t[s:e]
        for rx in RESTORED_RES + RESTORED_PARTIAL_RES:
            forced_partial = rx in RESTORED_PARTIAL_RES
            for m in rx.finditer(seg):
                if _AES_FEED.search(seg) or re.search(r"надійн\w+\s+живлення", seg, I):
                    continue                    # "надійність живлення ЗАЕС відновлено": a transmission line, not consumers
                if NOT_ELECTRIC.search(m.group()) and not ELECTRIC.search(m.group()):
                    continue                    # "подано тепло до будинків", "відновлено газопостачання": not electricity
                if RESTORE_NEG.search(seg[max(0, m.start() - 30):m.start()]) or re.search(r"\bне\s+\w*$", seg[:m.start()][-14:], I):
                    continue
                if "вдал" in m.group().lower() and "вдал" not in rx.pattern:
                    continue
                if re.search(r"\bне\b", m.group(), I):
                    continue                    # "електропостачання ще не відновлено"
                if re.search(r"\bбуд(?:е|уть)\b|щоб|коли|якщо", m.group(), I) or re.search(r"\b(?:аби|щоб)\b", seg[:m.end()], I):
                    continue                    # future / purpose: "so that everyone has power", "when it will be restored"
                partial_cue = bool(RESTORE_PARTIAL.search(seg))
                nxt = t[e:e + 120] if seg.rstrip().endswith(":") else ""       # list of counts follows the colon
                full = (not partial_cue) and (bool(RESTORE_ALL.search(seg)) or not (CONS_RE.search(seg) or CONS_RE.search(nxt)))
                kind = "full" if full and not forced_partial and not CRIT_OBJ.search(seg) else "partial"
                rest_hits.append((s + m.start(), s + m.end(), kind))
    if rest_hits:
        out["restoration_power"] = "full" if any(k == "full" for _, _, k in rest_hits) else "partial"
        s, e, k = next((h for h in rest_hits if h[2] == out["restoration_power"]), rest_hits[0])
        add("restoration_power", s, e, out["restoration_power"])
    for s, e in segs:
        seg = t[s:e]
        for rx in REPAIR_RES:
            m = rx.search(seg)
            if m and not re.search(r"\bне\s+\w*$", seg[:m.start()][-14:], I) and not re.search(r"етап|частков|планов|планово", seg, I):
                out["restoration_repairs"] = True
                add("restoration_repairs", s + m.start(), s + m.end(), True)
                break
        if out["restoration_repairs"]:
            break
    out["restoration_any"] = bool(out["restoration_power"] or out["restoration_repairs"] or out["schedules_lifted"])

    # ---- ETA -------------------------------------------------------------------------------------
    for s, e in segs:
        seg = t[s:e]
        if not (ETA_TOPIC.search(seg) and ETA_CONTEXT.search(seg)) or ETA_PAST.search(seg) or ETA_SCHEDULE.search(seg):
            continue
        m = ETA_RE.search(seg)
        if not m:
            continue
        unit = m.group("unit").lower()
        f = 1 / 60 if unit.startswith("хвилин") else 1.0 if unit.startswith("годин") else 168.0 if unit.startswith("тижн") else 24.0
        a = _qty(m.group("a"))
        b = _qty(m.group("b")) if m.group("b") else a
        out["eta_hours"] = max(a, b) * f
        out["eta_hours_min"] = min(a, b) * f
        add("eta_hours", s + m.start(), s + m.end(), out["eta_hours"])
        break

    # ---- consumption change --------------------------------------------------------------------------
    for k, (s, e) in enumerate(segs):
        seg = t[s:e]
        if not (CONS_WORD.search(seg) or (k and CONS_WORD.search(t[slice(*segs[k - 1])]))):
            continue
        done = False
        for m in CONS_PCT.finditer(seg):
            lo, hi = max(0, m.start() - 70), min(len(seg), m.end() + 40)
            if re.search(r"част[кк]\w*|температур|СЕС|ВДЕ|вироб", seg[max(0, m.start() - 25):m.start()], I):
                continue
            up = [x for x in UP.finditer(seg, lo, hi)]
            dn = [x for x in DOWN.finditer(seg, lo, hi)]
            cand = [(abs(x.start() - m.start()), 1) for x in up] + [(abs(x.start() - m.start()), -1) for x in dn]
            if not cand:
                continue
            sign = min(cand)[1]
            out["consumption_change_pct"] = sign * parse_number(m.group("pct"))
            add("consumption_change_pct", s + m.start(), s + m.end(), out["consumption_change_pct"])
            done = True
            break
        if not done:
            m = SAME_LEVEL.search(seg)
            if m and re.search(r"попередн|вчора|минул|цей же час|у\s+(?:понеділок|вівторок|середу|четвер|п'ятницю|суботу|неділю)", seg, I):
                out["consumption_change_pct"] = 0.0
                add("consumption_change_pct", s + m.start(), s + m.end(), 0.0)
                done = True
        if done:
            break

    out["spans"] = json.dumps({f: [h.as_dict() for h in hits if h.field == f]
                               for f in dict.fromkeys(h.field for h in hits)}, ensure_ascii=False)
    return out


def _weather_seg(t: str, segs: list[tuple[int, int]], k: int) -> bool:
    """Weather is the stated cause of the outage in segment k (or in the sentence just before it)."""
    s, e = segs[k]
    seg = t[s:e]
    if re.search(r"бойов|обстр|атак|удар|інших\s+причин|технолог", seg, I):
        return False
    if WEATHER.search(seg):
        return True
    for j in range(k - 1, max(-1, k - 4), -1):                  # up to three sentences back, until an attack is mentioned
        prev = t[slice(*segs[j])]
        if re.search(r"бойов|обстр|атак|удар", prev, I):
            return False
        if WEATHER.search(prev):
            return True
    return False


def _sched_kinds(seg: str) -> set[str]:
    k = set()
    if re.search(r"погодинн|\bГПВ\b|стабілізаційн|черг|графік\w*\s+відключ", seg, I):
        k.add("gpv")
    if re.search(r"обмеження\s+потужності|\bГОП\b|промислов", seg, I):
        k.add("gop")
    if re.search(r"аварійн|\bГАВ\b", seg, I):
        k.add("gav")
    if not k:
        k.add("other")
    return k


def _mw_class(t: str, s: int, e: int, ms: int, me: int) -> str:
    """Class of an MW mention: the nearest cue before it in the sentence, else after it, else earlier in the paragraph."""
    def nearest_before(a: int, b: int) -> str | None:
        best = None
        for name, rx in MW_CLASSES:
            for m in rx.finditer(t, a, b):
                if best is None or m.end() > best[0]:
                    best = (m.end(), name)
        return best[1] if best else None
    c = nearest_before(s, ms)
    if c:
        return c
    after = None
    for name, rx in MW_CLASSES:
        m = rx.search(t, me, e)
        if m and (after is None or m.start() < after[0]):
            after = (m.start(), name)
    if after:
        return after[1]
    ps, _ = paragraph_of(t, ms)
    c = nearest_before(max(ps, ms - 300), ms)
    return c if c in ("import", "export", "limit") else "other"      # tables "ЕКСПОРТ/ІМПОРТ" and lists of limits


# --------------------------------------------------------------------------------------------------
# frame level
# --------------------------------------------------------------------------------------------------
LIST_COLS = ["damaged_objects", "regions", "regions_affected", "schedule_kinds", "schedule_hours"]


def extract_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Run `extract_post` over a raw posts frame; keeps channel, post_id, datetime, url, views, has_media."""
    keep = [c for c in ("channel", "post_id", "datetime", "url", "views", "has_media", "forwarded_from") if c in raw.columns]
    rows = [extract_post(t) for t in raw["text"].fillna("")]
    ext = pd.DataFrame(rows, index=raw.index)
    out = pd.concat([raw[keep], ext], axis=1)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out["text_len"] = raw["text"].fillna("").str.len()
    return out.sort_values(["datetime", "channel", "post_id"]).reset_index(drop=True)


def load_raw(raw_dir: Path, channels: list[str] | None = None, exclude: list[str] | None = None) -> pd.DataFrame:
    """All data/raw/telegram/*.parquet (or the given channels), concatenated. Works for any channel."""
    files = sorted(raw_dir.glob("*.parquet"))
    frames = []
    for f in files:
        ch = f.stem
        if channels and ch not in channels:
            continue
        if not channels and exclude and ch in exclude:
            log.info("skipping channel %s (excluded in config.telegram.exclude_channels)", ch)
            continue
        df = pd.read_parquet(f)
        if "channel" not in df.columns:
            df["channel"] = ch
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no Telegram parquet files in {raw_dir} (run `deficit collect` first)")
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    return out.drop_duplicates(["channel", "post_id"])


def posts_before(posts: pd.DataFrame, issue_time) -> pd.DataFrame:
    """Leakage guard: only posts published at or before `issue_time` (tz-aware or naive UTC)."""
    ts = pd.Timestamp(issue_time)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return posts[posts["datetime"] <= ts]


# --------------------------------------------------------------------------------------------------
# daily table
# --------------------------------------------------------------------------------------------------
def add_derived(posts: pd.DataFrame) -> pd.DataFrame:
    """Columns derived from extracted fields: `regions_hit` and `impact_post` (outage/restriction evidence)."""
    p = posts.copy()
    hit = p["is_attack_report"] | p["emergency_outages"] | p["consumers_without_power"].notna()
    p["regions_hit"] = [list(r) if h else [] for r, h in zip(p["regions_affected"], hit)]
    p["impact_post"] = (p["outage_schedules_announced"] | p["emergency_outages"] | p["consumers_without_power"].notna())
    return p


_MAX_COLS = {
    "max_consumers_without_power": "consumers_without_power", "max_consumers_weather": "consumers_weather",
    "max_settlements_without_power": "settlements_without_power", "max_deficit_mw": "deficit_mw",
    "max_import_mw": "import_mw", "max_export_mw": "export_mw", "max_limit_mw": "limit_mw",
    "max_capacity_lost_mw": "capacity_lost_mw", "max_schedule_queues": "schedule_queues_max",
}
_TS_COLS = ("date", "available_from", "first_post_at", "last_post_at", "first_schedule_post_at", "consumption_change_known_at")


def _has(series: pd.Series, kind: str) -> int:
    return int(series.map(lambda x: kind in x).sum())


def _agg_day(g: pd.DataFrame) -> dict:
    r = {
        "n_posts": len(g), "n_attack_reports": int(g["is_attack_report"].sum()), "n_attack_refs": int(g["attack_ref"].sum()),
        "n_schedule_posts": int(g["outage_schedules_announced"].sum()), "n_emergency_posts": int(g["emergency_outages"].sum()),
        "n_lifted_posts": int(g["schedules_lifted"].sum()), "n_restored_posts": int(g["restoration_power"].notna().sum()),
        "n_restored_full_posts": int((g["restoration_power"] == "full").sum()),
        "n_repair_posts": int(g["restoration_repairs"].sum()), "n_weather_posts": int(g["weather_related"].sum()),
        "n_damaged_generation": _has(g["damaged_objects"], "generation"),
        "n_damaged_substation": _has(g["damaged_objects"], "substation"),
        "n_damaged_transmission": _has(g["damaged_objects"], "transmission"),
        "n_damaged_distribution": _has(g["damaged_objects"], "distribution"),
        "nationwide": bool(g["nationwide"].any()),
    }
    for out_col, col in _MAX_COLS.items():
        r[out_col] = g[col].max() if g[col].notna().any() else np.nan
    restored = g.loc[~g["restored_is_aggregate"], "consumers_restored"]
    r["max_consumers_restored"] = restored.max() if restored.notna().any() else np.nan
    regs = sorted({x for lst in g["regions_hit"] for x in lst})
    r["regions_hit"] = regs
    r["n_regions_hit"] = len(regs)
    cc = g.dropna(subset=["consumption_change_pct"])
    r["consumption_change_pct_first"] = cc["consumption_change_pct"].iloc[0] if len(cc) else np.nan
    r["consumption_change_known_at"] = cc["datetime"].iloc[0] if len(cc) else pd.NaT
    r["first_post_at"] = g["datetime"].min()
    r["last_post_at"] = g["datetime"].max()
    sched = g.loc[g["outage_schedules_announced"], "datetime"]
    r["first_schedule_post_at"] = sched.min() if len(sched) else pd.NaT
    return r


def daily_table(posts: pd.DataFrame) -> pd.DataFrame:
    """One row per (UTC date, channel) plus channel == "ALL" pooling every channel.

    Leakage: a row summarises a whole UTC day, so it is usable only from `available_from` (next 00:00 UTC);
    `last_post_at` is the timestamp of the latest post inside the row.
    """
    p = add_derived(posts)
    p["date"] = p["datetime"].dt.floor("D")
    rows = []
    for (date, ch), g in p.groupby(["date", "channel"], sort=True):
        rows.append({"date": date, "channel": ch, **_agg_day(g)})
    for date, g in p.groupby("date", sort=True):
        rows.append({"date": date, "channel": "ALL", **_agg_day(g)})
    out = pd.DataFrame(rows)
    out["available_from"] = out["date"] + pd.Timedelta(days=1)
    first = ["date", "channel", "available_from"]
    out = out[first + [c for c in out.columns if c not in first]]
    for c in _TS_COLS:
        out[c] = pd.to_datetime(out[c], utc=True)
    return out.sort_values(["channel", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# impact episodes
# --------------------------------------------------------------------------------------------------
@dataclass
class EpisodeParams:
    horizon_days: int = 30             # how long after the strike posts are searched
    relapse_days: int = 3              # impact evidence within this many days after the resolution = relapse
    national_channels: tuple[str, ...] = ("ukrenergo",)   # channels whose lifting statements define the ALL resolution
    pre_strike_days: int = 3           # look-back for "restrictions already in force before the strike"


def load_strikes(path: Path) -> pd.DataFrame:
    """data/external/attacks.csv -> one row per strike with tz-aware UTC `strike_at`."""
    df = pd.read_csv(path)
    time = df["time_utc"].fillna("00:00") if "time_utc" in df.columns else "00:00"
    df["strike_at"] = pd.to_datetime(df["date"].astype(str) + " " + time, utc=True)
    if "major" in df.columns:
        df["major"] = df["major"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["major"] = True
    keep = [c for c in ("strike_at", "description", "major", "weapons_fired", "weapons_through", "missiles_fired") if c in df.columns]
    return df[keep].drop_duplicates("strike_at").sort_values("strike_at").reset_index(drop=True)


def _first(df: pd.DataFrame, mask: pd.Series):
    sub = df[mask]
    return sub["datetime"].iloc[0] if len(sub) else pd.NaT


def compute_episode(p: pd.DataFrame, strike_at: pd.Timestamp, next_strike_at, prm: EpisodeParams,
                    resolution_posts: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> dict:
    """Impact episode of one strike from posts published in [strike day 00:00 UTC, +horizon] and <= `as_of`.

    `p` must carry the add_derived() columns. Every value is a function of posts with datetime <= as_of only
    (the *_at fields give the publication time of the post that determined the value).
    The window starts at 00:00 UTC of the strike day because attacks.csv only fixes the day (06:00 is a placeholder).
    """
    start = strike_at.floor("D")
    end = start + pd.Timedelta(days=prm.horizon_days)
    cut = end if as_of is None else min(end, as_of)
    w = p[(p["datetime"] >= start) & (p["datetime"] <= cut)]
    rp = resolution_posts if resolution_posts is not None else p
    rw = rp[(rp["datetime"] >= start) & (rp["datetime"] <= cut)]
    pre_mask = (p["datetime"] >= start - pd.Timedelta(days=prm.pre_strike_days)) & (p["datetime"] < start)
    if as_of is not None:
        pre_mask &= p["datetime"] <= as_of
    pre = p[pre_mask]
    ep: dict = {"window_start": start, "window_end": cut, "n_posts": len(w),
                "n_channels": int(w["channel"].nunique()) if len(w) else 0,
                "pre_strike_schedule_posts": int(pre["outage_schedules_announced"].sum()) if len(pre) else 0,
                "n_attack_reports": int(w["is_attack_report"].sum()) if len(w) else 0}
    ep["first_report_at"] = _first(w, w["is_attack_report"]) if len(w) else pd.NaT
    imp = w[w["impact_post"]] if len(w) else w
    ep["impact_observed"] = bool(len(imp))
    ep["first_impact_at"] = imp["datetime"].iloc[0] if len(imp) else pd.NaT
    ep["last_impact_at"] = imp["datetime"].iloc[-1] if len(imp) else pd.NaT
    sched = w[w["outage_schedules_announced"]] if len(w) else w
    ep["first_schedule_at"] = sched["datetime"].iloc[0] if len(sched) else pd.NaT
    ep["schedule_days"] = int(sched["datetime"].dt.floor("D").nunique()) if len(sched) else 0
    ep["max_schedule_queues"] = float(w["schedule_queues_max"].max()) if len(w) and w["schedule_queues_max"].notna().any() else np.nan
    em = w[w["emergency_outages"]] if len(w) else w
    ep["emergency_outages"] = bool(len(em))
    ep["first_emergency_at"] = em["datetime"].iloc[0] if len(em) else pd.NaT
    for out_col, col in (("max_consumers_without_power", "consumers_without_power"),
                         ("max_settlements_without_power", "settlements_without_power"),
                         ("max_deficit_mw", "deficit_mw"), ("max_import_mw", "import_mw"), ("max_limit_mw", "limit_mw"),
                         ("max_capacity_lost_mw", "capacity_lost_mw")):
        if len(w) and w[col].notna().any():
            i = w[col].idxmax()
            ep[out_col], ep[out_col + "_at"] = float(w.loc[i, col]), w.loc[i, "datetime"]
        else:
            ep[out_col], ep[out_col + "_at"] = np.nan, pd.NaT
    regs = sorted({r for lst in w["regions_hit"] for r in lst}) if len(w) else []
    ep["regions_hit"], ep["n_regions_hit"] = regs, len(regs)
    ep["nationwide"] = bool(w["nationwide"].any()) if len(w) else False
    ep["damaged_objects"] = sorted({o for lst in w["damaged_objects"] for o in lst}) if len(w) else []
    # ---- observed restoration ------------------------------------------------------------------
    # Track A (system level): restrictions (schedules / emergency outages) announced -> first post saying they are not
    # applied / cancelled.  Track B (consumer level): first post with consumers cut off or an attack report -> first post
    # saying all consumers have power again.  `resolution_*` follows track A when a restriction regime was seen, else B.
    restr = rw[rw["outage_schedules_announced"] | rw["emergency_outages"]] if len(rw) else rw
    ep["restrictions_observed"] = bool(len(restr))
    ep["first_restriction_at"] = restr["datetime"].iloc[0] if len(restr) else pd.NaT
    ep["last_restriction_at"] = restr["datetime"].iloc[-1] if len(restr) else pd.NaT
    ep["restriction_days"] = int(restr["datetime"].dt.floor("D").nunique()) if len(restr) else 0
    ep.update({"lifted_at": pd.NaT, "lifted_sustained_at": pd.NaT, "restored_at": pd.NaT, "relapse": False})
    if len(restr):
        t0 = ep["first_restriction_at"]
        after = rw[rw["datetime"] > t0]
        lifted = after[after["schedules_lifted"]]
        if len(lifted):
            ep["lifted_at"] = lifted["datetime"].iloc[0]
            for t in lifted["datetime"]:
                nxt = restr[(restr["datetime"] > t) & (restr["datetime"] <= t + pd.Timedelta(days=prm.relapse_days))]
                confirmed = cut >= t + pd.Timedelta(days=prm.relapse_days)
                if not len(nxt) and confirmed:
                    ep["lifted_sustained_at"] = t
                    break
            later = restr[(restr["datetime"] > ep["lifted_at"])
                          & (restr["datetime"] <= ep["lifted_at"] + pd.Timedelta(days=prm.relapse_days))]
            ep["relapse"] = bool(len(later))
    cons = w[w["consumers_without_power"].notna() | w["is_attack_report"]] if len(w) else w
    ep["first_outage_at"] = cons["datetime"].iloc[0] if len(cons) else pd.NaT
    if len(cons):
        after = w[w["datetime"] > ep["first_outage_at"]]
        ep["restored_at"] = _first(after, after["restoration_power"] == "full") if len(after) else pd.NaT
    if pd.notna(ep["lifted_at"]):
        res, kind = ep["lifted_at"], "restrictions_lifted"
    elif len(restr):
        res, kind = pd.NaT, None
    elif pd.notna(ep["restored_at"]):
        res, kind = ep["restored_at"], "power_restored"
    else:
        res, kind = pd.NaT, None
    ep["resolution_at"], ep["resolution_kind"] = res, kind

    def days(t):
        return max((t - strike_at).total_seconds() / 86400, 0.0) if pd.notna(t) else np.nan
    ep["days_to_resolution"] = days(res)
    ep["days_to_restrictions_lifted"] = days(ep["lifted_at"])
    ep["days_to_restrictions_lifted_sustained"] = days(ep["lifted_sustained_at"])
    ep["days_to_power_restored"] = days(ep["restored_at"])
    ep["days_to_last_restriction"] = days(ep["last_restriction_at"])
    ep["next_strike_gap_days"] = (next_strike_at - strike_at).total_seconds() / 86400 if pd.notna(next_strike_at) else np.nan
    ep["resolved_before_next_strike"] = bool(pd.notna(res) and (pd.isna(next_strike_at) or res < next_strike_at))
    ep["status"] = ("observed" if pd.notna(res) else "censored" if (len(restr) or ep["impact_observed"]) else "no_impact_reported")
    return ep


def build_episodes(posts: pd.DataFrame, strikes: pd.DataFrame, prm: EpisodeParams | None = None,
                   as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Episodes for every strike, for scope "ALL" (resolution from the national channels) and each channel with posts.

    With `as_of` only posts published at or before that instant are used (the leakage-safe view for a feature issued
    at `as_of`; strikes after `as_of` are dropped). Without it the table holds the full ex-post labels.
    """
    prm = prm or EpisodeParams()
    p = add_derived(posts) if "impact_post" not in posts.columns else posts
    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
        p = p[p["datetime"] <= as_of]
        strikes = strikes[strikes["strike_at"] <= as_of]
    nat = p[p["channel"].isin(prm.national_channels)]
    if nat.empty:
        nat = p
    strikes = strikes.reset_index(drop=True)
    rows = []
    for i, st in strikes.iterrows():
        nxt = strikes["strike_at"].iloc[i + 1] if i + 1 < len(strikes) else pd.NaT
        base = {"strike_at": st["strike_at"], "major": bool(st.get("major", True)), "description": st.get("description")}
        for extra in ("weapons_fired", "weapons_through", "missiles_fired"):
            if extra in st:
                base[extra] = st[extra]
        ep = compute_episode(p, st["strike_at"], nxt, prm, resolution_posts=nat, as_of=as_of)
        rows.append({**base, "scope": "ALL", **ep})
        window = p[(p["datetime"] >= ep["window_start"]) & (p["datetime"] <= ep["window_end"])]
        for ch in sorted(window["channel"].unique()):
            rows.append({**base, "scope": ch,
                         **compute_episode(p[p["channel"] == ch], st["strike_at"], nxt, prm, as_of=as_of)})
    return pd.DataFrame(rows)


def episode_as_of(posts: pd.DataFrame, strike_at, as_of, prm: EpisodeParams | None = None, scope: str = "ALL") -> dict:
    """Episode features of one strike as known at `as_of` (leakage-safe: later posts are ignored)."""
    strike_at = pd.Timestamp(strike_at)
    strike_at = strike_at.tz_localize("UTC") if strike_at.tzinfo is None else strike_at.tz_convert("UTC")
    ep = build_episodes(posts, pd.DataFrame({"strike_at": [strike_at], "major": [True]}), prm, as_of=as_of)
    row = ep[ep["scope"] == scope]
    return row.iloc[0].to_dict() if len(row) else {}


# --------------------------------------------------------------------------------------------------
# step entry point
# --------------------------------------------------------------------------------------------------
def run_extract(data_root: Path, channels: list[str] | None = None, exclude: list[str] | None = None,
                prm: EpisodeParams | None = None, strikes_path: Path | None = None,
                channel_oblast: dict[str, str] | None = None,
                window_info: dict[str, dict] | None = None, proc_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """raw/telegram/*.parquet -> processed/telegram_{posts,daily,episodes}.parquet and, when `channel_oblast`
    (channel -> oblast code from config/sources.yaml) is given, telegram_region_{posts,episodes}.parquet."""
    prm = prm or EpisodeParams()
    raw = load_raw(data_root / "raw" / "telegram", channels, exclude)
    log.info("extracting %d posts from %s", len(raw), ", ".join(sorted(raw["channel"].unique())))
    posts = add_derived(extract_frame(raw))
    proc = proc_dir or (data_root / "processed")      # scenario-scoped output; raw/ is shared
    proc.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(proc / "telegram_posts.parquet")
    daily = daily_table(posts)
    daily.to_parquet(proc / "telegram_daily.parquet")
    log.info("wrote %s (%d posts) and %s (%d rows)", proc / "telegram_posts.parquet", len(posts),
             proc / "telegram_daily.parquet", len(daily))
    out = {"posts": posts, "daily": daily}
    strikes_path = strikes_path or (data_root / "external" / "attacks.csv")
    if strikes_path.exists():
        eps = build_episodes(posts, load_strikes(strikes_path), prm)
        eps.to_parquet(proc / "telegram_episodes.parquet")
        out["episodes"] = eps
        if channel_oblast is not None:
            from .telegram_regional import run_regional
            out.update(run_regional(posts, raw, load_strikes(strikes_path), channel_oblast, proc, prm,
                                          window_info=window_info))
        a = eps[eps["scope"] == "ALL"]
        log.info("episodes: %d strikes, impact reported %d, observed restoration %d, censored %d (wrote %s)",
                 len(a), int(a["impact_observed"].sum()), int((a["status"] == "observed").sum()),
                 int((a["status"] == "censored").sum()), proc / "telegram_episodes.parquet")
    else:
        log.warning("%s missing: no impact episodes", strikes_path)
    return out
