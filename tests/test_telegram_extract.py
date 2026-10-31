"""Rule-based extraction of strike-impact facts from Ukrainian Telegram posts: fixed example texts, no network.

Texts are short excerpts / paraphrases of public posts of Ukrenergo, DTEK and the oblenergos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deficit.ingestion import telegram_extract as te
from deficit.ingestion.telegram_extract import EpisodeParams, extract_post


def ex(text: str) -> dict:
    return extract_post(text)


# ---------------------------------------------------------------- attack report and objects
def test_attack_report_with_objects_and_regions():
    r = ex("Вночі ворог здійснив масовану ракетно-дронову атаку на енергооб'єкти. Пошкоджено підстанцію 330 кВ у Харківській "
           "області та ТЕС у Київській області. Є знеструмлення.")
    assert r["is_attack_report"] and not r["attack_ref"]
    assert {"generation", "substation"} <= set(r["damaged_objects"])
    assert {"Kharkiv", "Kyiv"} <= set(r["regions"])


def test_reason_for_restrictions_is_a_reference_not_a_report():
    r = ex("Завтра, 18 листопада, в усіх регіонах будуть застосовуватись заходи обмеження споживання. "
           "Причина - наслідки попередніх російських ударів по енергетичній інфраструктурі.")
    assert not r["is_attack_report"] and r["attack_ref"]


def test_retrospective_and_commemoration_are_not_strike_reports():
    assert not ex("11 вересня 2022 року на підстанції полетіли перші російські ракети, загинули наші колеги. Пам'ятаємо.")["is_attack_report"]
    assert not ex("Понад 220 атак пережили наші ТЕС, серйозних пошкоджень зазнали й мережі.")["is_attack_report"]


def test_object_types_distribution_and_transmission():
    r = ex("Сьогодні вночі через атаку пошкоджено повітряну лінію 330 кВ та трансформаторні підстанції в мережах обленерго у Сумській області.")
    assert r["is_attack_report"]
    assert "transmission" in r["damaged_objects"] and "distribution" in r["damaged_objects"]


# ---------------------------------------------------------------- consumers
def test_consumers_without_power_thousands_and_millions():
    assert ex("Через обстріли в Херсоні знеструмлено понад 90 тисяч споживачів")["consumers_without_power"] == 90_000
    assert ex("Без світла залишається 1,4 млн родин Донеччини.")["consumers_without_power"] == 1_400_000
    assert ex("Без енергопостачання понад 16 500 споживачів.")["consumers_without_power"] == 16_500
    assert ex("На ранок знеструмлено близько 209 тис. побутових споживачів.")["consumers_without_power"] == 209_000


def test_consumers_restored_is_separate_from_without_power():
    r = ex("Енергетики повернули світло для 185 000 родин. Без світла залишаються майже 98 тисяч родин.")
    assert r["consumers_restored"] == 185_000 and r["consumers_without_power"] == 98_000


def test_weather_outages_do_not_count_as_strike_outages():
    r = ex("Через негоду знеструмлено 5 848 абонентів та 37 населених пунктів.")
    assert r["consumers_without_power"] is None and r["consumers_weather"] == 5848
    assert r["settlements_without_power"] is None and r["weather_related"]


def test_non_consumer_numbers_are_ignored():
    r = ex("760 тис. грн для ЗСУ скерували енергетики. Понад 130 тисяч користувачів Viber-бота. 2,4 млн кВт∙год.")
    assert r["consumers_without_power"] is None and r["consumers_restored"] is None


def test_queue_count_is_not_read_as_consumers():
    r = ex("Введені графіки аварійних відключень для 10 черг споживачів.")
    assert r["consumers_without_power"] is None and r["schedule_queues_max"] == 10


def test_settlements_without_power():
    assert ex("Через бойові дії та з інших причин без світла залишається 418 населених пунктів.")["settlements_without_power"] == 418


# ---------------------------------------------------------------- schedules, emergency, lifting
def test_schedules_hours_and_queues():
    r = ex("УВАГА! ІНФОРМАЦІЯ ПРО ВІДКЛЮЧЕННЯ\n\nЗавтра, 9 листопада, у більшості регіонів будуть застосовуватись заходи обмеження споживання:\n\n"
           "ГРАФІКИ ПОГОДИННИХ ВІДКЛЮЧЕНЬ\n• з 00:00 до 23:59 – обсягом від 2 до 4 черг\n\n"
           "ГРАФІКИ ОБМЕЖЕННЯ ПОТУЖНОСТІ\n• з 00:00 до 23:59 – для промислових споживачів\n")
    assert r["outage_schedules_announced"] and not r["schedules_lifted"]
    assert r["schedule_hours"] == ["00:00-23:59"]
    assert (r["schedule_queues_min"], r["schedule_queues_max"]) == (2, 4)
    assert {"gpv", "gop"} <= set(r["schedule_kinds"])


def test_appliance_advice_hours_are_not_schedule_hours():
    r = ex("Завтра, 17 вересня, заходи обмеження не прогнозуються.\n\nБудь ласка, користуйтесь потужними електроприладами з 10:00 до 16:00")
    assert r["schedule_hours"] == [] and r["schedules_lifted"] and not r["outage_schedules_announced"]


def test_lifting_statements():
    assert ex("УВАГА! 10 ЖОВТНЯ ВІДКЛЮЧЕННЯ НЕ ПРОГНОЗУЮТЬСЯ")["schedules_lifted"]
    assert ex("На завтра команди НЕК на застосування графіків погодинних відключень на Житомирщині не надходило.")["schedules_lifted"]
    assert ex("Графіки погодинних відключень, ГОП та ГАВ за попередню добу не застосовувалися.")["schedules_lifted"]
    r = ex("Виробленої електроенергії достатньо, не має підстав для запровадження графіків відключень.")
    assert r["schedules_lifted"] and not r["outage_schedules_announced"]


def test_not_lifting_when_cancellation_is_future_or_counterfactual():
    assert not ex("Обмеження діятимуть до скасування запроваджених обмежень командою Укренерго.")["schedules_lifted"]
    assert not ex("Якби атак не було, ми пройшли б період сильних морозів взагалі без відключень.")["schedules_lifted"]
    assert ex("АВАРІЙНІ ВІДКЛЮЧЕННЯ СКАСОВАНІ\n\nЗаходи обмеження споживання не застосовуються у жодному регіоні")["schedules_lifted"]


def test_partial_lifting_is_not_lifting():
    r = ex("Завтра з 16:00 до 19:00 будуть застосовуватись графіки обмеження потужності для промисловості.\n"
           "Відключення побутових споживачів наразі не прогнозується")
    assert r["outage_schedules_announced"] and not r["schedules_lifted"]


def test_hedged_schedules_are_not_announced():
    r = ex("Можливе застосування графіків аварійних відключень. Щоб не довелося застосовувати погодинні відключення, не вмикайте прилади.")
    assert not r["outage_schedules_announced"] and not r["emergency_outages"] and r["restrictions_possible"]


def test_emergency_outages_now_vs_equipment_vs_past():
    assert ex("У зв'язку з масованою ракетною атакою у Харківській та Сумській областях застосовані аварійні відключення")["emergency_outages"]
    assert not ex("Об 11.22 відбулось аварійне відключення ПЛ 750 кВ Чорнобильська АЕС - Київська.")["emergency_outages"]
    past = ex("Вчора, 10 червня, на Харківщині був вимушений застосувати аварійні відключення для споживачів.")
    assert not past["emergency_outages"] and past["emergency_past"]
    assert not ex("Аварійні відключення скасовано. Знову зі світлом 413 000 родин.")["emergency_outages"]


def test_emergency_schedules_in_force_are_both():
    r = ex("Шановні споживачі! На території Сумщини о 07:14 введені графіки аварійних відключень для 10 черг споживачів.")
    assert r["emergency_outages"] and r["outage_schedules_announced"] and r["schedule_queues_max"] == 10


def test_emergency_outages_in_force_block_lifting():
    r = ex("Попередньо оприлюднені графіки наразі не діють. Застосовані аварійні відключення.")
    assert r["emergency_outages"] and not r["schedules_lifted"]


# ---------------------------------------------------------------- megawatts
def test_mw_classes_and_units():
    r = ex("ІМПОРТ\nПротягом доби з Румунії. Загальним обсягом 28 139 МВт*год, з максимальною потужністю в окремі години до 1567 МВт.\n"
           "ЕКСПОРТ\nНе відбувається.\nДефіцит потужності сягав 1,2 ГВт. Встановлені ліміти споживання до 514 мегават.")
    assert r["import_mw"] == 1567 and r["deficit_mw"] == 1200 and r["limit_mw"] == 514
    assert r["export_mw"] is None


def test_energy_units_are_not_power():
    r = ex("Експорт: загальний обсяг 6341 МВт·год.")
    assert r["export_mw"] is None


def test_capacity_added_is_not_a_limit():
    assert ex("Сьогодні підключено блок, який додасть 1000 МВт потужності і дозволить не застосовувати обмеження.")["limit_mw"] is None


# ---------------------------------------------------------------- restoration
def test_restoration_full_and_partial_and_not_future():
    assert ex("Наразі всі споживачі — зі світлом ✅")["restoration_power"] == "full"
    assert ex("В Кіровоградській області заживлено всіх споживачів, знеструмлених вчора через атаку.")["restoration_power"] == "full"
    assert ex("Енергетики повернули світло для 8 000 родин.")["restoration_power"] == "partial"
    assert ex("Станом на ранок вдалося заживити майже усіх споживачів.")["restoration_power"] == "partial"
    # purpose / future / negation are not restoration statements
    assert ex("Енергетики роблять усе можливе, щоб якнайшвидше заживити усіх споживачів.")["restoration_power"] is None
    assert ex("Використовуйте техніку, коли електропостачання буде відновлене.")["restoration_power"] is None
    assert ex("Електропостачання ще не відновлено.")["restoration_power"] is None


def test_repairs_completed_and_ongoing():
    assert ex("Ремонтні бригади відновили надійність живлення ЗАЕС. Лінію 330 кВ включено в роботу.")["restoration_repairs"]
    assert not ex("Аварійно-відновлювальні роботи тривають.")["restoration_repairs"]
    assert not ex("Енергетики працюють над тим, щоб якнайшвидше повернути в роботу пошкоджене обладнання.")["restoration_repairs"]


def test_eta_hours_only_for_estimates():
    r = ex("Відновлення електропостачання очікується упродовж двох годин.")
    assert r["eta_hours"] == 2
    assert ex("Протягом 3-4 годин енергоживлення буде відновлено.")["eta_hours"] == 4
    assert ex("Використовуйте прилади через 15-30 хвилин після відновлення електропостачання.")["eta_hours"] is None
    assert ex("Через 30 хвилин енергоживлення було відновлено.")["eta_hours"] is None


# ---------------------------------------------------------------- consumption, regions
def test_consumption_change_sign_and_same_level():
    assert ex("Сьогодні станом на 9:30 споживання було на 5,8% нижчим, ніж у середу.")["consumption_change_pct"] == -5.8
    assert ex("Споживання зросло. Станом на 9:00 його рівень був на 3,3% вищим, ніж вчора.")["consumption_change_pct"] == 3.3
    assert ex("Споживання електроенергії тримається на високому рівні. Сьогодні воно було таким же, як в цей час попереднього дня.")["consumption_change_pct"] == 0
    assert ex("Частка ТЕС у покритті навантаження склала 22,2%.")["consumption_change_pct"] is None


def test_regions_normalised():
    r = ex("Знеструмлення у Києві, Київщині, на Одещині та в Сумах. Також Дніпропетровська область, м. Львів.")
    assert {"Kyiv_city", "Kyiv", "Odesa", "Sumy", "Dnipropetrovsk", "Lviv"} <= set(r["regions"])
    # company names and lower-case "суми" (sums of money) are not regions
    r = ex("Львівобленерго перерахувало значні суми на потреби ЗСУ.")
    assert r["regions"] == []


def test_nationwide_only_in_outage_context():
    assert ex("По всій території України застосовуються графіки погодинних відключень.")["nationwide"]
    assert not ex("Вітаємо всіх по всій Україні з Днем енергетика!")["nationwide"]


# ---------------------------------------------------------------- audit spans, robustness
def test_spans_point_to_the_matched_text():
    text = "​Вночі ворог атакував енергооб’єкти. Без енергопостачання понад 16 500 споживачів.\nТривають роботи."
    r = ex(text)
    spans = json.loads(r["spans"])
    assert text[spans["consumers_without_power"][0]["start"]:spans["consumers_without_power"][0]["end"]] == "16 500 споживачів"
    assert spans["consumers_without_power"][0]["value"] == 16_500
    for field, items in spans.items():
        for it in items:
            assert text[it["start"]:it["end"]] == it["text"], field


def test_normalize_preserves_length_so_spans_stay_valid():
    raw = "​​З обʼєкта’ та – «тест» "
    assert len(te.normalize(raw)) == len(raw)


def test_empty_and_non_string_input():
    for t in ("", None, "   ", float("nan")):
        r = ex(t)
        assert not r["is_attack_report"] and r["regions"] == [] and r["spans"] == "{}"


# ---------------------------------------------------------------- pipeline: frame, daily table, episodes
def _raw(rows) -> pd.DataFrame:
    return pd.DataFrame([{"channel": c, "post_id": i, "datetime": pd.Timestamp(t, tz="UTC"), "text": x, "url": f"https://t.me/{c}/{i}"}
                         for i, (c, t, x) in enumerate(rows, 1)])


STRIKE = pd.DataFrame({"strike_at": [pd.Timestamp("2024-03-22 06:00", tz="UTC"), pd.Timestamp("2024-04-27 06:00", tz="UTC")],
                       "major": [True, True]})
POSTS = [
    ("ukrenergo", "2024-03-21 08:00", "СТАН ЕНЕРГОСИСТЕМИ\nЗаходи обмеження сьогодні не застосовуються."),
    ("ukrenergo", "2024-03-22 05:30", "Вночі ворог здійснив масовану атаку на ТЕС у Харківській області. Пошкоджено обладнання. "
                                      "Без світла залишаються 500 тисяч споживачів."),
    ("ukrenergo", "2024-03-22 15:00", "Завтра, 23 березня, у всіх регіонах з 06:00 до 22:00 вимушено будуть застосовуватись заходи "
                                      "обмеження споживання: графіки погодинних відключень обсягом 2 черги."),
    ("ukrenergo", "2024-03-25 15:00", "Завтра, 26 березня, у всіх регіонах з 06:00 до 22:00 будуть застосовуватись графіки погодинних відключень."),
    ("ukrenergo", "2024-03-27 15:00", "УВАГА! 28 БЕРЕЗНЯ ВІДКЛЮЧЕННЯ НЕ ПРОГНОЗУЮТЬСЯ"),
    ("dtek_ua", "2024-03-23 09:00", "Харківщина: всі споживачі знову зі світлом після обстрілів."),
    ("ukrenergo", "2024-04-27 15:00", "Сьогодні, 27 квітня, діють графіки погодинних відключень."),
]


@pytest.fixture(scope="module")
def posts():
    return te.add_derived(te.extract_frame(_raw(POSTS)))


def test_extract_frame_keeps_timestamps_and_flags(posts):
    assert posts["datetime"].is_monotonic_increasing and str(posts["datetime"].dt.tz) == "UTC"
    row = posts[(posts.channel == "ukrenergo") & (posts.post_id == 2)].iloc[0]
    assert row["is_attack_report"] and row["consumers_without_power"] == 500_000 and "generation" in row["damaged_objects"]


def test_daily_table_is_usable_only_from_next_day(posts):
    d = te.daily_table(posts)
    a = d[(d.channel == "ALL") & (d.date == pd.Timestamp("2024-03-22", tz="UTC"))].iloc[0]
    assert a["n_posts"] == 2 and a["n_attack_reports"] == 1 and a["max_consumers_without_power"] == 500_000
    assert a["available_from"] == pd.Timestamp("2024-03-23", tz="UTC")
    assert a["last_post_at"] < a["available_from"]
    assert "Kharkiv" in a["regions_hit"]


def test_episode_observed_resolution(posts):
    eps = te.build_episodes(posts, STRIKE, EpisodeParams(horizon_days=20))
    e = eps[(eps.scope == "ALL") & (eps.strike_at == STRIKE.strike_at[0])].iloc[0]
    assert e["impact_observed"] and e["restrictions_observed"] and e["max_consumers_without_power"] == 500_000
    assert e["first_restriction_at"] == pd.Timestamp("2024-03-22 15:00", tz="UTC")
    assert e["lifted_at"] == pd.Timestamp("2024-03-27 15:00", tz="UTC") and e["status"] == "observed"
    assert e["days_to_resolution"] == pytest.approx(5.375, abs=1e-6) and e["resolution_kind"] == "restrictions_lifted"
    assert e["restored_at"] == pd.Timestamp("2024-03-23 09:00", tz="UTC")       # "all consumers with light" (consumer level)
    assert e["resolved_before_next_strike"] and e["n_regions_hit"] >= 1
    # the next strike has restrictions but no lifting yet -> censored, no duration
    e2 = eps[(eps.scope == "ALL") & (eps.strike_at == STRIKE.strike_at[1])].iloc[0]
    assert e2["status"] == "censored" and np.isnan(e2["days_to_resolution"])


def test_episode_as_of_ignores_later_posts(posts):
    """Leakage guard: the resolution published on 03-27 must be invisible at an issue time before it."""
    before = te.episode_as_of(posts, STRIKE.strike_at[0], pd.Timestamp("2024-03-26 00:00", tz="UTC"))
    assert before["status"] == "censored" and pd.isna(before["lifted_at"]) and np.isnan(before["days_to_resolution"])
    assert before["max_consumers_without_power"] == 500_000 and before["window_end"] <= pd.Timestamp("2024-03-26", tz="UTC")
    after = te.episode_as_of(posts, STRIKE.strike_at[0], pd.Timestamp("2024-03-28 00:00", tz="UTC"))
    assert after["status"] == "observed"
    # changing a future post must not change what was known earlier
    changed = [(c, t, x) for c, t, x in POSTS]
    changed[4] = ("ukrenergo", "2024-03-27 15:00", "Завтра будуть застосовуватись графіки погодинних відключень.")
    posts2 = te.add_derived(te.extract_frame(_raw(changed)))
    before2 = te.episode_as_of(posts2, STRIKE.strike_at[0], pd.Timestamp("2024-03-26 00:00", tz="UTC"))
    for k in ("status", "n_posts", "first_restriction_at", "max_consumers_without_power", "restriction_days"):
        assert before2[k] == before[k] or (pd.isna(before2[k]) and pd.isna(before[k]))


def test_posts_before_is_inclusive_and_utc(posts):
    cut = te.posts_before(posts, "2024-03-22 15:00")
    assert cut["datetime"].max() == pd.Timestamp("2024-03-22 15:00", tz="UTC")
    assert te.posts_before(posts, pd.Timestamp("2024-03-22 17:00", tz="Europe/Kyiv"))["datetime"].max() == pd.Timestamp("2024-03-22 15:00", tz="UTC")


def test_run_extract_any_channel_and_excluded_channels(tmp_path: Path):
    raw = tmp_path / "raw" / "telegram"
    raw.mkdir(parents=True)
    rows = _raw([("somechan", "2024-03-22 05:30", "Вночі ворог атакував підстанцію у Сумській області. Знеструмлено 1 200 споживачів."),
                 ("somechan", "2024-03-22 12:00", "Усі споживачі зі світлом.")])
    rows.to_parquet(raw / "somechan.parquet")
    _raw([("kpszsu", "2024-03-22 05:00", "Масована атака ракетами по енергетиці.")]).to_parquet(raw / "kpszsu.parquet")
    ext = tmp_path / "external"
    ext.mkdir()
    pd.DataFrame({"date": ["2024-03-22"], "time_utc": ["06:00"], "description": ["x"], "major": [True]}).to_csv(ext / "attacks.csv", index=False)
    out = te.run_extract(tmp_path, exclude=["kpszsu"], prm=EpisodeParams(national_channels=("somechan",)))
    assert set(out["posts"]["channel"]) == {"somechan"}
    assert (tmp_path / "processed" / "telegram_posts.parquet").exists() and (tmp_path / "processed" / "telegram_daily.parquet").exists()
    ep = pd.read_parquet(tmp_path / "processed" / "telegram_episodes.parquet")
    assert set(ep["scope"]) == {"ALL", "somechan"}
    # an explicit channel list overrides the exclusion
    out2 = te.run_extract(tmp_path, channels=["kpszsu"], exclude=["kpszsu"])
    assert set(out2["posts"]["channel"]) == {"kpszsu"}
