## How the gold labels were made

* **Sample (114 posts, "main")**: stratified by channel (ukrenergo 36, dtek_ua 30, Lvivoblenergo 20, ztoe_ua 16, sumyoblenergo 12). In every
  channel about 40 % are *random* posts (year-stratified, 49 in total, to measure false positives on ordinary traffic) and about 60 %
  *enriched* posts drawn from four keyword families (attack and outage words, schedule words, numbers or MW, restoration words; 65 in total) so
  that rare fields have positives. Seed 20261018, posts of at least 25 characters, long digests printed up to 4 200 characters.
* **Targeted supplement (72 posts)**, drawn after the main sample had been used to fix the extractor, so it is a partly held-out check of the
  fields that are rare in the main sample: 24 posts with "number + consumers" patterns (`targeted_K`), 14 with duration phrases (`targeted_E`),
  16 more with duration phrases next to restoration words (`targeted_eta`), 8 with MW values (`targeted_W`), 10 with restoration verbs
  (`targeted_R`). Posts of at most 3 000 characters, so long digests are under-represented there. Only the fields listed in the `scored` column
  are scored for these rows (consumers, consumers_restored, settlements, eta_h, mw, restored_*, repairs; `targeted_eta` only eta_h).
* Every post was read in full and labelled **before** looking at extractor output (the extractor was run only afterwards). Labels follow the
  definitions in the module docstring: `attack` = a new strike or shelling damage on energy objects is reported (references such as "restrictions
  because of earlier strikes", retrospectives, interviews, deaths of workers, weekly/monthly restoration totals are 0); regions = oblasts and cities
  mentioned (company names are not regions); `regions_affected` = regions named next to damage, outage, restriction or attack; consumers = non-weather
  count of consumers cut off; MW classes deficit/import/export/limit/capacity_lost with the largest stated value per class; `eta_h` = a stated
  *estimate* of restoration time (past durations and schedule lengths do not count).
* **Honest limits.** (1) The extractor was tuned on the main sample after a first blind pass (first-pass numbers below), so the main-sample
  numbers in the table are optimistic; only the supplement was partly held out, and its failures (e.g. 9 false ETAs, MW lists without cue word,
  weather look-back, "м. Житомир" splitting a sentence) also led to fixes. (2) 114 + 72 posts and few positives for some fields (eta_h: 1 positive in
  186 posts, consumers: 10, lifted: 3), so those precision/recall values are anecdotal. (3) One person labelled; borderline cases were decided by
  the text alone (see "label consistency" below). (4) Recall for consumers/ETA/MW is conditional on posts that contain a number or phrase of that
  kind (recall for facts phrased without digits, e.g. "сотні тисяч точок обліку", is not measured).

### First blind pass on the main sample (before any tuning; 114 posts)

| field | tp | fp | fn | precision | recall |
|---|---|---|---|---|---|
| attack | 12 | 2 | 18 | 0.86 | 0.40 |
| objects | 6 | 2 | 23 | 0.75 | 0.21 |
| regions | 145 | 10 | 5 | 0.94 | 0.97 |
| regions_affected | 95 | 6 | 34 | 0.94 | 0.74 |
| consumers | 1 | 4 | 0 | 0.20 | 1.00 |
| consumers_restored | 6 | 0 | 4 | 1.00 | 0.60 |
| settlements | 5 | 2 | 0 | 0.71 | 1.00 |
| schedules | 33 | 1 | 6 | 0.97 | 0.85 |
| hours | 19 | 0 | 2 | 1.00 | 0.90 |
| queues_max | 6 | 1 | 5 | 0.86 | 0.55 |
| emergency | 16 | 4 | 0 | 0.80 | 1.00 |
| lifted | 3 | 2 | 0 | 0.60 | 1.00 |
| mw | 7 | 3 | 1 | 0.70 | 0.88 |
| restored_any | 11 | 3 | 4 | 0.79 | 0.73 |
| restored_full | 1 | 0 | 4 | 1.00 | 0.20 |
| repairs | 3 | 0 | 6 | 1.00 | 0.33 |
| cons_pct | 6 | 3 | 6 | 0.67 | 0.50 |
| micro average | | | | 0.90 | 0.76 |

The first pass missed most strike reports because it demanded damage word, energy object and a time marker in the same sentence
(digest sentences are split: "Ворог влучив по об'єкту..." / "Є руйнування"). That was relaxed to a three-sentence window, plus event verbs
("атакував", "завдав удару"), "через бойові дії є нові пошкодження" and causal present-tense statements.

### Label consistency decisions made while scoring

Two gold labels were changed after the first scoring because they contradicted other labels of the same kind: a DTEK "last week 27 thousand families
lost power because of attacks ... restored" summary (dtek_ua 757) and a Sumyoblenergo "reason: damage caused by the attack" line under a ГАВ notice
(sumyoblenergo 2823) are now `attack = 0` (weekly totals and "reasons for restrictions" are references, like the analogous posts labelled 0).

## Known failure cases and design limits

* **What is *not* a strike report is partly a judgement call.** Posts that explain current restrictions by earlier damage are marked `attack_ref`,
  not `is_attack_report`. A report with no time marker, no event verb and no "через ... обстріли ... є" pattern is missed (e.g. an interview
  mentioning "ТЕС ... яка була атакована 13 лютого", ukrenergo 2654; a one-line "Енергетична інфраструктура Житомирщини зазнала ворожої атаки",
  ztoe_ua 1818). Conversely a ZNPP/NPP name next to "attack" is sometimes read as a regional strike.
* **Object types are the weakest field (recall 0.70).** Types are taken only from attack sentences; digests that say "енергооб'єкти" or
  "обладнання" give `unspecified`; a list "генерації, системі передачі, системі розподілу" gives generation + distribution but not transmission;
  "мережі обленерго" is distribution only under the digest header "МЕРЕЖІ ОБЛЕНЕРГО" or with the word обленерго.
* **Regions.** Matching is by stems and listed cities, so "львівські енергетики" (workers of Lviv) and NPP names ("Запорізька АЕС") count as mentions;
  the digests of 2025-26 often say "у кількох областях" without names, so the count of regions per strike is a lower bound. `regions_affected` is
  sentence-based plus list continuations ("Зокрема ..."), and ignores weather-only sentences. Kyiv oblast needs "Київщина" or "Київська/Київській
  області" (bare "Київська" is accepted only when not followed by ТЕЦ/район/...).
* **Consumers.** Counts need "number + noun" (споживачів, абонентів, родин, осель, домівок, будинків, мешканців, клієнтів, точок обліку is *not*
  covered when a noun sits between, e.g. "11, 5 тисяч" with a stray space is read as 5 000). Ukrenergo posts of 2025-26 almost never give consumer
  counts (2 posts in 2025, 0 in 2026), DTEK and the oblenergos do. Counts mix causes when a post gives a total including weather. Word numbers
  ("сотні тисяч", "дві тисячі") are not parsed.
* **Schedules.** FAQ and explanation posts that say schedules "діють" are not announcements, but some are still counted (e.g. a Lvivoblenergo
  explainer about queue groups); the reverse also occurs (a post saying "чинні і діють" about yesterday's schedules is treated as past).
  The hours of a schedule are the time ranges in lines that mention restrictions; advice such as "користуйтесь приладами з 10:00 до 16:00" is
  skipped. A partially cancelled queue ("скасовано одну чергу ГПВ") counts as an announcement, not as lifting.
* **Lifting statements** are system-level: "заходи обмеження не застосовуються" does not mean every region is restored. A post that lifts household
  restrictions but keeps industrial ones (ГОП) is *not* a lifting statement. Posts under emergency outages ("графіки не діють") are not lifting.
* **MW.** Only МВт/ГВт/мегават (not МВт·год). The class is the nearest cue before the number in the sentence, else after it, else (only for
  import / export / limit) earlier in the paragraph. Capacity additions and unavailable capacity ("понад 10 ГВт під окупацією") get no class.
  Import is the largest stated value (hourly maximum or the technical limit of 1 700 MW), so it is a ceiling, not the flow.
* **Restoration.** "full" requires an all-quantifier or a statement without any count; counts and "most/part" are "partial". Restoration stated only
  as "ліквідовано аварію" with a typo ("Знесттрумлених") or as "кількість знеструмлених знизилася до 2284" is missed. Statements about generation
  units returning from *planned* repair are excluded from `restoration_repairs`.
* **ETA** (`eta_hours`) is rare (6 posts in 16 000) and needs both a restoration word and an estimate cue (очікується, орієнтовно, прогнозується,
  "буде відновлено"); "протягом декількох годин" has no number and gives nothing. Treat it as an auxiliary field, not as a target.
* **Language and typos.** Ukrainian only; Russian-language or English posts (Ukrenergo's early bilingual posts) are only matched through shared
  abbreviations and numbers.

## What the episode table can and cannot say

* The strike time in `attacks.csv` is a day-level placeholder (06:00), so the episode window starts at 00:00 UTC of the strike day; a restriction
  announced earlier that day by a regime that pre-dates the strike is counted as its onset (`pre_strike_schedule_posts` shows how many restriction
  posts were published in the 3 days before). Chains of strikes within days overlap: several episodes then share the same lifting post
  (`resolved_before_next_strike` marks the stand-alone ones; `lifted_sustained_at` requires 3 quiet days afterwards, which most chain episodes never reach).
* The first "restrictions not applied" post is often a weekend break inside a long regime, hence the sustained variant and the relapse flag.
* The consumer-level track (`restored_at`: first "all consumers have power" post after the first outage post) is region-agnostic: in the pooled
  scope it can be triggered by a different region's restoration. Use per-channel scopes (DTEK, oblenergos) for regional questions.
* Durations are right-censored at 30 days; `status = censored` means "no lifting statement within the window", not "lasted 30 days".
