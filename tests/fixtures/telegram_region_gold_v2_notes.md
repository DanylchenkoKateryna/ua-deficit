## Oblast-level gold v2 (`telegram_region_gold_v2_dev.csv`, `telegram_region_gold_v2_holdout.csv`)

**Labels were produced automatically, not by a human annotator. Every post was read once and labelled from the text (long posts to
their end where it mattered); the only re-reading was of DEV posts where label and extractor disagreed. There was no second labeller and no
systematic spot-check by a human yet. The person responsible for the thesis must review (at least a sample of) the labels before the numbers are quoted.**

* **Population**: posts of the oblast channels in `config/sources.yaml` (oblenergos and regional military administrations, i.e. every
  channel with an `oblast` key; the national channels `ukrenergo` and `dtek_ua` are not sampled) that exist in
  `data/processed/telegram_posts.parquet`, at least 25 characters long, and that are not in `telegram_region_gold.csv`.
  The texts were read from `data/raw/telegram/*.parquet` (read only; the gold files hold labels only, not texts).
* **Sample** (`stratum` column), fixed seeds, channel allocation proportional to the square root of the channel size so that small
  oblenergos appear:
  * `kw` 148 posts: contain an electricity keyword (`світл|електро|знеструм|відключен|графік|відновлен|енерго`), seed 20261031;
  * `other` 51 posts: no such keyword (about 25 % of the 199 posts of the first draw);
  * `restore_kw` 61 posts: a second draw, added after the first one turned out to hold only 7 power-restoration items, of posts that
    contain a restoration / cancellation word (`відновил|відновлен|повернул|повернут|заживл|скасован|не застосов|не прогноз|знято|…`) and an electricity word,
    seed 777. These raise the share of positives, so precision / recall are not population values.
  * Each stratum was split at random (same seeds) into a **DEV half** (129 posts; may be used to improve the rules) and a **HOLDOUT half**
    (131 posts; evaluated once, at the end, before / after; never used to change a rule).
* **Labelling rules** (same definitions as `telegram_region_gold_notes.md`; items are `;`-separated oblast codes of `telegram_extract.REGIONS`,
  `consumers` as `Oblast:count`). A post of an oblast channel is about the channel's oblast unless the text names another one; the national statements
  without an oblast carry no label:
  * `attack` new strike / damage on energy objects reported for the oblast (ongoing outages caused by attacks included; retrospectives, statistics, reasons of a regime, 0);
  * `consumers` the largest stated non-weather count of consumers / families / houses / abonents without power in that oblast ("понад 90 тисяч споживачів" = 90000; heat, water, gas not counted);
  * `schedules` outage schedules or restrictions of any kind announced or in force for the oblast (GPV, GOP, GAV, "network restrictions"; changes of the volume
    count; **emergency outages in force count as schedules**, so `emergency` implies `schedules` in the labels; FAQ / explainers, service descriptions and "possible if ..." are 0);
  * `emergency` emergency outages (GAV, SGAV, "екстрені / аварійні відключення") in force in the oblast;
  * `lifted` restrictions are not applied / were cancelled for the oblast (nothing announced in the same post);
  * `restored_full` power is back for all affected consumers of the stated outage / unit ("повністю відновили", "всі заживлені", "відновили електропостачання" of a
    stated outage); `restored_any` any statement that power was returned, partial restoration and weather outages included (restoration of critical objects only, "most
    consumers", "back to the schedules" are partial). Water / heat / gas / repair of a line are not power restored.
* **Revisions after seeing extractor output**: one DEV label (`sumyoblenergo/3400`, dev `restore_kw`) got `schedules=Sumy` added to its `emergency=Sumy`,
  because the labelling rule "emergency implies schedules" had been applied everywhere else. No HOLDOUT label was changed after the extractor was first run on it.
  All HOLDOUT posts were read (and labelled) before the rules were changed, which is a mild leak through the labeller's memory but not through the extractor.
* **Scoring**: `telegram_regional.score_region_gold` (item level over oblast sets). Since v2 the scorer counts a predicted `emergency` as a predicted `schedules` item too
  (`emergency_is_schedule=True`, the gold definition above); `emergency_is_schedule=False` reproduces the old scorer. The old gold file (112 posts) was labelled
  with the same convention in 7 of its 9 emergency posts.
* **Limits**: one (artificial) labeller; the DEV and HOLDOUT halves each hold only 3 - 8 positives for some fields (`lifted` 3 / 1, `restored_full` 6 / 5, `consumers`
  7 / 3), so those precision / recall values are anecdotal; strata are enriched; posts that name no electricity are mostly irrelevant (`other`), so the share of positives is small.

### Results (item-level precision / recall; before = extractor and scorer as in the repository before this change, after = current rules and scorer)

Rules were changed using the old gold (112 posts) and the DEV half only; the old gold is therefore no longer held out. The HOLDOUT half was run once, after the last rule change.
`P` precision, `R` recall, `n` gold items. "after*" = current rules scored with the old scorer (`emergency_is_schedule=False`), which isolates the scoring convention from the rules.

| field | old gold before P / R | old gold after P / R | DEV before P / R | DEV after P / R | HOLDOUT before P / R | HOLDOUT after P / R | HOLDOUT after* P / R | n (old / dev / holdout) |
|---|---|---|---|---|---|---|---|---|
| attack | 0.70 / 0.86 | 0.69 / 0.88 | 0.86 / 0.57 | 0.86 / 0.57 | 0.85 / 0.63 | 0.85 / 0.63 | 0.85 / 0.63 | 56 / 42 / 35 |
| consumers | 0.89 / 0.62 | 1.00 / 1.00 | 0.71 / 0.71 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 13 / 7 / 3 |
| schedules | 0.95 / 0.67 | 0.94 / 0.97 | 0.92 / 0.55 | 1.00 / 0.91 | 1.00 / 0.62 | 0.86 / 0.86 | 1.00 / 0.71 | 60 / 22 / 21 |
| emergency | 0.85 / 0.94 | 0.95 / 1.00 | 0.71 / 0.71 | 1.00 / 1.00 | 0.57 / 0.80 | 0.57 / 0.80 | 0.57 / 0.80 | 18 / 7 / 5 |
| lifted | 1.00 / 1.00 | 1.00 / 1.00 | 0.40 / 0.67 | 1.00 / 1.00 | 0.50 / 1.00 | 0.50 / 1.00 | 0.50 / 1.00 | 8 / 3 / 1 |
| restored_full | 0.46 / 0.60 | 0.80 / 0.80 | 0.40 / 0.33 | 1.00 / 1.00 | 0.80 / 0.80 | 0.80 / 0.80 | 0.80 / 0.80 | 10 / 6 / 5 |
| restored_any | 0.83 / 0.66 | 0.89 / 0.89 | 0.88 / 0.50 | 1.00 / 1.00 | 0.75 / 0.60 | 0.82 / 0.90 | 0.82 / 0.90 | 38 / 14 / 10 |

How to read it: the old-gold and DEV "after" columns are fitted (the rules were written to remove exactly those errors) and show only that the rules can express the labels; the
HOLDOUT columns are the honest ones. On HOLDOUT the gains are in `schedules` recall (0.62 -> 0.71 with the old scorer, 0.86 with `emergency` counted as a schedule, at the
price of 3 false positives that are exactly the 3 false `emergency` items) and `restored_any` recall (0.60 -> 0.90); `restored_full`, `lifted` and `emergency` did not
change on HOLDOUT (1 - 5 gold items each, so a single post moves the value by 0.2 - 1.0), and `emergency` precision stays at 0.57. The HOLDOUT errors were not inspected to
keep it clean.
