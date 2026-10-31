## Oblast-level gold sample: how it was made and what it can tell

* **112 posts** in `telegram_region_gold.csv`, drawn with seed 7 / 11 from the regional channels (oblenergos, regional administrations)
  and the national channels `ukrenergo`, `dtek_ua`:
  `own_with_facts` 22 (own-channel posts where the extractor found something), `own_no_facts` 16 (own-channel posts without any fact, text
  of at least 40 characters), `national_text_facts` 24 and `national_region_no_fact` 14 (national posts that name an oblast, with and
  without an extracted oblast fact), and a second batch drawn after reading the first one to get positives of the rare fields:
  `ext_lifted` 8, `ext_restored_full` 8, `ext_consumers` 8 (posts where the extractor found that field) and `keyword_no_fact` 12 (posts with
  outage / restoration words and no extracted lifting, consumers or restoration, to look for misses).
* Every post was read (up to 2 600 characters, the long digests to their end where it mattered) and labelled **before** the extractor
  output for it was looked at. Labels per oblast: `attack` = the post reports damage or outages caused by a strike or shelling in that oblast
  (new or ongoing; retrospectives, statistics and "consequences of earlier attacks" as the reason of a nationwide regime are 0);
  `consumers` = `Oblast:count` of the largest stated non-weather count of consumers / families / houses cut off in that oblast;
  `schedules` = outage schedules of any kind (including emergency GAV / SGAV, industrial GOP) announced or in force for that oblast;
  `emergency` = emergency outages (GAV, SGAV, "екстрені / аварійні відключення") in force; `lifted` = restrictions are not applied or were
  cancelled for that oblast; `restored_full` = power is back for all consumers of that oblast / outage; `restored_any` = any statement that
  power was returned (partial included; weather outages included). Statements for the whole country carry no oblast and are 0.
* Scores are item-level over (oblast) sets: a post counts for an oblast only when label and extractor agree on that oblast.
* **Blind first pass** (the rules of `telegram_regional.py` as first written, before the gold errors were looked at; 112 posts):

  | field | tp | fp | fn | precision | recall |
  |---|---|---|---|---|---|
  | attack | 48 | 22 | 8 | 0.69 | 0.86 |
  | consumers | 8 | 2 | 5 | 0.80 | 0.62 |
  | schedules | 41 | 2 | 19 | 0.95 | 0.68 |
  | emergency | 17 | 3 | 1 | 0.85 | 0.94 |
  | lifted | 8 | 0 | 0 | 1.00 | 1.00 |
  | restored_full | 6 | 9 | 4 | 0.40 | 0.60 |
  | restored_any | 25 | 8 | 13 | 0.76 | 0.66 |

  The errors showed one structural problem: regional channels (Lvivoblenergo, Sumyoblenergo) repost Ukrenergo digests, and the own-channel
  rule attributed their facts (a full restoration for Kyiv, 380 000 consumers in Kyiv) to the channel's oblast. After that was fixed (a post that
  names another oblast is read sentence by sentence like a national post) the numbers in the current report are slightly better
  (consumers fp 2 -> 1, restored_full fp 9 -> 7, restored_any fp 8 -> 5); nothing else was changed after seeing the gold errors, so the
  remaining errors are listed as found.
* **Main remaining errors**: (1) schedules / consumers phrased without a number or without the oblast next to the cue ("Київ: графіки відключень на ...",
  "понад 335 тисяч киян", "178,7 тисячі сімей" parsed as 178 000); (2) generic nationwide digest sentences that name oblasts but are not new damage
  (attack false positives in `ukrenergo` digests); (3) restoration of a power line (nuclear plant feed) or a water pipeline read as power restored to
  consumers (`restored_full` false positives); (4) day-by-day oblenergo posts ("Командою Укренерго внесено зміни в обсяг ГПВ") that imply schedules
  without saying so.
* **Limits**: 112 posts, one labeller, borderline cases decided by the text alone; the strata are enriched, so the per-field numbers are
  not population precision / recall. On the two nearly unselected strata (`own_no_facts`, `national_region_no_fact`, 30 posts) the extractor found
  nothing and the labels hold 3 attack, 5 schedules and 3 restoration items, i.e. recall on ordinary traffic is low (about 0.0 on 11 items) while
  there were no false positives; the extractor is conservative (precision first).
