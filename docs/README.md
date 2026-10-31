# docs/ - practical part of the course work (Word)

`Практична_частина_курсової_роботи.docx` is the practical part in Ukrainian, in the layout of the original course work
(A4, Times New Roman 14 pt, 1.5 spacing). The topic on the title page is the topic given by the student:
"Прогнозування глибини та тривалості дефіциту електроенергії після атак на інфраструктуру на основі методів машинного навчання".

It is generated, so numbers always match the result files. Chapter 3 leads with the post-attack depth/duration forecast
(`reports/<sc>/post_event_*.csv`), then retraining policies (day-ahead forecasts), break detection, a hypotheses table and the
"Напрями вдосконалення" subsection for the results that did not beat the baselines (with their real numbers).

| File | Role |
|---|---|
| `make_figures.py` | charts (Ukrainian labels) from `artifacts/`, `reports/`, `data/processed/` -> `figures/*.png` (`FIG_MODE=slide` -> `s_*.png`) |
| `res_helpers.py` | small readers shared by figures and text: empty-CSV-safe loader, stars, paired Wilcoxon of every model against `xgb_rolling` (computed from the forecast files; `wilcoxon.csv` of the project has `xgb_adaptive` as reference) |
| `content_ch12.py`, `content_ch3.py` | text of the introduction, chapters and conclusions; chapter 3 reads its numbers from `reports/` and `artifacts/`; `content_ch3.CHECKS` lists narrative claims the data do not support (printed by `build_docx.py`) |
| `docx_helpers.py`, `build_docx.py` | page layout, tables, figures, table of contents, cross-reference numbers (two passes); writes `practical_part_raw.docx` |
| `finalize_docx.ps1` | Microsoft Word: updates the table of contents and page numbers, saves `practical_part.docx` and a PDF |
| `render_pages.ps1` | Microsoft Word: renders every page to PNG in `qa/` for a visual check |

New figures: `fig_post_depth.png` (depth MAE after a strike, test), `fig_retrain.png` (retraining triggers relative to `xgb_rolling`),
`fig_fc_price.png`, `fig_detect_price.png`; `fig_coverage.png` and `fig_mae.png` cover the strike list, the day-ahead market and four scenarios.

Rebuild after re-running experiments (from the project folder; python-docx is installed into a temporary folder, not into the image):

```bash
docker run --rm -v "%CD%:/work" -w /work --entrypoint python ua-deficit-forecast docs/make_figures.py
docker run --rm -e FIG_MODE=slide -v "%CD%:/work" -w /work --entrypoint python ua-deficit-forecast docs/make_figures.py
docker run --rm -v "<temp>:/pkgs" --entrypoint pip ua-deficit-forecast install --target /pkgs python-docx
docker run --rm -v "%CD%:/work" -v "<temp>:/pkgs" -w /work -e PYTHONPATH=/pkgs:/work/docs --entrypoint python ua-deficit-forecast docs/build_docx.py
powershell -File docs/finalize_docx.ps1
```

Then rename `practical_part.docx` as needed.

The reference list has no exact URLs for the two Energy Map datasets added in October 2026 (strike list, DAM results): only the portal is given, add the resource links by hand.

## Presentation and study guide

| File | Role |
|---|---|
| `Презентація_практична_частина.pptx` | slides with speaker notes; built by `build_pptx.py` (slide figures: `FIG_MODE=slide make_figures.py`) |
| `Підготовка_до_захисту.docx` | study guide: pitches, step by step, numbers, glossary, Q&A; built by `guide_content.py` + `build_guide.py`, then `update_fields.ps1` |
| `export_slides.ps1` | PowerPoint: renders slides to PNG for checking |
