"""Builds the practical part of the course work as a Word file (Ukrainian), in the layout of the original work.

Run inside the project image with python-docx on PYTHONPATH (see docs/README.md):
  docker run --rm -v "<project dir>:/work" -v "<pkgs dir>:/pkgs" -w /work -e PYTHONPATH=/pkgs:/work/docs \
      --entrypoint python ua-deficit-forecast docs/build_docx.py
The numbers in chapter 3 are read from reports/ and artifacts/, so rebuild after re-running experiments.
"""
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from content_ch12 import ch1, ch2, intro
import content_ch3
from content_ch3 import ch3, conclusions, load_results, sources
from docx_helpers import Builder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "practical_part_raw.docx"


def title_page(b):
    d = b.d

    def line(text="", bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
        p = d.add_paragraph(style="Титул")
        if text:
            r = p.add_run(text)
            r.bold = bold
        p.alignment = align
        return p

    for t in ("МІНІСТЕРСТВО ОСВІТИ І НАУКИ УКРАЇНИ", "НАЦІОНАЛЬНИЙ УНІВЕРСИТЕТ “ЛЬВІВСЬКА ПОЛІТЕХНІКА”",
              "Інститут прикладної математики та фундаментальних наук", "Кафедра прикладної математики"):
        line(t)
    for _ in range(4):
        line()
    line("ЗВІТ", bold=True)
    line("про проходження практики за темою кваліфікаційної роботи")
    line("«Прогнозування глибини та тривалості дефіциту електроенергії після атак на інфраструктуру "
         "на основі методів машинного навчання»", bold=True)
    for _ in range(2):
        line()

    def block(parts, size=14, after=0.12, indent=8.5):
        """One left-aligned line of the signature block; parts = [(text, bold, underline)]."""
        p = d.add_paragraph(style="Титул")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Cm(indent)
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.space_after = Cm(after)
        p.paragraph_format.space_before = Cm(0)
        p.paragraph_format.line_spacing = 1.0
        for text, bold, under in parts:
            r = p.add_run(text)
            r.bold, r.underline = bold, under
            r.font.size = Pt(size)
        return p

    block([("Виконала", True, False), (":", False, False)])
    block([("студентка групи ", False, False), ("ПМІД-11", False, True)])
    block([("Катерина ДАНИЛЬЧЕНКО", False, True)])
    block([("(прізвище та ініціали студента)", False, False)], size=8, after=0.5)
    block([("Керівник роботи:", True, False)])
    block([("Віталій ПОЛЬОВИЙ", False, True)])
    block([("(прізвище та ініціали керівника кваліф. роботи)", False, False)], size=8, after=0.3)
    block([("__________   ____________________", False, False)])
    block([("    (оцінка)              (підпис керівника кваліф. роботи)", False, False)], size=8, after=0.5)
    block([("Керівник практики:", True, False)])
    block([("Владислав АЛЄКСЄЄВ", False, True)])
    block([("(прізвище та ініціали керівника практики)", False, False)], size=8, after=0.3)
    block([("__________   ____________________", False, False)])
    block([("    (дата)                    (підпис керівника практики)", False, False)], size=8)
    for _ in range(2):
        line()
    line("Львів – 2026 р.")


def contents(b):
    d = b.d
    p = d.add_paragraph(style="Титул")
    r = p.add_run("ЗМІСТ")
    r.bold = True
    p.paragraph_format.page_break_before = True
    p.paragraph_format.space_after = Cm(0.85)
    b.toc()


def main():
    R = load_results(ROOT)
    scratch = Builder()                      # first pass only collects table and figure numbers for cross-references
    intro(scratch)
    ch1(scratch, {})
    ch2(scratch, R, {})
    ch3(scratch, R, {})
    X = scratch.keys
    b = Builder()
    title_page(b)
    contents(b)
    intro(b)
    ch1(b, X)
    ch2(b, R, X)
    ch3(b, R, X)
    conclusions(b, R)
    sources(b)
    b.save(OUT)
    print("saved", OUT)
    print("narrative checks failed:", [c for c in content_ch3.CHECKS if c] or "none")


if __name__ == "__main__":
    main()
