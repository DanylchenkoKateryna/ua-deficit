"""Builds the defence study guide (Word, Ukrainian) from docs/guide_content.py.

  docker run --rm -v "<project dir>:/work" -v "<pkgs dir>:/pkgs" -w /work -e PYTHONPATH=/pkgs:/work/docs \
      --entrypoint python ua-deficit-forecast docs/build_guide.py
Then docs/update_fields.ps1 refreshes the table of contents in Microsoft Word.
"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from guide_content import build, load_all

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guide_raw.docx"
NAVY = RGBColor(0x14, 0x21, 0x3D)
FONT = "Calibri"

D = load_all(ROOT)
C = build(D)
d = Document()
sec = d.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2)


def fonts(style, size, bold=False, color=None):
    style.font.name, style.font.size, style.font.bold = FONT, Pt(size), bold
    if color:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), FONT)


fonts(d.styles["Normal"], 11)
d.styles["Normal"].paragraph_format.space_after = Pt(6)
d.styles["Normal"].paragraph_format.line_spacing = 1.15
fonts(d.styles["Heading 1"], 20, True, NAVY)
fonts(d.styles["Heading 2"], 14, True, NAVY)
fonts(d.styles["Heading 3"], 12, True, NAVY)
d.styles["Heading 1"].paragraph_format.space_before = Pt(18)
d.styles["Heading 1"].paragraph_format.page_break_before = True
d.styles["Heading 2"].paragraph_format.space_before = Pt(14)
fonts(d.styles["List Bullet"], 11)

# footer page number
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER


def field(par, instr, placeholder="1"):
    for kind in ("begin", None, "separate", "txt", "end"):
        r = par.add_run()
        if kind == "txt":
            r.text = placeholder
            continue
        if kind is None:
            t = OxmlElement("w:instrText")
            t.set(qn("xml:space"), "preserve")
            t.text = f" {instr} "
            r._r.append(t)
            continue
        c = OxmlElement("w:fldChar")
        c.set(qn("w:fldCharType"), kind)
        r._r.append(c)


field(fp, "PAGE")


def para(text, bold=False, italic=False, indent=0.0, style=None, keep=False, color=None):
    p = d.add_paragraph(style=style)
    for i, seg in enumerate(text.split("**")):
        if seg:
            r = p.add_run(seg)
            r.bold = bold or (i % 2 == 1)
            r.italic = italic
            if color:
                r.font.color.rgb = color
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    p.paragraph_format.keep_with_next = keep
    return p


def bullets(items):
    for it in items:
        para(it, style="List Bullet")


def table(headers, rows, widths):
    t = d.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.width = Cm(widths[i])
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        sh = OxmlElement("w:shd")
        sh.set(qn("w:val"), "clear")
        sh.set(qn("w:color"), "auto")
        sh.set(qn("w:fill"), "E3E8F0")
        c._tc.get_or_add_tcPr().append(sh)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].width = Cm(widths[i])
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            for j, seg in enumerate(str(v).split("**")):
                if seg:
                    r = p.add_run(seg)
                    r.bold = j % 2 == 1
    d.add_paragraph()
    return t


# ---------------------------------------------------------------- title + how to use
tp = d.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_before = Pt(120)
r = tp.add_run("Підготовка до захисту")
r.bold, r.font.size, r.font.color.rgb = True, Pt(32), NAVY
tp2 = d.add_paragraph()
tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp2.add_run("Що було зроблено, що це означає і як відповісти на запитання")
r.font.size = Pt(16)
tp3 = d.add_paragraph()
tp3.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp3.add_run("Практична частина курсової роботи · Данильченко К.С., гр. ПМІД-11")
r.font.size, r.font.color.rgb = Pt(12), RGBColor(0x5B, 0x64, 0x75)
d.add_paragraph()
para("Зміст", bold=True)
tocp = d.add_paragraph()
field(tocp, 'TOC \\o "1-2" \\h \\z \\u', "Щоб оновити зміст, натисніть правою кнопкою миші й виберіть «Оновити поле».")

d.add_heading("Як користуватися цим файлом", level=1)
bullets([
    "Спершу прочитайте розділ «Історія роботи»: це 30 секунд і 3 хвилини, які треба вміти розповісти без паперу.",
    "Потім «Крок за кроком»: для кожного кроку — що зроблено, навіщо, простими словами, що означає результат і які запитання можуть поставити.",
    "«Числа, які треба знати» і «Глосарій» — для швидкого повторення перед захистом.",
    f"«Запитання і відповіді» — {len(C['qa'])} запитань із готовими відповідями, зокрема про прогноз глибини й тривалості, перенавчання та слабкі місця. Не вчіть дослівно: зрозумійте логіку й говоріть своїми словами.",
    "Презентація має нотатки до кожного слайда (режим доповідача): що казати й які запитання можливі саме тут.",
    "Усі числа тут взято з файлів результатів проєкту; якщо ви перезапустите експерименти, перебудуйте файл (docs/README.md)."])
para("**Головне правило захисту:** не стверджуйте більше, ніж показують дані. Ключовий результат — прямий прогноз глибини дефіциту після атаки значуще точніший за «як учора» (ціна, імпорт); слабкі місця (тривалість, відключення) подавайте як напрями вдосконалення й чесно кажіть, що потрібно для їх розв'язання.")

# ---------------------------------------------------------------- pitches
d.add_heading("Історія роботи: 30 секунд і 3 хвилини", level=1)
d.add_heading("Версія на 30 секунд", level=2)
para(C["pitch30"])
d.add_heading("Версія на 3 хвилини", level=2)
bullets(C["pitch3"])

# ---------------------------------------------------------------- map
d.add_heading("Карта проєкту: що де лежить", level=1)
para("Проєкт знаходиться в папці C:\\Projects\\ua-deficit-forecast. Усе запускається в Docker.")
table(["Що", "Де", "Навіщо"], C["map"], [3.6, 6.6, 6.4])

# ---------------------------------------------------------------- steps
d.add_heading("Крок за кроком: що було зроблено", level=1)
para("Кожен крок пояснено однаково: що зроблено → навіщо → простими словами → що це означає → які запитання можливі.", italic=True)
for s in C["steps"]:
    d.add_heading(s["t"], level=2)
    para("**Що зроблено.** " + s["what"])
    para("**Навіщо.** " + s["why"])
    para("**Простими словами.** " + s["simple"])
    para("**Що це означає.** " + s["means"])
    para("**Запитання, які можуть поставити:** " + "   •   ".join(s["q"]), keep=False)

# ---------------------------------------------------------------- how to read figures
d.add_heading("Як читати графіки", level=1)
bullets(C["figs"])

# ---------------------------------------------------------------- numbers
d.add_heading("Числа, які треба знати", level=1)
table(["Показник", "Значення"], C["numbers"], [9.0, 7.6])

# ---------------------------------------------------------------- glossary
d.add_heading("Глосарій", level=1)
table(["Термін", "Пояснення"], C["glossary"], [4.2, 12.4])

# ---------------------------------------------------------------- Q&A
d.add_heading("Запитання і відповіді", level=1)
para("Відповіді можна скорочувати. Якщо питання складніше за відповідь — спочатку коротко відповідайте по суті, а деталі додавайте, лише якщо просять.", italic=True)
last = None
n = 0
for group, qn_, ans in C["qa"]:
    if group != last:
        d.add_heading(group, level=2)
        last = group
    n += 1
    para(f"{n}. {qn_}", bold=True, keep=True)
    para(ans, indent=0.6)

# ---------------------------------------------------------------- demo
d.add_heading("Демонстрація: команди", level=1)
para("Docker Desktop має бути запущений. Команди виконуються в PowerShell із папки проєкту (C:\\Projects\\ua-deficit-forecast).")
table(["Що", "Команда"], C["demo"], [5.0, 11.6])

d.add_heading("Чого не варто стверджувати", level=1)
bullets(C["dont"])
d.add_heading("Що відповідати, коли не знаєш", level=1)
bullets(C["unknown"])
d.add_heading("Чекліст перед захистом", level=1)
bullets(C["checklist"])

d.save(str(OUT))
print("saved", OUT)
