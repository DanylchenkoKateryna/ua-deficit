"""Formatting helpers that reproduce the layout of the original course work (A4, Times New Roman 14 pt,
1.5 line spacing, 30 mm left margin, bold centred chapter titles, page number at the bottom centre)."""
from __future__ import annotations

from docx import Document
from docx.enum.section import WD_SECTION  # noqa: F401
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
TEXT_WIDTH_CM = 16.5
STAR = ""      # placeholder for a literal asterisk in table text (a plain "*" would collide with the **bold** markup)


def _star(t: str) -> str:
    return t.replace(STAR, "*")


def fm(x, d=1, sign=False) -> str:
    """Number with a decimal comma (Ukrainian typography)."""
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def _set_fonts(rpr_owner, size=None, bold=None):
    rpr = rpr_owner.element.get_or_add_rPr() if hasattr(rpr_owner, "element") else rpr_owner
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), FONT)


def _style(doc, name, size=14, bold=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=1.25, line=1.5,
           before=0, after=0, keep_next=False, page_break=False, base=None, italic=False):
    st = doc.styles[name] if name in [s.name for s in doc.styles] else doc.styles.add_style(name, 1)
    if base:
        st.base_style = doc.styles[base]
    f = st.font
    f.name, f.size, f.bold, f.italic, f.color.rgb = FONT, Pt(size), bold, italic, RGBColor(0, 0, 0)
    _set_fonts(st, size, bold)
    pf = st.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Cm(first_indent) if first_indent else Cm(0)
    pf.left_indent = Cm(0)
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    if line == "single":
        pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
    else:
        pf.line_spacing = line
    pf.keep_with_next = keep_next
    pf.page_break_before = page_break
    pf.widow_control = True
    return st


def new_document() -> Document:
    d = Document()
    s = d.sections[0]
    s.page_width, s.page_height = Cm(21), Cm(29.7)
    s.left_margin, s.right_margin, s.top_margin, s.bottom_margin = Cm(3), Cm(1.5), Cm(2), Cm(2)
    s.footer_distance = Cm(1.25)
    s.different_first_page_header_footer = True
    _style(d, "Normal")
    _style(d, "Heading 1", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=0, after=24, keep_next=True, page_break=True)
    _style(d, "Heading 2", bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=0, before=24, after=24, keep_next=True)
    _style(d, "Підпис таблиці", align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=0, keep_next=True, line="single", before=12, after=6)
    _style(d, "Підпис рисунка", align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=0, line="single", before=6, after=18)
    _style(d, "Текст таблиці", size=12, align=WD_ALIGN_PARAGRAPH.LEFT, first_indent=0, line="single")
    _style(d, "Рисунок", align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=0, line="single", keep_next=True, before=6)
    _style(d, "Титул", align=WD_ALIGN_PARAGRAPH.CENTER, first_indent=0, line=1.5)
    # headings must not use theme colours / fonts
    for n in ("Heading 1", "Heading 2"):
        d.styles[n].font.color.rgb = RGBColor(0, 0, 0)
    # page number in the footer (not on the first page)
    p = s.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    _field(p, "PAGE")
    return d


def _field(paragraph, instr: str, placeholder: str = "1"):
    def fld(kind):
        r = paragraph.add_run()
        c = OxmlElement("w:fldChar")
        c.set(qn("w:fldCharType"), kind)
        r._r.append(c)
        return r
    fld("begin")
    r = paragraph.add_run()
    t = OxmlElement("w:instrText")
    t.set(qn("xml:space"), "preserve")
    t.text = f" {instr} "
    r._r.append(t)
    fld("separate")
    paragraph.add_run(placeholder)
    fld("end")


class Builder:
    def __init__(self):
        self.d = new_document()
        self.chapter = 0
        self.tab = 0
        self.fig = 0
        self.keys = {}            # key -> 'chapter.number' of tables and figures, for cross-references

    # ---- text -------------------------------------------------------------
    def para(self, text="", bold=False, italic=False, align=None, indent=True, keep_next=False):
        p = self.d.add_paragraph(style="Normal")
        if text:
            self._runs(p, text, bold, italic)
        if align is not None:
            p.alignment = align
        if not indent:
            p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.keep_with_next = keep_next
        return p

    def _runs(self, p, text, bold=False, italic=False):
        """Text with **bold** segments."""
        parts = text.split("**")
        for i, seg in enumerate(parts):
            if seg:
                r = p.add_run(_star(seg))
                r.bold = bold or (i % 2 == 1)
                r.italic = italic

    def items(self, items, dash=True):
        for it in items:
            self.para(("– " if dash else "") + it)

    def numbered(self, items):
        for i, it in enumerate(items, 1):
            self.para(f"{i}. {it}")

    def h1(self, text, chapter=False):
        if chapter:
            self.chapter += 1
            self.tab = self.fig = 0
        return self.d.add_paragraph(text, style="Heading 1")

    def h2(self, text):
        return self.d.add_paragraph(text, style="Heading 2")

    def blank(self):
        return self.para("", indent=False)

    # ---- tables -----------------------------------------------------------
    def table(self, caption, headers, rows, widths_cm, align=None, bold_first_col=False, bold_rows=(), keep=True, key=None):
        self.tab += 1
        if key:
            self.keys[key] = f"{self.chapter}.{self.tab}"
        cap = self.d.add_paragraph(_star(f"Таблиця {self.chapter}.{self.tab} – {caption}"), style="Підпис таблиці")
        cap.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        self._borders(t)
        align = align or ["l"] * len(headers)
        for i, h in enumerate(headers):
            self._cell(t.rows[0].cells[i], h, widths_cm[i], bold=True, align="c", shade=True)
        self._repeat_header(t.rows[0])
        for ri, row in enumerate(rows):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                self._cell(cells[i], str(v), widths_cm[i], bold=(bold_first_col and i == 0) or ri in bold_rows, align=align[i])
            self._no_split(t.rows[-1])
        if keep:   # keep a small table on one page: every row except the last stays with the next
            for row in t.rows[:-1]:
                for c in row.cells:
                    for p in c.paragraphs:
                        p.paragraph_format.keep_with_next = True
        sp = self.d.add_paragraph(style="Normal")        # spacing after the table
        sp.paragraph_format.first_line_indent = Cm(0)
        sp.paragraph_format.space_after = Pt(0)
        return t

    def _cell(self, cell, text, width_cm, bold=False, align="l", shade=False):
        cell.width = Cm(width_cm)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.style = self.d.styles["Текст таблиці"]
        p.alignment = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}[align]
        for i, seg in enumerate(text.split("**")):
            if seg:
                r = p.add_run(_star(seg))
                r.bold = bold or (i % 2 == 1)
        tcpr = cell._tc.get_or_add_tcPr()
        mar = OxmlElement("w:tcMar")
        for side, w in (("top", 40), ("bottom", 40), ("left", 80), ("right", 80)):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:w"), str(w))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        tcpr.append(mar)
        if shade:
            sh = OxmlElement("w:shd")
            sh.set(qn("w:val"), "clear")
            sh.set(qn("w:color"), "auto")
            sh.set(qn("w:fill"), "EDEDED")
            tcpr.append(sh)

    def _borders(self, table):
        tblPr = table._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{side}")
            e.set(qn("w:val"), "single")
            e.set(qn("w:sz"), "4")
            e.set(qn("w:space"), "0")
            e.set(qn("w:color"), "000000")
            b.append(e)
        tblPr.append(b)

    def _repeat_header(self, row):
        trpr = row._tr.get_or_add_trPr()
        e = OxmlElement("w:tblHeader")
        e.set(qn("w:val"), "true")
        trpr.append(e)

    def _no_split(self, row):
        trpr = row._tr.get_or_add_trPr()
        e = OxmlElement("w:cantSplit")
        e.set(qn("w:val"), "true")
        trpr.append(e)

    # ---- figures ----------------------------------------------------------
    def figure(self, path, caption, width_cm=16.0, key=None):
        self.fig += 1
        if key:
            self.keys[key] = f"{self.chapter}.{self.fig}"
        p = self.d.add_paragraph(style="Рисунок")
        p.add_run().add_picture(str(path), width=Cm(width_cm))
        self.d.add_paragraph(f"Рис. {self.chapter}.{self.fig}. {caption}", style="Підпис рисунка")

    # ---- table of contents -----------------------------------------------
    def toc(self):
        p = self.d.add_paragraph(style="Normal")
        p.paragraph_format.first_line_indent = Cm(0)
        _field(p, 'TOC \\o "1-2" \\h \\z \\u', "Щоб оновити зміст, натисніть правою кнопкою миші та виберіть «Оновити поле».")

    def save(self, path):
        self.d.save(str(path))
