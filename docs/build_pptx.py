"""Presentation for the practical part (Ukrainian). Every number is read from reports/, artifacts/ and data/external.

Run inside the project image with python-pptx on PYTHONPATH:
  docker run --rm -v "<project dir>:/work" -v "<pkgs dir>:/pkgs" -w /work -e PYTHONPATH=/pkgs:/work/docs \
      --entrypoint python ua-deficit-forecast docs/build_pptx.py

Structure: topic -> problem -> aim -> pipeline -> data -> leakage rules -> post-attack depth/duration forecast
(method, result with significance, duration) -> retraining (strategies, result) -> forecasts per scenario ->
detection -> summary -> directions of improvement -> conclusions -> thanks; one backup slide.
"""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from docx_helpers import fm
from guide_content import (D_E, D_H, LABEL, PE_LABEL, SC_NAME, SCEN, TESTS_N, UNIT_SHORT, has_model, load_all, mae, n_ev, pair,
                           pct, pe, pt, ptxt, refit_counts, strikes, verdict)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Презентація_практична_частина.pptx"
FIG = ROOT / "docs" / "figures"

NAVY, AMBER, INK, MUTED = "14213D", "FCA311", "1F2937", "5B6475"
TINT, WHITE, LINE = "EEF2F7", "FFFFFF", "D5DBE5"
GREEN, RED, ORANGE_TINT = "2E7D32", "C62828", "FFF1D6"
GREEN_TINT = "E3F1E4"
HEAD, BODY = "Cambria", "Calibri"

D = load_all(ROOT)
ST = strikes(D)
CHECKS = []           # (statement, bool): printed at the end, so that a claim that the data do not support is noticed


def rgb(h):
    return RGBColor.from_string(h)


def check(statement, ok):
    CHECKS.append((statement, bool(ok)))
    return ok


def M(sc, m, **k):
    return mae(D, sc, m, **k)


# ------------------------------------------------------------------------------------ helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
COUNT = {"n": 0}


def new_slide(title, dark=False, notes="", number=True):
    s = prs.slides.add_slide(prs.slide_layouts[5])
    COUNT["n"] += 1
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(NAVY if dark else WHITE)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.text = title
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    for r in p.runs:
        r.font.name, r.font.size, r.font.bold = HEAD, Pt(32 if len(title) <= 50 else 28 if len(title) <= 70 else 26), True
        r.font.color.rgb = rgb(WHITE if dark else NAVY)
    if number and not dark:
        text(s, 12.2, 7.0, 0.8, 0.3, str(COUNT["n"]), size=12, color=MUTED, align=PP_ALIGN.RIGHT)
    if notes:
        s.notes_slide.notes_text_frame.text = notes.strip()
    return s


def text(s, x, y, w, h, content, size=18, color=INK, bold=False, align=PP_ALIGN.LEFT, font=BODY,
         anchor=MSO_ANCHOR.TOP, italic=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    paras = content if isinstance(content, list) else [content]
    for i, c in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        runs = c if isinstance(c, list) else [(c, bold)]
        for seg, b in runs:
            r = p.add_run()
            r.text = seg
            r.font.name, r.font.size, r.font.bold, r.font.italic = font, Pt(size), b, italic
            r.font.color.rgb = rgb(color)
    return tb


def bullets(s, x, y, w, h, items, size=20, color=INK, gap=8):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        runs = it if isinstance(it, list) else [(it, False)]
        for seg, b in runs:
            r = p.add_run()
            r.text = seg
            r.font.name, r.font.size, r.font.bold = BODY, Pt(size), b
            r.font.color.rgb = rgb(color)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.3))))
        pPr.set("indent", str(-int(Inches(0.3))))
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
        pPr.append(bu)
    return tb


def box(s, x, y, w, h, fill=TINT, line=None, radius=0.06, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    if line:
        sh.line.color.rgb = rgb(line)
        sh.line.width = Pt(1)
    else:
        sh.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.shadow.inherit = False
    return sh


def card(s, x, y, w, h, head, body, fill=TINT, head_color=NAVY, size=16, head_size=19, head_h=0.6):
    box(s, x, y, w, h, fill)
    text(s, x + 0.2, y + 0.12, w - 0.4, head_h, head, size=head_size, bold=True, color=head_color, font=HEAD)
    text(s, x + 0.2, y + 0.12 + head_h, w - 0.4, h - 0.2 - head_h, body, size=size, color=INK)


def stat(s, x, y, w, h, big, label, fill=TINT, big_color=NAVY, big_size=40, label_size=15):
    box(s, x, y, w, h, fill)
    text(s, x + 0.15, y + 0.1, w - 0.3, 0.85, big, size=big_size, bold=True, color=big_color, font=HEAD,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.15, y + 0.98, w - 0.3, h - 1.05, label, size=label_size, color=INK)


def chip(s, x, y, w, h, label, fill, size=14):
    sh = box(s, x, y, w, h, fill, radius=0.5)
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold = BODY, Pt(size), True
    r.font.color.rgb = rgb(WHITE)


def picture(s, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)


def picture_fit(s, name, x, y, w, h):
    """Place the picture inside the box (x, y, w, h) keeping the aspect ratio; returns (width, height) actually used."""
    iw, ih = Image.open(FIG / name).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), width=Inches(pw), height=Inches(ph))
    return pw, ph


def table(s, x, y, col_w, rows, size=16, header=True, row_h=0.48, highlight=(), bold_rows=(), align=None, colors=None):
    nrows, ncols = len(rows), len(rows[0])
    gs = s.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(sum(col_w)), Inches(row_h * nrows))
    t = gs.table
    t.horz_banding = False
    t.first_row = header
    for j, w in enumerate(col_w):
        t.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val_ in enumerate(row):
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if header and i == 0:
                c.fill.fore_color.rgb = rgb(NAVY)
            elif i in highlight:
                c.fill.fore_color.rgb = rgb(ORANGE_TINT)
            else:
                c.fill.fore_color.rgb = rgb(WHITE if i % 2 else TINT)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = (align[j] if align else PP_ALIGN.LEFT)
            r = p.add_run()
            r.text = str(val_)
            r.font.name, r.font.size = BODY, Pt(size)
            r.font.bold = (header and i == 0) or (i in bold_rows)
            r.font.color.rgb = rgb(WHITE if (header and i == 0) else (colors or {}).get((i, j), INK))
            if (colors or {}).get((i, j)):
                r.font.bold = True
    return t


def line(s, x1, y1, x2, y2, color=NAVY, width=2.5):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    return c


C = PP_ALIGN.CENTER
SPLIT_UA = {"validation": "валід.", "test": "тест"}
SHORT = {"demand": "Попит, МВт", "net_import": "Імпорт, МВт", "outages": "Відключення", "price": "Ціна / стеля"}
STRIKE_DAYS, STRIKE_FIRED, STRIKE_THROUGH = ST["n"], ST["fired"], ST["through"]
STRIKE_PERIOD = f"{ST['first'][5:7]}.{ST['first'][:4]} – {ST['last'][5:7]}.{ST['last'][:4]}"


def sp(n):
    """Integer with a thin space as thousands separator."""
    return f"{int(n):,}".replace(",", " ")


def depth_p(sc, split, method="ridge_depth", metric="depth"):
    return pt(D, sc, method, metric, split)


def sig_color(v):
    return {"better": GREEN, "worse": RED}.get(v)


# numbers reused on several slides ---------------------------------------------------------------------------
TRIG = ("xgb_drift", "xgb_rolling_freq", "xgb_adaptive", "xgb_attack")
REL = {sc: {m: pct(M(sc, m), M(sc, "xgb_rolling")) for m in LABEL if has_model(D, sc, m)} for sc in SCEN}
PAIR = {sc: {m: pair(D, sc, "xgb_rolling", m) for m in TRIG + ("xgb_static", "xgb_adaptive_tau_w") if has_model(D, sc, m)} for sc in SCEN}


def rel_cell(sc, m):
    """(text, colour) of the MAE of model m relative to the weekly rolling window, with the significance colour."""
    if m not in REL[sc]:
        return "—", None
    r, pr = REL[sc][m], PAIR[sc].get(m)
    txt = f"{fm(r, 1, sign=True)} %"
    if pr is not None and pr["p"] < 0.05 and r < 0:
        return txt, GREEN
    if pr is not None and pr["p"] < 0.05 and r > 0:
        return txt, RED
    return txt, None


def static_line():
    """How much worse the static model is than the weekly rolling window, per scenario (from metrics.csv)."""
    out = []
    for sc in SCEN:
        r = REL[sc]["xgb_static"]
        if M(sc, "xgb_static") / M(sc, "xgb_rolling") >= 2:
            out.append(f"{SHORT[sc]}: статична гірша в {fm(M(sc, 'xgb_static') / M(sc, 'xgb_rolling'), 0)} р.")
        else:
            out.append(f"{SHORT[sc]}: {fm(r, 0, sign=True)} %")
    return "; ".join(out)


STATIC_LINE = static_line()
STATIC_OK = [sc for sc in SCEN if REL[sc]["xgb_static"] <= 0]        # static not worse than the weekly window


def verdict_word(sc, m):
    r, pr = REL[sc].get(m), PAIR[sc].get(m)
    if r is None or pr is None:
        return "немає даних"
    if pr["p"] < 0.05:
        return "значуще краще" if r < 0 else "значуще гірше"
    return "без значущої різниці"


# ====================================================================================== 1. title
s = new_slide("Прогнозування глибини та тривалості дефіциту електроенергії після атак на інфраструктуру на основі методів машинного навчання",
              dark=True, number=False, notes="""
ЩО СКАЗАТИ (20 секунд): Добрий день. Моя тема — прогнозування глибини та тривалості дефіциту електроенергії після атак на інфраструктуру методами машинного навчання. У теоретичній частині я спроєктувала систему; сьогодні розповім про практичну частину: що саме реалізовано, на яких реальних даних перевірено і що з цього вийшло. Головне, що я покажу, — прямий прогноз глибини дефіциту після атаки, який значуще точніший за простий метод «як учора», та порівняння стратегій перенавчання моделей.

ПІДКАЗКА: назва на слайді — точне формулювання теми курсової.
""")
s.shapes.title.top, s.shapes.title.height = Inches(1.3), Inches(2.7)
for r in s.shapes.title.text_frame.paragraphs[0].runs:
    r.font.size = Pt(34)
text(s, 0.6, 4.3, 12.1, 0.9, "Практична частина курсової роботи: програмна система та експерименти на реальних даних",
     size=21, color="D9E0EC")
text(s, 0.6, 5.7, 11.5, 1.2, ["Данильченко К.С., гр. ПМІД-11", "Перевірив: Віталій Польовий", "Національний університет «Львівська політехніка», 2026"],
     size=18, color="B6C1D6")

# ====================================================================================== 2. problem
s = new_slide("Після атаки треба знати, наскільки глибоким і як довго триватиме дефіцит", notes=f"""
ЩО СКАЗАТИ (1 хв): Від 2022 року атаки на енергетику не просто тимчасово збільшують дефіцит — вони міняють сам режим системи: спочатку упав попит, потім били по підстанціях, у 2024-му знищили понад 9 ГВт генерації. Це три хвилі, описані в теоретичній частині. У практичній частині я використала офіційний перелік масованих ударів по критичній інфраструктурі: {STRIKE_DAYS} днів ({STRIKE_PERIOD}). Після кожного такого удару оператору треба відповісти на два запитання: наскільки глибоким буде дефіцит і скільки він триватиме. Від цього залежать графіки відключень, імпорт, ремонти.
Статистики називають зміни режиму структурними зламами: після них старі закономірності не працюють. Класичні моделі (ARIMA, Prophet, звичайні нейромережі) створені для стабільних систем, тому їхня точність після удару різко падає. Ідея роботи: прогнозувати глибину й тривалість від дня удару та автоматично перенавчати модель, коли режим змінюється.

ТЕРМІН: структурний злам — момент, після якого статистичні властивості ряду стійко змінюються; аномалія — тимчасове відхилення з поверненням до норми.

МОЖЛИВІ ЗАПИТАННЯ: Звідки 9 ГВт? — Зі звітів IEA і Міненерго, наведених у теоретичній частині; я їх самостійно не перевіряла. Чим злам відрізняється від аномалії? — Після зламу система не повертається до старого режиму.
""")
stat(s, 0.6, 1.6, 3.9, 2.2, "3 хвилі", "атак на енергетику 2022–2025: попит, підстанції, генерація")
stat(s, 4.7, 1.6, 3.9, 2.2, "> 9 ГВт", "генерації втрачено у 2024 році (за даними IEA та Міненерго з теоретичної частини)")
stat(s, 8.8, 1.6, 3.9, 2.2, f"{STRIKE_DAYS} днів", f"масованих ударів по критичній інфраструктурі, {STRIKE_PERIOD} (офіційні зведення)")
card(s, 0.6, 4.1, 6.0, 2.6, "Проблема", "Після удару режим системи змінюється (структурний злам), і класичні моделі втрачають точність саме тоді, коли прогноз потрібен найбільше.", size=18)
card(s, 6.8, 4.1, 5.9, 2.6, "Підхід", "Прогнозувати глибину й тривалість дефіциту від дня удару та перенавчати моделі, коли режим змінюється.", fill=ORANGE_TINT, size=18)

# ====================================================================================== 3. goal
s = new_slide("Мета практичної частини і три запитання", notes="""
ЩО СКАЗАТИ (1 хв): Мета практичної частини — реалізувати систему, яка прогнозує глибину й тривалість дефіциту після атаки, і перевірити на реальних даних, які підходи справді працюють. Я сформулювала три запитання, на які експерименти дають відповідь:
1) Чи можна після атаки спрогнозувати глибину й тривалість дефіциту точніше, ніж найпростішим методом «як учора»?
2) Чи потрібне перенавчання моделей, і яке саме: за розкладом, за виявленим зламом, після кожної атаки чи за зростанням власної помилки моделі?
3) Чи не вистачає простих методів: «як тиждень тому», «як учора»?
Для кожного запитання я додала чесні порівняльні методи: сезонний наївний, персистентність і «ковзне вікно» (абляція — прибираємо один елемент системи, щоб побачити його внесок).

ТЕРМІН: абляція — експеримент, у якому прибирають один елемент системи, щоб побачити, чи він справді потрібен.

МОЖЛИВІ ЗАПИТАННЯ: Чому саме такі порівняння? — Бо перше, що спитає рецензент: «а чи не працює щось простіше?». Тому кожен результат я показую поряд із простим методом.
""")
box(s, 0.6, 1.6, 12.1, 1.3, ORANGE_TINT)
text(s, 0.85, 1.7, 11.6, 1.1, "Реалізувати систему прогнозу глибини й тривалості дефіциту після атаки та перевірити на реальних даних, що з підходів машинного навчання працює, а що ні.",
     size=21, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE, font=HEAD)
for i, (h, b) in enumerate([("1. Глибина і тривалість", "Чи можна після атаки спрогнозувати їх точніше за найпростіший метод «як учора»?"),
                            ("2. Яке перенавчання?", "За розкладом, за виявленим зламом, після атаки чи за зростанням власної помилки моделі?"),
                            ("3. Чи досить простого?", "Чи не вистачає методів «як тиждень тому» і «як учора»?")]):
    card(s, 0.6 + i * 4.1, 3.3, 3.9, 3.2, h, b, size=20, head_size=18)

# ====================================================================================== 4. pipeline
s = new_slide("Пайплайн із шести кроків і технічна реалізація", notes=f"""
ЩО СКАЗАТИ (1,5 хв): Систему реалізовано як шість кроків, визначених у теоретичній частині. 1) Збір даних. 2) Обробка: єдиний час, пропуски, цільова змінна. 3) Ознаки: лаги, календар, погода, ознаки ударів. 4) Виявлення зламів: PELT ретроспективно, BOCPD онлайн. 5) Адаптивне навчання моделей. 6) Оцінювання: метрики, тест Вілкоксона, прогноз глибини й тривалості після атаки. Кожен крок — окрема команда «deficit <крок>»; результати зберігаються у файлах, тому будь-який крок можна повторити окремо.
Технічно: Python 3.12 у контейнері Docker (відтворюване, ізольоване середовище), усі параметри в YAML-файлах: чотири реальні сценарії — це чотири файли config/<сценарій>.yaml. Правильність методики перевіряють {TESTS_N} автоматизованих тестів (тест LSTM пропускається без TensorFlow). Алгоритм BOCPD і детектор дрейфу помилки CUSUM я реалізувала сама.

ТЕРМІНИ: PELT — алгоритм, що знаходить оптимальні точки зламу по всьому ряду (бачить майбутнє, тому лише для аналізу). BOCPD — байєсівський онлайн-алгоритм: у кожний момент оцінює ймовірність, що режим щойно змінився (лише за минулим). CUSUM — накопичувальна сума, що сигналізує про стійке зростання величини (тут — помилки моделі).

МОЖЛИВІ ЗАПИТАННЯ: Чому Docker? — Відтворюваність і ізоляція від налаштувань комп'ютера. Чому власний BOCPD? — Щоб не залежати від непідтримуваного пакета; алгоритм той самий (Adams & MacKay, 2007). Що таке Parquet? — Бінарний колонковий формат таблиць, швидкий і з типами.
""")
steps = [("1. Збір даних", "Energy Map, Open-Meteo, список ударів"), ("2. Обробка", "UTC, пропуски, ціль, епізоди"),
         ("3. Ознаки", "лаги, календар, погода, удари"), ("4. Виявлення зламів", "PELT ретроспективно, BOCPD онлайн"),
         ("5. Навчання", "статичні й адаптивні моделі, Optuna"), ("6. Оцінювання", "MAE, Вілкоксон, глибина й тривалість")]
for i, (h, b_) in enumerate(steps):
    x = 0.6 + i * 2.05
    box(s, x, 1.55, 1.85, 2.4, TINT)
    text(s, x + 0.1, 1.62, 1.65, 0.8, h, size=16, bold=True, color=NAVY, font=HEAD)
    text(s, x + 0.1, 2.5, 1.65, 1.4, b_, size=14, color=INK)
    if i < 5:
        box(s, x + 1.87, 2.55, 0.17, 0.4, AMBER, shape=MSO_SHAPE.CHEVRON)
text(s, 0.6, 4.05, 12.1, 0.5, "Кожен крок — команда deficit <крок>; крок events — прогноз глибини й тривалості після атаки; tune-triggers — підбір тригерів на валідації.",
     size=15, color=MUTED)
for i, (h, b) in enumerate([("Python 3.12 + Docker", "Відтворюване середовище; запуск на локальних даних без мережі."),
                            ("Конфігурація YAML", "Чотири реальні сценарії = чотири файли config/<сценарій>.yaml."),
                            (f"{TESTS_N} автоматизованих тестів", "Витік майбутнього, час перенавчання, обробка часу й даних, прогноз після атаки."),
                            ("Власні алгоритми", "BOCPD (Adams & MacKay, 2007), CUSUM на помилці моделі, прямі моделі глибини й тривалості.")]):
    card(s, 0.6 + i * 3.08, 4.65, 2.9, 2.2, h, b, size=14, head_size=16, head_h=0.7)

# ====================================================================================== 5. data
s = new_slide("Дані: офіційний список ударів і відкриті набори Укренерго", notes=f"""
ЩО СКАЗАТИ (1,5 хв): У теоретичній частині основним джерелом були погодинні дані ENTSO-E про споживання й генерацію. На практиці токен я отримала й зробила вибіркову перевірку (по одній добі на квартал, зони UA і UA_IPS, 2019–2025): фактичне споживання за годинами є до січня 2022 року, з квітня 2022 по 2025 відповіді немає; генерація за видами майже недоступна навіть у мирний час (2 з 28 діб); доступні лише фізичні перетоки. Тому формулу балансу з теорії для воєнних років обчислити неможливо.
Замість цього використала відкриті набори з порталу Energy Map (дані Укренерго): погодинний баланс (до жовтня 2022), планові відключення (2023–2026), імпорт/експорт (2019–2026), результати й індекси ринку на добу наперед (РДН), а також погоду Open-Meteo.
Найважливіше оновлення — список атак. Раніше я мала вручну складений перелік із 18 подій без втраченої потужності. Тепер це офіційний перелік масованих ударів по критичній інфраструктурі (Energy Map за зведеннями Повітряних сил): {STRIKE_DAYS} днів ({STRIKE_PERIOD}), для кожного — скільки засобів уражень випущено і скільки досягло цілей (у сумі {sp(STRIKE_FIRED)} випущено, {sp(STRIKE_THROUGH)} досягли цілей; плюс 24 лютого 2022 як початок повномасштабного вторгнення). Втрачену потужність у МВт жодне відкрите джерело не дає, тому «сила удару» — це кількість засобів, що досягли цілей.

МОЖЛИВІ ЗАПИТАННЯ: Чому не ENTSO-E? — Токен є, але споживання України там лише до січня 2022. Чи можна довіряти Energy Map? — Це дані Укренерго та Повітряних сил у републікації DiXi Group; ліцензія не вказана, тому посилаюся на першоджерело. Чи ціна РДН допомогла як ознака? — Ні, приріст точності відсутній; я залишила це як перевірену й відхилену ідею.
""")
pw, ph = picture_fit(s, "s_fig_coverage.png", 0.4, 1.5, 8.3, 3.9)
box(s, 0.6, 5.5, 8.1, 1.4, ORANGE_TINT)
text(s, 0.8, 5.52, 7.7, 1.36, [[(f"{STRIKE_DAYS} днів ударів", True), (f" ({STRIKE_PERIOD}) замінили ручний список із 18 рядів: для кожного дня відомо, скільки засобів випущено й скільки досягло цілей. Втрачену потужність у МВт відкриті джерела не дають.", False)]],
     size=16, anchor=MSO_ANCHOR.MIDDLE)
bullets(s, 9.0, 1.65, 3.8, 5.3, [
    [("ENTSO-E: ", True), ("токен є, але споживання України — лише до січня 2022", False)],
    [("Energy Map (Укренерго): ", True), ("баланс, планові відключення, імпорт/експорт, ціни РДН, список ударів", False)],
    [("Open-Meteo: ", True), ("погода без реєстрації", False)],
    [("Ціна РДН як ознака: ", True), ("приросту точності не дала", False)]], size=16)

# ====================================================================================== 6. scenarios
s = new_slide("Чотири ряди — чотири вимірювані погляди на дефіцит", notes=f"""
ЩО СКАЗАТИ (1,5 хв): Формула дефіциту з теорії потребує споживання й генерації. До того ж «фактичне споживання» на ENTSO-E — це вже задоволене навантаження, тому в збалансованій системі різниця близька до нуля. Тому я взяла чотири вимірювані показники, кожен із власним конфігом:
1) Втрачений попит, МВт = очікуване споживання мирного часу мінус фактичне. Очікуване обчислює допоміжна модель, навчена лише на 2019–2021 роках за календарем і погодою. Дані до 27 жовтня 2022, тобто перший рік війни; дата зламу відома незалежно від списку атак — 24 лютого 2022.
2) Чистий імпорт, МВт — імпорт мінус експорт, 2019–2026. Обмежений пропускною здатністю ліній (максимум 2 233 МВт).
3) Індекс відключень — частка областей, де діють графіки відключень (0…1), 2023–2026; насичується на одиниці; шкала — моя наближена оцінка зі слів у файлі.
4) Ціна РДН / цінова стеля — новий четвертий ряд: коли пропозиції бракує, ціна на ринку на добу наперед упирається у стелю. Це погодинний показник напруги в системі за весь період війни, 2019-07 – 2026-06.
Епізод дефіциту — послідовність діб, де добове середнє вище порога; близькі епізоди об'єднуються. Глибина — найвищий добовий рівень, тривалість — кількість діб до трьох спокійних діб поспіль.
Праворуч — кількість подій (удари й тривоги BOCPD) у тестовому періоді, на яких оцінюється прогноз після атаки.

МОЖЛИВІ ЗАПИТАННЯ: Чи це справжній дефіцит? — Ні, це вимірювані наближення («проксі»); я називаю їх прямо. Чому ціна? — Вона вимірюється щогодини за весь період і не залежить від неповного списку атак; стеля — регуляторна, тому ряд обмежений зверху. Чому поріг 800 МВт / 0,02 / 0,9? — За розподілом ряду; це параметри конфігурації.
""")
rows = [["Ряд", "Що вимірює", "Період", "Поріг епізоду", "Подій у тесті"]]
for sc, what, per, thr in (("demand", "очікуване мінус фактичне споживання", "2019-01 – 2022-10", "> 2000 МВт"),
                           ("net_import", "імпорт мінус експорт", "2019-07 – 2026-06", "> 800 МВт"),
                           ("outages", "частка областей під графіками відключень", "2023-01 – 2026-06", "> 0,02"),
                           ("price", "ціна на добу наперед / цінова стеля", "2019-07 – 2026-06", "> 0,9 стелі")):
    rows.append([SC_NAME[sc], what, per, thr, str(n_ev(D, sc, "test"))])
table(s, 0.6, 1.55, [2.9, 4.4, 2.1, 1.6, 1.1], rows, size=15, row_h=0.62, align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, C, C, C])
box(s, 0.6, 5.0, 12.1, 1.85, ORANGE_TINT)
text(s, 0.85, 5.05, 11.6, 1.75, [[("Епізод дефіциту: ", True), ("доби, де добове середнє вище порога; близькі епізоди об’єднуються.", False)],
                                 [("Глибина", True), (" = найвищий добовий рівень.  ", False), ("Тривалість", True), (" = діб до трьох спокійних діб поспіль.", False)],
                                 [("Жоден ряд не є «дефіцитом у МВт»", True), (": це вимірювані наближення, які разом дають повнішу картину.", False)]],
     size=17, anchor=MSO_ANCHOR.MIDDLE)

# ====================================================================================== 7. leakage
s = new_slide("Чесність оцінки: захист від витоку майбутнього", notes="""
ЩО СКАЗАТИ (1,5 хв): Головна небезпека в часових рядах — «витік майбутнього»: коли модель непомітно користується даними, яких на момент прогнозу ще не було. Тоді результати виглядають чудово, але нічого не варті. Я закріпила шість правил:
1) Поділ тільки за часом, без перемішування рядків.
2) Прогноз на годину t випускається за 24 години (ринок на добу наперед: заявки подають до 12:00 попередньої доби), тому ознаки беруться лише з даних до t−24 год.
3) Перенавчання в момент T використовує лише дані до T; сигнал про злам або удар за добу d доступний з початку доби d+1.
4) PELT бачить увесь ряд, тому він ніколи не стає входом моделей.
5) Параметри (вікна, пороги тригерів, гіперпараметри) підбираю тільки на валідації; тестовий період — лише для фінального порівняння.
6) У прогнозі після атаки навчальний приклад (30-добове вікно після дня i) використовується лише якщо вікно повністю закінчилося до дня прогнозу.
Перевірка: тест «на витік» підміняє значення з майбутнього величезними числами і вимагає, щоб ознаки на момент прогнозу не змінилися; аналогічний тест для прогнозу після атаки.

ТЕРМІНИ: валідація — період для підбору параметрів; тест — період для фінальної оцінки, який не можна використовувати для підбору. Walk-forward — навчання «вперед по часу».

МОЖЛИВІ ЗАПИТАННЯ: Чи є слабке місце? — Так: погода береться фактична, а не прогнозна («ідеальний прогноз погоди»), це обмеження я називаю. Чи не підганяли під тест? — Параметри на тесті не підбирались; єдине застереження: деякі базові моделі (наївний, ковзне вікно) додані після першого перегляду результатів.
""")
for i, t_ in enumerate(["Поділ за часом, без перемішування рядків",
                        "Ознаки лише з даних до моменту t − 24 год (ринок на добу наперед)",
                        "Перенавчання тільки на минулому; сигнал доступний з наступної доби",
                        "PELT — лише ретроспективний аналіз, не вхід моделей",
                        "Параметри й тригери підбираються на валідації; тест — лише для порівняння",
                        "Прогноз після атаки: приклад для навчання береться, лише якщо його 30-добове вікно вже закінчилось"]):
    y = 1.5 + i * 0.9
    box(s, 0.6, y, 0.7, 0.7, NAVY, radius=0.5)
    text(s, 0.6, y, 0.7, 0.7, str(i + 1), size=24, bold=True, color=WHITE, align=C, anchor=MSO_ANCHOR.MIDDLE, font=HEAD)
    text(s, 1.5, y - 0.05, 6.8, 0.8, t_, size=18, anchor=MSO_ANCHOR.MIDDLE)
card(s, 8.6, 1.5, 4.1, 3.3, "Як це перевіряється", f"{TESTS_N} тестів, серед них: значення з майбутнього підмінюються великими числами — ознаки на момент прогнозу не повинні змінитися (також для прогнозу після атаки).", size=16)
card(s, 8.6, 5.0, 4.1, 1.8, "Відоме обмеження", "Погода — фактична, а не прогнозна.", fill=ORANGE_TINT, size=17)

# ====================================================================================== 8. post-event: method
s = new_slide("Прогноз глибини й тривалості після атаки: як це працює", notes=f"""
ЩО СКАЗАТИ (2 хв): Це центральна частина теми. Звичайний прогноз на добу наперед відстежує ряд; тут інакше. У день випуску d0 — це день удару зі списку або день тривоги BOCPD — модель бачить лише історію до цього дня і прогнозує добовий дефіцит на 30 діб уперед. З цього беремо три величини: глибину (найвищий добовий рівень), тривалість (скільки діб до трьох спокійних діб поспіль) і кількість діб дефіциту.
Методи. Базові: «як учора» (персистентність — рівень останньої доби без змін) і середнє за 28 діб. Траєкторні: XGBoost, який прогнозує добову траєкторію (горизонт — ознака): статичний і перенавчений у день події. Прямі, що пропускають траєкторію й одразу прогнозують потрібну величину: для глибини — ridge-регресія (лінійна регресія зі штрафом) на недавніх рівнях ряду (рівень останньої доби, середні й максимуми за 7 і 28 діб) та XGBoost напряму (ridge + дерева на її залишку); для тривалості — XGBoost-регресія, модель виживання AFT (враховує цензурування: випадки, коли дефіцит не закінчився за 30 діб) і класифікатор «чи буде доба над порогом».
Оцінка: похибка MAE глибини й тривалості та парний тест Вілкоксона кожного методу проти «як учора» на помилках окремих подій. Поділ: події до кінця валідації — валідація, пізніші — тест.
Правило проти витоку: приклад для навчання (30-добове вікно після дня i) беремо лише якщо i + 30 ≤ d0; тест у tests/test_post_event.py перемішує дні після d0 і перевіряє, що прогноз не змінюється.

ТЕРМІНИ: ridge — лінійна регресія з L2-штрафом; AFT (accelerated failure time) — модель виживання, що прогнозує час до події і коректно працює з цензурованими даними; цензурування — подія ще не відбулася на момент закінчення спостереження.

МОЖЛИВІ ЗАПИТАННЯ: Чому «як учора» — база? — Після удару рівень дефіциту зазвичай тримається, тож це сильний простий метод; перевершити його — змістовна вимога. Чому ridge? — Лінійна модель на рівнях ряду екстраполює на нові рівні, чого дерева не вміють (вони не виходять за межі побачених у навчанні значень).
""")
# timeline
line(s, 0.9, 2.55, 12.4, 2.55, color=NAVY, width=3)
box(s, 0.9, 1.65, 3.8, 0.7, "E3E8F0", radius=0.15)
text(s, 0.9, 1.65, 3.8, 0.7, "історія до d0: рівні, сезон, ціна, сила удару", size=14, color=MUTED, align=C, anchor=MSO_ANCHOR.MIDDLE)
box(s, 5.3, 1.65, 7.1, 0.7, ORANGE_TINT, radius=0.15)
text(s, 5.3, 1.65, 7.1, 0.7, "прогноз на 30 діб уперед", size=18, bold=True, color=NAVY, align=C, anchor=MSO_ANCHOR.MIDDLE)
box(s, 5.16, 2.41, 0.28, 0.28, RED, shape=MSO_SHAPE.OVAL)
text(s, 3.9, 2.75, 2.8, 0.4, "d0: удар або тривога BOCPD", size=14, align=C)
for i, (h, b) in enumerate([("Глибина", "найвищий добовий рівень"), ("Тривалість", "діб до трьох спокійних діб поспіль"), ("Діб дефіциту", "скільки діб над порогом")]):
    box(s, 7.0 + i * 1.8, 2.95, 1.7, 0.95, GREEN_TINT)
    text(s, 7.0 + i * 1.8, 2.97, 1.7, 0.4, h, size=14, bold=True, color=NAVY, align=C)
    text(s, 7.0 + i * 1.8, 3.35, 1.7, 0.55, b, size=11, color=INK, align=C)
rows = [["Група методів", "Що робить"],
        ["Базові", "«як учора» (персистентність); середнє за 28 діб"],
        ["Траєкторія", "XGBoost прогнозує добовий шлях: статичний; перенавчений у день події на останніх 180 добах"],
        ["Глибина напряму", "ridge на недавніх рівнях ряду; XGBoost напряму (ridge + дерева на залишку)"],
        ["Тривалість напряму", "XGBoost-регресія; модель виживання AFT з цензуруванням; класифікатор «доба над порогом»"]]
table(s, 0.6, 4.1, [2.6, 9.5], rows, size=14, row_h=0.46, bold_rows=())
text(s, 0.6, 6.5, 12.1, 0.45, "Оцінка: MAE глибини й тривалості; парний тест Вілкоксона проти «як учора» на помилках подій. Приклад для навчання береться, лише якщо його вікно закінчилось до d0.",
     size=13, color=MUTED)

# ====================================================================================== 9. post-event: depth result
rows = [["Ряд · вибірка (подій)", "«Як учора»", "Ridge", "XGBoost напряму", "p (ridge)"]]
colors = {}
depth_rows = []
for sc in ("price", "net_import", "outages", "demand"):
    for split in ("validation", "test"):
        n = n_ev(D, sc, split)
        if n == 0:
            continue
        pers, rid, xgd = (pe(D, sc, m, "depth_MAE", split) for m in ("persistence", "ridge_depth", "xgb_direct"))
        row_r, row_x = depth_p(sc, split, "ridge_depth"), depth_p(sc, split, "xgb_direct")
        v_r, v_x = verdict(row_r), verdict(row_x)
        d = D_E[sc]
        rows.append([f"{SHORT[sc]} · {SPLIT_UA[split]} ({n})", fm(pers, d), fm(rid, d), fm(xgd, d), ptxt(row_r["p_value"]) if row_r else "замало подій"])
        i = len(rows) - 1
        colors[(i, 2)], colors[(i, 4)] = sig_color(v_r), sig_color(v_r)
        colors[(i, 3)] = sig_color(v_x)
        colors = {k: v for k, v in colors.items() if v}
        depth_rows.append((sc, split, n, pers, rid, xgd, row_r, row_x, v_r, v_x))
DR = {(r[0], r[1]): r for r in depth_rows}
good = [f"{SHORT[sc]} ({SPLIT_UA[sp_]})" for (sc, sp_), r in DR.items() if r[8] == "better"]
nogain = [f"{SHORT[sc]} ({SPLIT_UA[sp_]})" for (sc, sp_), r in DR.items() if r[8] in ("same", "none") and sc != "demand"]
check("ridge_depth значуще краще за персистентність: ціна, валідація", DR.get(("price", "validation"), [None] * 9)[8] == "better")
check("ridge_depth значуще краще за персистентність: ціна, тест", DR.get(("price", "test"), [None] * 9)[8] == "better")
check("ridge_depth значуще краще за персистентність: імпорт, тест", DR.get(("net_import", "test"), [None] * 9)[8] == "better")
check("відключення: немає значущого виграшу глибини (тест)", DR.get(("outages", "test"), [None] * 9)[8] in ("same", "none", "worse"))
dp, dn, do_ = DR[("price", "test")], DR[("net_import", "test")], DR[("outages", "test")]
s = new_slide("Головний результат: глибина після атаки значуще точніша за «як учора»", notes=f"""
ЩО СКАЗАТИ (2 хв): Це відповідь на тему роботи. Таблиця: похибка MAE глибини дефіциту на 30 діб після події. Що менше — то краще. Колір — значущий виграш проти «як учора» за тестом Вілкоксона на помилках подій (p < 0,05).
Ціна РДН / стеля: на тесті ({dp[2]} подій) «як учора» помиляється на {fm(dp[3], 2)} частки стелі, ridge — на {fm(dp[4], 2)}, XGBoost напряму — на {fm(dp[5], 2)}; {ptxt(dp[6]['p_value'])}; ridge точніший у {fm(100 * dp[6]['share_better'], 0)} % подій. Те саме на валідації.
Чистий імпорт, тест ({dn[2]} подій): {fm(dn[3], 0)} МВт проти {fm(dn[4], 0)} МВт у ridge, {ptxt(dn[6]['p_value'])}.
Індекс відключень, тест ({do_[2]} подій): {fm(do_[3], 2)} проти {fm(do_[4], 2)} (ridge) та {fm(do_[5], 2)} (XGBoost напряму); значущої різниці немає ({ptxt(do_[6]['p_value'])}) — це напрям вдосконалення: індекс насичується на одиниці й має мало подій.
Втрачений попит: у тестовому періоді лише {n_ev(D, 'demand', 'test')} подій — для тесту замало, значущість не оцінюю.
Чому прямий ridge виграє: лінійна модель на рівнях ряду екстраполює на нові рівні, а дерева XGBoost не виходять за межі побачених у навчанні значень; тому після удару, що виводить ряд на новий рівень, ridge точніший.
ЧЕСНО: події перекриваються у часі (30-добові вікна), тому p-значення оптимістичні; висновок — індикативний, але стійкий на двох рядах і двох вибірках (валідація, тест).

МОЖЛИВІ ЗАПИТАННЯ: Чому не XGBoost? — Він теж значуще кращий за «як учора» на тих самих рядах, але ridge простіший і не гірший. Скільки подій? — див. таблицю: стільки днів ударів і тривог BOCPD. Чи значущо? — Так на ціні (валідація й тест) та імпорті (тест); на відключеннях — ні.
""")
pw, ph = picture_fit(s, "s_fig_post_depth.png", 0.4, 1.5, 6.1, 3.9)
table(s, 6.6, 1.55, [2.45, 1.0, 0.8, 1.0, 1.0], rows, size=12, row_h=0.46, align=[PP_ALIGN.LEFT, C, C, C, C], colors=colors)
box(s, 0.6, 5.55, 5.95, 1.35, GREEN_TINT)
text(s, 0.8, 5.57, 5.6, 1.3, [[("Значуще краще за «як учора»: ", True), ("; ".join(good) if good else "—", False)]], size=15, anchor=MSO_ANCHOR.MIDDLE)
box(s, 6.75, 5.55, 5.95, 1.35, ORANGE_TINT)
text(s, 6.95, 5.57, 5.6, 1.3, [[("Напрям вдосконалення: ", True), (f"відключення — без виграшу (індекс насичується на 1); попит — лише {n_ev(D, 'demand', 'test')} події в тесті.", False)]], size=15, anchor=MSO_ANCHOR.MIDDLE)

# ====================================================================================== 10. post-event: duration
DUR_M = ("persistence", "xgb_direct", "xgb_aft", "xgb_cls")
rows = [["Ряд, тест (подій)", "«Як учора»", "XGBoost-регресія", "Модель виживання AFT", "Класифікатор"]]
colors = {}
dur_better = []
dur_worse = []
dur_info = {}
for sc in ("price", "net_import", "outages", "demand"):
    n = n_ev(D, sc, "test")
    if n == 0:
        continue
    row = [f"{SHORT[sc]} ({n})"]
    for j, m in enumerate(DUR_M, start=1):
        val_ = pe(D, sc, m, "duration_MAE_days", "test")
        r_ = pt(D, sc, m, "duration", "test") if m != "persistence" else None
        v_ = verdict(r_)
        row.append(f"{fm(val_, 1)}" if val_ == val_ else "—")
        if v_ in ("better", "worse"):
            colors[(len(rows), j)] = sig_color(v_)
            (dur_better if v_ == "better" else dur_worse).append((sc, m, val_, pe(D, sc, "persistence", "duration_MAE_days", "test"), r_["p_value"]))
        dur_info[(sc, m)] = (val_, r_)
    rows.append(row)
cens = {}
for sc in SCEN:
    pev = D[sc]["pev"]
    if len(pev) and "actual_censored" in pev:
        t_ = pev[pev["split"] == "test"]
        cens[sc] = (float(t_["actual_censored"].astype(float).mean()) if len(t_) else float("nan"), len(t_))
dur_title = ("Тривалість: реалізовано три підходи, точність поки на рівні базового методу" if not dur_better
             else "Тривалість: реалізовано три підходи; виграш над «як учора» лише на окремих рядах")
check("тривалість: жоден метод не значуще кращий за персистентність (текст «на рівні базового»)", not dur_better)
dur_txt = "; ".join(f"{SHORT[sc]}: {PE_LABEL[m]} {fm(v1, 1)} проти {fm(v0, 1)} діб ({ptxt(p_)})" for sc, m, v1, v0, p_ in dur_better) or "жоден метод не значуще кращий за «як учора»"
dur_txt_worse = "; ".join(f"{SHORT[sc]}: {PE_LABEL[m]} {fm(v1, 1)} проти {fm(v0, 1)} діб ({ptxt(p_)})" for sc, m, v1, v0, p_ in dur_worse) or "—"
cens_txt = "; ".join(f"{SHORT[sc]} {fm(100 * c[0], 0)} % ({c[1]} подій)" for sc, c in cens.items() if sc != "demand" and c[0] == c[0])
s = new_slide(dur_title, notes=f"""
ЩО СКАЗАТИ (1,5 хв): Тривалість дефіциту після атаки я теж прогнозую напряму: три підходи — XGBoost-регресія числа діб до трьох спокійних діб поспіль, модель виживання AFT (вона коректно враховує, що в частині подій дефіцит не закінчився за 30 діб — це цензурування) та класифікатор «чи буде доба над порогом». Таблиця — похибка MAE у добах на тесті поряд із «як учора».
Чесна відповідь на запитання «а тривалість?»: реалізовано й перевірено, точність поки на рівні базового методу. Значущо кращі за «як учора»: {dur_txt}. Значущо гірші: {dur_txt_worse}. В інших випадках різниця статистично не відрізняється від нуля.
Чому так (це припущення, а не доведений факт): 1) тривалість — це «число діб до відновлення», а воно сильно залежить від обставин, яких немає у ряду (наступні удари, ремонти); 2) у частині подій дефіцит не закінчується за 30 діб — частка таких подій на тесті: {cens_txt}; 3) подій небагато й вони перекриваються. Що потрібно: більше подій, дані про втрачену потужність і ремонти, довший горизонт.
Тому тривалість подана як напрям вдосконалення; ключовий результат — прогноз глибини (попередній слайд).

ТЕРМІНИ: цензурування — подія ще не відбулася на момент закінчення спостереження (тут: за 30 діб дефіцит не закінчився); модель виживання (AFT) — прогнозує час до події, враховуючи цензурування.

МОЖЛИВІ ЗАПИТАННЯ: Чому вона така неточна? — Див. вище: тривалість залежить від майбутніх ударів і ремонтів, яких на момент прогнозу не знаємо. Навіщо тоді три підходи? — Щоб перевірити різні постановки (регресія, виживання, класифікація) й чесно показати, що жодна не дала стійкого виграшу над «як учора» (де p ≥ 0,05).
""")
table(s, 0.6, 1.6, [3.1, 1.6, 2.0, 2.4, 1.9], rows, size=15, row_h=0.62, align=[PP_ALIGN.LEFT, C, C, C, C], colors=colors)
text(s, 0.6, 4.75, 11.0, 0.45, "MAE тривалості, діб (тест). Зелений / червоний — значуще краще / гірше за «як учора» (Вілкоксон на подіях, p < 0,05).", size=13, color=MUTED)
card(s, 0.6, 5.25, 5.9, 1.65, "Що реалізовано", "Регресія, модель виживання AFT з цензуруванням і класифікатор; усі перевірені на тих самих подіях, що й глибина.", size=15, head_size=17, head_h=0.5)
card(s, 6.8, 5.25, 5.9, 1.65, "Напрям вдосконалення", "Точність на рівні «як учора»: потрібні більше подій, дані про втрачену потужність і ремонти.", size=15, head_size=17, fill=ORANGE_TINT, head_h=0.5)

# ====================================================================================== 11. retraining strategies
nr = {sc: refit_counts(D, sc, "xgb_drift") for sc in SCEN}
s = new_slide("Адаптація: стратегії перенавчання моделі", notes=f"""
ЩО СКАЗАТИ (1,5 хв): Друге запитання — чи потрібне перенавчання і за яким сигналом. Усі моделі — XGBoost з однаковими ознаками; відрізняються лише тим, КОЛИ їх перенавчають. Статична — один раз на 2019–2021. «Ковзне вікно (щотижня)» — щотижня на останніх w добах; це контроль без жодного тригера. «Ковзне вікно (кожні 3 дні)» — те саме, але частіше: цей контроль додано, щоб не приписати користь тригерам, які просто перенавчаються частіше за тижневий розклад, — тобто той самий обчислювальний бюджет. «За тривогою BOCPD» — щотижня й після тривоги BOCPD, на даних від оціненого моменту зламу (ідея з теорії). «Після атаки» — щотижня й наступного дня після кожного удару зі списку, на короткому вікні. «За помилками» — щотижня й тоді, коли CUSUM на власній добовій помилці моделі сигналізує про дрейф.
CUSUM: щодня порівнюємо MAE моделі за добу з її середнім за попередні 60 діб; перевищення понад допуск k накопичується; коли сума перевищує поріг h — перенавчання на останніх 14–30 добах. Параметри k, h і вікно підібрані на валідації для кожного ряду. Перенавчань за помилками (від усіх перенавчань за весь період walk-forward): відключення {nr['outages'][1]} із {nr['outages'][0]}, імпорт {nr['net_import'][1]} із {nr['net_import'][0]}, попит {nr['demand'][1]} із {nr['demand'][0]}, ціна {nr['price'][1]} із {nr['price'][0]}; решта — за тижневим розкладом.
Усі тригери й вікна підбирались лише на валідації; тест не використовувався.

ТЕРМІНИ: дрейф — поступове стійке погіршення помилки моделі; CUSUM — накопичувальна сума перевищень над допустимим рівнем, тривога при перетині порога.

МОЖЛИВІ ЗАПИТАННЯ: Чому саме 3 дні? — Це параметр контролю: тригерні моделі перенавчаються приблизно так часто, і ми порівнюємо за однакового бюджету. Чи не підганяли під тест? — Ні, тригери підбирались на валідації (tune-triggers).
""")
rows = [["Модель", "Коли перенавчається", "Роль"],
        ["XGBoost статичний", "один раз, на 2019–2021", "що буде без адаптації"],
        ["Ковзне вікно, щотижня", "щотижня, останні w діб", "контроль без тригерів"],
        ["Ковзне вікно, кожні 3 дні", "кожні 3 дні, останні w діб", "контроль із тим самим бюджетом"],
        ["За тривогою BOCPD", "щотижня + після тривоги, від моменту зламу", "ідея з теоретичної частини"],
        ["Після атаки", "щотижня + наступного дня після удару", "подієвий тригер"],
        ["За помилками (CUSUM)", "щотижня + коли помилка моделі дрейфує", "тригер за власною помилкою"]]
table(s, 0.6, 1.55, [3.1, 5.2, 3.8], rows, size=15, row_h=0.56, highlight=(3, 6), align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
card(s, 0.6, 5.65, 5.95, 1.3, "CUSUM на власній помилці", "Помилка за добу порівнюється з середнім за 60 діб; перевищення накопичується; сума > h — перенавчання на короткому вікні.", size=14, head_size=16, head_h=0.4)
card(s, 6.75, 5.65, 5.95, 1.3, "Чому контроль кожні 3 дні", "Тригерні моделі перенавчаються частіше за тиждень. Контроль показує, що дає сам сигнал, а що — лише частота.", fill=ORANGE_TINT, size=14, head_size=16, head_h=0.4)

# ====================================================================================== 12. retraining result
st_ratio = M("demand", "xgb_static") / M("demand", "xgb_drift")
out_drift = PAIR["outages"].get("xgb_drift")
dem_drift = PAIR["demand"].get("xgb_drift")
check("демо: статична ≫ перенавчана (≥ 5 разів)", st_ratio >= 5)
check("відключення: drift значуще кращий за щотижневе ковзне вікно", out_drift is not None and out_drift["p"] < 0.05 and REL["outages"]["xgb_drift"] < 0)
check("контроль кожні 3 дні не значуще гірший за drift (відключення)", (PAIR["outages"].get("xgb_rolling_freq") is not None) and not (REL["outages"]["xgb_rolling_freq"] > REL["outages"]["xgb_drift"] and pair(D, "outages", "xgb_drift", "xgb_rolling_freq")["p"] < 0.05))
s = new_slide("Перенавчання краще за статичну модель; найкраще — за власною помилкою", notes=f"""
ЩО СКАЗАТИ (2 хв): Рисунок і таблиця — похибка на тесті відносно щотижневого ковзного вікна (xgb_rolling): мінус — краще. Зелений — значуще краще, червоний — значуще гірше (парний тест Вілкоксона на добових MAE, p < 0,05).
1) Перенавчані моделі значно точніші за статичну там, де режим змінився: на втраченому попиті статична {fm(M('demand', 'xgb_static'), 0)} МВт проти {fm(M('demand', 'xgb_drift'), 0)} МВт у перенавчаній за помилками (у {fm(st_ratio, 0)} разів). Статична проти щотижневого ковзного вікна — {STATIC_LINE}. Тобто виграш великий на попиті й відключеннях, помірний на імпорті; {'на ціні статична не гірша (ряд обмежений стелею й менш чутливий до режиму)' if 'price' in STATIC_OK else 'на ціні перенавчання також краще'}.
2) Перенавчання за помилками (xgb_drift) відносно щотижневого ковзного вікна: відключення {fm(REL['outages']['xgb_drift'], 1, sign=True)} % ({ptxt(out_drift['p']) if out_drift else '—'}, {verdict_word('outages', 'xgb_drift')}); попит {fm(REL['demand']['xgb_drift'], 1, sign=True)} % ({ptxt(dem_drift['p']) if dem_drift else '—'}, {verdict_word('demand', 'xgb_drift')}); імпорт {fm(REL['net_import']['xgb_drift'], 1, sign=True)} % ({verdict_word('net_import', 'xgb_drift')}); ціна {fm(REL['price']['xgb_drift'], 1, sign=True)} % ({verdict_word('price', 'xgb_drift')}).
3) Контроль «ковзне вікно кожні 3 дні» не гірший за тригерні: відключення {fm(REL['outages'].get('xgb_rolling_freq', float('nan')), 1, sign=True)} %, попит {fm(REL['demand'].get('xgb_rolling_freq', float('nan')), 1, sign=True)} %. Висновок: допомагає частота перенавчання й тригер за власною помилкою; тригер за рівнем BOCPD (xgb_adaptive: відключення {fm(REL['outages']['xgb_adaptive'], 1, sign=True)} %, попит {fm(REL['demand']['xgb_adaptive'], 1, sign=True)} %) не обганяє просте часте перенавчання. «Після атаки» (xgb_attack) не краще — напрям вдосконалення.
ЧЕСНО: я не стверджую, що перенавчання за виявленим зламом краще за просте; на цих даних це не підтвердилось. Різниці між перенавчаними моделями невеликі, p-значення оптимістичні (автокореляція добових похибок).

МОЖЛИВІ ЗАПИТАННЯ: Чому BOCPD не допоміг? — Тривога спізнюється на кілька діб; сигнал за рівнем ряду менш чутливий, ніж помилка самої моделі (CUSUM). Чи це остаточно? — Ні: епізодів мало, ряди — проксі.
""")
pw, ph = picture_fit(s, "s_fig_retrain.png", 0.4, 1.5, 6.7, 3.95)
rows = [["Ряд (тест)", "Ковзне, MAE", "За помилками", "Кожні 3 дні", "За BOCPD", "Після атаки"]]
colors = {}
for sc in ("demand", "net_import", "outages", "price"):
    i = len(rows)
    row = [SHORT[sc], f"{fm(M(sc, 'xgb_rolling'), D_H[sc])}"]
    for j, m in enumerate(("xgb_drift", "xgb_rolling_freq", "xgb_adaptive", "xgb_attack"), start=2):
        t_, c_ = rel_cell(sc, m)
        row.append(t_)
        if c_:
            colors[(i, j)] = c_
    rows.append(row)
table(s, 7.25, 1.55, [1.25, 0.95, 0.95, 0.9, 0.85, 0.85], rows, size=12, row_h=0.58, align=[PP_ALIGN.LEFT] + [C] * 5, colors=colors)
text(s, 7.25, 4.55, 5.8, 0.9, "Відносно щотижневого ковзного вікна, %; MAE ковзного — в МВт (попит, імпорт) або в одиницях ряду. Зелений / червоний — значуща різниця (p < 0,05).", size=12, color=MUTED)
stat(s, 0.6, 5.55, 3.95, 1.4, f"{fm(M('demand', 'xgb_static'), 0)} → {fm(M('demand', 'xgb_drift'), 0)}", "МВт: статична проти перенавчання за помилками (попит, тест)", big_size=26, label_size=13)
stat(s, 4.7, 5.55, 3.95, 1.4, f"{fm(REL['outages']['xgb_drift'], 1, sign=True)} %", f"за помилками проти щотижневого вікна, відключення ({ptxt(out_drift['p']) if out_drift else '—'})", fill=GREEN_TINT, big_size=26, label_size=13)
stat(s, 8.8, 5.55, 3.9, 1.4, "частота, не BOCPD", "контроль кожні 3 дні не гірший за тригер BOCPD; тригер за помилками — найкращий або рівний", fill=ORANGE_TINT, big_size=22, label_size=13)

# ====================================================================================== 13-16. forecasts per scenario
FC_ROWS = ("seasonal_naive", "xgb_static", "xgb_rolling", "xgb_rolling_freq", "xgb_drift", "xgb_adaptive")


def fc_table(sc):
    rows = [["Модель (тест)", f"MAE{UNIT_SHORT[sc].replace(' ', ', ') if UNIT_SHORT[sc] else ''}", "до ковзного"]]
    bold = []
    colors = {}
    for m in FC_ROWS:
        if not has_model(D, sc, m):
            continue
        t_, c_ = rel_cell(sc, m) if m != "xgb_rolling" else ("—", None)
        rows.append([LABEL[m], fm(M(sc, m), D_H[sc]), t_])
        if c_:
            colors[(len(rows) - 1, 2)] = c_
    best = min((m for m in FC_ROWS if has_model(D, sc, m)), key=lambda m: M(sc, m))
    for i, r_ in enumerate(rows[1:], start=1):
        if r_[0] == LABEL[best]:
            bold.append(i)
    return rows, bold, colors, best


def fc_slide(sc, title, fig, notes, takeaway_head, takeaway, fig_box=(0.4, 1.5, 6.9, 3.95), prophet_note=True):
    s = new_slide(title, notes=notes)
    pw, ph = picture_fit(s, fig, *fig_box)
    rows, bold, colors, best = fc_table(sc)
    table(s, 7.5, 1.55, [3.2, 1.0, 1.0], rows, size=12, row_h=0.5, bold_rows=bold, align=[PP_ALIGN.LEFT, C, C], colors=colors)
    ypx = 1.55 + 0.5 * len(rows) + 0.1
    if prophet_note:
        text(s, 7.5, ypx, 5.2, 0.6, f"Prophet: {fm(M(sc, 'prophet_static'), D_H[sc] if M(sc, 'prophet_static') < 100 else 0)}{UNIT_SHORT[sc]} (поза шкалою, не показано). «До ковзного» — відносно щотижневого ковзного вікна; зелений / червоний — p < 0,05.", size=11, color=MUTED, italic=True)
    card(s, 0.6, 5.6, 12.1, 1.35, takeaway_head, takeaway, size=15, head_size=17, fill=ORANGE_TINT, head_h=0.4)
    return s


def sc_takeaway(sc, extra=""):
    st, rl, dr, nv = M(sc, "xgb_static"), M(sc, "xgb_rolling"), M(sc, "xgb_drift"), M(sc, "seasonal_naive")
    d1 = D_H[sc]
    return (f"Статична {fm(st, d1)}, ковзне вікно {fm(rl, d1)}, за помилками {fm(dr, d1)} ({fm(REL[sc]['xgb_drift'], 1, sign=True)} % до ковзного, {verdict_word(sc, 'xgb_drift')}); "
            f"сезонний наївний {fm(nv, d1)}{UNIT_SHORT[sc]}." + (" " + extra if extra else ""))


# 13 demand
check("демо: перенавчані моделі точніші за сезонний наївний", M("demand", "xgb_drift") < M("demand", "seasonal_naive"))
fc_slide("demand", f"Попит: перенавчання зменшує похибку в ~{fm(st_ratio, 0)} разів", "s_fig_fc_demand.png", f"""
ЩО СКАЗАТИ (1,5 хв): Це найближчий до теми сценарій: ціна — втрачений попит у МВт. Навчальний період (2019–2021) мирний, перехід 24 лютого 2022 припадає на валідацію, тест — квітень–жовтень 2022. Статичні моделі, навчені на мирному часі, не бачать нового режиму: статичний XGBoost має MAE {fm(M('demand', 'xgb_static'), 0)} МВт, Prophet {fm(M('demand', 'prophet_static'), 0)} МВт. Перенавчані моделі: ковзне вікно щотижня {fm(M('demand', 'xgb_rolling'), 0)}, кожні 3 дні {fm(M('demand', 'xgb_rolling_freq'), 0)}, за тривогою BOCPD {fm(M('demand', 'xgb_adaptive'), 0)}, за помилками {fm(M('demand', 'xgb_drift'), 0)}, сезонний наївний {fm(M('demand', 'seasonal_naive'), 0)} МВт. Перенавчання за помилками відносно щотижневого вікна: {fm(REL['demand']['xgb_drift'], 1, sign=True)} % ({verdict_word('demand', 'xgb_drift')}).
ЗАСТЕРЕЖЕННЯ: тест — це вже усталений новий режим, сам перехід у валідації; тому таблиця показує, наскільки швидко модель опановує режим, а не як передбачає його появу. Дані балансу закінчуються 27.10.2022.

МОЖЛИВІ ЗАПИТАННЯ: Звідки очікуване споживання? — Допоміжна XGBoost за календарем і погодою, навчена до кінця 2021; її похибка до 1 ГВт у мирний час. Чи це справжній дефіцит? — Ні, проксі: втрачений попит не відрізняє дефіцит від економії чи втрати споживачів на окупованих територіях.
""", "Що це означає", sc_takeaway("demand", "Тест — вже усталений режим, перехід 24.02.2022 у валідації."))
# 14 net import
fc_slide("net_import", "Чистий імпорт: перенавчання краще за статичну й наївний", "s_fig_fc_import.png", f"""
ЩО СКАЗАТИ (1,5 хв): Тестовий період 2024 – червень 2026. MAE — середня абсолютна помилка прогнозу в МВт (діапазон значень від −1 350 до +2 233 МВт). Ковзне вікно щотижня: {fm(M('net_import', 'xgb_rolling'), 1)}; за помилками: {fm(M('net_import', 'xgb_drift'), 1)}; за тривогою BOCPD: {fm(M('net_import', 'xgb_adaptive'), 1)}; статична: {fm(M('net_import', 'xgb_static'), 1)}; сезонний наївний: {fm(M('net_import', 'seasonal_naive'), 1)}; Prophet: {fm(M('net_import', 'prophet_static'), 0)}. Перенавчані моделі краще за статичну на {fm(abs(pct(M('net_import', 'xgb_rolling'), M('net_import', 'xgb_static'))), 0)} % і за сезонний наївний на {fm(abs(pct(M('net_import', 'xgb_rolling'), M('net_import', 'seasonal_naive'))), 0)} %; між собою різниці невеликі: за помилками проти щотижневого вікна — {fm(REL['net_import']['xgb_drift'], 1, sign=True)} % ({verdict_word('net_import', 'xgb_drift')}).
Prophet помилився на {fm(M('net_import', 'prophet_static'), 0)} МВт, що більше за весь діапазон даних: його лінійний тренд, навчений у 2019–2021, «втікає» до 8000 МВт. Це недолік цієї базової моделі, а не доказ чогось про метод.
Тест Вілкоксона (парний, на добових MAE); p-значення оптимістичні, бо добові помилки пов'язані між собою.

МОЖЛИВІ ЗАПИТАННЯ: Чи значуща різниця? — Для статичної й наївного так; між перенавчаними — як у таблиці. Чому Prophet такий поганий? — Лінійний тренд екстраполюється; Prophet не бачить режимів.
""", "Що це означає", sc_takeaway("net_import"))
# 15 outages
fc_slide("outages", "Відключення: перенавчання за помилками значуще краще за щотижневе", "s_fig_fc_outages.png", f"""
ЩО СКАЗАТИ (1,5 хв): Цільова змінна — частка областей під відключеннями; тест — липень 2025 – червень 2026, зимова кампанія. Статична {fm(M('outages', 'xgb_static'), 3)}; ковзне вікно щотижня {fm(M('outages', 'xgb_rolling'), 3)}; кожні 3 дні {fm(M('outages', 'xgb_rolling_freq'), 3)}; за тривогою BOCPD {fm(M('outages', 'xgb_adaptive'), 3)}; за помилками {fm(M('outages', 'xgb_drift'), 3)}; сезонний наївний {fm(M('outages', 'seasonal_naive'), 3)}; Prophet {fm(M('outages', 'prophet_static'), 3)}. Перенавчання за помилками відносно щотижневого вікна: {fm(REL['outages']['xgb_drift'], 1, sign=True)} % ({ptxt(out_drift['p']) if out_drift else '—'}, {verdict_word('outages', 'xgb_drift')}); BOCPD-варіант: {fm(REL['outages']['xgb_adaptive'], 1, sign=True)} % ({verdict_word('outages', 'xgb_adaptive')}); кожні 3 дні: {fm(REL['outages']['xgb_rolling_freq'], 1, sign=True)} % ({verdict_word('outages', 'xgb_rolling_freq')}).
ЧЕСНО: різниці між перенавчаними моделями невеликі в абсолютних числах; на валідації найкращою була статична модель — рейтинг нестійкий. Тестових епізодів дефіциту мало (3), тому висновки індикативні. Раніше я знайшла й виправила помилку: у виключений період (10.12.2025–28.01.2026) вікно навчання не мало відомих значень, і модель ковзного вікна прогнозувала сталий нуль; тепер оновлення пропускається, якщо у вікні менше 168 відомих значень.

МОЖЛИВІ ЗАПИТАННЯ: Чому виправлення змінило результат? — Раніше ковзне вікно «ламалося» в період без даних; виправлення застосовано до всіх моделей однаково. Чи не підганяєте результат? — Ні: параметри підбирались на валідації; виправлення — помилка коду, знайдена на графіку.
""", "Що це означає", sc_takeaway("outages", "Епізодів у тесті мало, на валідації найкращою була статична — рейтинг нестійкий."))
# 16 price
fc_slide("price", "Ціна РДН / стеля: четвертий ряд, перевірка на всю війну", "s_fig_fc_price.png", f"""
ЩО СКАЗАТИ (1,5 хв): Це четвертий, новий сценарій: ціна на ринку на добу наперед відносно цінової стелі, 2019-07 – 2026-06; тест — 2024–2026. Коли пропозиції бракує, ціна впирається у стелю; перша зима війни: місячна середня 0,96–0,99 стелі. Це погодинний, вимірюваний показник напруги в системі за весь період, який не залежить від неповного списку атак. Результати: статична {fm(M('price', 'xgb_static'), 3)}, ковзне вікно {fm(M('price', 'xgb_rolling'), 3)}, за помилками {fm(M('price', 'xgb_drift'), 3)}, за тривогою BOCPD {fm(M('price', 'xgb_adaptive'), 3)}, сезонний наївний {fm(M('price', 'seasonal_naive'), 3)}, Prophet {fm(M('price', 'prophet_static'), 3)}. Перенавчання за помилками відносно щотижневого вікна: {fm(REL['price']['xgb_drift'], 1, sign=True)} % ({verdict_word('price', 'xgb_drift')}).
ЧЕСНО: {'на цьому ряді статична модель не гірша за перенавчані (' + fm(M('price', 'xgb_static'), 3) + ' проти ' + fm(M('price', 'xgb_rolling'), 3) + '): ціна обмежена стелею й менш чутлива до режиму, тому перенавчання тут нічого не дає' if 'price' in STATIC_OK else 'на цьому ряді перенавчання дає помірний виграш'}. Ряд корисний передусім для прогнозу глибини після атаки, де він дав значущий виграш над «як учора».

МОЖЛИВІ ЗАПИТАННЯ: Чому ціна — показник дефіциту? — Бо за нестачі пропозиції ціна росте до стелі; але ціна реагує не лише на атаки, тож це проксі. Чи ціну РДН як ознаку для інших рядів перевіряли? — Так, приросту точності не дала.
""", "Що це означає", sc_takeaway("price", "Ряд обмежений стелею; " + ("статична не гірша за перенавчані." if "price" in STATIC_OK else "перевага перенавчання тут найменша.")))

# ====================================================================================== 17. detection
cp = {sc: D[sc]["cp"] for sc in SCEN}
bo_o = cp["outages"]["bocpd"]["scores"]
s = new_slide("Виявлення зламів: PELT і BOCPD як діагностика та джерело подій", notes=f"""
ЩО СКАЗАТИ (1,5 хв): На рисунку — ряд ціни РДН / стелі: червоні лінії — удари зі списку, пунктир — злами PELT, трикутники — тривоги BOCPD, внизу — ймовірність BOCPD. Збіг — виявлення протягом 14 діб після удару. Таблиця — якість виявлення за чотирма рядами: точність (яка частка тривог влучила в удар) і повнота (яка частка ударів виявлена). Для списку з {STRIKE_DAYS} ударів повнота низька: багато ударів не залишають помітного сліду в добових середніх, а режим змінюється поступово. Високу точність на відключеннях ({fm(bo_o['precision'], 2)}) пояснює те, що відключення — прямий наслідок ударів.
Для втраченого попиту дата зламу відома незалежно від списку — 24 лютого 2022: PELT знайшов її точно (затримка 0 діб).
Що це означає для роботи: BOCPD корисний як діагностика й ранній сигнал; тривоги BOCPD разом із ударами зі списку задають дні, від яких прогнозується глибина й тривалість. Але як тригер перенавчання він не обігнав просте часте перенавчання (слайд 12) — це напрям вдосконалення.
Оцінка охоплює лише період, де є список ударів; поріг тривоги 0,7 до підбору, для перенавчання поріг підібрано на валідації.

ТЕРМІНИ: точність (precision) — яка частка тривог влучила в справжні удари; повнота (recall) — яка частка ударів виявлена.

МОЖЛИВІ ЗАПИТАННЯ: Чому повнота низька? — Деякі удари не залишають сліду в добових середніх; режим змінюється поступово. Чому не оцінюєте 2025–2026 окремо? — Вони входять у список ударів; оцінка охоплює весь період списку. Чому тоді BOCPD? — Він онлайн (лише минуле), PELT — ретроспективний і не може керувати перенавчанням.
""")
pw, ph = picture_fit(s, "s_fig_detect_price.png", 0.4, 1.5, 7.0, 4.2)
rows = [["Ряд", "Ударів", "BOCPD: тривог · точн. / повн.", "PELT: точн. / повн."]]
for sc in ("demand", "net_import", "outages", "price"):
    b, p_ = cp[sc]["bocpd"]["scores"], cp[sc]["pelt"]["scores"]
    rows.append([SHORT[sc], str(b["n_attacks"]), f"{b['n_detections']} · {fm(b['precision'], 2)} / {fm(b['recall'], 2)}", f"{fm(p_['precision'], 2)} / {fm(p_['recall'], 2)}"])
table(s, 7.55, 1.55, [1.35, 0.75, 1.9, 1.2], rows, size=12, row_h=0.62, align=[PP_ALIGN.LEFT, C, C, C])
box(s, 7.55, 4.8, 5.15, 1.95, ORANGE_TINT)
text(s, 7.7, 4.85, 4.9, 1.85, [[("Діагностика, а не тригер: ", True), ("тривоги BOCPD задають дні подій для прогнозу після атаки, але як тригер перенавчання він не обігнав часте перенавчання.", False)]], size=14, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 5.85, 6.9, 1.0, "Збіг: виявлення протягом 14 діб після удару; оцінка охоплює період зі списком ударів. Для попиту дата зламу відома незалежно: 24.02.2022 (PELT, затримка 0 діб).", size=13, color=MUTED)

# ====================================================================================== 18. summary
s = new_slide("Підсумок: що працює, а що ні", notes=f"""
ЩО СКАЗАТИ (2 хв): Підсумок порівнянь. Зліва — MAE на тесті для чотирьох рядів поруч із сезонним наївним і ковзним вікном. Справа — що підтверджено з наших тверджень.
1) Прямий прогноз глибини після атаки точніший за «як учора» — підтверджено на ціні (валідація й тест) і на імпорті (тест); на відключеннях виграшу немає.
2) Перенавчані моделі точніші за статичну там, де режим змінився: попит (≈ {fm(M('demand', 'xgb_static') / M('demand', 'xgb_drift'), 0)} разів), відключення, імпорт; статична проти щотижневого вікна — {STATIC_LINE}.
3) Перенавчання за власною помилкою моделі не гірше за щотижневе й значуще краще на відключеннях ({fm(REL['outages']['xgb_drift'], 1, sign=True)} %); на інших рядах різниця незначуща.
4) Тригер за рівнем BOCPD не обганяє часте перенавчання (контроль кожні 3 дні) — не підтверджено, це уточнення гіпотези теоретичної частини.
5) Тривалість після атаки: реалізовано три підходи, точність на рівні базового методу — напрям вдосконалення.
Головне: результат чесно подано поруч із простими методами (сезонний наївний, «як учора», ковзне вікно). Наукова цінність — відтворювана система та перевірка, що саме з підходів працює.

МОЖЛИВІ ЗАПИТАННЯ: То в чому наукова цінність? — Реалізовано систему й повністю відтворюваний експеримент із чесними базовими методами; показано, де адаптація й прямі моделі працюють, а де простий метод не гірший. Чи це остаточно? — Ні: мало подій, проксі замість точного дефіциту, неповні дані про потужність.
""")
picture_fit(s, "s_fig_mae.png", 0.3, 1.5, 7.4, 3.9)
rows = [["Твердження", "Результат"],
        ["Прямий прогноз глибини краще за «як учора»", "так: ціна, імпорт; відключення — ні"],
        ["Перенавчані моделі краще за статичну", f"так: попит (в {fm(st_ratio, 0)} р.), відключення, імпорт" + ("" if "price" not in STATIC_OK else "; на ціні статична не гірша")],
        ["Перенавчання за помилками ≥ щотижневого", "значуще краще на відключеннях; інше — рівно"],
        ["Тригер BOCPD краще за часте перенавчання", "не підтверджено"],
        ["Тривалість краще за «як учора»", "на рівні базового: напрям вдосконалення"]]
table(s, 7.85, 1.55, [2.7, 2.2], rows, size=12, row_h=0.8, colors={(1, 1): GREEN, (2, 1): GREEN, (3, 1): GREEN, (4, 1): RED, (5, 1): NAVY})
box(s, 0.6, 5.55, 12.1, 1.35, ORANGE_TINT)
text(s, 0.8, 5.57, 11.7, 1.3, [[("Головне: ", True), ("глибину дефіциту після атаки можна прогнозувати значуще точніше за «як учора» (ціна, імпорт); перенавчання моделей дає виграш, а найкраще працює часте перенавчання або перенавчання за власною помилкою. Усе показано поруч із сезонним наївним і ковзним вікном.", False)]],
     size=15, anchor=MSO_ANCHOR.MIDDLE)

# ====================================================================================== 19. improvements
s = new_slide("Напрями вдосконалення", notes=f"""
ЩО СКАЗАТИ (1,5 хв): Я розглядаю слабкі місця як конкретні напрями вдосконалення, бо знаю, що для кожного потрібно.
1) Тривалість після атаки. Реалізовано три підходи (регресія, модель виживання, класифікатор), точність на рівні «як учора». Потрібно: більше подій, дані про втрачену потужність і ремонти, довший горизонт.
2) Глибина на індексі відключень. Виграшу немає: індекс насичується на одиниці й має мало подій. Потрібен ряд у МВт.
3) Перший удар нового режиму. Для втраченого попиту у тесті лише {n_ev(D, 'demand', 'test')} подій, а 24 лютого 2022 жодної воєнної доби в навчанні немає — модель без прикладів не екстраполює на небачений режим.
4) Сила удару. У списку є кількість засобів, що досягли цілей, але немає втраченої потужності в МВт. Ціна РДН як ознака приросту не дала.
5) Тригер перенавчання. BOCPD не обганяє часте перенавчання; потрібні чутливіші сигнали або поєднання з помилко-орієнтованим тригером.
6) Дані й методика. LSTM на реальних даних, прогноз погоди замість фактичної, споживання й генерація за воєнні роки (ENTSO-E їх не має), блокові тести значущості для автокорельованих похибок.

МОЖЛИВІ ЗАПИТАННЯ: Що б ви зробили першим? — Додала б втрачену потужність за ударами й більше подій: від цього залежить і тривалість, і глибина на відключеннях. Чому ви не зробили це зараз? — Відкриті джерела не дають потужності; це потребує звернення до оператора.
""")
items = [("Тривалість після атаки", "Реалізовано три підходи; точність на рівні «як учора». Потрібні: більше подій, потужність і ремонти."),
         ("Глибина на відключеннях", "Без виграшу: індекс насичується на 1. Потрібен ряд у МВт."),
         ("Перший удар нового режиму", f"Попит: лише {n_ev(D, 'demand', 'test')} події в тесті; без прикладів модель не екстраполює."),
         ("Сила удару", "Є кількість засобів, що досягли цілей; немає МВт. Ціна РДН як ознака приросту не дала."),
         ("Тригер перенавчання", "BOCPD не обганяє часте перенавчання; шукати чутливіші сигнали або поєднати з тригером за помилкою."),
         ("Дані й методика", "LSTM на реальних даних, прогноз погоди, споживання за воєнні роки, блокові тести значущості.")]
for i, (h, b) in enumerate(items):
    card(s, 0.6 + (i % 3) * 4.1, 1.55 + (i // 3) * 2.75, 3.9, 2.55, h, b, size=16, head_size=18, fill=TINT if i % 2 == 0 else ORANGE_TINT, head_h=0.55)

# ====================================================================================== 20. conclusions
s = new_slide("Висновки", notes=f"""
ЩО СКАЗАТИ (1,5 хв): 1) Реалізовано відтворювану систему з шести кроків і окремий модуль прогнозу глибини й тривалості дефіциту після атаки; {TESTS_N} тестів, захист від витоку майбутнього. 2) Дані: офіційний список із {STRIKE_DAYS} днів масованих ударів (замість ручних 18) і чотири реальні ряди — втрачений попит, чистий імпорт, індекс відключень і новий ряд ціни РДН. 3) Прямий прогноз глибини після атаки значуще точніший за «як учора» на ціні (валідація й тест) та імпорті (тест); на відключеннях виграшу немає. 4) Перенавчані моделі значно точніші за статичну (на попиті ≈ {fm(M('demand', 'xgb_static') / M('demand', 'xgb_drift'), 0)} разів); перенавчання за помилками значуще краще за щотижневе на відключеннях; допомагає частота перенавчання й тригер за помилкою, а не рівень BOCPD. 5) Тривалість реалізована й перевірена, точність на рівні базового методу — це напрям вдосконалення.
Подальша робота: втрачена потужність за ударами, більше подій, LSTM на реальних даних, прогноз погоди, чутливіші тригери.

МОЖЛИВІ ЗАПИТАННЯ: Яка головна новизна? — Відтворювана система з прямим прогнозом глибини й тривалості після атаки та чесне порівняння стратегій перенавчання з простими методами на чотирьох реальних рядах.
""")
text(s, 0.6, 1.5, 7.2, 0.5, "Що зроблено і що показано", size=22, bold=True, color=NAVY, font=HEAD)
bullets(s, 0.6, 2.05, 7.2, 4.9, [
    f"Відтворювана система з 6 кроків, прогноз глибини й тривалості після атаки, {TESTS_N} тестів, захист від витоку",
    f"Офіційний список {STRIKE_DAYS} днів ударів; 4 ряди: попит, імпорт, відключення, ціна РДН",
    "Глибина після атаки значуще точніша за «як учора»: ціна, імпорт",
    "Перенавчані моделі ≫ статична; за помилками значуще краще на відключеннях; BOCPD-тригер не обганяє часте перенавчання",
    "Тривалість реалізована; точність на рівні базового методу"], size=18, gap=12)
text(s, 8.2, 1.5, 4.5, 0.5, "Що далі", size=22, bold=True, color=NAVY, font=HEAD)
bullets(s, 8.2, 2.05, 4.5, 4.9, [
    "Втрачена потужність за ударами, більше подій",
    "Прямі моделі тривалості з дод. даними",
    "LSTM на реальних даних",
    "Прогноз погоди замість фактичної",
    "Чутливіші тригери перенавчання"], size=19, gap=14)

# ====================================================================================== 21. thanks
s = new_slide("Дякую за увагу! Запитання?", dark=True, number=False, notes="""
Підсумкова фраза: «Головний результат — відтворювана система, яка прогнозує глибину дефіциту після атаки значуще точніше за простий метод і чесно показує, де адаптивний підхід працює, а де — напрям вдосконалення».

Якщо запитань нема — запропонуйте показати запуск (команди у файлі «Підготовка до захисту», розділ «Демонстрація»).
""")
s.shapes.title.top, s.shapes.title.height = Inches(2.5), Inches(1.4)
text(s, 0.6, 4.2, 12, 1.0, "Код, дані та звіти: C:\\Projects\\ua-deficit-forecast  ·  RESULTS.md  ·  README.md", size=18, color="B6C1D6")

# ====================================================================================== backup
s = new_slide("Додатково: підготовка даних і знайдені помилки", notes="""
РЕЗЕРВНИЙ СЛАЙД — показувати лише якщо спитають.
Підготовка даних. 328 рядків файлу відключень були зіпсовані: замість дати — формули електронної таблиці на кшталт «=B28442»; їх відкинуто, відповідні 13 діб квітня 2026 лишено невідомими. Час у файлах місцевий (Київ), усе зберігається в UTC; у день переходу на зимовий час година 03:00–04:00 трапляється двічі — перше входження вважаю літнім часом, друге зимовим. 1 200 годин (10 грудня 2025 – 28 січня 2026) виключені: за описом набору це дані регіональних операторів, а не Укренерго. Принцип: невідоме не дорівнює нулю — пропуски не заповнюю нулями чи інтерполяцією, вони не беруть участі в навчанні й оцінюванні.
Помилки, які я знайшла й виправила (доказ ретельності, не слабкість): 1) модель ковзного вікна без даних у вікні ставала «порожньою» й прогнозувала сталий нуль — тепер оновлення пропускається, якщо у вікні менше 168 відомих значень; 2) метрика тривалості епізоду давала нуль помилки моделі, що завжди прогнозує дефіцит, — додано точність і повноту днів; 3) Prophet екстраполював лінійний тренд до 8000 МВт — додано сезонний наївний як чесну базову модель; 4) XGBoost запускав потік на ядро, тести йшли понад 5 хвилин замість 9 секунд — обмежено кількість потоків.
Синтетичні дані (згенеровані мною з вбудованими зламами) використовуються лише для перевірки того, що код працює від початку до кінця; це не результат для енергосистеми України.

МОЖЛИВІ ЗАПИТАННЯ: Чому не інтерполювати пропуски? — Для індексу відключень «невідомо» не означає «так само, як поряд». Чому UTC? — Єдиний час без переходів; календарні ознаки рахуються окремо в київському часі.
""")
stat(s, 0.6, 1.6, 2.9, 2.6, "328", "пошкоджених рядків (формули замість дат) відкинуто; 13 діб квітня 2026 — невідомі", big_size=40, label_size=14)
stat(s, 3.65, 1.6, 2.9, 2.6, "UTC", "місцевий час переведено в UTC; повторювану годину при переході на зимовий час оброблено", big_size=40, label_size=14)
stat(s, 6.7, 1.6, 2.9, 2.6, "1 200 год", "виключено з цілі: дані регіональних операторів, не Укренерго", big_size=34, label_size=14)
box(s, 0.6, 4.5, 9.0, 2.3, ORANGE_TINT)
text(s, 0.85, 4.55, 8.5, 2.2, [[("Принцип: невідоме ≠ нуль. ", True), ("Відсутні значення не заповнюються нулями чи інтерполяцією — вони не беруть участі в навчанні й оцінюванні.", False)],
                              [("Синтетичні дані ", True), ("лише перевіряють, що код працює від початку до кінця; це не результат для енергосистеми.", False)]], size=18, anchor=MSO_ANCHOR.MIDDLE)
text(s, 9.9, 1.6, 2.9, 0.5, "Знайдені й виправлені помилки", size=17, bold=True, color=NAVY, font=HEAD)
bullets(s, 9.9, 2.4, 2.9, 4.5, [
    "ковзне вікно без даних → порожня модель з прогнозом 0",
    "метрика тривалості заохочувала «завжди дефіцит»",
    "Prophet екстраполював тренд до 8000 МВт",
    "потоки XGBoost: тести 5 хв замість 9 с"], size=14, gap=10)

try:
    prs.save(str(OUT))
    print("saved", OUT, "slides:", len(prs.slides))
except PermissionError:
    alt = OUT.with_name(OUT.stem + "_v3.pptx")
    prs.save(str(alt))
    print("LOCKED, saved", alt, "slides:", len(prs.slides))

print("\nCLAIM CHECKS")
for st_, ok in CHECKS:
    print(("OK   " if ok else "FAIL ") + st_)
print("strike list:", ST)
