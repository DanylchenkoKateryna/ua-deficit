"""Text of chapter 3, general conclusions and the reference list. All numbers are read from the result files.

Names used by other scripts (build_pptx.py, guide_content.py): load_results, val, ep, pval, pct, pf, po, ORDER, LABEL.
"""
import json
from pathlib import Path

import pandas as pd

from content_ch12 import N_TESTS, N_TESTS_RUN
from docx_helpers import fm
from res_helpers import NAME, NAME_POST, SCENARIOS, cap, read_csv_or_none, vs_reference

ORDER = ["seasonal_naive", "prophet_static", "xgb_static", "xgb_rolling", "xgb_adaptive", "xgb_adaptive_tau_w"]
LABEL = {k: cap(v) for k, v in NAME.items()}          # model names in tables and legends (the identifiers stay in the code)
ORDER_ALL = ["seasonal_naive", "prophet_static", "xgb_static", "xgb_rolling", "xgb_rolling_freq", "xgb_adaptive",
             "xgb_adaptive_tau_w", "xgb_attack", "xgb_drift"]
SCN = {"demand": "Втрачений попит", "net_import": "Чистий імпорт", "outages": "Індекс відключень", "price": "Ціна РДН"}
SCNL = {"demand": "втрачений попит", "net_import": "чистий імпорт", "outages": "індекс відключень", "price": "ціна РДН"}
UNIT = {"demand": "МВт", "net_import": "МВт", "outages": "частка областей", "price": "частка стелі цін"}
DIG = {"demand": 0, "net_import": 1, "outages": 3, "price": 3}
DIG_DEPTH = {"demand": 0, "net_import": 0, "outages": 3, "price": 3}
PM = ["persistence", "baseline28", "xgb_static", "xgb_expanding", "xgb_rolling"]
PLABEL = {k: cap(v) for k, v in NAME_POST.items()}    # post-attack methods
SEC = {"s_meth": "3.1", "s_depth": "3.2", "s_dur": "3.3", "s_adapt": "3.4", "s_trig": "3.5", "s_det": "3.6", "dir": "3.9"}   # section numbers used in cross-references
CHECKS: list[str] = []     # narrative claims that the data do not support are collected here and printed by build_docx.py


def chk(cond, msg):
    if not cond:
        CHECKS.append(msg)


def load_results(root: Path) -> dict:
    R = {"root": root}
    for sc in SCENARIOS:
        rep, art = root / "reports" / sc, root / "artifacts" / sc
        fdir = art / "forecasts"
        R[sc] = dict(post=read_csv_or_none(rep / "post_event_summary.csv"), post_events=read_csv_or_none(rep / "post_event.csv"),
                     post_tests=read_csv_or_none(rep / "post_event_tests.csv"), post_paths=read_csv_or_none(rep / "post_event_paths.csv"),
                     m=pd.read_csv(rep / "metrics.csv"), w=pd.read_csv(rep / "wilcoxon.csv"),
                     e=pd.read_csv(rep / "episodes_summary.csv"),
                     cp=json.loads((art / "changepoints.json").read_text(encoding="utf-8")),
                     tuned=json.loads((art / "tuned.json").read_text(encoding="utf-8")),
                     refits={p.name[:-len("_refits.csv")]: pd.read_csv(p) for p in sorted(fdir.glob("*_refits.csv"))},
                     vs_roll=vs_reference(root, sc, "xgb_rolling", "test"),
                     vs_roll_val=vs_reference(root, sc, "xgb_rolling", "validation"))
        tt = art / "tuned_triggers.json"
        R[sc]["trig"] = json.loads(tt.read_text(encoding="utf-8")) if tt.exists() else {}
    syn = root / "reports" / "synthetic"
    R["syn"] = dict(m=pd.read_csv(syn / "metrics.csv"),
                    cp=json.loads((root / "artifacts" / "synthetic" / "changepoints.json").read_text(encoding="utf-8")))
    R["data"] = _data_facts(root)
    return R


def _data_facts(root: Path) -> dict:
    """Facts about the strike list and the day-ahead market, read from the data files."""
    out = {}
    a = pd.read_csv(root / "data" / "external" / "attacks.csv")
    s = a[a["date"] >= "2022-09-01"]
    out.update(strikes_n=len(s), strikes_first=s["date"].min(), strikes_last=s["date"].max(),
               fired=int(s["weapons_fired"].sum()), through=int(s["weapons_through"].sum()), attack_dates=list(a["date"]))
    p = root / "data" / "raw" / "dam_hourly.parquet"
    if p.exists():
        r = pd.read_parquet(p)["dam_price_ratio"].resample("MS").mean()
        w = r["2022-10-01":"2023-03-31"]
        out.update(dam_win_lo=float(w.min()), dam_win_hi=float(w.max()))
    return out


def val(R, sc, model, split="test", subset="all", col="MAE"):
    d = R[sc]["m"]
    return float(d[(d["model"] == model) & (d["split"] == split) & (d["subset"] == subset)][col].iloc[0])


def ep(R, sc, model, split, col):
    d = R[sc]["e"]
    return float(d[(d["model"] == model) & (d["split"] == split)][col].iloc[0])


def pval(R, sc, model, split="test", subset="all"):
    d = R[sc]["w"]
    return float(d[(d["model"] == model) & (d["split"] == split) & (d["subset"] == subset)]["p_value"].iloc[0])


def pct(a, b):
    """Relative change of a with respect to b, in percent."""
    return 100.0 * (a - b) / b


def pf(p):
    return "< 0,001" if p < 0.001 else fm(p, 3)


def pp(p):
    """'p < 0,001' or 'p = 0,023' (two decimals from 0,1 up)."""
    if p != p:
        return "прогнози збігаються"
    return "p < 0,001" if p < 0.001 else "p = " + fm(p, 3 if p < 0.1 else 2)


def po(R, sc, m, col, trig="all", split="test"):
    """Post-attack summary value; split 'test' (default), 'validation' or 'all'."""
    d = R[sc]["post"]
    r = d[(d["trigger"] == trig) & (d["method"] == m) & (d["split"] == split)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def pt(R, sc, m, metric, split="test", col="p_value"):
    """Paired test of a post-attack method against persistence (nan if not available, e.g. fewer than 8 events)."""
    t = R[sc]["post_tests"]
    if t is None:
        return float("nan")
    r = t[(t["method"] == m) & (t["metric"] == metric) & (t["split"] == split)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def trv(n):
    return "тривогу" if n % 10 == 1 and n % 100 != 11 else "тривоги" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "тривог"


def pere(n):
    """'перенавчання' after 'з N' (genitive): singular after numbers ending in 1 (not 11), plural otherwise."""
    return "перенавчання" if n % 10 == 1 and n % 100 != 11 else "перенавчань"


def raz(n):
    return "раз" if n % 10 == 1 and n % 100 != 11 else "рази" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "разів"


def lohi(a, b, d):
    lo, hi = sorted((a, b))
    return f"{fm(lo, d)}–{fm(hi, d)}"


def n_events(R, sc, split="test"):
    d = R[sc]["post"]
    return int(d[(d["trigger"] == "all") & (d["split"] == split)]["n_events"].iloc[0])


def vroll(R, sc, m, split="test"):
    """(relative change of the MAE with respect to xgb_rolling in %, p) from the paired daily-MAE Wilcoxon test."""
    d = R[sc]["vs_roll" if split == "test" else "vs_roll_val"]
    r = d[d["model"] == m]
    return (float(r["rel_pct"].iloc[0]), float(r["p_value"].iloc[0])) if len(r) else (float("nan"), float("nan"))


def vtxt(R, sc, m, split="test"):
    rel, p = vroll(R, sc, m, split)
    return f"{fm(rel, 1, sign=True)} %, {pp(p)}"


def nref(R, sc, m):
    """(number of refits, number triggered by a signal) of a model."""
    d = R[sc]["refits"].get(m)
    if d is None:
        return 0, 0
    return len(d), int((d["reason"] != "scheduled").sum())


# --------------------------------------------------------------------------------------------- table rows
def _metrics_rows(R, sc, d_mae, d_rmse, d_cr, crisis=True):
    """Model rows of the day-ahead tables (ORDER); kept for the other documents."""
    rows, cols = [], {"MAE": [], "RMSE": [], "CR": [], "VAL": []}
    for m in ORDER:
        cols["MAE"].append(val(R, sc, m)); cols["RMSE"].append(val(R, sc, m, col="RMSE"))
        cols["CR"].append(val(R, sc, m, subset="crisis_14d")); cols["VAL"].append(val(R, sc, m, split="validation"))
    best = {k: min(v for v in vs if v == v) if any(v == v for v in vs) else None for k, vs in cols.items()}

    def cell(k, i, d):
        v = cols[k][i]
        if v != v:
            return "–"
        s = fm(v, d)
        return f"**{s}**" if best[k] is not None and v == best[k] else s
    for i, m in enumerate(ORDER):
        row = [LABEL[m], cell("MAE", i, d_mae), cell("RMSE", i, d_rmse)]
        if crisis:
            row.append(cell("CR", i, d_cr))
        rows.append(row + [cell("VAL", i, d_mae)])
    return rows


def _adapt_rows(R, sc):
    """Rows of the day-ahead table of one scenario: MAE test, RMSE test, MAE 14 days after strikes, MAE validation, change vs xgb_rolling."""
    d = DIG[sc]
    models = [m for m in ORDER_ALL if ((R[sc]["m"]["model"] == m).any())]
    cols = {k: {m: val(R, sc, m, **kw) for m in models} for k, kw in
            (("MAE", {}), ("RMSE", dict(col="RMSE")), ("CR", dict(subset="crisis_14d")), ("VAL", dict(split="validation")))}
    best = {k: min(v for v in c.values() if v == v) for k, c in cols.items()}
    rows = []
    for m in models:
        row = [LABEL[m]]
        for k in ("MAE", "RMSE", "CR", "VAL"):
            v = cols[k][m]
            s = "–" if v != v else fm(v, d)
            row.append(f"**{s}**" if v == best[k] else s)
        if m == "xgb_rolling":
            row.append("еталон")
        else:
            rel, p = vroll(R, sc, m)
            row.append("–" if rel != rel else fm(rel, 1, sign=True) + " %, " + pp(p))
        rows.append(row)
    return rows


PD_COLS = ["persistence", "baseline28", "xgb_static", "xgb_rolling", "ridge_depth", "xgb_direct"]
PD_SC = ("price", "net_import", "outages", "demand")


def vs_pers(R, sc, m, metric, split="test"):
    """Plain-words comparison of a post-attack method with persistence (paired test): better / worse (p) or no significant difference."""
    p = pt(R, sc, m, metric, split)
    if p != p:
        n = n_events(R, sc, split)
        return f"не оцінювалось (лише {n} подій)" if n < 8 else "не оцінено"
    if p >= 0.05:
        return "без значущої різниці"
    return ("краще " if pt(R, sc, m, metric, split, col="mean_error_minus_persistence") < 0 else "гірше ") + "(" + pp(p).replace(" ", " ") + ")"


def _depth_rows(R, split="test"):
    """Wide table: MAE of the depth forecast of every method; the last column compares ridge_depth with persistence in words."""
    rows = []
    for sc in PD_SC:
        vals = [po(R, sc, m, "depth_MAE", split=split) for m in PD_COLS]
        cells = [fm(v, DIG_DEPTH[sc]) for v in vals]
        rows.append([f"{SCN[sc]} ({n_events(R, sc, split)})"] + cells + [vs_pers(R, sc, "ridge_depth", "depth", split)])
    return rows


def _signif_rows(R):
    rows = []
    for sc in PD_SC:
        for split, lab in (("test", "тест"), ("validation", "валідація")):
            base = po(R, sc, "persistence", "depth_MAE", split=split)
            row = [SCN[sc], lab, str(n_events(R, sc, split))]
            for m in ("ridge_depth", "xgb_direct"):
                v = po(R, sc, m, "depth_MAE", split=split)
                p = pt(R, sc, m, "depth", split)
                row.append(fm(pct(v, base), 0, sign=True) + " %" + (", " + pp(p) if p == p else ", не оцінювалось (замало подій)"))
            rows.append(row)
    return rows


DU_COLS = ["persistence", "baseline28", "xgb_rolling", "xgb_direct", "xgb_aft", "xgb_cls"]


def _duration_rows(R, split="test"):
    """Wide table: one row per series, MAE of the duration forecast of every method (best value in bold)."""
    rows = []
    for sc in PD_SC:
        vals = [po(R, sc, m, "duration_MAE_days", split=split) for m in DU_COLS]
        cells = [("–" if v != v else fm(v, 1)) for v in vals]
        rows.append([f"{SCN[sc]} ({n_events(R, sc, split)})"] + cells)
    return rows


def _post_rows(R, sc, d_dep, d_traj):
    rows = []
    for m in PM:
        rows.append([PLABEL[m], fm(po(R, sc, m, "depth_MAE"), d_dep), fm(po(R, sc, m, "duration_MAE_days"), 1),
                     fm(po(R, sc, m, "trajectory_MAE"), d_traj)])
    return rows


def hit(s):
    return f"{round(s['precision'] * s['n_detections'])} з {s['n_detections']}"


# --------------------------------------------------------------------------------------------- chapter 3
def _dayahead_rows(R):
    """Compact table: rows = models, columns = the four scenarios, MAE on the test period (best value of a column in bold)."""
    scs = ("demand", "net_import", "outages", "price")
    cols = {}
    for sc in scs:
        d = {}
        for m in ORDER_ALL:
            try:
                d[m] = val(R, sc, m)
            except IndexError:
                d[m] = float("nan")
        cols[sc] = d
    rows = []
    for m in ORDER_ALL:
        row = [LABEL[m]]
        for sc in scs:
            v = cols[sc][m]
            row.append("–" if v != v else fm(v, DIG[sc]))
        rows.append(row)
    return rows


def ch3(b, R, X):
    ni, ou, de, pr, sy = R["net_import"], R["outages"], R["demand"], R["price"], R["syn"]

    def r(k):
        return SEC.get(k) or X.get(k, "?.?")

    def g(sc, m, **kw):
        return val(R, sc, m, **kw)

    ap = lambda x, d=0: fm(abs(x), d)          # size of a percentage change, without the sign

    CHECKS.clear()
    b.h1("РОЗДІЛ 3. РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТІВ", chapter=True)

    # ------------------------------------------------------------------ overview
    pd_p = {sc: (po(R, sc, "persistence", "depth_MAE"), po(R, sc, "ridge_depth", "depth_MAE"), po(R, sc, "xgb_direct", "depth_MAE")) for sc in PD_SC}
    b.para("**Основні результати розділу.** По-перше, після удару пряма модель глибини дефіциту точніша за персистентність. "
           "Персистентність – це найпростіший прогноз: рівень лишається таким, як в останній відомий день. "
           f"На ціні РДН похибка прямої моделі менша на {ap(pct(pd_p['price'][1], pd_p['price'][0]))} %, "
           f"на чистому імпорті – на {ap(pct(pd_p['net_import'][1], pd_p['net_import'][0]))} %. "
           "По-друге, моделі, які перенавчають на свіжих даних, значно точніші за статичні там, де режим змінився. "
           f"Найкраще це видно на втраченому попиті: похибка (MAE) {fm(g('demand', 'xgb_rolling'), 0)} проти "
           f"{fm(g('demand', 'xgb_static'), 0)} МВт. На ціні РДН статична модель не гірша. "
           "По-третє, перенавчання за зростанням власної похибки моделі на індексі відключень значуще точніше за щотижневе "
           f"(на {ap(vroll(R, 'outages', 'xgb_drift')[0], 1)} %; що означає «значуще», пояснено в п. " + r("s_meth") + "). "
           "На інших рядах воно не гірше. Прогноз тривалості й інші напрями, де виграшу не отримано, описано в п. " + r("dir") +
           ".")

    # ------------------------------------------------------------------ 3.1
    b.h2("3.1. Методика оцінювання")
    b.para("Якість прогнозу вимірюють похибкою. Основна метрика – **MAE**, середня абсолютна похибка. Це середнє відхилення "
           "прогнозу від факту без урахування знака. MAE виражено в одиницях самого ряду, тому її легко читати. Відсоткову "
           "похибку (MAPE) не використано: дефіцит перетинає нуль, і вона втрачає сенс. Окремо оцінено «кризові тижні» – перші 14 діб після кожного масованого удару. "
           "Відсоткова різниця в розділі – це зміна похибки відносно еталона. Знак «−» означає, що похибка менша, тобто модель точніша.")
    b.para("Щоб зрозуміти, чи різниця між двома моделями не випадкова, використано парний **тест Вілкоксона**. Він порівнює "
           "похибки двох моделей за одні й ті самі доби. Результат тесту – **p-значення**: це ймовірність випадково отримати таку "
           "різницю, якщо насправді моделі однакові. Якщо p менше за 0,05, різницю вважають значущою. Тест застосовано до добових MAE. "
           "Погодинні похибки автокорельовані (залежать одна від одної), тому p-значення слід вважати оптимістичними, "
           "тобто кращими, ніж є насправді.")
    b.para("Усі результати подано поруч із двома орієнтирами. Перший – сезонний наївний метод: прогноз дорівнює значенню тієї самої години "
           "тиждень тому. Другий – контрольне щотижневе перенавчання. Перенавчання – це повторне навчання моделі "
           "на свіжих даних. У контрольній моделі його роблять щотижня, без жодних сигналів. Тому вона показує, чи дає щось сам "
           "сигнал перенавчання.")
    b.para("Експерименти бувають двох типів. Перший – **прогноз глибини й тривалості дефіциту після удару** (пп. " + r("s_depth") + "–" + r("s_dur") + "). "
           "У день видачі d₀ метод бачить лише попередню історію й прогнозує 30 наступних діб. Прогнозна глибина – це найбільше "
           "добове значення. Тривалість – кількість діб до моменту, коли дефіцит не перевищує порога три доби поспіль. "
           "Події до кінця валідації належать до валідації, решта – до тесту.")
    b.para("Другий тип – **прогноз на добу наперед** (пп. " + r("s_adapt") + "–" + r("s_det") + "). Його випускають за 24 години до кожної "
           "години тестового періоду. Він потрібен, щоб порівняти політики перенавчання, тобто правила, коли модель навчати заново. "
           "Якість виявлення зламів оцінено двома показниками відносно переліку ударів. Збіг – це тривога протягом 14 діб після удару зі списку. "
           "**Точність** – частка тривог, які збіглися з ударом. **Повнота** – частка ударів, на які була тривога.")
    b.para("Усі параметри підбирали лише на валідації. Тест для підбору не використовували. Проте деякі рішення ухвалено після "
           "перегляду результатів на тесті. Це контрольні моделі, прямі методи глибини, тригери за зростанням похибки та після ударів, "
           "обмеження ознаки «години від останнього удару» й основний варіант навчання з оціненого моменту зламу. Тому ці рішення "
           "не є повністю незалежними від тесту. Водночас параметри тригерів вибрано на валідації, а прямі методи мають однакові "
           "параметри для всіх рядів.")

    # ------------------------------------------------------------------ 3.2 depth
    b.h2("3.2. Прогноз глибини дефіциту після удару")
    b.para("Це основний результат роботи. У момент удару модель називає, наскільки високим стане дефіцит у найближчі 30 діб. "
           "Повний опис методів наведено в таблиці " + r("t_post") + ". Коротко: «середнє за 28 діб» – це середнє за попередні 28 діб. "
           "XGBoost – метод машинного навчання на основі ансамблю рішальних дерев; статичний навчено один раз, а «ковзне вікно» "
           "перенавчено в день видачі на останніх 180 добах. регресія Ridge – проста лінійна модель на рівнях і максимумах "
           "останніх тижнів. «Ridge + XGBoost» – це регресія Ridge плюс дерева XGBoost, які виправляють залишок її похибки. "
           "Таблиця " + r("post_depth") + " показує похибку глибини на тесті (MAE в одиницях ряду: МВт для втраченого попиту й чистого імпорту, "
           "частка для інших рядів; у першому стовпці в дужках – кількість ударів). В останньому стовпці регресію Ridge порівняно з персистентністю "
           "за тестом Вілкоксона.")
    b.table("Похибка прогнозу глибини дефіциту на 30 діб після удару, тест",
            ["Ряд (кількість ударів)", "Без змін", "Середнє за 28 діб", *[PLABEL[m] for m in PD_COLS[2:]], "Регресія Ridge проти персистентності"],
            _depth_rows(R), [2.6, 1.3, 1.9, 2.0, 2.1, 1.8, 1.8, 3.0], align=["l"] + ["c"] * 6 + ["l"], key="post_depth")
    b.figure("docs/figures/fig_post_depth.png", "Похибка прогнозу глибини після удару на тесті: персистентність, XGBoost з перенавчанням у день видачі та прямі моделі глибини", 13.5, key="post_depth_fig")
    pp_ = {sc: (po(R, sc, "persistence", "depth_MAE"), po(R, sc, "ridge_depth", "depth_MAE"), po(R, sc, "xgb_direct", "depth_MAE"),
                po(R, sc, "xgb_rolling", "depth_MAE"), po(R, sc, "xgb_static", "depth_MAE")) for sc in PD_SC}
    P, I = pp_["price"], pp_["net_import"]
    chk(pt(R, "price", "ridge_depth", "depth") < 0.05 and pt(R, "price", "ridge_depth", "depth", "validation") < 0.05, "price: ridge_depth not significant on val and test")
    chk(pt(R, "price", "xgb_direct", "depth") < 0.05, "price: xgb_direct not significant on test")
    chk(all(pt(R, "price", m, "depth") < 0.05 and pt(R, "price", m, "depth", col="mean_error_minus_persistence") < 0 for m in ("xgb_static", "xgb_rolling", "xgb_direct"))
        and pt(R, "price", "baseline28", "depth") >= 0.05, "price: depth significance pattern changed")
    chk(pt(R, "net_import", "xgb_rolling", "depth") < 0.05 and pt(R, "net_import", "xgb_rolling", "depth", col="mean_error_minus_persistence") < 0
        and pt(R, "net_import", "xgb_static", "depth", col="mean_error_minus_persistence") > 0 and pt(R, "net_import", "xgb_static", "depth") < 0.05
        and pt(R, "net_import", "baseline28", "depth") >= 0.05, "net_import: depth significance pattern changed")
    chk(pt(R, "outages", "xgb_static", "depth", col="mean_error_minus_persistence") > 0 and pt(R, "outages", "xgb_static", "depth") < 0.05
        and all(pt(R, "outages", m, "depth") >= 0.05 for m in ("baseline28", "xgb_rolling", "ridge_depth", "xgb_direct")), "outages: depth significance pattern changed")
    b.para(f"**Ціна РДН.** Обидві прямі моделі значуще точніші за персистентність. Похибка зменшилась з {fm(P[0], 3)} до {fm(P[1], 3)} "
           f"у регресії Ridge, тобто на {ap(pct(P[1], P[0]))} % ({pp(pt(R, 'price', 'ridge_depth', 'depth'))}). "
           f"Регресія Ridge точніша за персистентність у {fm(100 * pt(R, 'price', 'ridge_depth', 'depth', col='share_better'), 0)} % подій. "
           f"Така сама перевага є й на валідації ({fm(po(R, 'price', 'ridge_depth', 'depth_MAE', split='validation'), 3)} проти "
           f"{fm(po(R, 'price', 'persistence', 'depth_MAE', split='validation'), 3)}, {pp(pt(R, 'price', 'ridge_depth', 'depth', 'validation'))}), "
           "тож результат стійкий між періодами. Статичний XGBoost і XGBoost з ковзним вікном теж значуще точніші, а середнє за 28 діб значущої різниці не дає.")
    chk(pt(R, "net_import", "ridge_depth", "depth") < 0.05 and pt(R, "net_import", "xgb_direct", "depth") < 0.05, "net_import: direct not significant on test")
    b.para(f"**Чистий імпорт.** На тесті обидві прямі моделі значуще точніші за персистентність. Для регресії Ridge похибка менша на "
           f"{ap(pct(I[1], I[0]))} % ({fm(I[1], 0)} проти {fm(I[0], 0)} МВт). Ковзне вікно теж значуще краще, а статична модель значуще гірша ({fm(I[4], 0)} МВт). "
           f"На валідації різниця не значуща ({pp(pt(R, 'net_import', 'ridge_depth', 'depth', 'validation'))}). "
           "Тому результат стосується тестового періоду 2024–2026 років, коли удари по генерації були найсильнішими, а подій було більше.")
    b.para(f"**Індекс відключень.** Жоден метод значуще не кращий за персистентність, а статичний XGBoost значуще гірший ({pp(pt(R, 'outages', 'xgb_static', 'depth'))}).")
    b.para("Чому так? Лінійна модель може давати значення, яких не було в навчальних даних. Дерева XGBoost цього не вміють. "
           "Тому проста регресія Ridge не гірша, а часто й краща за складнішу модель. Варіант «Ridge + XGBoost» з додатковими ознаками "
           "напруженості ринку стабільного виграшу не дав. Подій небагато, і вони частково перекриваються в часі. "
           "Тому p-значення оптимістичні, а відмінності орієнтовні. Результати для індексу відключень і втраченого попиту розглянуто в п. " + r("dir") + ".")

    # ------------------------------------------------------------------ 3.3 duration
    b.h2("3.3. Прогноз тривалості дефіциту після удару")
    b.para("Тривалість прогнозують чотирма способами (таблиця " + r("t_post") + "): траєкторним підходом, прямою регресією («Ridge + XGBoost»), "
           "моделлю виживання XGBoost та класифікатором XGBoost. Модель виживання вміє працювати з цензурованими даними. "
           "**Цензурування** – це випадок, коли дефіцит до кінця вікна не закінчився. Тоді точну тривалість невідомо, "
           "відомо лише, що вона не менша за спостережену. Результати на тесті наведено в таблиці " + r("post_dur") + ".")
    b.table("Похибка прогнозу тривалості дефіциту після удару, тест (MAE, діб)",
            ["Ряд (кількість ударів)", "Без змін", "Середнє за 28 діб", *[PLABEL[m] for m in DU_COLS[2:]]],
            _duration_rows(R), [3.2, 1.8, 2.0, 2.4, 2.2, 2.4, 2.5], align=["l"] + ["c"] * 6, key="post_dur")
    pdur = {sc: po(R, sc, "persistence", "duration_MAE_days") for sc in PD_SC}
    sig_d = {(sc, m): pt(R, sc, m, "duration") for sc in PD_SC for m in DU_COLS[1:] if pt(R, sc, m, "duration") < 0.05}
    chk(set(sig_d) == {("price", "baseline28"), ("net_import", "xgb_direct")}, "duration: set of significant differences from persistence changed")
    b.para("Усі підходи реалізовано й оцінено, але надійної переваги над персистентністю вони не мають. "
           "Різницю з персистентністю перевірено тестом Вілкоксона; для втраченого попиту його не проводили, бо подій лише "
           f"{n_events(R, 'demand')}. "
           f"На ціні РДН «Ridge + XGBoost» і модель виживання дають {lohi(po(R, 'price', 'xgb_aft', 'duration_MAE_days'), po(R, 'price', 'xgb_direct', 'duration_MAE_days'), 1)} доби "
           f"проти {fm(pdur['price'], 1)} у персистентності. Проте значущою є лише різниця для середнього за 28 діб "
           f"({fm(po(R, 'price', 'baseline28', 'duration_MAE_days'), 1)} доби, {pp(pt(R, 'price', 'baseline28', 'duration'))}). "
           f"Для моделі виживання різниця не значуща ({pp(pt(R, 'price', 'xgb_aft', 'duration'))}). "
           f"На чистому імпорті «Ridge + XGBoost» значуще гірший за персистентність ({fm(po(R, 'net_import', 'xgb_direct', 'duration_MAE_days'), 1)} проти {fm(pdur['net_import'], 1)} доби, {pp(pt(R, 'net_import', 'xgb_direct', 'duration'))}). "
           "Усі інші відмінності від персистентності не значущі. "
           "Причину й шлях до покращення описано в п. " + r("dir") + ". Тривалість залежить від швидкості ремонтів, "
           "а відкритих даних про терміни відновлення немає.")

    # ------------------------------------------------------------------ 3.4 adaptation
    b.h2("3.4. Адаптація моделей до нового режиму: перенавчання")
    b.para("Тепер розглянемо прогноз на добу наперед (по годинах). Він показує, наскільки модель відстежує ряд після зміни режиму. "
           "Статична модель навчається один раз. Інші моделі оновлюють навчання на свіжих даних. Вони відрізняються тим, коли саме це "
           "роблять: щотижня («" + NAME["xgb_rolling"] + "», контроль), кожні 3 доби («" + NAME["xgb_rolling_freq"] + "», ще один контроль), "
           "коли помилка самої моделі помітно зростає («" + NAME["xgb_drift"] + "», сигнал CUSUM), наступної доби після удару («" + NAME["xgb_attack"] + "») "
           "або після тривоги BOCPD («" + NAME["xgb_adaptive"] + "» і «" + NAME["xgb_adaptive_tau_w"] + "», де вікно навчання захоплює частину старого режиму). **BOCPD** – це алгоритм, який у реальному часі "
           "подає тривогу, коли ряд раптово змінює рівень.")
    b.para("Таблиця " + r("dayahead") + " зводить MAE на тесті для всіх моделей і чотирьох рядів. "
           "Різниці відносно щотижневого перенавчання та їхню значущість наведено в таблиці " + r("hyp") + " і на рис. " + r("retrain") + ".")
    b.table("Прогноз на добу наперед: MAE на тесті",
            ["Модель", "Втрачений попит, МВт", "Чистий імпорт, МВт", "Індекс відключень, частка", "Ціна РДН, частка"],
            _dayahead_rows(R), [5.6, 2.7, 2.6, 2.8, 2.6], align=["l", "c", "c", "c", "c"], key="dayahead", keep=False)

    # ---- demand
    a_, rr_, st_ = g("demand", "xgb_adaptive"), g("demand", "xgb_rolling"), g("demand", "xgb_static")
    nv_, tw_, pr_, dr_ = g("demand", "seasonal_naive"), g("demand", "xgb_adaptive_tau_w"), g("demand", "prophet_static"), g("demand", "xgb_drift")
    chk(st_ / rr_ > 5, "demand: static not far worse than retrained")
    b.para("**Втрачений попит.** Після лютого 2022 року цей показник стрибнув з нуля до 5–6 ГВт (рис. " + r("fc_dem") + "). "
           "Статичні моделі навчено на мирному часі, тому вони не бачать нового режиму. "
           f"Статичний XGBoost помиляється на {fm(st_, 0)} МВт, а Prophet – ще більше. "
           f"Після перенавчання похибка падає до {lohi(rr_, dr_, 0)} МВт, тобто приблизно в {fm(st_ / dr_, 0)} разів. "
           f"Усі такі моделі точніші за сезонний наївний метод ({pp(pval(R, 'demand', 'seasonal_naive'))} для моделі «{NAME['xgb_adaptive']}»). "
           f"Найменшу похибку має «{NAME['xgb_drift']}», але її перевага над щотижневим перенавчанням ({vtxt(R, 'demand', 'xgb_drift')}) – "
           "лише тенденція. Варіант зі ширшим вікном навчання гірший, бо вікно залишає частину мирного часу. "
           "Тестовий період – це вже усталений новий режим, а сам перехід припадає на валідацію. "
           "Тому результат показує, як швидко модель опановує режим, а не чи передбачає вона його появу.")
    b.figure("docs/figures/fig_fc_demand.png", "Втрачений попит: факт і прогнози (добові середні; Prophet не показано, його похибка близько 5000 МВт)", 13.5, key="fc_dem")

    # ---- net import
    a_, rr_, st_ = g("net_import", "xgb_adaptive"), g("net_import", "xgb_rolling"), g("net_import", "xgb_static")
    nv_, pr_, tw_ = g("net_import", "seasonal_naive"), g("net_import", "prophet_static"), g("net_import", "xgb_adaptive_tau_w")
    dr_ = g("net_import", "xgb_drift")
    na_ = nref(R, "net_import", "xgb_adaptive")
    b.para(f"**Чистий імпорт.** Моделі з перенавчанням точніші за статичну на {ap(pct(rr_, st_))} %, а «{NAME['xgb_drift']}» – ще й за сезонний наївний метод "
           f"на {ap(pct(dr_, nv_))} %. Prophet тут непридатний: його лінійний тренд навчено в 2019–2021 роках, і в наступні роки він зростає "
           f"до 8000 МВт (похибка {fm(pr_, 0)} МВт). Це недолік цієї базової моделі, а не методу. "
           f"Між перенавчанням за зростанням похибки і щотижневим різниці немає ({vtxt(R, 'net_import', 'xgb_drift')}). "
           f"Адаптивна модель за тривогою BOCPD від щотижневого перенавчання також не відрізняється ({vtxt(R, 'net_import', 'xgb_adaptive')}). "
           f"Із {na_[0]} {pere(na_[0])} лише {na_[1]} запустили тривоги.")

    # ---- outages
    a_, rr_, st_ = g("outages", "xgb_adaptive"), g("outages", "xgb_rolling"), g("outages", "xgb_static")
    nv_, dr_, fq_ = g("outages", "seasonal_naive"), g("outages", "xgb_drift"), g("outages", "xgb_rolling_freq")
    chk(vroll(R, "outages", "xgb_drift")[0] < 0 and vroll(R, "outages", "xgb_drift")[1] < 0.05, "outages: xgb_drift not significantly better than rolling")
    nd_o = nref(R, "outages", "xgb_drift")
    b.para(f"**Індекс відключень.** Тест (липень 2025 – червень 2026) містить зимову кампанію 2025/26 року. Серед запропонованих тригерів "
           f"найточніше працює перенавчання за зростанням похибки: MAE {fm(dr_, 3)} проти {fm(rr_, 3)} у щотижневого ({vtxt(R, 'outages', 'xgb_drift')}). "
           f"Загалом модель перенавчалася {nd_o[0]} {raz(nd_o[0])}, із них {nd_o[1]} – за сигналом дрейфу. "
           f"Адаптивна модель за BOCPD гірша за щотижневу, але різниця не значуща ({vtxt(R, 'outages', 'xgb_adaptive')}). "
           f"Статична модель ({fm(st_, 3)}) недооцінює охоплення відключень у піку кампанії. "
           f"Контроль – перенавчання кожні 3 доби – дає {fm(fq_, 3)} ({vtxt(R, 'outages', 'xgb_rolling_freq')} відносно щотижневого) "
           "і не гірше за перенавчання за зростанням похибки. Тому виграш пояснюється головним чином частішим перенавчанням на свіжих даних (п. " + r("s_trig") + ").")

    # ---- price
    a_, rr_, st_ = g("price", "xgb_adaptive"), g("price", "xgb_rolling"), g("price", "xgb_static")
    nv_, dr_, pr_ = g("price", "seasonal_naive"), g("price", "xgb_drift"), g("price", "prophet_static")
    fq_ = g("price", "xgb_rolling_freq")
    nd_ = nref(R, "price", "xgb_drift")
    chk(abs(vroll(R, "price", "xgb_drift")[0]) < 0.05, "price: xgb_drift forecasts differ from xgb_rolling on the test")
    b.para(f"**Ціна РДН.** Тут перенавчання виграшу не дає. На тесті статична модель не гірша за щотижневу ({fm(st_, 3)} проти {fm(rr_, 3)}; "
           f"{vtxt(R, 'price', 'xgb_static')} відносно щотижневого перенавчання). На валідації перенавчання було трохи точнішим "
           f"({fm(g('price', 'xgb_rolling', split='validation'), 3)} проти {fm(g('price', 'xgb_static', split='validation'), 3)}). "
           "Ціна після 2024 року не змінила режиму так, як втрачений попит. "
           f"Сезонний наївний метод помиляється суттєво більше ({vtxt(R, 'price', 'seasonal_naive')}), а Prophet – ще більше. "
           f"На тесті сигнал дрейфу не спрацював жодного разу (за весь час лише {nd_[1]} із {nd_[0]} {pere(nd_[0])} було за сигналом). "
           f"Тому перенавчання за зростанням похибки збігається зі щотижневим. Перенавчання кожні 3 доби трохи точніше за щотижневе ({vtxt(R, 'price', 'xgb_rolling_freq')}).")

    # ------------------------------------------------------------------ 3.5 triggers
    b.h2("3.5. Порівняння політик перенавчання")
    sc_ = ("demand", "net_import", "outages", "price")
    vr = lambda sc, m: vroll(R, sc, m)
    chk(all(vr(sc, "xgb_rolling_freq")[0] < 0 and vr(sc, "xgb_rolling_freq")[1] < 0.05 for sc in ("net_import", "outages", "price")) and vr("demand", "xgb_rolling_freq")[1] >= 0.05,
        "xgb_rolling_freq: not significantly better on exactly three series")
    chk(not any(vr(sc, "xgb_adaptive")[0] < 0 and vr(sc, "xgb_adaptive")[1] < 0.05 for sc in sc_) and vr("price", "xgb_adaptive")[0] > 0 and vr("price", "xgb_adaptive")[1] < 0.05,
        "xgb_adaptive: better somewhere or not significantly worse on price")
    chk(vr("net_import", "xgb_attack")[0] > 0 and vr("net_import", "xgb_attack")[1] < 0.05 and vr("price", "xgb_attack")[0] > 0 and vr("price", "xgb_attack")[1] < 0.05
        and vr("outages", "xgb_attack")[1] >= 0.05, "xgb_attack: pattern of significance changed")
    b.para("Рисунок " + r("retrain") + " зіставляє всі політики перенавчання зі щотижневим ковзним вікном. Значення нижче 100 % означає, що "
           "політика точніша за щотижневе перенавчання. Точні відсотки й p-значення наведено в таблиці " + r("hyp") + ". Висновки такі.", keep_next=True)
    b.items([
        "**Перенавчання за зростанням похибки** не гірше за щотижневе на всіх рядах. "
        f"На індексі відключень перевага значуща ({vtxt(R, 'outages', 'xgb_drift')}). На втраченому попиті є лише тенденція "
        f"({vtxt(R, 'demand', 'xgb_drift')}). На чистому імпорті та ціні РДН різниці немає.",
        "**Контроль – перенавчання кожні 3 доби** значуще точніше за щотижневе на чистому імпорті, індексі "
        f"відключень ({vtxt(R, 'outages', 'xgb_rolling_freq')}) і ціні РДН, але не на втраченому попиті. На індексі відключень воно не гірше "
        "за перенавчання за зростанням похибки. Тому найпростіше пояснення виграшу – частіше перенавчання на свіжих даних, а не сам сигнал дрейфу.",
        "**Перенавчання за тривогою BOCPD** переваги не дає. Воно не точніше за щотижневе на жодному ряді, "
        f"а на ціні РДН навіть значуще гірше ({vtxt(R, 'price', 'xgb_adaptive')}).",
        "**Перенавчання після ударів** виграшу не дає. На чистому імпорті та ціні РДН воно значуще гірше, "
        f"а на індексі відключень різниця не значуща ({vtxt(R, 'outages', 'xgb_attack')}). "
        "Тобто знання дат ударів саме по собі нічого не додає (див. п. " + r("dir") + ").",
        "**Сезонний наївний метод** гірший за моделі з перенавчанням на всіх рядах. Найменша відмінність – на індексі "
        f"відключень ({fm(g('outages', 'seasonal_naive'), 3)} проти {fm(g('outages', 'xgb_drift'), 3)}). "
        f"Різниця з щотижневим перенавчанням там не значуща ({pp(vroll(R, 'outages', 'seasonal_naive')[1])})."])
    b.figure("docs/figures/fig_retrain.png", "Похибка MAE на тесті відносно щотижневого перенавчання для тригерів перенавчання та сезонного наївного методу", 13.5, key="retrain")
    chk(all(vroll(R, sc, "xgb_drift")[0] < 1.0 for sc in sc_), "xgb_drift worse than rolling by >1% somewhere")

    # ------------------------------------------------------------------ 3.6 detection
    b.h2("3.6. Виявлення структурних зламів")
    bo_d, pe_d = de["cp"]["bocpd"]["scores"], de["cp"]["pelt"]["scores"]
    bo_i, pe_i = ni["cp"]["bocpd"]["scores"], ni["cp"]["pelt"]["scores"]
    bo_o, pe_o = ou["cp"]["bocpd"]["scores"], ou["cp"]["pelt"]["scores"]
    bo_p, pe_p = pr["cp"]["bocpd"]["scores"], pr["cp"]["pelt"]["scores"]
    b.para("Структурний злам – це момент, коли ряд різко змінює свою поведінку. Для виявлення зламів у добових середніх значеннях "
           "цільового ряду використано два методи. BOCPD працює в реальному часі й подає тривогу. PELT аналізує увесь ряд уже після події. "
           "Для оцінки взято весь офіційний перелік ударів. Затримка – це скільки діб у середньому минуло від удару до виявлення. "
           "Результати наведено в таблиці " + r("det") + " та на рисунку " + r("det_dem") + ".")

    def det_row(name, bo, pe):
        return [[name, "BOCPD", bo["n_detections"], hit(bo), bo["n_attacks"], fm(bo["precision"], 2), fm(bo["recall"], 2), fm(bo["mean_lag_days"], 1)],
                ["", "PELT", pe["n_detections"], hit(pe), pe["n_attacks"], fm(pe["precision"], 2), fm(pe["recall"], 2), fm(pe["mean_lag_days"], 1)]]
    b.table("Якість виявлення структурних зламів",
            ["Ряд", "Метод", "Виявлено", "З них збіги", "Ударів у періоді", "Точність", "Повнота", "Затримка, діб"],
            det_row("Втрачений попит", bo_d, pe_d) + det_row("Індекс відключень", bo_o, pe_o) + det_row("Чистий імпорт", bo_i, pe_i) + det_row("Ціна РДН", bo_p, pe_p),
            [2.6, 1.7, 2.4, 1.7, 1.7, 2.0, 2.0, 2.3], align=["l", "c", "c", "c", "c", "c", "c", "c"], key="det")
    al_d = [a["alarm"] for a in de["cp"]["bocpd"]["alarms"]]
    pel_d = de["cp"]["pelt"]["breakpoints"]
    inv = [x for x in pel_d if abs((pd.Timestamp(x) - pd.Timestamp("2022-02-24")).days) <= 3]
    b.para("Ряд втраченого попиту єдиний, де дата зламу відома незалежно від переліку ударів. Споживання впало приблизно "
           "з 18,5 до 13 ГВт за п’ять діб від 24 лютого 2022 року. "
           + (f"PELT вказав дату {inv[0][8:10]}.{inv[0][5:7]}.{inv[0][:4]}, що збігається з вторгненням. " if inv else "PELT не вказав дату вторгнення з точністю до трьох діб. ")
           + f"BOCPD подав {len(al_d)} {trv(len(al_d))}, {hit(bo_d).split(' ')[0]} із них припадають на удари. "
           f"На індексі відключень BOCPD має високу точність ({fm(bo_o['precision'], 2)}), але низьку повноту ({fm(bo_o['recall'], 2)}). "
           "Це означає, що частина ударів не призводить до видимих відключень. "
           "На чистому імпорті та ціні РДН обидва показники невисокі. Ці ряди зашумлені або реагують на сезонні й ринкові зміни, "
           "а не на окремі удари. PELT виявляє злами лише ретроспективно, тому для керування перенавчанням він не годиться. "
           "Загалом BOCPD корисний як діагностичний сигнал, але повнота невисока. Це узгоджується з висновком п. " + r("s_trig") + ".")

    
    b.figure("docs/figures/fig_detect_demand.png", "Втрачений попит, удари зі списку, злами PELT і тривоги BOCPD", 13.5, key="det_dem")

    # ------------------------------------------------------------------ 3.7 synthetic
    b.h2("3.7. Перевірка працездатності на синтетичних даних")
    s_b = sy["cp"]["bocpd"]["scores"]
    mm = sy["m"]
    g2 = lambda m: float(mm[(mm["model"] == m) & (mm["split"] == "test") & (mm["subset"] == "all")]["MAE"].iloc[0])
    b.para("Щоб перевірити роботу пайплайну від початку до кінця, його запущено на згенерованих даних із заздалегідь відомими зламами. "
           "У них попит падає на 35 % у лютому 2022 року, а потім є хвилі ударів із частковим відновленням. "
           f"Похибка адаптивної моделі у {fm(g2('xgb_static') / g2('xgb_adaptive'), 1)} раза менша за статичну. "
           f"BOCPD виявляє злами з точністю {fm(s_b['precision'], 2)} і повнотою {fm(s_b['recall'], 2)}, середня затримка – {fm(s_b['mean_lag_days'], 1)} доби. "
           "Ці дані створено автором, і злами в них вбудовано. Тому перевага адаптивної моделі закладена в самих даних. "
           "Експеримент підтверджує, що реалізація працює правильно, але не є доказом для енергосистеми України.")

    # ------------------------------------------------------------------ 3.8 comparison
    b.h2("3.8. Порівняльний аналіз")
    b.figure("docs/figures/fig_mae.png", "Похибка MAE на тесті для чотирьох сценаріїв (прогноз на добу наперед)", 12.5, key="mae")
    b.para("Результати щодо гіпотез роботи та запропонованих методів зведено в таблиці " + r("hyp") + ". У кожній клітинці вказано, "
           "на скільки відсотків змінилася похибка відносно еталона, і p-значення.")

    def rel_row(m, ref="xgb_rolling"):
        return [f"{fm(vroll(R, sc, m)[0], 1, sign=True)} %, {pp(vroll(R, sc, m)[1])}" for sc in ("demand", "net_import", "outages", "price")]

    def stat_row(sc):
        return vtxt(R, sc, "xgb_static")

    pdz = lambda sc, m: f"{fm(pct(po(R, sc, m, 'depth_MAE'), po(R, sc, 'persistence', 'depth_MAE')), 0, sign=True)} %" + (", " + pp(pt(R, sc, m, 'depth')) if pt(R, sc, m, 'depth') == pt(R, sc, m, 'depth') else ", не оцінювалось (замало подій)")
    b.table("Висновки щодо гіпотез і нових методів, тест",
            ["Гіпотеза", "Попит", "Імпорт", "Відклю- чення", "Ціна РДН", "Оцінка"],
            [["Пряма модель глибини (регресія Ridge) краща за персистентність", pdz("demand", "ridge_depth"), pdz("net_import", "ridge_depth"),
              pdz("outages", "ridge_depth"), pdz("price", "ridge_depth"), "підтверджено для ціни РДН і чистого імпорту"],
             ["Прогноз тривалості кращий за персистентність", *[f"виживання: {fm(pct(po(R, sc, 'xgb_aft', 'duration_MAE_days'), po(R, sc, 'persistence', 'duration_MAE_days')), 0, sign=True)} %" if po(R, sc, 'persistence', 'duration_MAE_days') > 0 else "–" for sc in ("demand", "net_import", "outages", "price")],
              "не підтверджено (напрям вдосконалення)"],
             ["Статичні моделі погіршуються після зміни режиму (статичний XGBoost відносно щотижневого перенавчання)", stat_row("demand"), stat_row("net_import"), stat_row("outages"), stat_row("price"), "підтверджено на трьох рядах; на ціні РДН – ні"],
             ["Перенавчання за зростанням похибки краще за щотижневе", *rel_row("xgb_drift"), "значуще на індексі відключень, на інших – не гірше"],
             ["Частіше перенавчання (кожні 3 доби) краще за щотижневе", *rel_row("xgb_rolling_freq"), "залежить від ряду"],
             ["Перенавчання за тривогою BOCPD краще за щотижневе", *rel_row("xgb_adaptive"), "не підтверджено"],
             ["Перенавчання після ударів краще за щотижневе", *rel_row("xgb_attack"), "не підтверджено"],
             ["Перенавчання за зростанням похибки точніше за сезонний наївний метод", *[f"{fm(pct(g(sc, 'xgb_drift'), g(sc, 'seasonal_naive')), 0, sign=True)} %" for sc in ("demand", "net_import", "outages", "price")], "моделі з перенавчанням краще"],
             ["Ширше вікно навчання (із частиною старого режиму) краще за навчання від моменту зламу", *[f"{fm(pct(g(sc, 'xgb_adaptive_tau_w'), g(sc, 'xgb_adaptive')), 0, sign=True)} %" for sc in ("demand", "net_import", "outages", "price")], "залежить від ряду"],
             ["BOCPD виявляє структурні злами (точність / повнота)", f"{fm(bo_d['precision'], 2)} / {fm(bo_d['recall'], 2)}", f"{fm(bo_i['precision'], 2)} / {fm(bo_i['recall'], 2)}",
              f"{fm(bo_o['precision'], 2)} / {fm(bo_o['recall'], 2)}", f"{fm(bo_p['precision'], 2)} / {fm(bo_p['recall'], 2)}", "діагностика; повнота невисока"]],
            [3.5, 2.4, 2.4, 2.4, 2.4, 3.4], keep=False, key="hyp")
    b.para("**Підсумок порівняння.** Користь дає свіже навчання: часте або кероване похибкою перенавчання. Виявлення зламу за рівнем "
           "ряду користі не дає, але корисне як діагностика та ранній сигнал.")

    # ------------------------------------------------------------------ 3.9 directions
    b.h2("3.9. Напрями вдосконалення")
    X_ = {sc: (po(R, sc, "persistence", "depth_MAE"), po(R, sc, "ridge_depth", "depth_MAE")) for sc in ("outages", "demand")}
    dl = {sc: (po(R, sc, "persistence", "duration_MAE_days"), po(R, sc, "xgb_direct", "duration_MAE_days"), po(R, sc, "xgb_aft", "duration_MAE_days")) for sc in PD_SC}
    b.para("Нижче зібрано результати, які поки не дали виграшу. Для кожного вказано реальні показники й конкретний крок вдосконалення.", keep_next=True)
    b.items([
        "**Прогноз тривалості.** Пряма регресія, модель виживання та класифікатор не точніші за персистентність. "
        f"Для моделі виживання XGBoost (тест, MAE, доби; персистентність → модель виживання) це видно на чистому імпорті ({fm(dl['net_import'][0], 1)} → {fm(dl['net_import'][2], 1)}) "
        f"та індексі відключень ({fm(dl['outages'][0], 1)} → {fm(dl['outages'][2], 1)}). "
        f"На ціні РДН модель виживання формально краща ({fm(dl['price'][0], 1)} → {fm(dl['price'][2], 1)}), але різниця не значуща. "
        "Тривалість залежить від швидкості ремонтів. Тому потрібні дані про втрачену потужність кожного удару та терміни відновлення, "
        "які варто додати як ознаки.",
        f"**Глибина на індексі відключень.** Прогноз глибини не кращий за персистентність (регресія Ridge {fm(X_['outages'][1], 3)} проти {fm(X_['outages'][0], 3)}, "
        f"{pp(pt(R, 'outages', 'ridge_depth', 'depth'))}). Індекс насичується на одиниці й змінюється лише в періоди кампаній. "
        "Потрібен показник у МВт (потужність відключень по областях) замість словесної шкали.",
        f"**Перший удар і втрачений попит.** Подій дуже мало: {n_events(R, 'demand', 'validation')} на валідації й {n_events(R, 'demand')} на тесті. "
        f"На них персистентність ({fm(po(R, 'demand', 'persistence', 'depth_MAE'), 0)} МВт) точніша за регресію Ridge ({fm(po(R, 'demand', 'ridge_depth', 'depth_MAE'), 0)} МВт). "
        f"Модель «{NAME_POST['xgb_rolling']}» ({fm(po(R, 'demand', 'xgb_rolling', 'depth_MAE'), 0)} МВт) на цих п’яти подіях краще за персистентність, але значущість не оцінювалась. "
        "Першого дня вторгнення навчальні дані не містять жодного прикладу, тому модель не екстраполює на небачений режим. "
        "Потрібні погодинні дані після жовтня 2022 року й більше прикладів різного масштабу.",
        f"**Перенавчання після ударів і за тривогою BOCPD.** Знання дат ударів («{NAME['xgb_attack']}»: {vtxt(R, 'outages', 'xgb_attack')} на індексі відключень) "
        "та тривоги за рівнем ряду не дають виграшу порівняно з частим перенавчанням. Далі варто спробувати сигнал, що поєднує рівень ряду, похибку й силу удару.",
        "**Дані та валідація.** Потрібні довші ряди подій. Незалежних подій мало, тому відмінності в п. " + r("s_depth") + " орієнтовні.",
    ])
    b.figure("docs/figures/fig_post_event.png", "Прогноз траєкторії втраченого попиту на 30 діб: факт і три методи", 12.0, key="post")
    b.para("Рисунок " + r("post") + " ілюструє перший удар на втраченому попиті. 24 лютого 2022 року в навчальних даних немає жодного дня дефіциту, "
           "тому перенавчена модель прогнозує лише близько 2 ГВт, персистентність – 2,7 ГВт, а статична – нуль, хоча фактично було 5–7 ГВт. "
           "У жовтні 2022 року модель, навчена на воєнних даних, відтворює рівень, але не провал до 4,5 ГВт (пунктир – поріг епізоду 2 ГВт).")

    # ------------------------------------------------------------------ conclusions
    b.h2("Висновки до розділу 3")
    b.items([
        "Після удару пряма модель глибини дефіциту на 30 діб значуще точніша за персистентність. На ціні РДН похибка менша на "
        f"{ap(pct(pd_p['price'][1], pd_p['price'][0]))} % (підтверджено й на валідації). "
        f"На чистому імпорті похибка менша на {ap(pct(pd_p['net_import'][1], pd_p['net_import'][0]))} % (на тесті).",
        "Моделі з перенавчанням набагато точніші за статичні там, де режим змінився. На втраченому попиті MAE "
        f"{fm(g('demand', 'xgb_drift'), 0)} проти {fm(g('demand', 'xgb_static'), 0)} МВт.",
        f"Перенавчання за зростанням похибки значуще краще за щотижневе на індексі відключень ({vtxt(R, 'outages', 'xgb_drift')}) і не гірше на інших рядах; "
        "контроль із частішим перенавчанням не гірший, а тригер BOCPD переваги не дає.",
        "Прогноз тривалості, глибини на індексі відключень та першої хвилі 2022 року поки на рівні персистентності. "
        "Конкретні кроки вдосконалення названо в п. " + r("dir") + "."])


def conclusions(b, R):
    de, ni, ou, pr = R["demand"], R["net_import"], R["outages"], R["price"]
    g = lambda sc, m: val(R, sc, m)
    pdp = lambda sc, m: po(R, sc, m, "depth_MAE")
    b.h1("ЗАГАЛЬНІ ВИСНОВКИ")
    b.para("У результаті виконання роботи отримано такі результати.")
    b.numbered([
        "Реалізовано програмну систему на Python у середовищі Docker. Вона має шість кроків пайплайну (збір даних, обробка, ознаки, "
        "виявлення зламів, адаптивне навчання, оцінювання) і модуль прогнозу глибини й тривалості дефіциту після удару. "
        f"Систему можна відтворити, її параметри задано файлами конфігурації, а правильність роботи перевіряють {N_TESTS} автоматизованих тестів.",
        "Методологічну коректність закладено в код: дані поділено за часом, ознаки використовують лише інформацію, відому на момент "
        "випуску прогнозу, сигнал BOCPD причинний (не використовує майбутнього), PELT застосовано лише ретроспективно, параметри "
        "підбирають лише на валідації, невідомі значення не заповнюють. Тести перевіряють відсутність витоку майбутнього та коректний час перенавчання.",
        "Для експериментів використано відкриті дані Energy Map, офіційний перелік "
        f"{R['data']['strikes_n']} масованих ударів 2022–2026 рр. і погоду Open-Meteo. Дефіцит описано чотирма показниками: "
        "втраченим попитом, чистим імпортом, індексом відключень і ціною РДН відносно граничної ціни.",
        "Прямий прогноз глибини дефіциту на 30 діб після удару точніший за персистентність. "
        f"На ціні РДН похибка зменшується з {fm(pdp('price', 'persistence'), 3)} до {fm(pdp('price', 'ridge_depth'), 3)} "
        f"({fm(pct(pdp('price', 'ridge_depth'), pdp('price', 'persistence')), 0, sign=True)} %, {pp(pt(R, 'price', 'ridge_depth', 'depth'))}); "
        f"перевагу підтверджено й на валідації. На чистому імпорті на тесті похибка зменшується з {fm(pdp('net_import', 'persistence'), 0)} до "
        f"{fm(pdp('net_import', 'ridge_depth'), 0)} МВт ({fm(pct(pdp('net_import', 'ridge_depth'), pdp('net_import', 'persistence')), 0, sign=True)} %, "
        f"{pp(pt(R, 'net_import', 'ridge_depth', 'depth'))}). Проста лінійна модель на рівнях останніх тижнів не гірша за XGBoost, "
        "бо вміє екстраполювати на небачені рівні.",
        "Моделі з перенавчанням на останніх даних точніші за статичні на трьох рядах із чотирьох. "
        f"На втраченому попиті MAE {fm(g('demand', 'xgb_drift'), 0)} проти {fm(g('demand', 'xgb_static'), 0)} МВт, "
        f"на чистому імпорті {fm(g('net_import', 'xgb_drift'), 1)} проти {fm(g('net_import', 'xgb_static'), 1)} МВт, "
        f"на індексі відключень {fm(g('outages', 'xgb_drift'), 3)} проти {fm(g('outages', 'xgb_static'), 3)}. "
        f"На ціні РДН статична модель не гірша ({fm(g('price', 'xgb_static'), 3)} проти {fm(g('price', 'xgb_drift'), 3)}).",
        "Запропоноване перенавчання за дрейфом власної похибки моделі (CUSUM) значуще точніше за щотижневе на індексі "
        f"відключень ({vtxt(R, 'outages', 'xgb_drift')}). На втраченому попиті є лише тенденція ({vtxt(R, 'demand', 'xgb_drift')}). "
        "На чистому імпорті та ціні РДН різниці немає. Контроль – перенавчання кожні 3 доби – на індексі відключень не гірший. "
        "Тому корисним є часте або кероване похибкою перенавчання. Перенавчання за тривогою BOCPD, що спирається на рівень ряду, "
        "щотижневе не перевершує. Перенавчання після ударів виграшу не дає.",
        "BOCPD як діагностичний сигнал має точність "
        f"{fm(de['cp']['bocpd']['scores']['precision'], 2)} (втрачений попит), {fm(ou['cp']['bocpd']['scores']['precision'], 2)} (індекс відключень), "
        f"{fm(ni['cp']['bocpd']['scores']['precision'], 2)} (чистий імпорт) і {fm(pr['cp']['bocpd']['scores']['precision'], 2)} (ціна РДН). "
        "Повнота невисока. PELT придатний лише ретроспективно.",
        "Прогноз тривалості дефіциту (пряма регресія, модель виживання, класифікатор), прогноз глибини на індексі відключень і прогноз "
        "першої хвилі 2022 року реалізовано, але їхня точність на рівні персистентності. Це напрями вдосконалення. Для них потрібні "
        "дані про втрачену потужність і терміни ремонтів, показники в МВт та довші ряди подій.",
        ])
    b.para("**Напрями подальших досліджень:** додати втрачену потужність і терміни відновлення кожного удару як ознаки прогнозу тривалості; "
           "знайти джерело погодинного споживання та генерації після жовтня 2022 року, щоб продовжити сценарій втраченого попиту на "
           "пізніші хвилі; поєднати сигнали рівня, похибки та сили удару в одному тригері перенавчання; розширити оцінку на більшу "
           "кількість незалежних подій і регіональні ряди.")


def sources(b):
    b.h1("СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ")
    refs = [
        "ENTSO-E Transparency Platform. URL: https://transparency.entsoe.eu (дата звернення: 04.10.2026).",
        "Energy Map (DiXi Group). Електроенергія: заходи щодо обмеження споживання. URL: https://energy-map.info/en/resources/0f8f9882-1fb2-47c6-81dc-31fbad914f16 (дата звернення: 04.10.2026).",
        "Energy Map (DiXi Group). Погодинний імпорт та експорт електричної енергії. URL: https://energy-map.info/en/resources/56df70b0-6bc1-4c7d-a82f-284cf723438d/ (дата звернення: 04.10.2026).",
        "Energy Map (DiXi Group). Погодинний баланс ОЕС України. URL: https://energy-map.info/en/resources/8998f2ed-379f-4fa9-9076-88782b32ee4f/ (дата звернення: 04.10.2026).",
        "Open-Meteo. Free Weather API. URL: https://open-meteo.com (дата звернення: 04.10.2026).",
        "Armed Conflict Location & Event Data Project (ACLED). URL: https://acleddata.com (дата звернення: 04.10.2026).",
        "Taylor S.J., Letham B. Forecasting at scale. The American Statistician. 2018. Vol. 72, No. 1. P. 37–45.",
        "Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD. 2016. P. 785–794.",
        "Adams R.P., MacKay D.J.C. Bayesian Online Changepoint Detection. arXiv:0710.3742. 2007.",
        "Killick R., Fearnhead P., Eckley I.A. Optimal Detection of Changepoints With a Linear Computational Cost. Journal of the American Statistical Association. 2012. Vol. 107, No. 500. P. 1590–1598.",
        "Truong C., Oudre L., Vayatis N. Selective review of offline change point detection methods. Signal Processing. 2020. Vol. 167. Article 107299.",
        "Akiba T., Sano S., Yanase T., Ohta T., Koyama M. Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD. 2019. P. 2623–2631.",
        "Pedregosa F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research. 2011. Vol. 12. P. 2825–2830.",
        "Wilcoxon F. Individual Comparisons by Ranking Methods. Biometrics Bulletin. 1945. Vol. 1, No. 6. P. 80–83.",
        "Box G.E.P., Jenkins G.M., Reinsel G.C., Ljung G.M. Time Series Analysis: Forecasting and Control. 5th ed. Wiley, 2015.",
        "Міністерство енергетики України. Звіти про пошкодження об’єктів енергетики. URL: https://mev.gov.ua (дата звернення: 04.10.2026).",
        "НЕК «Укренерго». Оперативна інформація. URL: https://ua.energy (дата звернення: 04.10.2026).",
        "Energy Map (DiXi Group). Інформація про масовані удари по критичній інфраструктурі (за зведеннями Повітряних сил ЗС України), набір даних. URL: https://energy-map.info/en/datasets/12f3148d-841a-478d-b9ed-72bf0764b286 (дата звернення: 06.10.2026).",
        "Energy Map (DiXi Group). Результати торгів на ринку «на добу наперед» (РДН), набір даних. URL: https://energy-map.info/en/datasets/5a616fba-fbc9-4073-9532-9161592faca8 (дата звернення: 06.10.2026).",
        "Page E.S. Continuous Inspection Schemes. Biometrika. 1954. Vol. 41, No. 1–2. P. 100–115.",
        "Hoerl A.E., Kennard R.W. Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics. 1970. Vol. 12, No. 1. P. 55–67.",
        "Wei L.J. The accelerated failure time model: a useful alternative to the Cox regression model in survival analysis. Statistics in Medicine. 1992. Vol. 11, No. 14–15. P. 1871–1879.",
    ]
    b.numbered(refs)
