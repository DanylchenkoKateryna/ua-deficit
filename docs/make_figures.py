"""Figures for the practical part of the course work (Ukrainian labels), built from the real-data results.

Run inside the project image (matplotlib, pandas, pyarrow are installed there):
  docker run --rm -v "<project dir>:/work" -w /work --entrypoint python ua-deficit-forecast docs/make_figures.py
Reads artifacts/<scenario>/ and data/processed/<scenario>/, writes docs/figures/*.png.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from res_helpers import NAME, NAME_POST, cap  # noqa: E402  (reader-facing model names)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
SLIDE = os.environ.get("FIG_MODE") == "slide"        # larger fonts for the presentation
K = 1.15 if SLIDE else 1.0
PREFIX = "s_" if SLIDE else ""
W_SLIDE = {"cov": 8.6, "det": 8.4, "fc": 5.7, "mae": 8.6, "pd": 8.6, "rt": 8.6}


def F(x):
    return x * K


def size(w, h, key):
    if not SLIDE:
        return (w, h)
    f = W_SLIDE[key] / w
    return (W_SLIDE[key], h * f)

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9 * K, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
C_ACT, C_NAIVE, C_STATIC, C_ROLL, C_ADAPT, C_PROPH = "#222222", "#8c8c8c", "#4C78A8", "#F58518", "#D62728", "#9467BD"
NAMES = {k: cap(v) for k, v in NAME.items()}
NAMES["seasonal_naive"] = "Сезонний наївний (тиждень тому)"
COLORS = {"seasonal_naive": C_NAIVE, "xgb_static": C_STATIC, "xgb_rolling": C_ROLL, "xgb_adaptive": C_ADAPT,
          "prophet_static": C_PROPH, "xgb_adaptive_tau_w": "#2CA02C", "xgb_drift": "#0F8B8D", "xgb_attack": "#BCBD22",
          "xgb_rolling_freq": "#E377C2"}
SCEN_TITLE = {"demand": "Втрачений попит", "net_import": "Чистий імпорт", "outages": "Індекс відключень", "price": "Ціна РДН"}
UNIT = {"outages": "частка областей з відключеннями", "net_import": "чистий імпорт, МВт"}


def load(sc):
    art, proc = ROOT / "artifacts" / sc, ROOT / "data" / "processed" / sc
    master = pd.read_parquet(proc / "master.parquet")
    fcs = {p.stem: pd.read_parquet(p) for p in sorted((art / "forecasts").glob("*.parquet"))}
    sig = pd.read_parquet(art / "bocpd_signal.parquet")
    cp = json.loads((art / "changepoints.json").read_text(encoding="utf-8"))
    return master, fcs, sig, cp


def attacks():
    a = pd.read_csv(ROOT / "data" / "external" / "attacks.csv")
    return pd.to_datetime(a["date"] + " " + a["time_utc"].fillna("00:00"), utc=True)


def _comma(v, _pos=None):
    return f"{v:g}".replace(".", ",").replace("-", "−")


def finish(fig, name):
    for a in fig.axes:
        if a.get_visible() and a.get_ylabel() != "" or a.get_yticklabels():
            if not any(t.get_text() and not t.get_text().replace("−", "-").replace(".", "", 1).lstrip("-").isdigit() for t in a.get_yticklabels() if t.get_text()):
                a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_comma))
    fig.tight_layout()
    fig.savefig(OUT / (PREFIX + name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Fig 1.1 architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(8, 50); ax.axis("off"); ax.grid(False)
    steps = [("1. Збір даних", "ingestion/\nOpen-Meteo, Energy Map\n(баланс, РДН, удари)"),
             ("2. Обробка", "preprocessing/\nUTC, пропуски, ціль,\nепізоди дефіциту"),
             ("3. Ознаки", "features/\nлаги, ковзні, календар,\nпогода, атаки"),
             ("4. Виявлення\nзламів", "changepoints/\nPELT (ретроспективно),\nBOCPD (онлайн)"),
             ("5. Адаптивне\nнавчання", "training/\nперенавчання: розклад,\nBOCPD, дрейф, Optuna"),
             ("6. Оцінювання", "evaluation/\nMAE/RMSE, Вілкоксон,\nпрогноз після атаки")]
    w, gap, x0, y0, h = 14.0, 2.4, 1.8, 22, 16
    for i, (title, mods) in enumerate(steps):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#E8EEF5", ec="#4C78A8", lw=1.2))
        ax.text(x + w / 2, y0 + h - 2.2, title, ha="center", va="top", fontsize=F(8.6), fontweight="bold")
        ax.text(x + w / 2, y0 + 7.0, mods, ha="center", va="top", fontsize=F(6.9), color="#333")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.3, y0 + h / 2), (x + w + gap - 0.3, y0 + h / 2), arrowstyle="-|>", mutation_scale=10, color="#444"))
    ax.text(50, 47, "Вхід: файли даних (CSV), перелік масованих ударів  →  Parquet-файли у data/", ha="center", fontsize=F(8.6), color="#222")
    ax.text(50, 15.5, "Вихід: таблиці метрик, глибина і тривалість дефіциту, графіки (reports/<сценарій>/)", ha="center", fontsize=F(8.6), color="#222")
    ax.text(50, 10.5, "Усі кроки запускаються командою deficit <крок> у контейнері Docker; параметри – у config/*.yaml", ha="center", fontsize=F(7.8), color="#555")
    finish(fig, "fig_architecture.png")


# ---------------------------------------------------------------- Fig 2.1 data coverage
def fig_coverage():
    rows = [("Погода (Open-Meteo)", "2019-01-01", "2026-06-30", "#4C78A8", "використано"),
            ("Імпорт/експорт (Energy Map)", "2019-07-01", "2026-06-30", "#4C78A8", "використано"),
            ("Планові відключення (Energy Map)", "2023-01-01", "2026-06-30", "#4C78A8", "використано"),
            ("Результати торгів РДН (Energy Map)", "2019-07-01", "2026-06-30", "#4C78A8", "використано"),
            ("Масовані удари (Energy Map)", "2022-09-11", "2026-06-15", "#4C78A8", "використано"),
            ("Баланс за годинами (Energy Map)", "2019-01-01", "2022-10-27", "#4C78A8", "використано"),
            ("ENTSO-E: споживання України", "2019-01-01", "2022-01-31", "#B0B0B0", "перевірка даних")]
    fig, ax = plt.subplots(figsize=size(9, 4.0, "cov"))
    for i, (name, a, b, col, note) in enumerate(rows[::-1]):
        if a:
            s, e = mdates.date2num(pd.Timestamp(a)), mdates.date2num(pd.Timestamp(b))
            ax.barh(i, e - s, left=s, height=0.55, color=col)
            ax.text(e + 25, i, note, va="center", fontsize=F(7.6), color="#333")
        else:
            ax.text(mdates.date2num(pd.Timestamp("2019-02-01")), i, "токен очікується, покриття для воєнних років не перевірено", va="center", fontsize=F(7.6), color="#333")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.xaxis_date(); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(mdates.date2num(pd.Timestamp("2019-01-01")), mdates.date2num(pd.Timestamp("2027-09-01")))
    ax.set_ylim(-0.8, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
    finish(fig, "fig_coverage.png")


# ---------------------------------------------------------------- Fig 3.x detection
def fig_detection(sc, name, ylabel, thr):
    master, _, sig, cp = load(sc)
    daily = master["deficit"].resample("D").mean()
    att = attacks()
    att = att[(att >= daily.index[0]) & (att <= daily.index[-1])]
    thr = cp["bocpd"].get("threshold", thr)
    fig, ax = plt.subplots(2, 1, figsize=size(9, 5.2, "det"), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax[0].plot(daily.index, daily, color=C_ACT, lw=0.9, label="щоденне середнє")
    for i, t in enumerate(att):
        ax[0].axvline(t, color="#E45756", alpha=0.45, lw=1, label="масований удар" if i == 0 else None)
    for i, b in enumerate(cp["pelt"]["breakpoints"]):
        ax[0].axvline(pd.Timestamp(b, tz="UTC"), color="#4C78A8", ls="--", lw=1.1, label="злам PELT" if i == 0 else None)
    al = [pd.Timestamp(a["alarm"], tz="UTC") for a in cp["bocpd"]["alarms"]]
    top = daily.max()
    ax[0].scatter(al, [top * 1.07] * len(al), marker="v", color="#2CA02C", s=26, zorder=5, label="тривога BOCPD")
    ax[0].set_ylim(top=top * 1.14)
    ax[0].set_ylabel(ylabel)
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=F(7.8), ncol=4, frameon=False)
    ax[1].plot(sig.index, sig["cp_prob"], color="#7D3C98", lw=0.9)
    ax[1].axhline(thr, color="#777", ls=":", lw=1)
    ax[1].set_ylim(0, 1.02); ax[1].set_ylabel("P(злам), BOCPD")
    ax[1].xaxis.set_major_locator(mdates.YearLocator()); ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    finish(fig, name)


# ---------------------------------------------------------------- Fig 3.x forecasts
def fig_forecasts(sc, name, ylabel, models, daily=True, start=None, end=None, split_line=True, slide_label=None):
    if SLIDE:
        ylabel = slide_label or ("МВт" if "МВт" in ylabel else "частка областей")
    master, fcs, _, _ = load(sc)
    first = next(iter(fcs.values()))
    y = first["y"].resample("D").mean() if daily else first["y"]
    fig, ax = plt.subplots(figsize=size(9, 3.9, "fc"))
    ax.plot(y.index, y, color=C_ACT, lw=1.3, label="факт")
    for m in models:
        s = fcs[m]["yhat"].resample("D").mean() if daily else fcs[m]["yhat"]
        ax.plot(s.index, s, color=COLORS[m], lw=0.9, label=NAMES[m])
    if start:
        ax.set_xlim(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
    if split_line and not start:
        ax.axvline(pd.Timestamp({"outages": "2025-06-30", "demand": "2022-03-31"}.get(sc, "2023-12-31"), tz="UTC"), color="#999", ls=":", lw=1.2)
        ax.text(pd.Timestamp({"outages": "2025-07-10", "demand": "2022-04-05"}.get(sc, "2024-01-10"), tz="UTC"), ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03, "тест →", fontsize=F(8), color="#555")
    if SLIDE:
        span = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 365.0
        if span > 2:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y"))
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=F(7.6), ncol=2 if SLIDE else 3, frameon=False)
    finish(fig, name)


# ---------------------------------------------------------------- Fig 2.x real consumption and its peacetime expectation
def fig_demand_series():
    master, _, _, _ = load("demand")
    d = master[["actual_load_MW", "expected_load_MW"]].resample("D").mean()
    d = d["2021-09-01":]
    fig, ax = plt.subplots(figsize=size(9, 3.9, "fc"))
    ax.plot(d.index, d["actual_load_MW"] / 1000, color=C_ACT, lw=1.3, label="фактичне споживання")
    ax.plot(d.index, d["expected_load_MW"] / 1000, color=C_STATIC, lw=1.1, ls="--", label="очікуване в мирний час (за календарем і погодою)")
    ax.axvline(pd.Timestamp("2022-02-24", tz="UTC"), color="#E45756", lw=1.2)
    ax.text(pd.Timestamp("2022-02-27", tz="UTC"), ax.get_ylim()[1] * 0.97, "24.02.2022", color="#E45756", fontsize=F(9), va="top")
    ax.set_ylabel("ГВт")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=F(8), ncol=1 if SLIDE else 2, frameon=False)
    finish(fig, "fig_demand_series.png")


# ---------------------------------------------------------------- Fig 3.x forecast of the trajectory after an attack
def fig_post_event():
    t = pd.read_csv(ROOT / "reports" / "demand" / "post_event_paths.csv")
    fig, axs = plt.subplots(1, 2, figsize=size(9, 3.6, "det"), sharey=True)
    for ax, day, title in ((axs[0], "2022-02-24", "Вторгнення, 24.02.2022"), (axs[1], "2022-10-10", "Атаки жовтня, видача 10.10.2022")):
        g = t[(t["issue_day"] == day) & (t["trigger"] == "attack")]
        ax.plot(g["k"], g["actual"] / 1000, color=C_ACT, lw=2, label="факт")
        for m, c, lab in (("persistence", C_NAIVE, cap(NAME_POST["persistence"])), ("xgb_static", C_STATIC, cap(NAME_POST["xgb_static"])),
                          ("xgb_rolling", C_ADAPT, NAME_POST["xgb_rolling"])):
            ax.plot(g["k"], g[m] / 1000, color=c, lw=1.3, label=lab)
        ax.axhline(2.0, color="#999", ls=":", lw=1)
        ax.set_title(title, fontsize=F(9.5)); ax.set_xlabel("днів після видачі прогнозу")
    axs[0].set_ylabel("втрачений попит, ГВт")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", fontsize=F(7.8), ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))
    finish(fig, "fig_post_event.png")


# ---------------------------------------------------------------- Fig 3.x MAE comparison
def fig_mae():
    scen = (("demand", "Втрачений попит, МВт", "{:.0f}"), ("net_import", "Чистий імпорт, МВт", "{:.0f}"),
            ("outages", "Індекс відключень", "{:.3f}"), ("price", "Ціна РДН / стеля", "{:.3f}"))
    keys = ("seasonal_naive", "xgb_static", "xgb_rolling", "xgb_drift")
    fig, axs = plt.subplots(2, 2, figsize=size(9.5, 6.6, "mae"))
    for ax, (sc, title, fmt) in zip(axs.ravel(), scen):
        t = pd.read_csv(ROOT / "reports" / sc / "metrics.csv")
        t = t[(t["split"] == "test") & (t["subset"] == "all")].set_index("model")["MAE"]
        order = [m for m in keys if m in t.index]
        vals = [t[m] for m in order]
        cap = 1000 if sc == "demand" else None
        bars = ax.bar(range(len(order)), [min(v, cap) if cap else v for v in vals], color=[COLORS[m] for m in order], width=0.62)
        for b, v in zip(bars, vals):
            if cap and v > cap:
                continue
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v).replace(".", ","), ha="center", va="bottom", fontsize=F(7.0))
        labels = {"seasonal_naive": "Наїв." if SLIDE else "Сезонний\nнаївний", "xgb_static": "Стат." if SLIDE else "XGB\nстатичний",
                  "xgb_rolling": "Ковзн." if SLIDE else "XGB\nщотижня", "xgb_drift": "Похибка" if SLIDE else "XGB, за\nзростанням\nпохибки"}
        ax.set_xticks(range(len(order))); ax.set_xticklabels([labels[m] for m in order], fontsize=F(6.6))
        ax.set_title(title, fontsize=F(9.0)); ax.set_ylabel("MAE на тесті" if ax is axs[0] else "")
        ax.set_ylim(0, (1100 if cap else max(vals) * 1.18))
        if cap:
            ax.text(0.98, 0.97, "Статичний XGBoost: " + fmt.format(t["xgb_static"]).replace(".", ",") + "\n(стовпчик обрізано)", transform=ax.transAxes, ha="right", va="top", fontsize=F(6.4), color=C_STATIC)
        elif "prophet_static" in t.index:
            ax.text(0.98, 0.97, "Prophet: " + fmt.format(t["prophet_static"]).replace(".", ",") + "\n(поза шкалою)", transform=ax.transAxes, ha="right", va="top", fontsize=F(6.4), color=C_PROPH)
    finish(fig, "fig_mae.png")


# ---------------------------------------------------------------- Fig 3.x depth of the deficit after an attack
POST_M = (("persistence", cap(NAME_POST["persistence"]), C_NAIVE), ("xgb_rolling", NAME_POST["xgb_rolling"], C_ROLL),
          ("ridge_depth", cap(NAME_POST["ridge_depth"]), "#2CA02C"), ("xgb_direct", NAME_POST["xgb_direct"], "#0F8B8D"))


def fig_post_depth():
    scen = (("price", "Ціна РДН / стеля", "{:.3f}"), ("net_import", "Чистий імпорт, МВт", "{:.0f}"), ("outages", "Індекс відключень", "{:.3f}"))
    fig, axs = plt.subplots(1, 3, figsize=size(11.0, 3.6, "pd"))
    for ax, (sc, title, fmt) in zip(axs, scen):
        summ = pd.read_csv(ROOT / "reports" / sc / "post_event_summary.csv")
        summ = summ[(summ["split"] == "test") & (summ["trigger"] == "all")].set_index("method")
        n = int(summ["n_events"].iloc[0])
        vals = [summ.loc[m, "depth_MAE"] for m, _, _ in POST_M]
        bars = ax.bar(range(len(POST_M)), vals, color=[c for _, _, c in POST_M], width=0.66)
        for b, v in zip(bars, vals):      # significance of the differences is given in words in the table of the text
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v).replace(".", ","), ha="center", va="bottom", fontsize=F(7.0))
        ax.set_xticks([]); ax.set_ylim(0, max(vals) * 1.28)
        ax.set_title(f"{title}\n(тест, {n} подій)", fontsize=F(9.0))
        ax.set_ylabel("MAE глибини" if ax is axs[0] else "")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=c) for _, _, c in POST_M], [l for _, l, _ in POST_M], loc="lower center", ncol=2 if SLIDE else 4,
               fontsize=F(7.6), frameon=False, bbox_to_anchor=(0.5, -0.12 if not SLIDE else -0.2))
    finish(fig, "fig_post_depth.png")


# ---------------------------------------------------------------- Fig 3.x retraining triggers relative to the weekly rolling window
def fig_retrain():
    models = ("xgb_adaptive", "xgb_drift", "xgb_attack", "xgb_rolling_freq", "seasonal_naive")
    short = {"xgb_adaptive": "за тривогою BOCPD", "xgb_drift": "за зростанням похибки", "xgb_attack": "після удару", "xgb_rolling_freq": "кожні 3 доби", "seasonal_naive": "сезонний наївний"}
    scen = ("demand", "net_import", "outages", "price")
    fig, ax = plt.subplots(figsize=size(9, 3.9, "rt"))
    w = 0.16
    top, bot = 0.0, 0.0
    for j, m in enumerate(models):
        for i, sc in enumerate(scen):
            t = pd.read_csv(ROOT / "reports" / sc / "metrics.csv")
            t = t[(t["split"] == "test") & (t["subset"] == "all")].set_index("model")["MAE"]
            if m not in t.index:
                continue
            rel = 100.0 * t[m] / t["xgb_rolling"] - 100.0
            x = i + (j - (len(models) - 1) / 2) * w
            ax.bar(x, rel, width=w * 0.92, bottom=100.0, color=COLORS[m], label=short[m] if i == 0 else None)
            top, bot = max(top, rel), min(bot, rel)
            lab = f"{rel:+.1f}".replace(".", ",").replace("-", "−")
            ax.text(x, 100.0 + rel + (0.5 if rel >= 0 else -0.5), lab, ha="center", va="bottom" if rel >= 0 else "top", fontsize=F(5.6), rotation=90)
    ax.axhline(100, color="#222", lw=1.1)
    ax.set_xticks(range(len(scen))); ax.set_xticklabels([SCEN_TITLE[s] for s in scen], fontsize=F(8.6))
    ax.set_ylim(100 + min(bot - 8, -10), 100 + top + 12)
    ax.set_ylabel("MAE на тесті, % від щотижневого\nперенавчання (= 100 %)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=F(7.6), ncol=5 if not SLIDE else 3, frameon=False,
              title="Запуск перенавчання (нижче 100 % – точніше за щотижневе)", title_fontsize=F(7.6))
    finish(fig, "fig_retrain.png")


if __name__ == "__main__":
    fig_architecture()
    fig_coverage()
    fig_detection("outages", "fig_detect_outages.png", "частка областей з відключеннями", 0.7)
    fig_detection("net_import", "fig_detect_import.png", "чистий імпорт, МВт (щоденне середнє)", 0.7)
    fig_detection("price", "fig_detect_price.png", "ціна РДН / стеля (щоденне середнє)", 0.7)
    fig_forecasts("net_import", "fig_fc_import.png", "чистий імпорт, МВт (щоденне середнє)", ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_drift"])
    fig_forecasts("outages", "fig_fc_outages.png", "частка областей з відключеннями", ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_drift"])
    fig_forecasts("price", "fig_fc_price.png", "ціна РДН / стеля (щоденне середнє)", ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_drift"],
                  slide_label="частка стелі цін")
    fig_forecasts("outages", "fig_fc_winter.png", "частка областей з відключеннями", ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_drift"],
                  start="2025-10-01", end="2026-04-15", split_line=False)
    fig_mae()
    fig_demand_series()
    fig_post_event()
    fig_post_depth()
    fig_retrain()
    fig_detection("demand", "fig_detect_demand.png", "втрачений попит, МВт (щоденне середнє)", 0.7)
    fig_forecasts("demand", "fig_fc_demand.png", "втрачений попит, МВт (щоденне середнє)",
                  ["seasonal_naive", "xgb_static", "xgb_rolling", "xgb_drift"])
