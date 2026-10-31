"""Command-line entry point:  deficit <step> [--synthetic] [--config path]

Steps (thesis §3.3):  ingest -> preprocess -> features -> detect -> train -> evaluate
(`collect` downloads Telegram posts, `extract` turns them into the structured strike-impact dataset)
`run` executes all of them; `synth` replaces `ingest` with generated demo data; `tune`
searches hyper-parameters on the validation period; `tune-triggers` does the same for the attack and drift
refit triggers (needs the features and BOCPD signal, run after `detect`).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from dotenv import load_dotenv

from .config import PROJECT_ROOT, load_config
from .io import read_parquet, setup_logging

log = logging.getLogger("deficit")


def step_ingest(cfg, sources: list[str]) -> None:
    if cfg.synthetic:
        from .ingestion.synthetic import generate
        generate(cfg)
        return
    if "strikes" in sources:
        from .ingestion.strikes import build_attacks
        build_attacks(cfg)
    if "dam" in sources:
        from .ingestion.dam import fetch_dam
        fetch_dam(cfg)
    if "balance" in sources:
        from .ingestion.balance import fetch_balance
        fetch_balance(cfg)
    if "trade" in sources:
        from .ingestion.trade import fetch_trade
        fetch_trade(cfg)
    if "outages" in sources:
        from .ingestion.outages import fetch_outages
        fetch_outages(cfg)
    if "entsoe" in sources:
        from .ingestion.entsoe_client import fetch_all
        fetch_all(cfg)
    if "weather" in sources:
        from .ingestion.weather import fetch_weather
        fetch_weather(cfg)
    if "acled" in sources:
        from .ingestion.acled import fetch_acled
        try:
            fetch_acled(cfg)
        except RuntimeError as e:  # ACLED is optional: the curated attacks.csv is the main input
            log.warning("ACLED skipped: %s", e)


def step_collect_windows(cfg, channels: list[str], since: str, window_days: int, pre_days: int, share: float,
                         max_hours: float, until: dict[str, str], plan_only: bool) -> int:
    """Window mode: posts around the strikes of attacks.csv only (see ingestion/telegram_public.py). Returns an exit code."""
    import pandas as pd

    from .ingestion import telegram_public as tp
    data_root = cfg._abs(cfg.paths.data_dir)
    attacks = pd.read_csv(data_root / "external" / "attacks.csv")
    clocks = {}
    for ch in channels:
        clk = tp.IdClock()
        clk.add_frame(tp.load_store(tp.store_path(data_root, ch)))
        if len(clk) < 2:
            raise SystemExit(f"{ch}: needs at least 2 stored posts to estimate ids (run a normal/query collect first)")
        clocks[ch] = clk
    plan = tp.select_strikes(attacks, clocks, since, window_days, pre_days, share, max_hours, until)
    plan_path = cfg._abs(cfg.paths.reports_dir) / "telegram_windows_plan.md"
    tp.write_plan(plan, plan_path, window_days, pre_days, max_hours, since, until)
    c = plan["cost"]
    print(f"PLAN: {len(plan['selected'])}/{plan['n_strikes']} strikes ({plan['rule']}); estimated {c['requests']} requests = "
          f"{c['hours']:.2f} h at {plan['rate_per_hour']:.0f} requests/h (all strikes: {plan['full_cost']['requests']} requests, "
          f"{plan['full_cost']['hours']:.2f} h)", flush=True)
    for ch, e in c["per_channel"].items():
        print(f"  {ch}: {len(e['windows'])} windows, ~{e['ids']:.0f} ids, ~{e['requests']} requests", flush=True)
    if plan_only:
        return 0
    fetcher = tp.Fetcher()
    t0 = time.time()
    code, all_stats = 0, []
    try:
        for i, ch in enumerate(channels):
            if i:
                log.info("pausing %ds between channels", tp.CHANNEL_PAUSE)
                time.sleep(tp.CHANNEL_PAUSE)
            ds = [d for d in plan["selected"] if d <= until.get(ch, "9999")]
            try:
                all_stats.append(tp.collect_channel_windows(ch, ds, data_root, window_days, pre_days, fetcher))
            except tp.ChannelUnavailable as e:
                log.error("skipped: %s", e)
    except tp.TooManyRequests as e:
        log.error("STOPPED: %s", e)
        code = 3
    hours = (time.time() - t0) / 3600
    tp.quality_report(data_root, cfg._abs(cfg.paths.reports_dir))
    tp.append_actuals(plan_path, all_stats, fetcher.n_requests, hours, fetcher.n429, code)
    log.info("windows done: %d requests in %.2f h (%.0f/h), %d x HTTP 429", fetcher.n_requests, hours,
             fetcher.n_requests / max(hours, 1e-9), fetcher.n429)
    return code


def step_collect(cfg, channels: list[str] | None, since: str, update: bool,
                 query_terms: list[str] | None = None) -> None:
    """Public Telegram posts -> data/raw/telegram/<channel>.parquet (+ manifest and quality report)."""
    import yaml

    from .ingestion.telegram_public import (CHANNEL_PAUSE, ChannelUnavailable, Fetcher,
                                                    collect_channel, collect_channel_query, quality_report)
    if not channels:
        src = yaml.safe_load((PROJECT_ROOT / "config" / "sources.yaml").read_text(encoding="utf-8"))
        channels = [c["name"] for c in src["telegram"] if c.get("enabled", True)]
    data_root = cfg._abs(cfg.paths.data_dir)   # real data even if --synthetic is passed
    if query_terms and update:
        log.warning("--update is ignored in query mode")
    fetcher = Fetcher()
    for i, ch in enumerate(channels):
        if i:
            log.info("pausing %ds between channels", CHANNEL_PAUSE)
            time.sleep(CHANNEL_PAUSE)
        try:
            if query_terms:
                collect_channel_query(ch, query_terms, data_root, since=since, fetcher=fetcher)
            else:
                collect_channel(ch, data_root, since=since, update=update, fetcher=fetcher)
        except ChannelUnavailable as e:
            log.error("skipped: %s", e)
    quality_report(data_root, cfg._abs(cfg.paths.reports_dir))


def step_extract(cfg, channels: list[str] | None) -> None:
    """data/raw/telegram/*.parquet -> structured posts / daily table / strike impact episodes + quality report."""
    from .ingestion.telegram_extract import EpisodeParams, run_extract
    from .ingestion.telegram_quality import write_quality_report
    t = cfg.telegram
    data_root = cfg._abs(cfg.paths.data_dir)   # real data even if --synthetic is passed
    prm = EpisodeParams(horizon_days=t.episode_horizon_days, relapse_days=t.relapse_days,
                        national_channels=tuple(t.national_channels), pre_strike_days=t.pre_strike_days)
    from .ingestion.telegram_regional import load_channel_oblasts, load_window_info
    channel_oblast = load_channel_oblasts(PROJECT_ROOT / "config" / "sources.yaml")
    winfo = load_window_info(PROJECT_ROOT / "config" / "sources.yaml", data_root)
    out = run_extract(data_root, channels, t.exclude_channels, prm, channel_oblast=channel_oblast, window_info=winfo)
    write_quality_report(out["posts"], out.get("episodes"), cfg._abs(cfg.paths.reports_dir),
                         region_posts=out.get("region_posts"), region_episodes=out.get("region_episodes"))


def step_preprocess(cfg) -> None:
    from .preprocessing.master import build_master
    build_master(cfg)


def step_features(cfg) -> None:
    from .features.build import build_features
    build_features(read_parquet(cfg.processed_dir / "master.parquet"), cfg)


def step_detect(cfg) -> None:
    from .changepoints.detect import detect
    from .ingestion.attacks import load_attacks
    detect(read_parquet(cfg.processed_dir / "master.parquet"), load_attacks(cfg), cfg)


def step_train(cfg, models) -> None:
    from .training.run import train_all
    train_all(cfg, models)


def step_tune(cfg) -> None:
    from .training.tune import tune
    tune(cfg)


def step_evaluate(cfg) -> None:
    from .evaluation.report import evaluate
    evaluate(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="deficit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["ingest", "synth", "coverage", "preprocess", "features", "detect", "tune", "tune-triggers",
                                     "train", "evaluate", "events", "restoration", "region-restoration", "run", "collect", "extract", "tg-login"])
    ap.add_argument("--synthetic", action="store_true", help="use generated demo data (no API keys)")
    ap.add_argument("--config", default=None, help="path to config YAML")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="ingest only these sources (default depends on target.definition)")
    ap.add_argument("--models", nargs="+", default=None, help="train only these models")
    ap.add_argument("--tune", action="store_true", help="with `run`: tune before training")
    ap.add_argument("--channels", nargs="+", default=None,
                    help="collect/extract: Telegram channels (collect: enabled ones in config/sources.yaml; "
                         "extract: every parquet in data/raw/telegram except telegram.exclude_channels)")
    ap.add_argument("--since", default=None, help="collect: oldest post date to backfill to (default 2022-02-01); in "
                    "--windows-from-attacks mode: first strike date considered (default 2022-09-01)")
    ap.add_argument("--update", action="store_true", help="collect: only fetch posts newer than stored")
    ap.add_argument("--query-terms", nargs="+", default=None,
                    help="collect: query mode, use the preview search (t.me/s/<ch>?q=<term>) for each term (whole words) "
                         "and store only the hits (column matched_query)")
    ap.add_argument("--windows-from-attacks", action="store_true",
                    help="collect: window mode, only posts from [strike day - 1, strike day + window-days] (UTC) for the strikes "
                         "of data/external/attacks.csv (needs --channels; writes reports/telegram_windows_plan.md first)")
    ap.add_argument("--window-days", type=int, default=7, help="window mode: days after the strike day")
    ap.add_argument("--window-pre-days", type=int, default=1, help="window mode: days before the strike day")
    ap.add_argument("--strike-share", type=float, default=1.0,
                    help="window mode: use at most this share of the strikes (largest weapons_through first)")
    ap.add_argument("--max-hours", type=float, default=6.0,
                    help="window mode: time budget; when the estimate exceeds it, strikes are taken in descending weapons_through")
    ap.add_argument("--channel-until", nargs="+", default=[], metavar="CH=YYYY-MM-DD",
                    help="window mode: no windows after this date for that channel (e.g. dnipropetrovskaoda=2024-10-31)")
    ap.add_argument("--plan-only", action="store_true", help="window mode: print/write the plan, make no request")
    ap.add_argument("--api", action="store_true",
                    help="collect: use the official Telegram API (Telethon, needs `deficit tg-login` once) instead of "
                         "the public t.me/s previews")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    load_dotenv(PROJECT_ROOT / ".env")
    synthetic = a.synthetic or a.step == "synth"
    cfg = load_config(a.config, synthetic=synthetic)
    log.info("data root: %s%s", cfg.data_root, "  [SYNTHETIC]" if synthetic else "")
    if synthetic and cfg.target.definition in ("outage_extent", "net_import", "demand_loss", "dam_price"):
        ap.error("--synthetic has no data for this target; use the real CSV files")
    if a.sources is None:
        a.sources = {"outage_extent": ["strikes", "dam", "outages", "trade", "weather"],
                     "net_import": ["strikes", "dam", "trade", "weather"],
                     "demand_loss": ["strikes", "dam", "balance", "weather"],
                     "dam_price": ["strikes", "dam", "weather"]}.get(cfg.target.definition,
                                                               ["entsoe", "weather", "acled"])

    if a.step == "tg-login":
        from .ingestion.telegram_api import login
        login(cfg._abs(cfg.paths.data_dir))
    elif a.step == "collect" and a.api:
        from .ingestion.telegram_api import TelegramApiError, run_collect_api
        import yaml
        sources_yaml = PROJECT_ROOT / "config" / "sources.yaml"
        chans = a.channels or [c["name"] for c in yaml.safe_load(sources_yaml.read_text(encoding="utf-8"))["telegram"]
                               if c.get("enabled", True)]
        try:
            run_collect_api(chans, cfg._abs(cfg.paths.data_dir), cfg._abs(cfg.paths.reports_dir), sources_yaml,
                            a.since or "2022-02-01", a.update)
        except TelegramApiError as e:
            ap.error(str(e))
    elif a.step == "collect" and a.windows_from_attacks:
        if not a.channels:
            ap.error("--windows-from-attacks needs --channels")
        until = dict(x.split("=", 1) for x in a.channel_until)
        return step_collect_windows(cfg, a.channels, a.since or "2022-09-01", a.window_days, a.window_pre_days,
                                    a.strike_share, a.max_hours, until, a.plan_only)
    elif a.step == "collect":
        step_collect(cfg, a.channels, a.since or "2022-02-01", a.update, a.query_terms)
    elif a.step == "extract":
        step_extract(cfg, a.channels)
    elif a.step == "coverage":
        from .ingestion.entsoe_coverage import run_coverage
        run_coverage(cfg)
    elif a.step in ("ingest", "synth"):
        step_ingest(cfg, a.sources)
    elif a.step == "preprocess":
        step_preprocess(cfg)
    elif a.step == "features":
        step_features(cfg)
    elif a.step == "detect":
        step_detect(cfg)
    elif a.step == "tune":
        step_tune(cfg)
    elif a.step == "tune-triggers":
        from .training.tune import tune_triggers
        tune_triggers(cfg)
    elif a.step == "train":
        step_train(cfg, a.models)
    elif a.step == "evaluate":
        step_evaluate(cfg)
    elif a.step == "restoration":
        from .evaluation.restoration import run_restoration
        run_restoration(cfg)
    elif a.step == "region-restoration":
        from .evaluation.region_restoration import run_region_restoration
        run_region_restoration(cfg)
    elif a.step == "events":
        from .evaluation.post_event import run_post_event
        run_post_event(cfg)
    elif a.step == "run":
        step_ingest(cfg, a.sources)
        step_preprocess(cfg)
        step_features(cfg)
        step_detect(cfg)
        if a.tune:
            step_tune(cfg)
        step_train(cfg, a.models)
        step_evaluate(cfg)
        from .evaluation.post_event import run_post_event
        run_post_event(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
