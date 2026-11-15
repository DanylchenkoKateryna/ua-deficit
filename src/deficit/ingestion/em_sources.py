"""Polite collector of the source pages linked from the Energy Map records (column ``link``).

Input: data/raw/energy_map/*_oblasti_osr.csv and *_oblasti_minenergo.csv.  Output: gzipped HTML pages in
``data/raw/em_sources/html/<sha1(url)>.html.gz`` and a resumable ``_manifest.csv``
(url, domain, status, fetched_at, sha1, reason).  Only raw pages are stored; parsing is a later step.

Politeness: robots.txt per domain, one request at a time, random 3-5 s delay, 30 s timeout, 2 retries with backoff on
5xx / timeouts, a domain is dropped after 3 consecutive 403/429.  Login-wall domains (facebook, instagram, x/twitter) are
skipped; t.me posts already stored in data/raw/telegram/<channel>.parquet are skipped, others are fetched via the public
preview page ``https://t.me/s/<channel>/<id>`` (the project's Telegram rule: t.me/s pages only).  The User-Agent is generic (no personal data).
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import random
import re
import time
import urllib.robotparser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

log = logging.getLogger("deficit")

USER_AGENT = "Mozilla/5.0 (research crawler; thesis project)"
MIN_DELAY, MAX_DELAY = 3.0, 5.0
TIMEOUT = 30
MAX_RETRIES = 2
BACKOFF = 5.0
MAX_BLOCKED = 3
SKIP_DOMAINS = ("facebook.com", "fb.com", "instagram.com", "x.com", "twitter.com")
LINK_FILES = ("*_oblasti_osr.csv", "*_oblasti_minenergo.csv")
MANIFEST_COLS = ["url", "domain", "status", "fetched_at", "sha1", "reason"]
_URL_RE = re.compile(r"https?://[^\s\"'<>]+")
_TG_RE = re.compile(r"^/(?:s/)?([A-Za-z0-9_]+)/(\d+)")


def clean_links(cell) -> list[str]:
    """All URLs of one ``link`` cell (several may be glued by spaces/commas/newlines; quotes are stripped)."""
    if not isinstance(cell, str):
        return []
    out = []
    for u in _URL_RE.findall(cell.replace(",http", " http").replace(";http", " http")):
        u = u.strip().rstrip(".,;)\"'»”’")
        if u:
            out.append(u)
    return out


def domain_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def is_skipped_domain(dom: str) -> bool:
    return any(dom == d or dom.endswith("." + d) for d in SKIP_DOMAINS)


BOT_MARKERS = (b"_Incapsula_Resource", b"cf-browser-verification", b"challenge-platform", b"Just a moment...")


def is_bot_challenge(body: bytes) -> bool:
    """Anti-bot interstitial (Incapsula/Cloudflare) instead of the page: not stored, counts as a block (never bypassed)."""
    return len(body) < 20000 and any(m in body for m in BOT_MARKERS)


def url_sha1(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def tg_parts(url: str) -> tuple[str, int] | None:
    if domain_of(url) not in ("t.me", "telegram.me"):
        return None
    m = _TG_RE.match(urlparse(url).path)
    return (m.group(1), int(m.group(2))) if m else None


def load_urls(em_dir: Path) -> list[str]:
    seen, out = set(), []
    for pat in LINK_FILES:
        for f in sorted(em_dir.glob(pat)):
            col = pd.read_csv(f, dtype=str, usecols=["link"])["link"]
            for cell in col:
                for u in clean_links(cell):
                    if u not in seen:
                        seen.add(u)
                        out.append(u)
    return out


def stored_tg_posts(tg_dir: Path) -> set[tuple[str, int]]:
    have = set()
    for f in tg_dir.glob("*.parquet"):
        try:
            d = pd.read_parquet(f, columns=["post_id"])
        except Exception:      # unreadable file: treat the channel as not stored
            continue
        ch = f.stem.lower()
        have.update((ch, int(i)) for i in d["post_id"].dropna())
    return have


def read_manifest(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=MANIFEST_COLS)


def plan(urls: list[str], tg_have: set[tuple[str, int]], done: set[str]) -> tuple[list[tuple[str, str]], list[dict]]:
    """Split URLs into ``(url, fetch_url)`` to fetch and manifest rows decided without a request (skips)."""
    todo, skipped = [], []
    for u in urls:
        if u in done:
            continue
        dom = domain_of(u)
        if not dom:
            skipped.append(dict(url=u, domain="", status="skipped", reason="bad_url"))
        elif is_skipped_domain(dom):
            skipped.append(dict(url=u, domain=dom, status="skipped", reason="login_wall"))
        elif (tg := tg_parts(u)) is not None:
            if (tg[0].lower(), tg[1]) in tg_have:
                skipped.append(dict(url=u, domain=dom, status="skipped", reason="in_telegram_parquet"))
            else:
                todo.append((u, f"https://t.me/s/{tg[0]}/{tg[1]}"))      # public preview page only (t.me/s)
        elif dom in ("t.me", "telegram.me"):
            skipped.append(dict(url=u, domain=dom, status="skipped", reason="tg_unparsed"))
        else:
            todo.append((u, u))
    return todo, skipped


def requests_fetcher(url: str) -> tuple[int, bytes]:
    """Return (status, body); raises requests exceptions on network errors."""
    r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept-Language": "uk,en;q=0.8"},
                     allow_redirects=True)
    return r.status_code, r.content


class Robots:
    def __init__(self, fetcher, sleep):
        self.fetcher, self.sleep, self.cache = fetcher, sleep, {}

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        key = f"{p.scheme}://{p.netloc}"
        if key not in self.cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                st, body = self.fetcher(key + "/robots.txt")
                if st == 200:
                    rp.parse(body.decode("utf-8", "replace").splitlines())
                elif st in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.allow_all = True
            except Exception:
                rp.allow_all = True
            self.cache[key] = rp
            self.sleep()          # the robots.txt fetch is a request too
        return self.cache[key].can_fetch(USER_AGENT, url)


def _fetch_with_retries(fetcher, url, sleep_fn):
    last, st = "", None
    for attempt in range(MAX_RETRIES + 1):
        try:
            st, body = fetcher(url)
        except (requests.Timeout, requests.ConnectionError, TimeoutError) as e:
            st, body, last = None, b"", type(e).__name__
        else:
            if st < 500:
                return st, body, ""
            last = f"http_{st}"
        if attempt < MAX_RETRIES:
            sleep_fn(BACKOFF * (attempt + 1))
    return st, b"", last


def collect(em_dir: Path, tg_dir: Path, out_dir: Path, fetcher=requests_fetcher, sleep=time.sleep,
            rng: random.Random | None = None, limit: int | None = None) -> Counter:
    rng = rng or random.Random()
    html_dir = out_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "_manifest.csv"
    man = read_manifest(mpath)
    todo, skipped = plan(load_urls(em_dir), stored_tg_posts(tg_dir), set(man["url"]))
    stats: Counter = Counter()

    def now():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def append(row):
        row = {c: row.get(c, "") for c in MANIFEST_COLS}
        pd.DataFrame([row]).to_csv(mpath, mode="a", header=not mpath.exists(), index=False)

    for r in skipped:
        append(dict(r, fetched_at=now()))
        stats[("skipped:" + r["reason"], r["domain"])] += 1

    def delay():
        sleep(rng.uniform(MIN_DELAY, MAX_DELAY))

    robots, blocked, dead = Robots(fetcher, delay), Counter(), set()
    n = 0
    for url, furl in todo:
        if limit is not None and n >= limit:
            break
        dom = domain_of(url)
        if dom in dead:
            append(dict(url=url, domain=dom, status="skipped", reason="domain_blocked", fetched_at=now()))
            stats[("skipped:domain_blocked", dom)] += 1
            continue
        if not robots.allowed(furl):
            append(dict(url=url, domain=dom, status="skipped", reason="robots_disallow", fetched_at=now()))
            stats[("skipped:robots_disallow", dom)] += 1
            continue
        st, body, why = _fetch_with_retries(fetcher, furl, sleep)
        n += 1
        if st == 200 and is_bot_challenge(body):
            st, why = 403, "bot_challenge"
        if st in (403, 429):
            blocked[dom] += 1
            if blocked[dom] >= MAX_BLOCKED:
                dead.add(dom)
                log.warning("%s: %d consecutive %s answers, domain dropped", dom, blocked[dom], st)
        else:
            blocked[dom] = 0
        if st == 200:
            sha = url_sha1(url)
            (html_dir / f"{sha}.html.gz").write_bytes(gzip.compress(body))
            append(dict(url=url, domain=dom, status="200", fetched_at=now(), sha1=sha))
            stats[("ok", dom)] += 1
        else:
            append(dict(url=url, domain=dom, status=str(st or "error"), fetched_at=now(), reason=why or f"http_{st}"))
            stats[("failed", dom)] += 1
        log.info("[%d/%d] %s %s (%d bytes)", n, len(todo), st, url[:100], len(body))
        delay()
    return stats


def plan_report(em_dir: Path, tg_dir: Path, out_dir: Path) -> dict:
    man = read_manifest(out_dir / "_manifest.csv")
    urls = load_urls(em_dir)
    todo, skipped = plan(urls, stored_tg_posts(tg_dir), set(man["url"]))
    per_dom = Counter(domain_of(u) for u, _ in todo)
    per_skip = Counter((r["reason"], r["domain"]) for r in skipped)
    seconds = (len(todo) + len(per_dom)) * (MIN_DELAY + MAX_DELAY) / 2     # + one robots.txt request per domain
    return dict(unique_urls=len(urls), already_done=len(man), to_fetch=len(todo), per_domain=per_dom,
                skipped=per_skip, est_hours=seconds / 3600)


def run_collect(cfg, plan_only: bool = False) -> None:
    root = cfg._abs(cfg.paths.data_dir)
    em_dir, tg_dir, out_dir = root / "raw" / "energy_map", root / "raw" / "telegram", root / "raw" / "em_sources"
    rep = plan_report(em_dir, tg_dir, out_dir)
    print(f"unique URLs: {rep['unique_urls']}, already in manifest: {rep['already_done']}, to fetch: {rep['to_fetch']}")
    for d, c in rep["per_domain"].most_common():
        print(f"  fetch {d}: {c}")
    for (why, d), c in sorted(rep["skipped"].items(), key=lambda x: -x[1]):
        print(f"  skip[{why}] {d}: {c}")
    print(f"estimated duration: {rep['est_hours']:.1f} h (mean 4 s per request, robots.txt included; "
          "robots-disallowed URLs make it shorter)")
    if plan_only:
        return
    stats = collect(em_dir, tg_dir, out_dir)
    for (k, d), c in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k} {d}: {c}")
