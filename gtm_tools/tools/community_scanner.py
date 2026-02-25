"""
community_scanner.py
--------------------
Watch Reddit, Hacker News, dev.to, GIS Stack Exchange, and IndieHackers
for questions about satellite detection / geospatial AI / insurance imagery.
Surface threads where a genuinely helpful answer creates natural brand awareness.

Usage:
    python tools/community_scanner.py --dry-run
    python tools/community_scanner.py --output signals/community_2026_02.json
    python tools/community_scanner.py --platform reddit --min-score 0.7
    python tools/community_scanner.py --platform stackexchange --min-score 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402

_SIGNALS_DIR = Path(__file__).resolve().parent.parent / "signals"

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")

# ---------------------------------------------------------------------------
# Sources config
# ---------------------------------------------------------------------------
SOURCES: dict[str, Any] = {
    "reddit": {
        "subreddits": ["gis", "remotesensing", "MachineLearning", "insurtech", "geospatial", "datascience"],
        "api_url": "https://www.reddit.com/r/{sub}/search.json?q={query}&sort=new&limit=10&t=week",
        "user_agent": "KestrelAI-GTM-Scanner/1.0 (by /u/kestrel_ai_bot)",
    },
    "hackernews": {
        "api_url": "https://hn.algolia.com/api/v1/search?query={query}&tags=comment,story&numericFilters=created_at_i>{ts}",
        "lookback_days": 7,
    },
    "devto": {
        "api_url": "https://dev.to/api/articles?tag={tag}&per_page=10",
        "tags": ["gis", "machinelearning", "remotesensing", "geospatial",
                 "satellite", "computervision", "python", "api", "insurance"],
    },
    "stackexchange": {
        "api_url": "https://api.stackexchange.com/2.3/search/advanced",
        "site": "gis",
        "tags": ["remote-sensing", "satellite-image", "object-detection",
                 "image-classification", "yolo", "building-footprint"],
        "pagesize": 15,
    },
    "indiehackers": {
        "api_url": "https://www.indiehackers.com/search?q={query}",
    },
}

SEARCH_QUERIES = [
    "satellite object detection",
    "aerial imagery AI",
    "geospatial machine learning",
    "insurance satellite imagery",
    "building detection satellite",
    "Picterra alternative",
    "affordable satellite API",
    "satellite image analysis API",
    "overhead imagery detection",
    "property risk satellite",
    "geospatial API pricing",
    "satellite detection cost",
]

# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def _score_thread(title: str, body: str = "", score: int = 0, age_hours: float = 48.0) -> float:
    text = (title + " " + body).lower()
    s = 0.0

    if "?" in title:
        s += 0.2
    if any(w in text for w in ["budget", "cost", "price", "affordable", "cheap", "expensive"]):
        s += 0.2
    if any(w in text for w in ["insurance", "property", "construction", "real estate", "underwriting"]):
        s += 0.15
    if any(w in text for w in ["picterra", "planet labs", "maxar", "eosda", "flypix"]):
        s += 0.15
    if any(w in text for w in ["object detection", "building detection", "yolo", "neural network"]):
        s += 0.2
    if age_hours < 24:
        s += 0.1

    return round(min(s, 1.0), 2)


def _draft_response(thread: dict[str, Any]) -> str:
    title = thread.get("title", "")
    score = thread.get("relevance_score", 0.0)
    text_lower = title.lower()

    if "affordable" in text_lower or "cheap" in text_lower or "cost" in text_lower:
        opener = (
            "Great question on cost. Most enterprise satellite detection platforms "
            "(Maxar, Planet Labs) require $10k+/month contracts — way out of reach for "
            "smaller teams. There are a few more accessible options now."
        )
        kestrel_mention = (
            "One I've been building is Kestrel AI — YOLOv8-based satellite object detection "
            "at $99/month (200 searches) via the web app, with API access on the $399 plan. "
            "Sub-3s detection, 88.7% mAP on buildings/vehicles/aircraft/ships."
        )
    elif "alternative" in text_lower or "picterra" in text_lower:
        opener = (
            "Picterra is solid but the pricing starts around $490/month and goes up quickly. "
            "A few alternatives worth evaluating depending on your use case:"
        )
        kestrel_mention = (
            "**Kestrel AI** — $99/month, focuses on building/vehicle/aircraft/ship detection "
            "via Google Maps satellite imagery. Fast (sub-3s), good for insurance/construction "
            "workflows. Full disclosure: I built it, so take that with appropriate salt."
        )
    else:
        opener = (
            "For satellite object detection, there are a few tiers of tooling depending on "
            "your scale and budget."
        )
        kestrel_mention = (
            "For mid-market use cases (insurance, construction, real estate) — Kestrel AI "
            "does this with YOLOv8 at $99/month. Address → satellite image → detection results "
            "in under 3 seconds. Works well for building footprint and vehicle detection."
        )

    cta = "Happy to share more or answer specific technical questions if useful."
    return f"{opener}\n\n{kestrel_mention}\n\n{cta}"


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

_SYNTHETIC_THREADS = [
    {
        "platform": "reddit", "subreddit": "gis",
        "title": "What's the most affordable satellite object detection API for a small startup?",
        "url": "https://reddit.com/r/gis/comments/mock1",
        "score": 234, "comments": 18, "age_hours": 6.0,
        "body": "We're building an insurtech product and need satellite detection but Maxar is way out of budget.",
    },
    {
        "platform": "reddit", "subreddit": "insurtech",
        "title": "Anyone using aerial imagery for automated roof inspection?",
        "url": "https://reddit.com/r/insurtech/comments/mock2",
        "score": 87, "comments": 11, "age_hours": 14.0,
        "body": "Looking for a Picterra alternative that doesn't cost a fortune.",
    },
    {
        "platform": "hackernews",
        "title": "Ask HN: Best tools for detecting buildings from satellite images?",
        "url": "https://news.ycombinator.com/item?id=mock3",
        "score": 156, "comments": 43, "age_hours": 20.0,
        "body": "Need object detection on satellite imagery for a government project. Open to SaaS or self-hosted.",
    },
    {
        "platform": "devto",
        "title": "Building a geospatial AI pipeline for property risk assessment",
        "url": "https://dev.to/mock_author/geospatial-ai-pipeline",
        "score": 42, "comments": 7, "age_hours": 48.0,
        "body": "Exploring ML approaches for satellite-based property inspection.",
    },
    {
        "platform": "reddit", "subreddit": "MachineLearning",
        "title": "YOLOv8 performance on satellite imagery — anyone benchmarked this?",
        "url": "https://reddit.com/r/MachineLearning/comments/mock5",
        "score": 312, "comments": 29, "age_hours": 36.0,
        "body": "Testing object detection models on aerial/satellite imagery for a research project.",
    },
    {
        "platform": "hackernews",
        "title": "Show HN: Open-source building footprint detection from satellite images",
        "url": "https://news.ycombinator.com/item?id=mock6",
        "score": 89, "comments": 15, "age_hours": 72.0,
        "body": "We built a tool for detecting buildings from satellite images.",
    },
    {
        "platform": "stackexchange",
        "title": "How to detect buildings from satellite imagery using deep learning?",
        "url": "https://gis.stackexchange.com/questions/mock7",
        "score": 12, "comments": 4, "age_hours": 18.0,
        "body": "remote-sensing satellite-image object-detection building-footprint. Looking for an affordable API or model for automated building detection from overhead imagery.",
    },
    {
        "platform": "stackexchange",
        "title": "Comparing satellite object detection platforms for insurance use case?",
        "url": "https://gis.stackexchange.com/questions/mock8",
        "score": 8, "comments": 3, "age_hours": 30.0,
        "body": "remote-sensing image-classification. We need to detect property structures for insurance underwriting. Evaluating Picterra vs other options.",
    },
    {
        "platform": "devto",
        "title": "Building an insurtech MVP with satellite imagery and computer vision",
        "url": "https://dev.to/mock_author2/insurtech-satellite-mvp",
        "score": 28, "comments": 5, "age_hours": 12.0,
        "body": "How we built a property risk assessment tool using affordable satellite detection APIs for our insurance startup.",
    },
]


def _dry_run_threads(min_score: float, platform_filter: str | None) -> list[dict[str, Any]]:
    threads = []
    for t in _SYNTHETIC_THREADS:
        if platform_filter and t["platform"] != platform_filter:
            continue
        relevance = _score_thread(t["title"], t.get("body", ""), t["score"], t["age_hours"])
        entry: dict[str, Any] = {
            "platform": t["platform"],
            "title": t["title"],
            "url": t["url"],
            "score": t["score"],
            "comments": t["comments"],
            "relevance_score": relevance,
            "engagement_opportunity": "high" if relevance >= 0.7 else ("medium" if relevance >= 0.5 else "low"),
            "keywords_matched": [w for w in ["satellite", "affordable", "insurance", "detection", "picterra"] if w in t["title"].lower() or w in t.get("body", "").lower()],
        }
        if t.get("subreddit"):
            entry["subreddit"] = t["subreddit"]
        if relevance >= 0.7:
            entry["draft_response"] = _draft_response(entry)
        threads.append(entry)

    threads = [t for t in threads if t["relevance_score"] >= min_score]
    threads.sort(key=lambda x: -x["relevance_score"])
    return threads


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

def _fetch_reddit(subreddit: str, query: str) -> list[dict[str, Any]]:
    try:
        import urllib.request
        url = SOURCES["reddit"]["api_url"].format(sub=subreddit, query=query.replace(" ", "+"))
        req = urllib.request.Request(url, headers={"User-Agent": SOURCES["reddit"]["user_agent"]})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        results = []
        now_ts = datetime.now(timezone.utc).timestamp()
        for post in data.get("data", {}).get("children", []):
            d = post.get("data", {})
            age_hours = (now_ts - d.get("created_utc", now_ts)) / 3600
            results.append({
                "platform": "reddit",
                "subreddit": subreddit,
                "title": d.get("title", ""),
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "score": d.get("score", 0),
                "comments": d.get("num_comments", 0),
                "age_hours": age_hours,
                "body": d.get("selftext", "")[:500],
            })
        time.sleep(1.5)
        return results
    except Exception as exc:
        print(f"⚠️  Reddit fetch failed ({subreddit}/{query}): {exc}")
        return []


def _fetch_hn(query: str) -> list[dict[str, Any]]:
    try:
        import urllib.request
        lookback = 7 * 86400
        ts = int(datetime.now(timezone.utc).timestamp()) - lookback
        url = SOURCES["hackernews"]["api_url"].format(query=query.replace(" ", "+"), ts=ts)
        req = urllib.request.Request(url, headers={"User-Agent": "KestrelAI-GTM/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        now_ts = datetime.now(timezone.utc).timestamp()
        results = []
        for hit in data.get("hits", [])[:5]:
            created = hit.get("created_at_i", now_ts)
            age_hours = (now_ts - created) / 3600
            results.append({
                "platform": "hackernews",
                "title": hit.get("title") or hit.get("story_title") or "",
                "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                "score": hit.get("points") or 0,
                "comments": hit.get("num_comments") or 0,
                "age_hours": age_hours,
                "body": hit.get("comment_text") or "",
            })
        time.sleep(1.5)
        return results
    except Exception as exc:
        print(f"⚠️  HN fetch failed: {exc}")
        return []


def _fetch_stackexchange(tags: list[str] | None = None) -> list[dict[str, Any]]:
    """Fetch recent questions from GIS Stack Exchange (public API, no auth needed)."""
    try:
        import urllib.request
        import urllib.parse
        cfg = SOURCES["stackexchange"]
        params: dict[str, str] = {
            "order": "desc",
            "sort": "creation",
            "site": cfg["site"],
            "pagesize": str(cfg["pagesize"]),
            "filter": "default",
        }
        if tags:
            params["tagged"] = ";".join(tags[:3])
        url = f"{cfg['api_url']}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "KestrelAI-GTM/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            import gzip
            raw_data = resp.read()
            try:
                data = json.loads(gzip.decompress(raw_data))
            except (OSError, gzip.BadGzipFile):
                data = json.loads(raw_data)
        now_ts = datetime.now(timezone.utc).timestamp()
        results = []
        for item in data.get("items", []):
            age_hours = (now_ts - item.get("creation_date", now_ts)) / 3600
            results.append({
                "platform": "stackexchange",
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "score": item.get("score", 0),
                "comments": item.get("answer_count", 0),
                "age_hours": age_hours,
                "body": " ".join(item.get("tags", [])),
            })
        time.sleep(1.0)
        return results
    except Exception as exc:
        print(f"⚠️  Stack Exchange fetch failed: {exc}")
        return []


def _fetch_devto(tag: str) -> list[dict[str, Any]]:
    """Fetch recent articles from dev.to by tag (public API, no auth needed)."""
    try:
        import urllib.request
        url = SOURCES["devto"]["api_url"].format(tag=tag)
        req = urllib.request.Request(url, headers={"User-Agent": "KestrelAI-GTM/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            articles = json.loads(resp.read())
        now_ts = datetime.now(timezone.utc).timestamp()
        results = []
        for article in articles:
            published = article.get("published_at", "")
            try:
                pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                age_hours = (now_ts - pub_dt.timestamp()) / 3600
            except (ValueError, AttributeError):
                age_hours = 48.0
            results.append({
                "platform": "devto",
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "score": article.get("positive_reactions_count", 0),
                "comments": article.get("comments_count", 0),
                "age_hours": age_hours,
                "body": article.get("description", ""),
            })
        time.sleep(1.0)
        return results
    except Exception as exc:
        print(f"⚠️  dev.to fetch failed ({tag}): {exc}")
        return []


def _live_threads(min_score: float, platform_filter: str | None) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []

    # Reddit — only if credentials are available (requires OAuth approval)
    if not platform_filter or platform_filter == "reddit":
        if REDDIT_CLIENT_ID:
            for sub in SOURCES["reddit"]["subreddits"][:3]:
                for q in SEARCH_QUERIES[:3]:
                    raw.extend(_fetch_reddit(sub, q))
        else:
            print("ℹ️  Reddit skipped — REDDIT_CLIENT_ID not set. Add credentials to .env when approved.")

    if not platform_filter or platform_filter == "hackernews":
        for q in SEARCH_QUERIES[:5]:
            raw.extend(_fetch_hn(q))

    # Stack Exchange GIS — public API, no auth needed
    if not platform_filter or platform_filter == "stackexchange":
        se_tags = SOURCES["stackexchange"]["tags"]
        # Fetch in batches of 3 tags (SE API supports semicolon-separated tags)
        for i in range(0, len(se_tags), 3):
            raw.extend(_fetch_stackexchange(se_tags[i:i+3]))

    # dev.to — public API, no auth needed
    if not platform_filter or platform_filter == "devto":
        for tag in SOURCES["devto"]["tags"]:
            raw.extend(_fetch_devto(tag))

    # Pre-filter: discard posts with no relevant keywords in title or body
    KEYWORD_FILTER = [
        "satellite", "aerial", "geospatial", "gis", "remote sensing",
        "imagery", "object detection", "insurance", "insurtech",
        "picterra", "planet labs", "maxar", "eosda", "flypix",
        "yolo", "building detection", "affordable api",
        "building footprint", "overhead", "property risk",
        "satellite api", "detection cost",
    ]
    raw = [
        t for t in raw
        if any(kw in (t.get("title", "") + " " + t.get("body", "")).lower() for kw in KEYWORD_FILTER)
    ]

    # Deduplicate by URL
    seen: set[str] = set()
    deduped = []
    for t in raw:
        if t["url"] not in seen:
            seen.add(t["url"])
            deduped.append(t)

    threads = []
    for t in deduped:
        relevance = _score_thread(t["title"], t.get("body", ""), t.get("score", 0), t.get("age_hours", 48))
        if relevance < min_score:
            continue
        entry: dict[str, Any] = {
            "platform": t["platform"],
            "title": t["title"],
            "url": t["url"],
            "score": t.get("score", 0),
            "comments": t.get("comments", 0),
            "relevance_score": relevance,
            "engagement_opportunity": "high" if relevance >= 0.7 else "medium",
            "keywords_matched": [w for w in ["satellite", "affordable", "insurance", "detection"] if w in t["title"].lower()],
        }
        if t.get("subreddit"):
            entry["subreddit"] = t["subreddit"]
        if relevance >= 0.7:
            entry["draft_response"] = _draft_response(entry)
        threads.append(entry)

    threads.sort(key=lambda x: -x["relevance_score"])
    return threads[:20]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_scanner(
    dry_run: bool = False,
    platform: str | None = None,
    min_score: float = 0.6,
    output_path: Path | None = None,
) -> dict[str, Any]:
    mode = "dry_run" if dry_run else "live"
    print(f"📊 Community Scanner — {mode} mode")
    if platform:
        print(f"📊 Platform filter: {platform}")
    print(f"📊 Min relevance score: {min_score}")
    print()

    threads = _dry_run_threads(min_score, platform) if dry_run else _live_threads(min_score, platform)

    platform_counts: dict[str, int] = {}
    for t in threads:
        platform_counts[t["platform"]] = platform_counts.get(t["platform"], 0) + 1

    avg_score = round(sum(t["relevance_score"] for t in threads) / max(len(threads), 1), 2)
    platform_str = ", ".join(f"{k}: {v}" for k, v in platform_counts.items())
    summary = f"{len(threads)} high-relevance threads found. {platform_str}."

    report: dict[str, Any] = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "platform_filter": platform or "all",
        "min_score": min_score,
        "threads_surfaced": len(threads),
        "avg_relevance_score": avg_score,
        "threads": threads,
        "summary": summary,
    }

    print("=" * 60)
    print("📊 Community Scan Results")
    print("=" * 60)
    for t in threads:
        icon = "🔥" if t["relevance_score"] >= 0.8 else ("✅" if t["relevance_score"] >= 0.6 else "  ")
        print(f"  {icon} [{t['relevance_score']:.2f}] [{t['platform']:<11}] {t['title'][:55]}")
    print("=" * 60)
    print(f"\n{summary}")

    if output_path is None:
        _SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = _SIGNALS_DIR / f"community_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Report saved → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan Reddit/HN/dev.to for satellite detection discussion opportunities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic threads, no network calls")
    parser.add_argument("--output", type=Path, metavar="PATH", help="Save JSON report")
    parser.add_argument("--platform", choices=["reddit", "hackernews", "devto", "stackexchange"], help="Scan one platform only")
    parser.add_argument("--min-score", type=float, default=0.6, metavar="FLOAT", help="Minimum relevance score (0-1)")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_scanner(
        dry_run=args.dry_run,
        platform=args.platform,
        min_score=args.min_score,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
