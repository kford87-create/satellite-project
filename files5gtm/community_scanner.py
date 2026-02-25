"""
tools/community_scanner.py  —  Tool 6: Community Mention & Opportunity Scanner

Watches Reddit (r/gis, r/remotesensing, r/MachineLearning, r/insurtech),
Hacker News, and dev.to for questions about satellite detection, geospatial AI,
and insurance imagery. Surfaces threads where a genuinely helpful answer
also creates natural brand awareness. Drafts response starters.

Think of this like a fishing net — it finds the exact conversations already
happening where your expertise is relevant, so you show up at the right moment
rather than cold-posting promotional content.

Usage:
    python tools/community_scanner.py --dry-run
    python tools/community_scanner.py --platform reddit
    python tools/community_scanner.py --platform hackernews
    python tools/community_scanner.py --min-score 0.7
    python tools/community_scanner.py --output signals/community_2026_02.json
"""

import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field, asdict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

# ── Community Config ──────────────────────────────────────────────────────────

REDDIT_SUBREDDITS = [
    "gis", "remotesensing", "MachineLearning",
    "insurtech", "computervision", "geospatial",
]

HN_SEARCH_QUERIES = [
    "satellite imagery detection",
    "geospatial object detection",
    "satellite AI insurance",
    "aerial image analysis",
]

DEVTO_TAGS = ["geospatial", "computervision", "satellite", "gis"]

# Keywords that signal high relevance for Kestrel AI
HIGH_RELEVANCE_KEYWORDS = [
    "satellite", "aerial", "object detection", "building detection",
    "change detection", "insurance imagery", "geospatial ai",
    "maxar alternative", "planet labs", "picterra", "remote sensing ai",
    "yolo satellite", "satellite api", "gis automation",
]

# Keywords that signal lower relevance (skip these)
LOW_RELEVANCE_KEYWORDS = [
    "sar", "radar", "multispectral", "lidar", "hyperspectral",
    "gps tracking", "navigation", "cartography",
]

# Response templates by topic type
RESPONSE_TEMPLATES = {
    "tool_recommendation": (
        "For {use_case}, Kestrel AI ({url}) is worth checking out — "
        "it detects buildings, vehicles, aircraft, and ships from satellite/aerial imagery "
        "and returns GeoJSON in under 60 seconds via API. Starts at $99/month. "
        "Full disclosure: I'm the founder."
    ),
    "technical_help": (
        "Happy to help with {topic}. The approach that works well for satellite imagery is {advice}. "
        "If you're looking for a managed solution rather than building from scratch, "
        "Kestrel AI handles the pipeline and gives you GeoJSON output at $99/month. "
        "But totally understand if DIY is the goal — happy to go deeper on the technical side."
    ),
    "general_discussion": (
        "Great thread. One thing worth adding: {insight}. "
        "This is something we deal with directly at Kestrel AI — "
        "we built an active learning bootstrapping pipeline to handle exactly this problem."
    ),
}


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class CommunityThread:
    thread_id: str
    platform: str
    subreddit: Optional[str]
    title: str
    url: str
    author: str
    posted_date: str
    score: int                    # Upvotes/points
    comment_count: int
    body_excerpt: str
    matched_keywords: list
    relevance_score: float        # 0-1 computed from keyword matches + engagement
    opportunity_type: str         # tool_recommendation / technical_help / general_discussion
    draft_response: str
    response_urgency: str         # high / medium / low (based on recency + score)


@dataclass
class CommunityReport:
    run_date: str
    platform_filter: Optional[str]
    min_score_filter: float
    total_threads_scanned: int
    relevant_threads: int
    high_opportunity_count: int
    threads: list = field(default_factory=list)
    by_platform: dict = field(default_factory=dict)
    by_opportunity_type: dict = field(default_factory=dict)


# ── Relevance Scorer ──────────────────────────────────────────────────────────

def score_relevance(title: str, body: str, score: int, age_hours: float) -> tuple:
    """
    Score a thread's relevance for Kestrel AI engagement.
    Returns (relevance_score, opportunity_type, matched_keywords).
    """
    text = (title + " " + body).lower()

    # Keyword matching
    matched = [kw for kw in HIGH_RELEVANCE_KEYWORDS if kw in text]
    low_match = [kw for kw in LOW_RELEVANCE_KEYWORDS if kw in text]

    if not matched:
        return 0.0, "irrelevant", []

    relevance = min(len(matched) * 0.15, 0.60)

    # Penalty for off-topic keywords
    relevance -= len(low_match) * 0.10

    # Engagement bonus
    if score > 50:
        relevance += 0.15
    elif score > 10:
        relevance += 0.08

    # Recency bonus
    if age_hours < 6:
        relevance += 0.15
    elif age_hours < 24:
        relevance += 0.10
    elif age_hours < 72:
        relevance += 0.05

    relevance = max(0.0, min(1.0, relevance))

    # Classify opportunity type
    question_signals = ["?", "how do", "what is", "best way", "recommend", "looking for",
                        "suggestions", "alternatives", "which tool"]
    technical_signals = ["code", "python", "api", "model", "inference", "training",
                         "yolo", "detection pipeline"]

    has_question = any(s in text for s in question_signals)
    has_technical = any(s in text for s in technical_signals)

    if has_question and ("recommend" in text or "alternative" in text or "tool" in text):
        opp_type = "tool_recommendation"
    elif has_technical or has_question:
        opp_type = "technical_help"
    else:
        opp_type = "general_discussion"

    return round(relevance, 2), opp_type, matched


def draft_response(thread: dict, opp_type: str) -> str:
    """Draft a contextual response starter for the thread."""
    title_lower = thread.get("title", "").lower()

    if opp_type == "tool_recommendation":
        use_case = "satellite object detection" if "satellite" in title_lower else "aerial imagery analysis"
        return RESPONSE_TEMPLATES["tool_recommendation"].format(
            use_case=use_case, url="kestrelai.com"
        )
    elif opp_type == "technical_help":
        if "active learning" in title_lower or "label" in title_lower:
            advice = "active learning with uncertainty sampling — lets you bootstrap a detector from very few labels"
        elif "yolo" in title_lower or "detection" in title_lower:
            advice = "YOLOv8 with satellite-specific augmentations (rotation, scale) and confidence calibration"
        else:
            advice = "starting with pre-trained weights on SpaceNet or xView datasets before fine-tuning"
        return RESPONSE_TEMPLATES["technical_help"].format(
            topic="satellite imagery detection", advice=advice
        )
    else:
        insight = "false negative rate in satellite detection has real operational cost — worth quantifying per use case"
        return RESPONSE_TEMPLATES["general_discussion"].format(insight=insight)


def urgency(age_hours: float, score: int) -> str:
    if age_hours < 12 and score > 5:
        return "high"
    elif age_hours < 48:
        return "medium"
    return "low"


# ── Platform Scanners ─────────────────────────────────────────────────────────

class RedditScanner:
    """Scans Reddit via the public JSON API — no auth required."""

    BASE = "https://www.reddit.com"
    HEADERS = {"User-Agent": "KestrelAI-CommunityScanner/1.0 (community research bot)"}

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def scan(self) -> list:
        if self.dry_run or not REQUESTS_AVAILABLE:
            return self._mock_threads()

        threads = []
        for sub in REDDIT_SUBREDDITS:
            try:
                url = f"{self.BASE}/r/{sub}/new.json?limit=25"
                resp = requests.get(url, headers=self.HEADERS, timeout=10)
                data = resp.json()
                for post in data.get("data", {}).get("children", []):
                    p = post["data"]
                    age_h = (time.time() - p["created_utc"]) / 3600
                    title = p.get("title", "")
                    body = p.get("selftext", "")[:500]

                    rel, opp, kws = score_relevance(title, body, p.get("score", 0), age_h)
                    if rel < 0.20:
                        continue

                    draft = draft_response({"title": title, "body": body}, opp)
                    threads.append(CommunityThread(
                        thread_id=p["id"],
                        platform="reddit",
                        subreddit=sub,
                        title=title,
                        url=f"{self.BASE}{p['permalink']}",
                        author=p.get("author", "unknown"),
                        posted_date=datetime.fromtimestamp(p["created_utc"]).strftime("%Y-%m-%d %H:%M"),
                        score=p.get("score", 0),
                        comment_count=p.get("num_comments", 0),
                        body_excerpt=body[:200],
                        matched_keywords=kws,
                        relevance_score=rel,
                        opportunity_type=opp,
                        draft_response=draft,
                        response_urgency=urgency(age_h, p.get("score", 0)),
                    ))
                time.sleep(1.5)
            except Exception:
                continue

        return threads

    def _mock_threads(self) -> list:
        mocks = [
            {
                "id": "abc123", "sub": "gis",
                "title": "Best tools for automated building detection from aerial imagery?",
                "body": "I'm working on a project to detect buildings from satellite images. Looking for tools that don't require a massive labeled dataset. Any recommendations?",
                "url": "https://reddit.com/r/gis/comments/abc123",
                "author": "gis_curious",
                "score": 45, "comments": 12, "age_h": 8,
            },
            {
                "id": "def456", "sub": "remotesensing",
                "title": "Object detection on satellite imagery — labeling cost is killing our budget",
                "body": "We're trying to build a vehicle detector for satellite imagery. The labeling cost is insane. Has anyone found a way to do active learning or semi-supervised approaches here?",
                "url": "https://reddit.com/r/remotesensing/comments/def456",
                "author": "satellite_dev",
                "score": 23, "comments": 7, "age_h": 18,
            },
            {
                "id": "ghi789", "sub": "insurtech",
                "title": "Anyone using satellite imagery for property risk assessment?",
                "body": "Our team is evaluating satellite imagery tools for property underwriting. Maxar seems too expensive for our scale. Any alternatives?",
                "url": "https://reddit.com/r/insurtech/comments/ghi789",
                "author": "underwriter_ai",
                "score": 67, "comments": 19, "age_h": 3,
            },
            {
                "id": "jkl012", "sub": "MachineLearning",
                "title": "Change detection in satellite imagery — what's the state of the art?",
                "body": "Looking for approaches that work well for detecting new building construction between two timestamps. Open to both ML approaches and traditional CV.",
                "url": "https://reddit.com/r/MachineLearning/comments/jkl012",
                "author": "ml_researcher",
                "score": 31, "comments": 9, "age_h": 36,
            },
        ]
        threads = []
        for m in mocks:
            rel, opp, kws = score_relevance(m["title"], m["body"], m["score"], m["age_h"])
            draft = draft_response(m, opp)
            threads.append(CommunityThread(
                thread_id=m["id"], platform="reddit",
                subreddit=m["sub"], title=m["title"],
                url=m["url"], author=m["author"],
                posted_date=(datetime.now() - timedelta(hours=m["age_h"])).strftime("%Y-%m-%d %H:%M"),
                score=m["score"], comment_count=m["comments"],
                body_excerpt=m["body"][:200],
                matched_keywords=kws, relevance_score=rel,
                opportunity_type=opp, draft_response=draft,
                response_urgency=urgency(m["age_h"], m["score"]),
            ))
        return threads


class HackerNewsScanner:
    """Scans HN via the Algolia search API — public, no auth required."""

    ALGOLIA = "https://hn.algolia.com/api/v1/search"

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def scan(self) -> list:
        if self.dry_run or not REQUESTS_AVAILABLE:
            return self._mock_threads()

        threads = []
        for query in HN_SEARCH_QUERIES:
            try:
                resp = requests.get(
                    self.ALGOLIA,
                    params={"query": query, "tags": "story", "numericFilters": "created_at_i>%d" % (int(time.time()) - 604800)},
                    timeout=10
                )
                data = resp.json()
                for hit in data.get("hits", [])[:5]:
                    title = hit.get("title", "")
                    body = hit.get("story_text", "")[:500]
                    age_h = (time.time() - hit.get("created_at_i", time.time())) / 3600
                    pts = hit.get("points", 0)

                    rel, opp, kws = score_relevance(title, body, pts, age_h)
                    if rel < 0.20:
                        continue

                    draft = draft_response({"title": title, "body": body}, opp)
                    threads.append(CommunityThread(
                        thread_id=hit["objectID"],
                        platform="hackernews", subreddit=None,
                        title=title,
                        url=f"https://news.ycombinator.com/item?id={hit['objectID']}",
                        author=hit.get("author", ""),
                        posted_date=hit.get("created_at", ""),
                        score=pts,
                        comment_count=hit.get("num_comments", 0),
                        body_excerpt=body[:200],
                        matched_keywords=kws, relevance_score=rel,
                        opportunity_type=opp, draft_response=draft,
                        response_urgency=urgency(age_h, pts),
                    ))
                time.sleep(0.5)
            except Exception:
                continue

        return threads

    def _mock_threads(self) -> list:
        mocks = [
            {
                "id": "hn001",
                "title": "Ask HN: What's a good satellite imagery API for a small startup?",
                "body": "We're building a property risk tool and want to add satellite object detection. Maxar is way out of budget. What are realistic options for early-stage companies?",
                "url": "https://news.ycombinator.com/item?id=hn001",
                "score": 89, "comments": 34, "age_h": 5,
            },
        ]
        threads = []
        for m in mocks:
            rel, opp, kws = score_relevance(m["title"], m["body"], m["score"], m["age_h"])
            draft = draft_response(m, opp)
            threads.append(CommunityThread(
                thread_id=m["id"], platform="hackernews", subreddit=None,
                title=m["title"], url=m["url"], author="hn_user",
                posted_date=(datetime.now() - timedelta(hours=m["age_h"])).strftime("%Y-%m-%d %H:%M"),
                score=m["score"], comment_count=m["comments"],
                body_excerpt=m["body"][:200], matched_keywords=kws,
                relevance_score=rel, opportunity_type=opp, draft_response=draft,
                response_urgency=urgency(m["age_h"], m["score"]),
            ))
        return threads


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(threads: list, platform_filter: Optional[str], min_score: float) -> CommunityReport:
    filtered = [t for t in threads if t.relevance_score >= min_score]
    high_opp = [t for t in filtered if t.relevance_score >= 0.70]

    by_platform = {}
    by_opp = {}
    for t in filtered:
        by_platform[t.platform] = by_platform.get(t.platform, 0) + 1
        by_opp[t.opportunity_type] = by_opp.get(t.opportunity_type, 0) + 1

    return CommunityReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        platform_filter=platform_filter,
        min_score_filter=min_score,
        total_threads_scanned=len(threads),
        relevant_threads=len(filtered),
        high_opportunity_count=len(high_opp),
        threads=[asdict(t) for t in sorted(filtered, key=lambda x: -x.relevance_score)],
        by_platform=by_platform,
        by_opportunity_type=by_opp,
    )


def print_report(report: CommunityReport):
    print(f"\n{'='*60}")
    print(f"💬 COMMUNITY SCANNER REPORT")
    print(f"   {report.run_date}")
    print(f"{'='*60}")
    print(f"\n   Threads scanned:  {report.total_threads_scanned}")
    print(f"   Relevant (≥{report.min_score_filter}):   {report.relevant_threads}")
    print(f"   High opportunity: {report.high_opportunity_count}")
    print(f"   By platform: {report.by_platform}")
    print(f"   By type:     {report.by_opportunity_type}")

    if report.threads:
        print(f"\n🎯 Top Opportunities:")
        for t in report.threads[:4]:
            urg = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(t["response_urgency"], "⚪")
            print(f"\n   {urg} [{t['relevance_score']:.2f}] {t['platform'].upper()} — {t.get('subreddit', '')}")
            print(f"   \"{t['title'][:65]}\"")
            print(f"   👍 {t['score']} pts · 💬 {t['comment_count']} comments · 🔗 {t['url'][:50]}")
            print(f"   Type: {t['opportunity_type']}")
            print(f"   Draft: {t['draft_response'][:100]}...")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scan communities for Kestrel AI engagement opportunities"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data (no real API calls)")
    parser.add_argument("--platform", choices=["reddit", "hackernews"],
                        help="Scan a specific platform only")
    parser.add_argument("--min-score", type=float, default=0.30,
                        help="Minimum relevance score to include (default 0.30)")
    parser.add_argument("--output", type=str,
                        help="Save report to this JSON path")
    args = parser.parse_args()

    print(f"🛰️  Kestrel AI — Community Scanner")
    print(f"   Platform: {args.platform or 'all'}")
    print(f"   Min score: {args.min_score}")
    print(f"   Mode: {'🧪 DRY RUN' if args.dry_run else '🔴 LIVE'}")

    threads = []
    if not args.platform or args.platform == "reddit":
        print("\n   📡 Scanning Reddit...")
        r = RedditScanner(dry_run=args.dry_run).scan()
        threads.extend(r)
        print(f"   Found {len(r)} Reddit threads")

    if not args.platform or args.platform == "hackernews":
        print("\n   📡 Scanning Hacker News...")
        h = HackerNewsScanner(dry_run=args.dry_run).scan()
        threads.extend(h)
        print(f"   Found {len(h)} HN threads")

    report = build_report(threads, args.platform, args.min_score)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2))
        print(f"\n✅ Report saved: {args.output}")


if __name__ == "__main__":
    main()
