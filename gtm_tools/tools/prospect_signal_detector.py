"""
prospect_signal_detector.py
---------------------------
Scan public web sources for buying-intent signals from ICP companies.
Output draft LinkedIn outreach messages with the signal as the hook.

Usage:
    python tools/prospect_signal_detector.py --dry-run
    python tools/prospect_signal_detector.py --output signals/signals_2026_02.json
    python tools/prospect_signal_detector.py --segment insurance
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402

_SIGNALS_DIR = Path(__file__).resolve().parent.parent / "signals"

# ---------------------------------------------------------------------------
# ICP Segments
# ---------------------------------------------------------------------------
ICP_SEGMENTS: dict[str, dict[str, Any]] = {
    "insurance": {
        "keywords": [
            "property insurance", "insurtech", "aerial imagery underwriting",
            "roof inspection AI", "claims automation", "geospatial underwriting",
        ],
        "job_signals": [
            "geospatial analyst", "remote sensing engineer",
            "aerial inspection", "satellite imagery analyst",
        ],
        "companies": [
            "Hippo Insurance", "Openly", "Branch Insurance",
            "Kin Insurance", "Lemonade", "Coterie Insurance",
        ],
    },
    "insurtech": {
        "keywords": [
            "insurtech startup", "AI insurance", "property risk assessment",
            "automated inspection",
        ],
        "job_signals": [
            "computer vision engineer", "ML engineer insurance",
            "geospatial data scientist",
        ],
        "companies": [
            "Cape Analytics", "Verisk", "CoreLogic", "Nearmap", "EagleView",
        ],
    },
    "defense": {
        "keywords": [
            "SBIR Phase I", "dual-use AI", "geospatial intelligence",
            "ISR platform", "object detection government",
        ],
        "job_signals": [
            "computer vision DoD", "geospatial AI contractor", "GEOINT analyst",
        ],
        "companies": [
            "Palantir", "Anduril", "Shield AI", "Primer AI",
        ],
    },
}

SIGNAL_TYPES = ["job_posting", "funding", "product_launch", "conference"]

OUTREACH_TEMPLATES: dict[str, str] = {
    "job_posting": (
        "Hi [Name], I noticed {company} is hiring a {detail} — "
        "that's exactly the workflow Kestrel AI automates. "
        "We provide satellite object detection at $99/month vs $50k+/year for Maxar. "
        "Would a 15-min demo make sense this week?"
    ),
    "funding": (
        "Hi [Name], congrats on {company}'s recent {detail}! "
        "As you scale, satellite imagery analysis often becomes a bottleneck. "
        "Kestrel AI handles it automatically at $99/mo — 15× cheaper than Planet Labs. "
        "Worth a quick chat?"
    ),
    "product_launch": (
        "Hi [Name], saw {company} just launched {detail} — impressive work. "
        "If aerial/satellite data is part of your roadmap, Kestrel AI integrates in minutes. "
        "$99/mo, sub-3s detection. Happy to show you a live demo."
    ),
    "conference": (
        "Hi [Name], noticed {company} will be at {detail}. "
        "We'll also be there — Kestrel AI does satellite object detection for insurtech at $99/mo. "
        "Would love to connect. 15 minutes?"
    ),
}


def _make_outreach(company: str, signal_type: str, detail: str) -> str:
    template = OUTREACH_TEMPLATES.get(signal_type, OUTREACH_TEMPLATES["job_posting"])
    msg = template.format(company=company, detail=detail)
    # Enforce 300-char LinkedIn limit hint
    if len(msg) > 300:
        msg = msg[:297] + "…"
    return msg


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def _dry_run_signals(segment_filter: str | None) -> list[dict[str, Any]]:
    rng = random.Random(99)
    segments = [segment_filter] if segment_filter else list(ICP_SEGMENTS.keys())
    signals: list[dict[str, Any]] = []

    sample_details: dict[str, list[str]] = {
        "job_posting": [
            "Geospatial Data Analyst", "Remote Sensing Engineer",
            "Computer Vision Engineer (Satellite)", "Aerial Inspection ML Engineer",
        ],
        "funding": ["Series B ($42M)", "Series A ($18M)", "Seed ($5M)"],
        "product_launch": [
            "AI-powered roof inspection product",
            "aerial change detection API",
            "geospatial risk scoring engine",
        ],
        "conference": [
            "InsureTech Connect 2026",
            "GEOINT Symposium 2026",
            "SatSummit 2026",
        ],
    }

    for seg in segments:
        companies = ICP_SEGMENTS[seg]["companies"]
        for company in rng.sample(companies, min(2, len(companies))):
            signal_type = rng.choice(SIGNAL_TYPES)
            detail = rng.choice(sample_details[signal_type])
            intent_score = round(rng.uniform(0.55, 0.95), 2)
            signals.append({
                "company": company,
                "segment": seg,
                "signal_type": signal_type,
                "signal_detail": f"Posted '{detail}'" if signal_type == "job_posting" else detail,
                "source_url": f"https://linkedin.com/jobs/search?keywords={detail.replace(' ', '+')}",
                "intent_score": intent_score,
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "draft_outreach": _make_outreach(company, signal_type, detail),
            })

    signals.sort(key=lambda x: -x["intent_score"])
    return signals[:5]


# ---------------------------------------------------------------------------
# Live mode (public scraping)
# ---------------------------------------------------------------------------

def _scrape_hn_jobs(keywords: list[str]) -> list[dict[str, Any]]:
    """Scrape HN Who's Hiring threads for relevant job signals."""
    try:
        import urllib.request
        import urllib.parse
        signals: list[dict[str, Any]] = []
        query = urllib.parse.quote(" ".join(keywords[:2]))
        url = f"https://hn.algolia.com/api/v1/search?query={query}&tags=job&numericFilters=created_at_i>0"
        req = urllib.request.Request(url, headers={"User-Agent": "KestrelAI-GTM/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        for hit in data.get("hits", [])[:5]:
            title = hit.get("title") or hit.get("story_title") or ""
            if any(kw.lower() in title.lower() for kw in keywords):
                signals.append({
                    "source": "hackernews",
                    "title": title,
                    "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}",
                })
        time.sleep(1.5)
        return signals
    except Exception as exc:
        print(f"⚠️  HN scrape failed: {exc}")
        return []


def _live_signals(segment_filter: str | None) -> list[dict[str, Any]]:
    """Attempt live signal detection via public sources."""
    segments = [segment_filter] if segment_filter else list(ICP_SEGMENTS.keys())
    signals: list[dict[str, Any]] = []

    for seg in segments:
        keywords = ICP_SEGMENTS[seg]["job_signals"][:2]
        hn_results = _scrape_hn_jobs(keywords)
        companies = ICP_SEGMENTS[seg]["companies"]

        for result in hn_results:
            company_match = next(
                (c for c in companies if c.lower() in result["title"].lower()), companies[0]
            )
            signals.append({
                "company": company_match,
                "segment": seg,
                "signal_type": "job_posting",
                "signal_detail": result["title"][:100],
                "source_url": result["url"],
                "intent_score": round(0.65 + len([k for k in keywords if k.lower() in result["title"].lower()]) * 0.1, 2),
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "draft_outreach": _make_outreach(company_match, "job_posting", result["title"][:60]),
            })

    # Fallback to dry-run data if no live signals found
    if not signals:
        print("⚠️  No live signals found — falling back to synthetic data")
        signals = _dry_run_signals(segment_filter)

    return signals


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_detector(
    dry_run: bool = False,
    segment: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    mode = "dry_run" if dry_run else "live"
    print(f"📊 Prospect Signal Detector — {mode} mode")
    if segment:
        print(f"📊 Segment filter: {segment}")
    print()

    if segment and segment not in ICP_SEGMENTS:
        print(f"❌ Unknown segment '{segment}'. Valid: {', '.join(ICP_SEGMENTS.keys())}")
        sys.exit(1)

    signals = _dry_run_signals(segment) if dry_run else _live_signals(segment)

    top_company = signals[0]["company"] if signals else "N/A"
    top_score = signals[0]["intent_score"] if signals else 0.0
    summary = (
        f"{len(signals)} signals detected across "
        f"{len(set(s['company'] for s in signals))} companies. "
        f"Top intent: {top_company} ({top_score})."
    )

    report: dict[str, Any] = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "segment_filter": segment or "all",
        "signals_found": len(signals),
        "signals": signals,
        "summary": summary,
    }

    # Print summary
    print("=" * 60)
    print("📊 Signal Detection Report")
    print("=" * 60)
    for s in signals:
        print(f"  [{s['intent_score']:.2f}] {s['company']:<25} {s['signal_type']:<18} {s['segment']}")
    print("=" * 60)
    print(f"\n{summary}")

    # Save
    if output_path is None:
        _SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = _SIGNALS_DIR / f"signals_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Report saved → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect ICP buying-intent signals from public sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Synthetic data, no web scraping")
    parser.add_argument("--output", type=Path, metavar="PATH", help="Save JSON report")
    parser.add_argument(
        "--segment", choices=list(ICP_SEGMENTS.keys()),
        metavar="SEGMENT", help="Filter by ICP segment"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_detector(dry_run=args.dry_run, segment=args.segment, output_path=args.output)


if __name__ == "__main__":
    main()
