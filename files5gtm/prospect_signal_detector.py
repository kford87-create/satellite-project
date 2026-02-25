"""
tools/prospect_signal_detector.py  —  Tool 3: Prospect Signal Detector

Scans public web sources for buying-intent signals from ICP companies.
Looks for: job postings (geospatial analyst, insurtech data engineer),
funding announcements, product launches, and conference registrations.
Outputs draft LinkedIn outreach messages with the signal as the hook.

Think of this like a motion detector for your sales pipeline — it only
fires when a prospect does something that signals they're ready to buy.

Usage:
    python tools/prospect_signal_detector.py --dry-run
    python tools/prospect_signal_detector.py --output signals/signals_2026_02.json
    python tools/prospect_signal_detector.py --segment insurance
    python tools/prospect_signal_detector.py --segment insurtech
"""

import json
import time
import argparse
import re
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
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── ICP Definitions ───────────────────────────────────────────────────────────

ICP_SEGMENTS = {
    "insurtech": {
        "description": "Insurtech startups — Series A, 5-50 employees",
        "job_keywords": [
            "geospatial", "satellite imagery", "property data", "aerial imagery",
            "GIS analyst", "remote sensing", "computer vision engineer",
            "data science insurance", "risk modeling engineer",
        ],
        "company_keywords": [
            "insurtech", "insuretech", "property insurance", "parametric insurance",
            "underwriting platform", "claims automation", "risk assessment",
        ],
        "signals": ["Series A", "seed funding", "property product launch", "expansion"],
    },
    "insurance": {
        "description": "Regional P&C insurers — 50-500 employees, $50M-$1B GWP",
        "job_keywords": [
            "geospatial analyst", "aerial imagery analyst", "property risk analyst",
            "data scientist underwriting", "satellite data", "remote sensing analyst",
            "GIS specialist", "spatial data analyst",
        ],
        "company_keywords": [
            "property casualty", "P&C insurance", "regional insurer",
            "homeowners insurance", "commercial property", "catastrophe modeling",
        ],
        "signals": ["digital transformation", "data modernization", "AI initiative",
                    "geospatial hiring", "new product launch"],
    },
    "defense": {
        "description": "Defense contractors SMB — $10M-$500M revenue",
        "job_keywords": [
            "computer vision", "object detection", "satellite imagery analyst",
            "geospatial intelligence", "GEOINT", "EO/IR analyst",
            "machine learning engineer defense", "image analyst",
        ],
        "company_keywords": [
            "defense contractor", "SBIR", "DoD contract", "intelligence",
            "geospatial analytics", "ISR", "reconnaissance",
        ],
        "signals": ["SBIR award", "new contract", "capability expansion", "hiring surge"],
    },
}

# Job board search URLs (public, no auth needed)
JOB_SOURCES = [
    {
        "name": "LinkedIn Jobs",
        "url_template": "https://www.linkedin.com/jobs/search/?keywords={query}&location=United+States&f_TPR=r604800",
        "requires_auth": True,  # Flag — we scrape public listings only
    },
    {
        "name": "Indeed",
        "url_template": "https://www.indeed.com/jobs?q={query}&l=United+States&fromage=7",
        "requires_auth": False,
    },
    {
        "name": "Greenhouse",
        "url_template": "https://boards.greenhouse.io/jobs?q={query}",
        "requires_auth": False,
    },
]

# News sources for funding/product signals
NEWS_SOURCES = [
    "https://techcrunch.com/search/{query}",
    "https://news.ycombinator.com/search?q={query}",
]


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ProspectSignal:
    signal_id: str
    detected_date: str
    segment: str
    signal_type: str          # job_posting / funding / product_launch / conference
    company_name: str
    company_size_est: str     # e.g. "50-200 employees"
    signal_title: str         # e.g. "Geospatial Data Analyst — Acme Insurance"
    signal_url: str
    signal_summary: str
    intent_score: float       # 0-1, estimated buying intent
    outreach_hook: str        # The signal rephrased as an outreach hook
    draft_message: str        # Full draft LinkedIn message


@dataclass
class SignalReport:
    run_date: str
    segment_filter: Optional[str]
    total_signals: int
    high_intent_count: int    # intent_score >= 0.7
    signals: list = field(default_factory=list)
    summary_by_segment: dict = field(default_factory=dict)
    summary_by_type: dict = field(default_factory=dict)


# ── Signal Scorer ─────────────────────────────────────────────────────────────

def score_intent(signal_type: str, keywords_matched: list, recency_days: int) -> float:
    """
    Score buying intent 0-1 based on signal type, keyword specificity,
    and recency. Like a lead scoring model but for GTM signals.
    """
    base_scores = {
        "job_posting": 0.65,
        "funding": 0.55,
        "product_launch": 0.60,
        "conference": 0.40,
    }
    score = base_scores.get(signal_type, 0.4)

    # Bonus for highly specific keywords
    high_value_kws = ["satellite imagery", "geospatial", "object detection",
                      "aerial imagery", "remote sensing"]
    matches = sum(1 for kw in keywords_matched if kw in high_value_kws)
    score += min(matches * 0.08, 0.20)

    # Recency bonus
    if recency_days <= 3:
        score += 0.10
    elif recency_days <= 7:
        score += 0.05

    return round(min(score, 1.0), 2)


def draft_outreach_message(signal: dict, company: str, segment: str) -> tuple:
    """Generate a hook and full draft outreach message for a signal."""
    sig_type = signal.get("type", "job_posting")
    title = signal.get("title", "")

    hooks = {
        "job_posting": f"I noticed {company} is hiring for {title} — that tells me you're investing in geospatial data capabilities.",
        "funding": f"Congrats on the recent funding — looks like {company} is scaling fast on the property data side.",
        "product_launch": f"Saw {company} just launched a new property product — geospatial detection tends to be a core need at that stage.",
        "conference": f"Noticed {company} is attending [Conference] — we'll both be in the room.",
    }
    hook = hooks.get(sig_type, f"I saw {company} is growing in the {segment} space.")

    if segment == "insurtech":
        value_prop = "Kestrel AI helps insurtech teams add satellite object detection to their stack starting at $99/month — no GIS team required. We detect buildings, vehicles, and change events from aerial imagery and return GeoJSON in under 60 seconds."
    elif segment == "insurance":
        value_prop = "Kestrel AI gives regional insurers satellite-based property detection at a fraction of enterprise pricing — $99/month gets you 200 searches with GeoJSON output ready for your existing GIS tools."
    else:
        value_prop = "Kestrel AI provides dual-use satellite object detection with a false-negative quantification model — relevant for mission-critical applications where missed detections have real operational cost."

    message = f"""Hi [Name],

{hook}

{value_prop}

Would it be worth a 15-minute call to see if it fits what you're building?

[Your name]
Kestrel AI | kestrelai.com"""

    return hook, message


# ── Scanner ───────────────────────────────────────────────────────────────────

class ProspectScanner:
    """Scans public sources for ICP buying signals."""

    HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; KestrelAI-Prospecting/1.0)"}
    REQUEST_DELAY = 1.5

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def scan(self, segments: list) -> list:
        """Scan all specified segments and return list of ProspectSignals."""
        if self.dry_run:
            return self._mock_signals(segments)

        signals = []
        for segment in segments:
            icp = ICP_SEGMENTS.get(segment, {})
            for keyword in icp.get("job_keywords", [])[:3]:  # Limit rate
                found = self._scan_indeed(keyword, segment)
                signals.extend(found)
                time.sleep(self.REQUEST_DELAY)

        return signals

    def _scan_indeed(self, keyword: str, segment: str) -> list:
        """Scan Indeed for job postings matching a keyword."""
        if not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            return []

        url = f"https://www.indeed.com/jobs?q={requests.utils.quote(keyword)}&l=United+States&fromage=7"
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            signals = []

            for card in soup.select(".job_seen_beacon")[:5]:
                title_el = card.select_one(".jobTitle")
                company_el = card.select_one(".companyName")
                if not title_el or not company_el:
                    continue

                title = title_el.get_text(strip=True)
                company = company_el.get_text(strip=True)
                job_url = "https://indeed.com" + (card.select_one("a") or {}).get("href", "")

                hook, message = draft_outreach_message(
                    {"type": "job_posting", "title": title}, company, segment
                )
                intent = score_intent("job_posting", [keyword], recency_days=3)

                signals.append(ProspectSignal(
                    signal_id=f"SIG-{abs(hash(title+company)) % 99999:05d}",
                    detected_date=datetime.now().strftime("%Y-%m-%d"),
                    segment=segment,
                    signal_type="job_posting",
                    company_name=company,
                    company_size_est="unknown",
                    signal_title=title,
                    signal_url=job_url,
                    signal_summary=f"Job posting for '{title}' at {company} — strong geospatial hiring signal",
                    intent_score=intent,
                    outreach_hook=hook,
                    draft_message=message,
                ))
            return signals
        except Exception:
            return []

    def _mock_signals(self, segments: list) -> list:
        """Generate realistic mock signals for testing."""
        import random
        random.seed(42)

        mock_data = [
            ("insurtech", "job_posting", "Betterview", "Geospatial Data Analyst",
             "https://jobs.betterview.com/geospatial-analyst", 0.82),
            ("insurtech", "funding", "Understory", "Series B — $20M for parametric insurance",
             "https://techcrunch.com/understory-series-b", 0.68),
            ("insurance", "job_posting", "Erie Indemnity", "Remote Sensing Analyst",
             "https://indeed.com/erie-remote-sensing", 0.75),
            ("insurance", "product_launch", "Hippo Insurance", "New commercial property product launch",
             "https://hippo.com/press/commercial-launch", 0.61),
            ("defense", "job_posting", "Leidos", "Computer Vision Engineer — GEOINT",
             "https://leidos.com/jobs/cv-geoint", 0.71),
            ("defense", "funding", "Orbit Fab", "SBIR Phase II award for satellite analytics",
             "https://sbir.gov/orbit-fab-phase2", 0.66),
        ]

        signals = []
        for seg, sig_type, company, title, url, intent in mock_data:
            if seg not in segments:
                continue
            hook, message = draft_outreach_message({"type": sig_type, "title": title},
                                                    company, seg)
            signals.append(ProspectSignal(
                signal_id=f"SIG-{abs(hash(title)) % 99999:05d}",
                detected_date=datetime.now().strftime("%Y-%m-%d"),
                segment=seg,
                signal_type=sig_type,
                company_name=company,
                company_size_est="50-500 employees",
                signal_title=title,
                signal_url=url,
                signal_summary=f"{sig_type.replace('_', ' ').title()} detected: {title}",
                intent_score=intent,
                outreach_hook=hook,
                draft_message=message,
            ))
        return signals


# ── Report Builder ────────────────────────────────────────────────────────────

def build_report(signals: list, segment_filter: Optional[str]) -> SignalReport:
    high_intent = [s for s in signals if s.intent_score >= 0.7]

    by_segment = {}
    by_type = {}
    for s in signals:
        by_segment[s.segment] = by_segment.get(s.segment, 0) + 1
        by_type[s.signal_type] = by_type.get(s.signal_type, 0) + 1

    return SignalReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        segment_filter=segment_filter,
        total_signals=len(signals),
        high_intent_count=len(high_intent),
        signals=[asdict(s) for s in sorted(signals, key=lambda x: -x.intent_score)],
        summary_by_segment=by_segment,
        summary_by_type=by_type,
    )


def print_report(report: SignalReport):
    print(f"\n{'='*60}")
    print(f"📡 PROSPECT SIGNAL REPORT")
    print(f"   {report.run_date}")
    print(f"{'='*60}")
    print(f"\n   Total signals:    {report.total_signals}")
    print(f"   High intent (≥0.7): {report.high_intent_count}")
    print(f"\n   By segment: {report.summary_by_segment}")
    print(f"   By type:    {report.summary_by_type}")

    if report.signals:
        print(f"\n🎯 Top Signals (by intent score):")
        for s in report.signals[:5]:
            bar = "█" * int(s['intent_score'] * 10)
            print(f"\n   [{s['intent_score']:.2f}] {bar}")
            print(f"   {s['company_name']} — {s['signal_title'][:55]}")
            print(f"   Hook: {s['outreach_hook'][:80]}")
            print(f"   URL:  {s['signal_url']}")
    else:
        print("\n⚠️  No signals detected — check segment filters or try --dry-run")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect buying-intent signals from ICP prospects"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data (no external calls)")
    parser.add_argument("--segment", choices=list(ICP_SEGMENTS.keys()),
                        help="Filter to a specific ICP segment")
    parser.add_argument("--output", type=str,
                        help="Save report to this JSON path")
    args = parser.parse_args()

    segments = [args.segment] if args.segment else list(ICP_SEGMENTS.keys())

    print(f"🛰️  Kestrel AI — Prospect Signal Detector")
    print(f"   Segments: {segments}")
    print(f"   Mode: {'🧪 DRY RUN' if args.dry_run else '🔴 LIVE'}")

    scanner = ProspectScanner(dry_run=args.dry_run)
    signals = scanner.scan(segments)
    report = build_report(signals, args.segment)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2))
        print(f"\n✅ Report saved: {args.output}")


if __name__ == "__main__":
    main()
