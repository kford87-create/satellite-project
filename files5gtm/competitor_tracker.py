"""
tools/competitor_tracker.py  —  Tool 5: Competitor Price & Feature Tracker

Monitors Picterra, FlyPix AI, EOSDA, and Geospatial Insight for pricing
changes, new feature announcements, and press releases. Diffs against the
last snapshot to surface only what changed. Stale competitor data on
comparison pages kills LLM credibility — this keeps them accurate.

Think of this as a smoke alarm for competitive drift. It only fires when
something meaningfully changed, not every time you run it.

Usage:
    python tools/competitor_tracker.py
    python tools/competitor_tracker.py --diff
    python tools/competitor_tracker.py --competitor picterra
    python tools/competitor_tracker.py --output reports/competitor_snapshot.json
    python tools/competitor_tracker.py --dry-run
"""

import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
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

# ── Competitor Registry ───────────────────────────────────────────────────────
# Keep URLs here — never hardcode inside functions

COMPETITORS = {
    "picterra": {
        "name": "Picterra",
        "website": "https://picterra.ch",
        "pricing_url": "https://picterra.ch/pricing/",
        "blog_url": "https://picterra.ch/blog/",
        "known_pricing": "€50/month starter, €500/month professional, €2000/month enterprise",
        "known_features": ["object detection", "change detection", "API", "no-code platform"],
        "hq": "Switzerland",
    },
    "flypix": {
        "name": "FlyPix AI",
        "website": "https://flypix.ai",
        "pricing_url": "https://flypix.ai/pricing/",
        "blog_url": "https://flypix.ai/blog/",
        "known_pricing": "€50-€2000/month depending on tier",
        "known_features": ["aerial imagery detection", "drone imagery", "API"],
        "hq": "Germany",
    },
    "eosda": {
        "name": "EOSDA",
        "website": "https://eos.com",
        "pricing_url": "https://eos.com/pricing/",
        "blog_url": "https://eos.com/blog/",
        "known_pricing": "Custom enterprise pricing",
        "known_features": ["satellite analytics", "crop monitoring", "change detection",
                           "climate analytics"],
        "hq": "USA",
    },
    "geospatial_insight": {
        "name": "Geospatial Insight",
        "website": "https://geospatial-insight.com",
        "pricing_url": "https://geospatial-insight.com/solutions/",
        "blog_url": "https://geospatial-insight.com/news/",
        "known_pricing": "Custom enterprise pricing, insurance focus",
        "known_features": ["insurance analytics", "property monitoring",
                           "flood risk", "vegetation risk"],
        "hq": "UK",
    },
}

SNAPSHOT_DIR = Path("reports/competitor_snapshots")


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class CompetitorSnapshot:
    competitor_id: str
    name: str
    scraped_date: str
    pricing_text: str
    pricing_hash: str           # MD5 of pricing text — detect changes
    features_mentioned: list
    recent_blog_titles: list
    press_mentions: list
    page_word_count: int
    scrape_success: bool
    error: Optional[str] = None


@dataclass
class CompetitorDiff:
    competitor_id: str
    name: str
    compared_date: str
    pricing_changed: bool
    pricing_before: str
    pricing_after: str
    new_features: list
    removed_features: list
    new_blog_posts: list
    action_required: bool
    action_note: str


@dataclass
class TrackerReport:
    run_date: str
    competitors_tracked: int
    changes_detected: int
    snapshots: list = field(default_factory=list)
    diffs: list = field(default_factory=list)
    alerts: list = field(default_factory=list)


# ── Scraper ───────────────────────────────────────────────────────────────────

class CompetitorScraper:
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; KestrelAI-CompetitiveIntel/1.0)"
    }
    REQUEST_DELAY = 2.0

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def scrape(self, competitor_id: str) -> CompetitorSnapshot:
        """Scrape a competitor's pricing and blog pages."""
        comp = COMPETITORS[competitor_id]
        if self.dry_run or not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            return self._mock_snapshot(competitor_id, comp)

        pricing_text, features = self._fetch_pricing(comp["pricing_url"])
        blog_titles = self._fetch_blog_titles(comp["blog_url"])
        time.sleep(self.REQUEST_DELAY)

        pricing_hash = hashlib.md5(pricing_text.encode()).hexdigest()[:12]

        return CompetitorSnapshot(
            competitor_id=competitor_id,
            name=comp["name"],
            scraped_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            pricing_text=pricing_text[:500],
            pricing_hash=pricing_hash,
            features_mentioned=features,
            recent_blog_titles=blog_titles,
            press_mentions=[],
            page_word_count=len(pricing_text.split()),
            scrape_success=bool(pricing_text),
            error=None if pricing_text else "Could not fetch pricing page",
        )

    def _fetch_pricing(self, url: str) -> tuple:
        """Fetch pricing page and extract text + features."""
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

            # Extract feature mentions from pricing page
            feature_keywords = [
                "api", "object detection", "change detection", "batch", "export",
                "geojson", "arcgis", "qgis", "custom model", "active learning",
                "false negative", "insurance", "drone", "satellite",
            ]
            features = [kw for kw in feature_keywords if kw in text.lower()]
            return text[:2000], features
        except Exception as e:
            return "", []

    def _fetch_blog_titles(self, url: str) -> list:
        """Fetch recent blog post titles."""
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            titles = []
            for el in soup.select("h2, h3, .post-title, .entry-title, article h2")[:5]:
                t = el.get_text(strip=True)
                if len(t) > 10:
                    titles.append(t)
            return titles[:5]
        except Exception:
            return []

    def _mock_snapshot(self, competitor_id: str, comp: dict) -> CompetitorSnapshot:
        """Return a realistic mock snapshot for testing."""
        import random
        random.seed(hash(competitor_id))

        mock_pricing = {
            "picterra": "Starter €50/month · 5 detectors · API access | Professional €500/month · unlimited detectors | Enterprise €2000+/month",
            "flypix": "Starter €50/month · Basic detection | Professional €500/month · Advanced | Enterprise €2000/month · Full API",
            "eosda": "Contact sales for enterprise pricing. Free trial available. Custom licensing for large deployments.",
            "geospatial_insight": "Enterprise pricing only. Contact our team for insurance, energy, and infrastructure solutions.",
        }

        mock_blogs = {
            "picterra": ["New: Multi-class detection in one pass", "Case study: Urban mapping with AI",
                         "API v3 release notes"],
            "flypix": ["How aerial AI is changing insurance", "Product update: batch processing",
                       "Partnership with DJI announced"],
            "eosda": ["2026 satellite analytics trends", "EOSDA expands to agricultural markets",
                      "New climate risk module"],
            "geospatial_insight": ["UK flood risk assessment with AI", "Insurance case study: property portfolio",
                                   "New partnership with Verisk"],
        }

        pricing = mock_pricing.get(competitor_id, "Contact for pricing")
        return CompetitorSnapshot(
            competitor_id=competitor_id,
            name=comp["name"],
            scraped_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            pricing_text=pricing,
            pricing_hash=hashlib.md5(pricing.encode()).hexdigest()[:12],
            features_mentioned=comp["known_features"][:4],
            recent_blog_titles=mock_blogs.get(competitor_id, ["New blog post"]),
            press_mentions=[],
            page_word_count=random.randint(800, 2000),
            scrape_success=True,
        )


# ── Diff Engine ───────────────────────────────────────────────────────────────

def load_last_snapshot(competitor_id: str) -> Optional[dict]:
    """Load the most recent saved snapshot for comparison."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SNAPSHOT_DIR.glob(f"{competitor_id}_*.json"), reverse=True)
    if not files:
        return None
    return json.loads(files[0].read_text())


def save_snapshot(snapshot: CompetitorSnapshot):
    """Save a snapshot to disk for future diffing."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    path = SNAPSHOT_DIR / f"{snapshot.competitor_id}_{date_str}.json"
    path.write_text(json.dumps(asdict(snapshot), indent=2))
    return path


def diff_snapshots(old: dict, new: CompetitorSnapshot) -> CompetitorDiff:
    """Compare old and new snapshots — return only what changed."""
    pricing_changed = old.get("pricing_hash") != new.pricing_hash

    old_features = set(old.get("features_mentioned", []))
    new_features = set(new.features_mentioned)
    added = list(new_features - old_features)
    removed = list(old_features - new_features)

    old_blogs = set(old.get("recent_blog_titles", []))
    new_blogs = [t for t in new.recent_blog_titles if t not in old_blogs]

    action_required = pricing_changed or bool(added) or bool(new_blogs)
    action_note = ""
    if pricing_changed:
        action_note += f"⚠️  Pricing changed — update comparison page at kestrelai.com/comparison. "
    if added:
        action_note += f"New features: {', '.join(added)}. "
    if new_blogs:
        action_note += f"New content: {new_blogs[0][:50]}."

    return CompetitorDiff(
        competitor_id=new.competitor_id,
        name=new.name,
        compared_date=datetime.now().strftime("%Y-%m-%d"),
        pricing_changed=pricing_changed,
        pricing_before=old.get("pricing_text", "")[:200],
        pricing_after=new.pricing_text[:200],
        new_features=added,
        removed_features=removed,
        new_blog_posts=new_blogs,
        action_required=action_required,
        action_note=action_note,
    )


# ── Report Builder ────────────────────────────────────────────────────────────

def print_report(report: TrackerReport):
    print(f"\n{'='*60}")
    print(f"🔍 COMPETITOR TRACKER REPORT")
    print(f"   {report.run_date}")
    print(f"{'='*60}")
    print(f"\n   Competitors tracked:  {report.competitors_tracked}")
    print(f"   Changes detected:     {report.changes_detected}")

    if report.alerts:
        print(f"\n🚨 ALERTS — Action Required:")
        for alert in report.alerts:
            print(f"\n   {alert['name']}")
            print(f"   {alert['note']}")

    print(f"\n📊 Snapshots:")
    for s in report.snapshots:
        status = "✅" if s["scrape_success"] else "❌"
        print(f"\n   {status} {s['name']}")
        print(f"      Pricing: {s['pricing_text'][:80]}...")
        print(f"      Features: {', '.join(s['features_mentioned'][:4])}")
        if s.get("recent_blog_titles"):
            print(f"      Latest post: {s['recent_blog_titles'][0][:60]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track competitor pricing and features"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data (no scraping)")
    parser.add_argument("--diff", action="store_true",
                        help="Compare to last snapshot and show only changes")
    parser.add_argument("--competitor", choices=list(COMPETITORS.keys()),
                        help="Track a single competitor")
    parser.add_argument("--output", type=str,
                        help="Save report to this JSON path")
    args = parser.parse_args()

    targets = [args.competitor] if args.competitor else list(COMPETITORS.keys())

    print(f"🛰️  Kestrel AI — Competitor Tracker")
    print(f"   Competitors: {targets}")
    print(f"   Mode: {'🧪 DRY RUN' if args.dry_run else '🔴 LIVE'}")

    scraper = CompetitorScraper(dry_run=args.dry_run)
    snapshots = []
    diffs = []
    alerts = []

    for comp_id in targets:
        print(f"\n   Scanning {COMPETITORS[comp_id]['name']}...")
        snap = scraper.scrape(comp_id)
        snapshots.append(snap)
        print(f"   {'✅' if snap.scrape_success else '❌'} {snap.name} — {snap.pricing_hash}")

        if args.diff:
            old = load_last_snapshot(comp_id)
            if old:
                d = diff_snapshots(old, snap)
                diffs.append(d)
                if d.action_required:
                    alerts.append({"name": d.name, "note": d.action_note})
                    print(f"   🚨 Changes detected: {d.action_note[:60]}")
            else:
                print(f"   ℹ️  No previous snapshot to diff against")

        if not args.dry_run:
            save_snapshot(snap)

    report = TrackerReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        competitors_tracked=len(snapshots),
        changes_detected=len(alerts),
        snapshots=[asdict(s) for s in snapshots],
        diffs=[asdict(d) for d in diffs],
        alerts=alerts,
    )
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2))
        print(f"\n✅ Report saved: {args.output}")


if __name__ == "__main__":
    main()
