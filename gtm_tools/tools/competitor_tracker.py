"""
competitor_tracker.py
---------------------
Monitor Picterra, FlyPix AI, EOSDA, and Geospatial Insight for pricing
changes, new features, and press releases.

Usage:
    python tools/competitor_tracker.py --dry-run
    python tools/competitor_tracker.py --diff
    python tools/competitor_tracker.py --output reports/competitor_snapshot.json
    python tools/competitor_tracker.py --competitor picterra
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
_SNAPSHOT_FILE = _REPORTS_DIR / "competitor_snapshot_latest.json"

# ---------------------------------------------------------------------------
# Competitor config
# ---------------------------------------------------------------------------
COMPETITOR_URLS: dict[str, dict[str, str]] = {
    "Picterra": {
        "pricing": "https://picterra.ch/pricing/",
        "blog": "https://picterra.ch/blog/",
        "homepage": "https://picterra.ch/",
    },
    "FlyPix AI": {
        "pricing": "https://flypix.ai/pricing/",
        "blog": "https://flypix.ai/blog/",
        "homepage": "https://flypix.ai/",
    },
    "EOSDA": {
        "pricing": "https://eos.com/pricing/",
        "blog": "https://eos.com/blog/",
        "homepage": "https://eos.com/",
    },
    "Geospatial Insight": {
        "pricing": "https://www.geospatialinsight.co.uk/",
        "blog": "https://www.geospatialinsight.co.uk/news/",
        "homepage": "https://www.geospatialinsight.co.uk/",
    },
}

PRICE_REGEX = re.compile(r"[\$€£][\d,]+(?:\.\d+)?(?:\s*(?:/\s*mo(?:nth)?|/\s*yr|/\s*year|annually))?", re.IGNORECASE)
FEATURE_KEYWORDS = [
    "object detection", "change detection", "building detection", "vehicle detection",
    "API", "batch processing", "GeoJSON", "custom model", "aerial", "satellite",
    "deep learning", "neural network", "cloud platform",
]
PRESS_KEYWORDS = ["partnership", "raises", "series", "acqui", "launch", "announce", "award", "contract"]

# ---------------------------------------------------------------------------
# Synthetic snapshot (dry-run)
# ---------------------------------------------------------------------------

_SYNTHETIC_SNAPSHOT: dict[str, dict[str, Any]] = {
    "Picterra": {
        "prices_found": ["$490/month", "$1,200/month", "$2,800/month"],
        "features_found": ["object detection", "custom model", "API", "GeoJSON", "batch processing"],
        "recent_posts": [
            "How Picterra is transforming infrastructure inspection",
            "New: Multi-class detection for urban mapping",
            "Case study: Picterra for utilities monitoring",
        ],
        "press_signals": [],
        "raw_text_hash": hashlib.sha256(b"picterra_mock_v1").hexdigest(),
    },
    "FlyPix AI": {
        "prices_found": ["Contact for pricing"],
        "features_found": ["object detection", "aerial", "API", "satellite", "deep learning"],
        "recent_posts": [
            "FlyPix AI raises $8M Series A",
            "Introducing real-time aerial detection",
        ],
        "press_signals": ["raises", "Series A"],
        "raw_text_hash": hashlib.sha256(b"flypix_mock_v1").hexdigest(),
    },
    "EOSDA": {
        "prices_found": ["$299/month", "$799/month", "Enterprise — contact us"],
        "features_found": ["object detection", "change detection", "satellite", "API", "cloud platform"],
        "recent_posts": [
            "EOSDA partners with Airbus on satellite analytics",
            "New vegetation + building detection model",
            "EOSDA crop monitoring now covers 50+ countries",
        ],
        "press_signals": ["partnership", "Airbus"],
        "raw_text_hash": hashlib.sha256(b"eosda_mock_v1").hexdigest(),
    },
    "Geospatial Insight": {
        "prices_found": [],
        "features_found": ["object detection", "change detection", "building detection", "GeoJSON"],
        "recent_posts": [
            "Geospatial Insight wins MOD contract",
            "Expanding into North American insurance market",
        ],
        "press_signals": ["contract", "MOD"],
        "raw_text_hash": hashlib.sha256(b"gi_mock_v1").hexdigest(),
    },
}

# ---------------------------------------------------------------------------
# Live scraping
# ---------------------------------------------------------------------------

def _fetch_page_text(url: str) -> str:
    """Fetch a URL and return cleaned text. Returns '' on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KestrelAI-GTM/1.0)"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")

        # Try BeautifulSoup for clean text
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(raw, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except ImportError:
            # Fallback: strip HTML tags with regex
            return re.sub(r"<[^>]+>", " ", raw)
    except Exception as exc:
        print(f"⚠️  Failed to fetch {url}: {exc}")
        return ""


def _scrape_competitor(name: str, urls: dict[str, str]) -> dict[str, Any]:
    all_text = ""
    for page_type, url in urls.items():
        print(f"    Fetching {page_type}: {url}", end=" ", flush=True)
        text = _fetch_page_text(url)
        all_text += " " + text
        print("✅" if text else "⚠️")
        time.sleep(2.0)  # respect rate limits

    prices_found = list(set(PRICE_REGEX.findall(all_text)))[:8]
    features_found = [kw for kw in FEATURE_KEYWORDS if kw.lower() in all_text.lower()]
    press_signals = [kw for kw in PRESS_KEYWORDS if kw.lower() in all_text.lower()]

    # Extract recent blog post titles (h2/h3 within first 5000 chars of blog page)
    recent_posts: list[str] = []
    try:
        from bs4 import BeautifulSoup
        blog_url = urls.get("blog", "")
        if blog_url:
            req_text = _fetch_page_text(blog_url)
            soup = BeautifulSoup(req_text, "html.parser") if "<" not in req_text else None
            if soup:
                for h in soup.find_all(["h2", "h3"])[:6]:
                    t = h.get_text(strip=True)
                    if t and len(t) > 10:
                        recent_posts.append(t)
    except Exception:
        pass

    return {
        "prices_found": prices_found,
        "features_found": features_found,
        "recent_posts": recent_posts[:3],
        "press_signals": press_signals,
        "raw_text_hash": hashlib.sha256(all_text.encode()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _compute_diff(old: dict, new: dict) -> list[str]:
    alerts: list[str] = []
    for comp in new:
        if comp not in old:
            alerts.append(f"➕ New competitor tracked: {comp}")
            continue
        if old[comp]["raw_text_hash"] != new[comp]["raw_text_hash"]:
            alerts.append(f"⚠️  {comp} page content changed since last snapshot")
        old_prices = set(old[comp].get("prices_found", []))
        new_prices = set(new[comp].get("prices_found", []))
        added_prices = new_prices - old_prices
        removed_prices = old_prices - new_prices
        if added_prices:
            alerts.append(f"💰 {comp} new price(s) detected: {', '.join(added_prices)}")
        if removed_prices:
            alerts.append(f"💰 {comp} price(s) removed: {', '.join(removed_prices)}")
        new_press = set(new[comp].get("press_signals", []))
        old_press = set(old[comp].get("press_signals", []))
        if new_press - old_press:
            alerts.append(f"📰 {comp} new press signal(s): {', '.join(new_press - old_press)}")
    return alerts


def _build_positioning(competitors: dict[str, dict[str, Any]]) -> dict[str, str]:
    # Find cheapest competitor price
    all_prices: list[float] = []
    for comp_data in competitors.values():
        for p in comp_data.get("prices_found", []):
            nums = re.findall(r"[\d,]+", p)
            if nums:
                try:
                    val = float(nums[0].replace(",", ""))
                    if val > 10:
                        all_prices.append(val)
                except ValueError:
                    pass

    cheapest_competitor = min(all_prices) if all_prices else 490.0
    price_mult = round(cheapest_competitor / 99, 1)

    feature_gaps = []
    all_features = set()
    for cd in competitors.values():
        all_features.update(cd.get("features_found", []))
    if "custom model" in all_features:
        feature_gaps.append("Custom model training not yet available in Kestrel AI")

    opportunities = ["No competitor publicly offers sub-3s detection SLA"]
    for comp, cd in competitors.items():
        if not cd.get("prices_found"):
            opportunities.append(f"{comp} hides pricing — Kestrel AI's transparent $99 entry is a differentiator")

    return {
        "price_advantage": f"Kestrel Starter ($99) is {price_mult}× cheaper than cheapest competitor (${cheapest_competitor:.0f})",
        "feature_gaps": feature_gaps,
        "opportunities": opportunities[:3],
    }


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_tracker(
    dry_run: bool = False,
    diff: bool = False,
    competitor_filter: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    mode = "dry_run" if dry_run else "live"
    print(f"📊 Competitor Tracker — {mode} mode")
    if competitor_filter:
        print(f"📊 Competitor filter: {competitor_filter}")
    print()

    # Filter competitors
    target = {}
    if competitor_filter:
        match = next((k for k in COMPETITOR_URLS if competitor_filter.lower() in k.lower()), None)
        if match:
            target[match] = COMPETITOR_URLS[match]
        else:
            print(f"❌ Unknown competitor '{competitor_filter}'. Options: {', '.join(COMPETITOR_URLS.keys())}")
            return {}
    else:
        target = COMPETITOR_URLS

    # Collect data
    if dry_run:
        competitors = {k: v for k, v in _SYNTHETIC_SNAPSHOT.items() if k in target}
    else:
        competitors = {}
        for name, urls in target.items():
            print(f"\n  Scraping {name}…")
            competitors[name] = _scrape_competitor(name, urls)

    # Diff against last snapshot
    diff_alerts: list[str] = []
    if diff and _SNAPSHOT_FILE.exists():
        try:
            old_snapshot = json.loads(_SNAPSHOT_FILE.read_text())
            old_competitors = old_snapshot.get("competitors", {})
            diff_alerts = _compute_diff(old_competitors, competitors)
        except Exception as exc:
            print(f"⚠️  Could not load previous snapshot for diff: {exc}")

    positioning = _build_positioning(competitors)

    report: dict[str, Any] = {
        "snapshot_date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "competitors": competitors,
        "diff": diff_alerts if diff else None,
        "alerts": diff_alerts,
        "kestrel_positioning": positioning,
    }

    # Print summary
    print()
    print("=" * 60)
    print("📊 Competitor Snapshot")
    print("=" * 60)
    for name, data in competitors.items():
        prices = ", ".join(data["prices_found"][:2]) or "Not found"
        features = len(data["features_found"])
        press = len(data["press_signals"])
        print(f"  {name:<25} prices={prices:<25} features={features} press={press}")

    if diff_alerts:
        print()
        print("⚠️  Changes detected:")
        for a in diff_alerts:
            print(f"  {a}")

    print()
    print(f"  {positioning['price_advantage']}")
    print("=" * 60)

    # Save
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    if output_path is None:
        output_path = _REPORTS_DIR / f"competitor_snapshot_{date_str}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Snapshot saved → {output_path}")

    # Update latest snapshot
    _SNAPSHOT_FILE.write_text(json.dumps(report, indent=2))
    print(f"✅ Latest snapshot updated → {_SNAPSHOT_FILE}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track competitor pricing, features, and press.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data, no scraping")
    parser.add_argument("--diff", action="store_true", help="Compare to last snapshot")
    parser.add_argument("--output", type=Path, metavar="PATH", help="Save JSON snapshot")
    parser.add_argument("--competitor", type=str, metavar="NAME", help="Focus on one competitor")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_tracker(
        dry_run=args.dry_run,
        diff=args.diff,
        competitor_filter=args.competitor,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
