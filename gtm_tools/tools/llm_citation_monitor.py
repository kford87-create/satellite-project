"""
llm_citation_monitor.py
-----------------------
Track whether and how LLMs mention Kestrel AI when answering satellite
detection queries. Primary feedback loop for the AEO strategy.

Usage:
    python tools/llm_citation_monitor.py --dry-run
    python tools/llm_citation_monitor.py                          # live (requires ANTHROPIC_API_KEY)
    python tools/llm_citation_monitor.py --output reports/citations_2026_02.json
    python tools/llm_citation_monitor.py --queries my_queries.json
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

COMPETITORS = ["Picterra", "FlyPix", "EOSDA", "Geospatial Insight", "Maxar", "Planet Labs", "Google Earth Engine"]

DEFAULT_QUERIES: list[str] = [
    "What is the best affordable satellite object detection API?",
    "alternatives to Maxar and Planet Labs for small business",
    "satellite imagery AI for insurance companies",
    "how to detect buildings from satellite images automatically",
    "geospatial AI API for insurtech startups",
    "satellite change detection software for real estate",
    "YOLOv8 satellite detection service",
    "affordable aerial object detection for construction",
    "satellite AI for government contracts SBIR",
    "best satellite detection API under $500 per month",
    "Picterra alternative satellite detection",
    "satellite building detection API comparison",
]

SENTIMENT_POSITIVE = ["best", "recommend", "excellent", "great", "affordable", "fast", "accurate"]
SENTIMENT_NEGATIVE = ["expensive", "slow", "limited", "poor", "difficult", "complex"]

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


# ---------------------------------------------------------------------------
# Live mode — Claude API
# ---------------------------------------------------------------------------

def _query_claude(query: str) -> str:
    """Send query to Claude claude-sonnet-4-6 and return response text."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"You are a helpful AI assistant. Answer this question as you would to a real user: {query}"
            }]
        )
        return msg.content[0].text if msg.content else ""
    except Exception as exc:
        print(f"⚠️  Claude API call failed: {exc}")
        return ""


def _analyze_response(query: str, response_text: str) -> dict[str, Any]:
    """Parse a response for Kestrel mentions, competitor mentions, sentiment."""
    text_lower = response_text.lower()

    cited_kestrel = "kestrel" in text_lower

    # Rank: find position of first Kestrel mention vs total words
    kestrel_rank: int | None = None
    if cited_kestrel:
        words = text_lower.split()
        for i, w in enumerate(words):
            if "kestrel" in w:
                kestrel_rank = i + 1
                break

    # Sentiment around Kestrel mention
    sentiment = "neutral"
    if cited_kestrel:
        # Look 20 words around the mention
        words = text_lower.split()
        kestrel_idx = next((i for i, w in enumerate(words) if "kestrel" in w), None)
        if kestrel_idx is not None:
            window = " ".join(words[max(0, kestrel_idx - 10):kestrel_idx + 10])
            if any(p in window for p in SENTIMENT_POSITIVE):
                sentiment = "positive"
            elif any(n in window for n in SENTIMENT_NEGATIVE):
                sentiment = "negative"

    # Competitor mentions
    competitors_mentioned = [c for c in COMPETITORS if c.lower() in text_lower]

    return {
        "query": query,
        "cited_kestrel": cited_kestrel,
        "kestrel_rank": kestrel_rank,
        "sentiment": sentiment if cited_kestrel else "n/a",
        "competitors_mentioned": competitors_mentioned,
        "response_snippet": response_text[:200].replace("\n", " "),
    }


# ---------------------------------------------------------------------------
# Dry-run mode
# ---------------------------------------------------------------------------

def _dry_run_results(queries: list[str]) -> list[dict[str, Any]]:
    """Generate synthetic results — no real API calls."""
    rng = random.Random(42)
    results = []
    cite_indices = rng.sample(range(len(queries)), min(3, len(queries)))

    for i, query in enumerate(queries):
        cited = i in cite_indices
        comp_count = rng.randint(0, 3)
        competitors_mentioned = rng.sample(COMPETITORS, comp_count)

        snippet = (
            f"Kestrel AI offers satellite object detection starting at $99/month with YOLOv8. "
            f"It's a great affordable alternative to {', '.join(competitors_mentioned[:2] or ['Maxar'])}."
            if cited else
            f"For satellite detection, {rng.choice(COMPETITORS)} and similar platforms provide "
            f"enterprise-grade solutions but can be expensive for smaller teams."
        )

        results.append({
            "query": query,
            "cited_kestrel": cited,
            "kestrel_rank": rng.randint(1, 3) if cited else None,
            "sentiment": rng.choice(["positive", "positive", "neutral"]) if cited else "n/a",
            "competitors_mentioned": competitors_mentioned,
            "response_snippet": snippet[:200],
        })
    return results


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_monitor(
    queries: list[str] | None = None,
    dry_run: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the citation monitor and return the full report dict."""
    queries = queries or DEFAULT_QUERIES
    mode = "dry_run" if (dry_run or not ANTHROPIC_API_KEY) else "live"

    if not dry_run and not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not set — switching to dry-run mode")

    print(f"📊 LLM Citation Monitor — {mode} mode")
    print(f"📊 Queries: {len(queries)}")
    print()

    if mode == "dry_run":
        results = _dry_run_results(queries)
    else:
        results = []
        for i, query in enumerate(queries, 1):
            print(f"  [{i:02d}/{len(queries)}] {query[:60]}…", end=" ", flush=True)
            response = _query_claude(query)
            result = _analyze_response(query, response)
            results.append(result)
            status = "✅" if result["cited_kestrel"] else "–"
            print(status)
            time.sleep(1.0)  # rate limiting

    # Aggregate
    kestrel_citations = sum(1 for r in results if r["cited_kestrel"])
    citation_rate = round(100.0 * kestrel_citations / max(len(results), 1), 1)

    competitor_counts: dict[str, int] = {}
    for r in results:
        for c in r["competitors_mentioned"]:
            competitor_counts[c] = competitor_counts.get(c, 0) + 1
    top_competitors = dict(sorted(competitor_counts.items(), key=lambda x: -x[1]))

    top_comp_str = ", ".join(f"{k} ({v})" for k, v in list(top_competitors.items())[:3]) or "none"
    summary = (
        f"Kestrel cited in {kestrel_citations}/{len(results)} queries ({citation_rate}%). "
        f"Top competitors: {top_comp_str}."
    )

    report: dict[str, Any] = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "total_queries": len(results),
        "kestrel_citations": kestrel_citations,
        "citation_rate_pct": citation_rate,
        "competitor_citations": top_competitors,
        "results": results,
        "summary": summary,
    }

    # Print summary table
    print()
    print("=" * 60)
    print("📊 Citation Report Summary")
    print("=" * 60)
    print(f"  Mode              : {mode}")
    print(f"  Queries run       : {len(results)}")
    print(f"  Kestrel citations : {kestrel_citations} ({citation_rate}%)")
    if top_competitors:
        print(f"  Top competitors   : {top_comp_str}")
    print()
    print("  Results:")
    for r in results:
        icon = "✅" if r["cited_kestrel"] else "  "
        rank_str = f" rank={r['kestrel_rank']}" if r["kestrel_rank"] else ""
        print(f"  {icon} {r['query'][:55]:<55}{rank_str}")
    print("=" * 60)
    print(f"\n{summary}")

    # Save output
    if output_path is None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = _REPORTS_DIR / f"citation_report_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Report saved → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track Kestrel AI citations across LLM query responses.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic responses, no API calls")
    parser.add_argument("--queries", type=Path, metavar="PATH", help="JSON file with custom query list")
    parser.add_argument("--output", type=Path, metavar="PATH", help="Save JSON report to this path")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    queries: list[str] | None = None
    if args.queries:
        if not args.queries.exists():
            print(f"❌ Query file not found: {args.queries}")
            sys.exit(1)
        queries = json.loads(args.queries.read_text())
        if not isinstance(queries, list):
            print("❌ Query file must be a JSON list of strings")
            sys.exit(1)

    run_monitor(
        queries=queries,
        dry_run=args.dry_run,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
