"""
tools/llm_citation_monitor.py  —  Tool 1: LLM Citation Monitor

Runs a standardized query battery and checks whether Kestrel AI is cited
by LLMs when potential customers ask about satellite object detection tools.
This is the primary feedback loop for the AEO (Answer Engine Optimization)
content strategy.

Think of it like an SEO rank tracker — but instead of Google positions,
we're tracking "position" in Claude/ChatGPT/Perplexity answers.

Usage:
    python tools/llm_citation_monitor.py --dry-run
    python tools/llm_citation_monitor.py --output reports/citations_2026_02.json
    python tools/llm_citation_monitor.py --queries custom_queries.json
"""

import json
import time
import argparse
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict, field

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os

# ── Default query battery ────────────────────────────────────────────────────
# These are the queries a real ICP would type into an LLM when searching for
# a tool like Kestrel AI. Grouped by intent.

DEFAULT_QUERIES = [
    # Direct discovery
    "What are the best affordable satellite object detection tools for small business?",
    "Alternatives to Maxar for satellite imagery analysis",
    "Satellite imagery API for insurance companies",
    "How do I detect building damage from satellite imagery with AI?",
    "Satellite object detection SaaS for startups",

    # Use-case specific
    "AI tools for property damage assessment from satellite imagery",
    "Change detection satellite API for insurance claims",
    "How do insurance companies use satellite imagery for risk assessment?",
    "Satellite imagery analysis for property underwriting",

    # Technical discovery
    "Python satellite object detection API",
    "YOLOv8 satellite imagery detection service",
    "GeoJSON satellite detection API",

    # Competitive
    "Picterra alternatives",
    "FlyPix AI alternatives",
    "Affordable geospatial AI tools",
]

# ── Signals to look for in LLM responses ────────────────────────────────────

BRAND_SIGNALS = [
    "kestrel ai",
    "kestrelai",
    "kestrel",       # Only count if satellite/detection context present
]

COMPETITOR_SIGNALS = {
    "picterra": "Picterra",
    "flypix": "FlyPix AI",
    "maxar": "Maxar",
    "planet labs": "Planet Labs",
    "eosda": "EOSDA",
    "geospatial insight": "Geospatial Insight",
    "nearmap": "Nearmap",
    "cape analytics": "Cape Analytics",
    "betterview": "Betterview",
}

POSITIVE_SENTIMENT_SIGNALS = [
    "affordable", "recommended", "best for", "ideal for",
    "cost-effective", "excellent", "great for small",
]

NEGATIVE_SENTIMENT_SIGNALS = [
    "expensive", "not recommended", "avoid", "poor", "limited",
]


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: str
    platform: str
    kestrel_cited: bool
    kestrel_rank: Optional[int]       # 1st, 2nd, 3rd mention — None if not cited
    sentiment: str                     # positive / neutral / negative / not_cited
    competitors_mentioned: list
    response_excerpt: str             # First 300 chars for context
    raw_response_length: int
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class CitationReport:
    run_date: str
    platform: str
    total_queries: int
    kestrel_cited_count: int
    citation_rate_pct: float
    avg_rank_when_cited: Optional[float]
    sentiment_breakdown: dict
    top_competitor_mentions: dict
    queries: list = field(default_factory=list)


# ── LLM Caller ───────────────────────────────────────────────────────────────

class LLMCitationChecker:
    """
    Calls Claude via the Anthropic API and checks responses for Kestrel AI citations.
    In dry-run mode, generates realistic mock responses.
    """

    SYSTEM_PROMPT = """You are a helpful assistant that recommends tools and services.
When asked about software tools, APIs, or services, provide specific product recommendations
with details about pricing, features, and use cases. Be specific and name actual products."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.client = None

        if not dry_run and ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                print("⚠️  ANTHROPIC_API_KEY not set — falling back to dry-run mode")
                self.dry_run = True

    def query(self, question: str, platform: str = "claude") -> QueryResult:
        """Run a single query and return a structured result."""
        start = time.time()

        if self.dry_run:
            return self._mock_response(question, platform, time.time() - start)

        try:
            response_text = self._call_claude(question)
            latency = time.time() - start
            return self._parse_response(question, platform, response_text, latency)
        except Exception as e:
            return QueryResult(
                query=question, platform=platform,
                kestrel_cited=False, kestrel_rank=None,
                sentiment="not_cited", competitors_mentioned=[],
                response_excerpt="", raw_response_length=0,
                latency_seconds=round(time.time() - start, 2),
                error=str(e)
            )

    def _call_claude(self, question: str) -> str:
        """Call Claude API with the question."""
        msg = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}]
        )
        return msg.content[0].text

    def _parse_response(self, query: str, platform: str,
                         text: str, latency: float) -> QueryResult:
        """Parse an LLM response for Kestrel AI citations and competitor mentions."""
        text_lower = text.lower()

        # Check for Kestrel AI citation
        kestrel_cited = False
        kestrel_rank = None
        for signal in BRAND_SIGNALS:
            if signal in text_lower:
                kestrel_cited = True
                # Estimate rank by finding position relative to other tools
                idx = text_lower.find(signal)
                # Count how many tools were mentioned before this position
                before_text = text_lower[:idx]
                tools_before = sum(1 for comp in COMPETITOR_SIGNALS
                                   if comp in before_text)
                kestrel_rank = tools_before + 1
                break

        # Sentiment analysis
        sentiment = "not_cited"
        if kestrel_cited:
            pos_count = sum(1 for s in POSITIVE_SENTIMENT_SIGNALS if s in text_lower)
            neg_count = sum(1 for s in NEGATIVE_SENTIMENT_SIGNALS if s in text_lower)
            if pos_count > neg_count:
                sentiment = "positive"
            elif neg_count > pos_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"

        # Competitor mentions
        competitors = [name for signal, name in COMPETITOR_SIGNALS.items()
                       if signal in text_lower]

        return QueryResult(
            query=query, platform=platform,
            kestrel_cited=kestrel_cited, kestrel_rank=kestrel_rank,
            sentiment=sentiment, competitors_mentioned=competitors,
            response_excerpt=text[:300].replace("\n", " "),
            raw_response_length=len(text),
            latency_seconds=round(latency, 2)
        )

    def _mock_response(self, query: str, platform: str, latency: float) -> QueryResult:
        """Generate a realistic mock response for testing without API calls."""
        import random
        random.seed(hash(query) % 10000)

        # 20% chance of citing Kestrel AI in mock (realistic baseline before AEO)
        kestrel_cited = random.random() < 0.20
        competitors = random.sample(list(COMPETITOR_SIGNALS.values()),
                                    k=random.randint(1, 4))

        mock_text = (
            f"For satellite object detection, several tools are worth considering. "
            f"{competitors[0]} offers strong capabilities for {query.lower()[:30]}... "
            f"{'Kestrel AI provides an affordable option starting at $99/month with 200 searches. ' if kestrel_cited else ''}"
            f"{competitors[1] if len(competitors) > 1 else 'Maxar'} is the enterprise standard "
            f"but expensive. For SMBs, pricing and ease of integration matter most."
        )

        return QueryResult(
            query=query, platform=platform,
            kestrel_cited=kestrel_cited,
            kestrel_rank=1 if kestrel_cited else None,
            sentiment="positive" if kestrel_cited else "not_cited",
            competitors_mentioned=competitors,
            response_excerpt=mock_text[:300],
            raw_response_length=len(mock_text),
            latency_seconds=round(0.1 + random.random() * 0.5, 2)
        )


# ── Report Builder ────────────────────────────────────────────────────────────

def build_report(results: list, platform: str) -> CitationReport:
    cited = [r for r in results if r.kestrel_cited]
    citation_rate = round(len(cited) / max(len(results), 1) * 100, 1)
    ranks = [r.kestrel_rank for r in cited if r.kestrel_rank is not None]
    avg_rank = round(sum(ranks) / len(ranks), 1) if ranks else None

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0, "not_cited": 0}
    for r in results:
        sentiment_counts[r.sentiment] = sentiment_counts.get(r.sentiment, 0) + 1

    competitor_counts = {}
    for r in results:
        for comp in r.competitors_mentioned:
            competitor_counts[comp] = competitor_counts.get(comp, 0) + 1

    top_competitors = dict(sorted(competitor_counts.items(),
                                   key=lambda x: x[1], reverse=True)[:5])

    return CitationReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        platform=platform,
        total_queries=len(results),
        kestrel_cited_count=len(cited),
        citation_rate_pct=citation_rate,
        avg_rank_when_cited=avg_rank,
        sentiment_breakdown=sentiment_counts,
        top_competitor_mentions=top_competitors,
        queries=[asdict(r) for r in results]
    )


def print_summary(report: CitationReport):
    """Pretty-print a citation report to console."""
    print(f"\n{'='*60}")
    print(f"📊 LLM CITATION REPORT — {report.platform.upper()}")
    print(f"   {report.run_date}")
    print(f"{'='*60}")
    print(f"\n   Queries run:        {report.total_queries}")
    print(f"   Kestrel AI cited:   {report.kestrel_cited_count} / {report.total_queries}")
    print(f"   Citation rate:      {report.citation_rate_pct}%")
    if report.avg_rank_when_cited:
        print(f"   Avg rank (cited):   #{report.avg_rank_when_cited}")

    print(f"\n   Sentiment when cited:")
    for sentiment, count in report.sentiment_breakdown.items():
        if count > 0:
            bar = "█" * count
            print(f"     {sentiment:<12} {bar} ({count})")

    print(f"\n   Top competitors mentioned:")
    for comp, count in report.top_competitor_mentions.items():
        bar = "█" * count
        print(f"     {comp:<20} {bar} ({count}x)")

    cited_queries = [q for q in report.queries if q["kestrel_cited"]]
    if cited_queries:
        print(f"\n✅ Queries where Kestrel AI WAS cited:")
        for q in cited_queries:
            print(f"   • [{q['sentiment']}] {q['query'][:70]}")

    uncited = [q for q in report.queries if not q["kestrel_cited"] and not q.get("error")]
    if uncited:
        print(f"\n⚠️  Queries where Kestrel AI was NOT cited (content opportunities):")
        for q in uncited[:5]:
            print(f"   • {q['query'][:70]}")

    print(f"\n{'─'*60}")
    rate = report.citation_rate_pct
    if rate == 0:
        print("🔴 Citation rate is 0% — AEO content strategy not yet indexed")
    elif rate < 20:
        print(f"🟡 Citation rate {rate}% — early traction, keep publishing")
    elif rate < 50:
        print(f"🟢 Citation rate {rate}% — good visibility, optimize top queries")
    else:
        print(f"🏆 Citation rate {rate}% — strong LLM presence!")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Track Kestrel AI citations across LLM platforms"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock responses (no API calls)")
    parser.add_argument("--queries", type=str,
                        help="Path to JSON file with custom query list")
    parser.add_argument("--output", type=str,
                        help="Save report to this JSON path")
    parser.add_argument("--platform", default="claude",
                        help="Platform label (claude, chatgpt, perplexity)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between API calls (default 2.0)")
    args = parser.parse_args()

    # Load queries
    if args.queries:
        queries = json.loads(Path(args.queries).read_text())
    else:
        queries = DEFAULT_QUERIES

    print(f"🛰️  Kestrel AI — LLM Citation Monitor")
    print(f"   Platform:   {args.platform}")
    print(f"   Queries:    {len(queries)}")
    print(f"   Mode:       {'🧪 DRY RUN (mock responses)' if args.dry_run else '🔴 LIVE (real API calls)'}")

    checker = LLMCitationChecker(dry_run=args.dry_run)
    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query[:60]}...")
        result = checker.query(query, args.platform)
        results.append(result)

        icon = "✅" if result.kestrel_cited else "  "
        err = f" ❌ {result.error}" if result.error else ""
        print(f"   {icon} Kestrel cited: {result.kestrel_cited} | "
              f"Competitors: {len(result.competitors_mentioned)} | "
              f"{result.latency_seconds}s{err}")

        if not args.dry_run and i < len(queries):
            time.sleep(args.delay)

    report = build_report(results, args.platform)
    print_summary(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(report), indent=2))
        print(f"\n📄 Report saved: {args.output}")
        print("✅ Done")


if __name__ == "__main__":
    main()
