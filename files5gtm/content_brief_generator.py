"""
tools/content_brief_generator.py  —  Tool 2: Content Brief Generator

Takes a target query, researches the competitive content landscape,
and produces a structured AEO (Answer Engine Optimization) content brief
in BLUF format (Bottom Line Up Front) — the format that maximizes the
probability of LLM citation by 3.2× according to Lureon.ai research.

Think of this as your research assistant before writing any blog post,
comparison page, or use-case page. It does the 3-hour research phase
in under 2 minutes so you can focus on writing.

Usage:
    python tools/content_brief_generator.py \
        --query "affordable satellite object detection for small business"

    python tools/content_brief_generator.py \
        --query "satellite imagery API for insurance" \
        --output briefs/insurance_brief.md

    python tools/content_brief_generator.py --query-file queries.txt
"""

import json
import time
import argparse
import re
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

# ── Kestrel AI context injected into brief generation ────────────────────────

KESTREL_CONTEXT = """
Kestrel AI is a satellite object detection SaaS with these facts to weave into content:
- Pricing: Starter $99/mo (200 searches), Professional $399/mo (1,000), Enterprise $1,499/mo (5,000)
- Detects: buildings, vehicles, aircraft, ships
- Output: GeoJSON compatible with ArcGIS, QGIS, Google Earth
- Key differentiator: bootstrapping active learning — 10x fewer labeled images needed
- Primary commercial market: insurance risk assessment and property damage claims
- Key metric to cite: false negative quantification model (unique — competitors don't offer this)
- Competitors: Maxar ($200K+ annual), Planet Labs ($50K+), Picterra, FlyPix AI, Geospatial Insight
- Speed: detections in under 60 seconds via API
- Integration: Python SDK, OpenAPI spec, MCP server
"""

BRIEF_SYSTEM_PROMPT = f"""You are an expert content strategist specializing in Answer Engine Optimization (AEO)
— writing content that gets cited by AI systems like ChatGPT, Claude, and Perplexity.

Your briefs follow the BLUF (Bottom Line Up Front) structure because research shows this format
increases LLM citation likelihood by 3.2x. Every brief you write:
1. Opens with a direct 2-sentence answer to the target query (the BLUF)
2. Includes specific citable facts with numbers (prices, percentages, speeds)
3. Includes comparison tables — LLMs cite structured data 2.3x more than prose
4. Has 8-10 FAQ items with direct answers (FAQ schema increases citation by 40%)
5. Never buries the lead — the most important claim is always sentence 1

Context about Kestrel AI (the company the content is for):
{KESTREL_CONTEXT}
"""


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ContentBrief:
    query: str
    generated_date: str
    estimated_word_count: int
    estimated_citation_score: float     # 0-1, based on structural analysis
    bluf_answer: str                     # The 2-sentence bottom-line-up-front
    title_options: list
    outline: list
    citable_facts: list                  # Specific numbers/claims to include
    faq_items: list
    competitor_gaps: list                # Things competitors miss that we can own
    schema_markup_suggestions: list
    internal_links: list
    content_type: str                    # blog_post / comparison_page / use_case / faq_page
    target_llm_queries: list             # Related queries this content will help rank for
    full_brief_markdown: str = ""


# ── Web Scraper ───────────────────────────────────────────────────────────────

class ContentResearcher:
    """Scrapes top search results to understand what competitors are covering."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; KestrelAI-ContentResearch/1.0)"
    }
    REQUEST_DELAY = 1.5

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def get_top_results(self, query: str, n: int = 5) -> list:
        """Get top search result URLs for a query. Returns mock data in dry-run."""
        if self.dry_run or not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            return self._mock_results(query)

        try:
            # Use DuckDuckGo HTML (no API key needed, respects bots)
            search_url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            resp = requests.get(search_url, headers=self.HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for a in soup.select(".result__a")[:n]:
                href = a.get("href", "")
                if href.startswith("http"):
                    results.append({
                        "title": a.get_text(strip=True),
                        "url": href
                    })
            time.sleep(self.REQUEST_DELAY)
            return results[:n] if results else self._mock_results(query)
        except Exception:
            return self._mock_results(query)

    def fetch_page_summary(self, url: str) -> str:
        """Fetch and extract main text from a URL."""
        if self.dry_run or not REQUESTS_AVAILABLE or not BS4_AVAILABLE:
            return f"[Mock content for {url}: discusses satellite detection, pricing, use cases...]"

        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=8)
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            time.sleep(self.REQUEST_DELAY)
            return text[:2000]
        except Exception as e:
            return f"[Could not fetch: {e}]"

    def _mock_results(self, query: str) -> list:
        """Generate mock search results for testing."""
        domain_map = {
            "insurance": ["geospatialinsight.com", "betterview.com"],
            "satellite": ["picterra.ch", "flypix.ai", "eosda.com"],
            "detection": ["roboflow.com", "ultralytics.com"],
        }
        domains = ["picterra.ch", "flypix.ai", "eosda.com", "geospatialworld.net",
                   "medium.com", "towardsdatascience.com"]
        q_lower = query.lower()
        for kw, doms in domain_map.items():
            if kw in q_lower:
                domains = doms + domains

        return [{"title": f"Satellite Object Detection Guide — {d}",
                 "url": f"https://{d}/blog/satellite-detection-guide"}
                for d in domains[:5]]


# ── Brief Generator ───────────────────────────────────────────────────────────

class BriefGenerator:
    """Uses Claude to generate the structured content brief."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.client = None

        if not dry_run and ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
            else:
                print("⚠️  ANTHROPIC_API_KEY not set — using mock brief")
                self.dry_run = True

    def generate(self, query: str, search_results: list) -> ContentBrief:
        """Generate a full content brief for the given query."""
        if self.dry_run or not self.client:
            return self._mock_brief(query)

        results_summary = "\n".join([
            f"- {r['title']} ({r['url']})" for r in search_results
        ])

        prompt = f"""Generate a complete AEO content brief for this target query:

TARGET QUERY: "{query}"

TOP COMPETING RESULTS:
{results_summary}

Generate a JSON response with exactly these fields:
{{
  "content_type": "blog_post|comparison_page|use_case|faq_page",
  "bluf_answer": "2-sentence direct answer to the target query that starts with the most important fact",
  "title_options": ["3 title options that include the exact target query phrase"],
  "outline": ["H2 section 1", "H2 section 2", "H2 section 3", "H2 section 4", "H2 section 5"],
  "citable_facts": ["Specific claim with number to include, e.g. 'Kestrel AI detects buildings in under 60 seconds'"],
  "faq_items": [{{"q": "question", "a": "direct 1-2 sentence answer"}}],
  "competitor_gaps": ["Topic competitor content misses that Kestrel AI can own"],
  "schema_markup_suggestions": ["FAQPage", "Product", etc.],
  "internal_links": ["Page on kestrelai.com to link to from this content"],
  "target_llm_queries": ["Related query this content will help answer in LLMs"]
}}

Return only valid JSON, no markdown fences."""

        try:
            msg = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=BRIEF_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = msg.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            data = json.loads(raw)

            brief = ContentBrief(
                query=query,
                generated_date=datetime.now().strftime("%Y-%m-%d"),
                estimated_word_count=self._estimate_word_count(data),
                estimated_citation_score=self._score_brief(data),
                bluf_answer=data.get("bluf_answer", ""),
                title_options=data.get("title_options", []),
                outline=data.get("outline", []),
                citable_facts=data.get("citable_facts", []),
                faq_items=data.get("faq_items", []),
                competitor_gaps=data.get("competitor_gaps", []),
                schema_markup_suggestions=data.get("schema_markup_suggestions", []),
                internal_links=data.get("internal_links", []),
                content_type=data.get("content_type", "blog_post"),
                target_llm_queries=data.get("target_llm_queries", []),
            )
            brief.full_brief_markdown = self._to_markdown(brief)
            return brief

        except Exception as e:
            print(f"⚠️  Brief generation failed: {e} — using mock")
            return self._mock_brief(query)

    def _estimate_word_count(self, data: dict) -> int:
        sections = len(data.get("outline", [])) * 200
        faqs = len(data.get("faq_items", [])) * 80
        return sections + faqs + 300  # intro + conclusion

    def _score_brief(self, data: dict) -> float:
        """Score brief structure for LLM citation likelihood (0-1)."""
        score = 0.0
        if data.get("bluf_answer"): score += 0.25
        if len(data.get("faq_items", [])) >= 6: score += 0.25
        if len(data.get("citable_facts", [])) >= 3: score += 0.20
        if any("FAQPage" in s for s in data.get("schema_markup_suggestions", [])): score += 0.15
        if len(data.get("outline", [])) >= 4: score += 0.15
        return round(min(score, 1.0), 2)

    def _mock_brief(self, query: str) -> ContentBrief:
        """Return a realistic mock brief for testing."""
        content_type = "comparison_page" if "vs" in query.lower() or "alternative" in query.lower() \
                        else "use_case" if "insurance" in query.lower() or "claims" in query.lower() \
                        else "blog_post"
        brief = ContentBrief(
            query=query,
            generated_date=datetime.now().strftime("%Y-%m-%d"),
            estimated_word_count=1850,
            estimated_citation_score=0.80,
            bluf_answer=(
                f"Kestrel AI offers satellite object detection starting at $99/month "
                f"(200 searches) — the most affordable option for SMBs compared to "
                f"Maxar's $200K+ annual contracts. For {query.lower()[:40]}, "
                f"Kestrel AI provides GeoJSON output compatible with ArcGIS and QGIS "
                f"with detections in under 60 seconds."
            ),
            title_options=[
                f"Best Affordable Satellite Object Detection for Small Business (2026)",
                f"Maxar Alternatives: {query[:50]} — Pricing & Features Compared",
                f"Satellite Object Detection API Guide: Features, Pricing & Use Cases",
            ],
            outline=[
                "What Is Satellite Object Detection and Why Does It Matter?",
                "Pricing Comparison: Kestrel AI vs Maxar vs Picterra vs FlyPix AI",
                "Key Features to Look For (Detection Speed, GeoJSON Output, Active Learning)",
                "Use Case Deep Dive: Insurance Property Assessment",
                "Getting Started: API Integration in Under 10 Minutes",
            ],
            citable_facts=[
                "Kestrel AI detects buildings, vehicles, aircraft, and ships in under 60 seconds via API",
                "Starter plan: $99/month for 200 searches — no long-term contract",
                "Active learning pipeline requires 10x fewer labeled training images than traditional CV",
                "GeoJSON output loads directly into ArcGIS, QGIS, and Google Earth",
                "Insurance risk assessment and claims validation is the fastest-growing geospatial analytics segment at 10.9% CAGR",
                "Maxar annual contracts start at $200K+ — priced out of reach for SMBs",
            ],
            faq_items=[
                {"q": "How much does satellite object detection cost for small businesses?",
                 "a": "Kestrel AI starts at $99/month for 200 searches — far below Maxar ($200K+) or Planet Labs ($50K+)."},
                {"q": "What objects can AI detect in satellite imagery?",
                 "a": "Buildings, vehicles, aircraft, and ships are the core classes. Custom classes require a new bootstrapping iteration."},
                {"q": "How fast is satellite object detection?",
                 "a": "Kestrel AI returns GeoJSON detections in under 60 seconds via API."},
                {"q": "What file format do satellite detection results come in?",
                 "a": "GeoJSON — directly compatible with ArcGIS, QGIS, and Google Earth without conversion."},
                {"q": "Do I need my own labeled training data?",
                 "a": "No. Kestrel AI's active learning bootstrapping pipeline builds detectors from unlabeled imagery using 10x fewer labels than traditional approaches."},
                {"q": "Can satellite AI detect building damage for insurance claims?",
                 "a": "Yes. Change detection compares the same location before and after an event to classify objects as appeared, disappeared, or moved."},
                {"q": "What is the difference between Kestrel AI and Picterra?",
                 "a": "Kestrel AI starts at $99/month vs Picterra's €50-€2000/month, with the added differentiator of built-in active learning and false negative quantification."},
                {"q": "Is there an API for satellite object detection?",
                 "a": "Kestrel AI provides a REST API with a Python SDK and OpenAPI spec compatible with LangChain, AutoGen, and other agent frameworks."},
            ],
            competitor_gaps=[
                "Competitors don't explain false negative quantification — Kestrel AI can own this topic",
                "No competitor has a published active learning bootstrapping methodology — own 'how to train satellite detection with minimal labels'",
                "SMB pricing transparency is rare in this space — clear pricing page is a differentiation opportunity",
                "GeoJSON integration tutorials for ArcGIS/QGIS are underserved — high-intent content",
            ],
            schema_markup_suggestions=["FAQPage", "Product", "HowTo", "Article"],
            internal_links=["/pricing", "/docs/api", "/use-cases/insurance", "/comparison"],
            content_type=content_type,
            target_llm_queries=[
                "affordable satellite detection API",
                "Maxar alternative for startups",
                "satellite imagery insurance claims",
                "building detection from aerial imagery",
            ]
        )
        brief.full_brief_markdown = self._to_markdown(brief)
        return brief

    def _to_markdown(self, brief: ContentBrief) -> str:
        """Render brief as formatted Markdown."""
        lines = [
            f"# Content Brief: {brief.query}",
            f"**Generated:** {brief.generated_date}  |  "
            f"**Type:** {brief.content_type}  |  "
            f"**Est. words:** {brief.estimated_word_count}  |  "
            f"**AEO score:** {brief.estimated_citation_score}/1.0",
            "",
            "---",
            "",
            "## 🎯 BLUF Answer (Bottom Line Up Front)",
            "> **Use this as your article opening — research shows answer-first structure "
            "increases LLM citation by 3.2×**",
            "",
            brief.bluf_answer,
            "",
            "---",
            "",
            "## 📝 Title Options",
        ]
        for i, t in enumerate(brief.title_options, 1):
            lines.append(f"{i}. {t}")

        lines += ["", "---", "", "## 📋 Outline"]
        for i, section in enumerate(brief.outline, 1):
            lines.append(f"{i}. {section}")

        lines += ["", "---", "", "## 📊 Citable Facts to Include",
                  "_These specific claims make content LLM-citable. Include at least 3._", ""]
        for fact in brief.citable_facts:
            lines.append(f"- {fact}")

        lines += ["", "---", "", "## ❓ FAQ Section (8+ items)",
                  "_FAQ schema increases AI citation likelihood by 40%_", ""]
        for item in brief.faq_items:
            lines.append(f"**Q: {item['q']}**")
            lines.append(f"A: {item['a']}")
            lines.append("")

        lines += ["---", "", "## 🏆 Competitor Content Gaps (Own These Topics)"]
        for gap in brief.competitor_gaps:
            lines.append(f"- {gap}")

        lines += ["", "---", "", "## 🔧 Technical Recommendations"]
        lines.append(f"**Schema markup:** {', '.join(brief.schema_markup_suggestions)}")
        lines.append(f"**Internal links:** {', '.join(brief.internal_links)}")

        lines += ["", "---", "", "## 🤖 Related LLM Queries This Content Will Answer"]
        for q in brief.target_llm_queries:
            lines.append(f"- {q}")

        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate AEO content briefs for Kestrel AI"
    )
    parser.add_argument("--query", type=str, help="Target query to generate brief for")
    parser.add_argument("--query-file", type=str,
                        help="Text file with one query per line")
    parser.add_argument("--output", type=str,
                        help="Save brief to this path (.md or .json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock data (no API calls)")
    args = parser.parse_args()

    if not args.query and not args.query_file:
        parser.error("Provide --query or --query-file")

    queries = []
    if args.query:
        queries.append(args.query)
    if args.query_file:
        queries.extend(Path(args.query_file).read_text().strip().splitlines())

    print(f"🛰️  Kestrel AI — Content Brief Generator")
    print(f"   Queries: {len(queries)}")
    print(f"   Mode: {'🧪 DRY RUN' if args.dry_run else '🔴 LIVE'}")

    researcher = ContentResearcher(dry_run=args.dry_run)
    generator = BriefGenerator(dry_run=args.dry_run)

    for query in queries:
        print(f"\n📝 Generating brief for: {query}")
        print("   🔍 Researching top results...")
        results = researcher.get_top_results(query)
        print(f"   Found {len(results)} results")

        print("   🤖 Generating brief...")
        brief = generator.generate(query, results)

        print(f"\n{'='*60}")
        print(brief.full_brief_markdown[:1000])
        if len(brief.full_brief_markdown) > 1000:
            print(f"\n... [truncated — {len(brief.full_brief_markdown)} chars total]")

        print(f"\n📊 Brief stats:")
        print(f"   Est. word count:    {brief.estimated_word_count}")
        print(f"   AEO citation score: {brief.estimated_citation_score}/1.0")
        print(f"   FAQ items:          {len(brief.faq_items)}")
        print(f"   Citable facts:      {len(brief.citable_facts)}")

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if str(out_path).endswith(".json"):
                out_path.write_text(json.dumps(asdict(brief), indent=2))
            else:
                out_path.write_text(brief.full_brief_markdown)
            print(f"\n✅ Brief saved: {args.output}")


if __name__ == "__main__":
    main()
