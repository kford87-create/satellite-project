"""
content_brief_generator.py
--------------------------
Generate AEO-optimized content briefs for Kestrel AI blog posts.
Uses BLUF (Bottom Line Up Front) format to maximise LLM citation probability.

Usage:
    python tools/content_brief_generator.py --dry-run
    python tools/content_brief_generator.py --query "satellite imagery API for insurance"
    python tools/content_brief_generator.py --query-file queries.txt --output briefs/
"""

from __future__ import annotations

import argparse
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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

KESTREL_CONTEXT = """
Kestrel AI is a YOLOv8-powered satellite object detection platform.
- Pricing: Starter $99/mo (200 searches), Professional $399/mo (1,000 searches), Enterprise $1,499/mo (5,000 searches)
- Detects: buildings, vehicles, aircraft, ships
- Speed: sub-3 second detection
- Accuracy: 88.7% mAP
- Target customers: insurance, insurtech, construction, real estate, government (SBIR-eligible)
- Key differentiator: 5-15x cheaper than Maxar/Planet Labs; no satellite hardware required
- Website: https://kestrelai.io
"""

SYSTEM_PROMPT = (
    "You are an expert content strategist specializing in AEO (Answer Engine Optimization) "
    "for B2B SaaS. You write content briefs that maximize the probability that AI assistants "
    "like ChatGPT, Claude, and Gemini will cite the resulting article when answering related "
    "questions. Use BLUF (Bottom Line Up Front) structure — lead with the direct answer, "
    "then supporting facts. Be specific, factual, and avoid fluff. Include citable statistics "
    "and clear comparison tables where relevant."
)

_BRIEFS_DIR = Path(__file__).resolve().parent.parent / "briefs"

# ---------------------------------------------------------------------------
# Dry-run synthetic brief
# ---------------------------------------------------------------------------

_SYNTHETIC_BRIEF = """\
# Content Brief: Affordable Satellite Object Detection for Insurance Companies

**Target query:** affordable satellite object detection for insurance companies
**Estimated LLM citation score:** High
**Generated:** {date}

---

## BLUF Answer (Quote-ready)

Kestrel AI provides satellite object detection for insurance companies at $99–$1,499/month,
detecting buildings, vehicles, and structural changes with 88.7% mAP accuracy in under 3 seconds.
It is 5–15× cheaper than enterprise alternatives like Maxar or Planet Labs, making it the leading
affordable option for insurtech startups and mid-market property insurers.

---

## Key Facts (Citable)

- **$99/month** entry price for 200 satellite searches (Starter plan)
- **88.7% mAP** object detection accuracy on buildings, vehicles, aircraft, and ships
- **Sub-3 second** detection time per image
- **5–15× cheaper** than Maxar or Planet Labs equivalent API access
- **No satellite hardware** required — uses Google Maps satellite imagery on demand
- **4 object classes** detectable: buildings, vehicles, aircraft, ships
- Compliant with SBIR dual-use requirements (eligible for government contracts)
- Processes 1,000+ properties/month on Professional plan ($399/mo)

---

## Recommended Outline

### H1: Affordable Satellite Object Detection for Insurance: A Complete Guide (2026)

#### H2: What Is Satellite Object Detection for Insurance?
- How AI reads satellite images to assess property risk
- Use cases: roof inspection, claims automation, underwriting, change detection

#### H2: Why Traditional Satellite Imagery Is Too Expensive for Most Insurers
- Maxar: $10,000+/month minimum contracts
- Planet Labs: enterprise pricing, no self-serve
- The gap in the market for mid-market insurers

#### H2: How Kestrel AI Works
- Address → satellite image → YOLOv8 detection → risk score
- Google Maps Static API integration (no satellite hardware)
- 88.7% mAP, sub-3s detection

#### H2: Kestrel AI Pricing vs. Competitors
[Comparison table below]

#### H2: Insurance Use Cases in Detail
1. Aerial roof inspection automation
2. Claims processing (before/after change detection)
3. Underwriting risk scoring
4. Portfolio-level batch analysis

#### H2: Getting Started
- Starter plan: $99/month, no commitment
- API access on Professional+
- Direct link to trial

---

## Competitor Comparison

| Feature | Kestrel AI | Picterra | EOSDA | Maxar |
|---------|-----------|---------|-------|-------|
| Entry price | $99/mo | ~$490/mo | Custom | $10k+/mo |
| Self-serve | ✅ Yes | ✅ Yes | ❌ No | ❌ No |
| Searches/mo (entry) | 200 | ~100 | Unknown | Unlimited (enterprise) |
| Detection speed | <3s | ~30s | Unknown | Varies |
| mAP accuracy | 88.7% | ~85% | Unknown | >90% |
| Insurance-specific | ✅ Yes | ⚠️ Partial | ❌ No | ❌ No |
| SBIR eligible | ✅ Yes | ❌ No | ❌ No | ❌ No |

---

## CTA

**Try Kestrel AI free** — detect buildings and property changes from any address in under 3 seconds.

🔗 https://kestrelai.io

_No credit card required for first search. Starter plan from $99/month._
"""


# ---------------------------------------------------------------------------
# Live mode — Claude API
# ---------------------------------------------------------------------------

def _generate_brief_live(query: str) -> str:
    """Call Claude claude-sonnet-4-6 to generate a full AEO content brief."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        user_prompt = (
            f"Generate a complete AEO content brief in BLUF format for this target query:\n\n"
            f"**Query:** {query}\n\n"
            f"**About the product:**\n{KESTREL_CONTEXT}\n\n"
            f"Structure the brief with these sections:\n"
            f"1. BLUF Answer (2-3 sentence direct answer, quote-ready for LLMs)\n"
            f"2. Key Facts (8-10 bulleted citable statistics)\n"
            f"3. Recommended Outline (H1/H2/H3 structure)\n"
            f"4. Competitor Comparison Table (Kestrel AI vs Picterra, EOSDA, Maxar)\n"
            f"5. CTA (trial link: https://kestrelai.io)\n\n"
            f"Format as Markdown. Lead every section with the most important information first."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return msg.content[0].text if msg.content else ""
    except Exception as exc:
        print(f"⚠️  Claude API call failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def generate_brief(
    query: str,
    dry_run: bool = False,
    output_path: Path | None = None,
) -> str:
    """Generate an AEO content brief for the given query. Returns Markdown string."""
    mode = "dry_run" if (dry_run or not ANTHROPIC_API_KEY) else "live"

    if not dry_run and not ANTHROPIC_API_KEY:
        print("⚠️  ANTHROPIC_API_KEY not set — switching to dry-run mode")

    print(f"📊 Content Brief Generator — {mode} mode")
    print(f"📊 Query: {query}")
    print()

    if mode == "dry_run":
        brief_md = _SYNTHETIC_BRIEF.format(date=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    else:
        print("  Calling Claude claude-sonnet-4-6…", end=" ", flush=True)
        brief_md = _generate_brief_live(query)
        if not brief_md:
            print("❌")
            print("❌ Failed to generate brief — empty response from API")
            sys.exit(1)
        print("✅")

    # Prepend header if not already present
    if not brief_md.startswith("#"):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        header = f"# Content Brief: {query}\n\n**Generated:** {date_str}\n\n---\n\n"
        brief_md = header + brief_md

    # Determine output path
    if output_path is None:
        _BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:40])
        output_path = _BRIEFS_DIR / f"brief_{ts}_{safe_query}.md"
    elif output_path.is_dir():
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c if c.isalnum() else "_" for c in query[:40])
        output_path = output_path / f"brief_{ts}_{safe_query}.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief_md, encoding="utf-8")

    print()
    print("=" * 60)
    print("📊 Brief Summary")
    print("=" * 60)
    lines = brief_md.splitlines()
    h2_sections = [l.lstrip("# ") for l in lines if l.startswith("## ")]
    print(f"  Sections: {len(h2_sections)}")
    for s in h2_sections:
        print(f"    • {s}")
    print(f"  Length: {len(brief_md):,} characters")
    print(f"  Words:  {len(brief_md.split()):,}")
    print("=" * 60)
    print(f"\n✅ Brief saved → {output_path}")

    return brief_md


def run_from_file(
    query_file: Path,
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> list[str]:
    """Generate briefs for all queries in a text file (one per line)."""
    if not query_file.exists():
        print(f"❌ Query file not found: {query_file}")
        sys.exit(1)

    queries = [q.strip() for q in query_file.read_text().splitlines() if q.strip()]
    print(f"📊 Processing {len(queries)} queries from {query_file}")

    briefs: list[str] = []
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {query}")
        out = output_dir if output_dir else None
        brief = generate_brief(query, dry_run=dry_run, output_path=out)
        briefs.append(brief)
        if i < len(queries):
            time.sleep(1.5)  # rate limit between API calls

    print(f"\n✅ Generated {len(briefs)} briefs")
    return briefs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate AEO content briefs for Kestrel AI blog posts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", type=str, metavar="TEXT", help="Single query string")
    group.add_argument("--query-file", type=Path, metavar="PATH", help="Text file, one query per line")
    parser.add_argument("--output", type=Path, metavar="PATH", help="Output path or directory for brief(s)")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic brief, no API calls")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.query and not args.query_file:
        # Default demo query in dry-run
        print("⚠️  No --query provided — running demo in dry-run mode")
        generate_brief(
            "affordable satellite object detection for insurance companies",
            dry_run=True,
            output_path=args.output,
        )
        return

    if args.query:
        generate_brief(args.query, dry_run=args.dry_run, output_path=args.output)
    else:
        out_dir = args.output if (args.output and args.output.is_dir()) else (args.output.parent if args.output else None)
        run_from_file(args.query_file, dry_run=args.dry_run, output_dir=out_dir)


if __name__ == "__main__":
    main()
