"""
run_daily.py
------------
Daily GTM orchestrator — runs all 6 tools in recommended order,
collects results into a single summary, and posts to Basecamp.

Usage:
    python gtm_tools/run_daily.py              # Full live run + Basecamp post
    python gtm_tools/run_daily.py --dry-run    # Synthetic data, no API calls
    python gtm_tools/run_daily.py --no-basecamp # Run tools but skip Basecamp post
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tool imports
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).resolve().parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from customer_health_monitor import run_monitor as run_health          # noqa: E402
from prospect_signal_detector import run_detector as run_prospects     # noqa: E402
from community_scanner import run_scanner as run_community             # noqa: E402
from llm_citation_monitor import run_monitor as run_citations          # noqa: E402
from competitor_tracker import run_tracker as run_competitor            # noqa: E402

# ---------------------------------------------------------------------------
# Tool registry — order matters (daily first, weekly appended on schedule)
# ---------------------------------------------------------------------------
DAILY_TOOLS = [
    {
        "name": "Customer Health Monitor",
        "key": "health",
        "run": lambda dry: run_health(dry_run=dry),
        "summary_fn": lambda r: (
            f"{r.get('customers_monitored', 0)} customers monitored. "
            f"{len([s for s in r.get('signals', []) if s.get('signal_type') == 'churn_risk'])} churn risks, "
            f"{len([s for s in r.get('signals', []) if s.get('signal_type') == 'upsell'])} upsell opportunities."
        ),
    },
    {
        "name": "Prospect Signal Detector",
        "key": "prospects",
        "run": lambda dry: run_prospects(dry_run=dry),
        "summary_fn": lambda r: (
            f"{r.get('signals_found', 0)} buying-intent signals detected. "
            f"Top segment: {r.get('top_segment', 'N/A')}."
        ),
    },
    {
        "name": "Community Scanner",
        "key": "community",
        "run": lambda dry: run_community(dry_run=dry),
        "summary_fn": lambda r: (
            f"{r.get('threads_surfaced', 0)} threads found "
            f"(avg relevance: {r.get('avg_relevance_score', 0):.2f}). "
            f"{len([t for t in r.get('threads', []) if t.get('engagement_opportunity') == 'high'])} high-opportunity."
        ),
    },
]

WEEKLY_TOOLS = [
    {
        "name": "LLM Citation Monitor",
        "key": "citations",
        "run": lambda dry: run_citations(dry_run=dry),
        "schedule": "monday",
        "summary_fn": lambda r: (
            f"{len(r.get('results', []))} queries tested. "
            f"{sum(1 for x in r.get('results', []) if x.get('cited_kestrel'))} cited Kestrel AI."
        ),
    },
    {
        "name": "Competitor Tracker",
        "key": "competitor",
        "run": lambda dry: run_competitor(dry_run=dry),
        "schedule": "sunday",
        "summary_fn": lambda r: (
            f"{len(r.get('competitors', []))} competitors tracked. "
            f"{r.get('price_changes', 0)} price changes detected."
        ),
    },
]


def _should_run_weekly(tool: dict, today: datetime) -> bool:
    day = today.strftime("%A").lower()
    return tool.get("schedule", "") == day


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_daily(dry_run: bool = False, include_weekly: bool = True) -> dict[str, Any]:
    today = datetime.now(timezone.utc)
    date_str = today.strftime("%Y-%m-%d")
    day_name = today.strftime("%A")

    print("=" * 60)
    print(f"  Kestrel AI — Daily GTM Run ({date_str}, {day_name})")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print("=" * 60)
    print()

    tools_to_run = list(DAILY_TOOLS)
    if include_weekly:
        for tool in WEEKLY_TOOLS:
            if _should_run_weekly(tool, today):
                tools_to_run.append(tool)
                print(f"  + Including weekly tool: {tool['name']} (scheduled for {day_name})")

    results: dict[str, Any] = {
        "run_date": today.isoformat(),
        "mode": "dry_run" if dry_run else "live",
        "tools_run": [],
        "tools_failed": [],
        "summaries": {},
    }

    for tool in tools_to_run:
        print(f"\n{'—' * 40}")
        print(f"  Running: {tool['name']}")
        print(f"{'—' * 40}")
        try:
            report = tool["run"](dry_run)
            summary = tool["summary_fn"](report)
            results["tools_run"].append(tool["name"])
            results["summaries"][tool["key"]] = summary
            print(f"  Result: {summary}")
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            results["tools_failed"].append({"name": tool["name"], "error": error_msg})
            print(f"  FAILED: {error_msg}")
            if dry_run:
                traceback.print_exc()

    # Build Basecamp-ready summary
    results["basecamp_summary"] = _format_basecamp_summary(results, date_str)

    # Save daily report
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"daily_{today.strftime('%Y%m%d')}.json"
    report_path.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  Daily run complete. {len(results['tools_run'])} succeeded, {len(results['tools_failed'])} failed.")
    print(f"  Report saved: {report_path}")
    print(f"{'=' * 60}")

    return results


def _format_basecamp_summary(results: dict[str, Any], date_str: str) -> str:
    lines = [f"Daily GTM Summary — {date_str}", ""]

    for key, summary in results["summaries"].items():
        label = key.replace("_", " ").title()
        lines.append(f"- {label}: {summary}")

    if results["tools_failed"]:
        lines.append("")
        lines.append("Failures:")
        for fail in results["tools_failed"]:
            lines.append(f"- {fail['name']}: {fail['error']}")

    lines.append("")
    lines.append(f"Tools run: {len(results['tools_run'])} | Failed: {len(results['tools_failed'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Basecamp posting
# ---------------------------------------------------------------------------

def post_to_basecamp(summary: str, date_str: str) -> bool:
    """Post daily summary to Basecamp message board.

    This function is designed to be called from Claude Code with MCP tools.
    When run standalone, it prints the summary for manual posting.
    """
    print("\n--- Basecamp Summary ---")
    print(summary)
    print("--- End Summary ---")
    print("\nTo post to Basecamp: use Claude Code MCP tools or copy the above.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all GTM tools and produce a daily summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data, no API calls")
    parser.add_argument("--no-basecamp", action="store_true", help="Skip Basecamp posting")
    parser.add_argument("--include-weekly", action="store_true", help="Force-include weekly tools regardless of day")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results = run_daily(dry_run=args.dry_run, include_weekly=args.include_weekly or True)

    if not args.no_basecamp and not args.dry_run:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        post_to_basecamp(results["basecamp_summary"], date_str)


if __name__ == "__main__":
    main()
