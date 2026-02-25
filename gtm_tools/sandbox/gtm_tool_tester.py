"""
gtm_tool_tester.py
------------------
Sandbox tester for all 6 Kestrel AI GTM tools.
Runs every tool against synthetic data — no real API calls unless --live.

Usage:
    python sandbox/gtm_tool_tester.py --all
    python sandbox/gtm_tool_tester.py --tool competitor_tracker
    python sandbox/gtm_tool_tester.py --list
    python sandbox/gtm_tool_tester.py --all --verbose
    python sandbox/gtm_tool_tester.py --tool community_scanner --keep-data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Path setup — make gtm_tools/ importable as root
# ---------------------------------------------------------------------------
_SANDBOX_DIR = Path(__file__).resolve().parent
_GTM_ROOT = _SANDBOX_DIR.parent
if str(_GTM_ROOT) not in sys.path:
    sys.path.insert(0, str(_GTM_ROOT))

ALL_TOOLS = [
    "llm_citation_monitor",
    "content_brief_generator",
    "prospect_signal_detector",
    "competitor_tracker",
    "community_scanner",
    "customer_health_monitor",
]

# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------

def test_llm_citation_monitor(sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    try:
        from tools.llm_citation_monitor import run_monitor  # type: ignore

        result = run_monitor(dry_run=True, output_path=sandbox_dir / "citations.json")

        assert isinstance(result, dict), "Result is not a dict"
        assert "citation_rate_pct" in result, "Missing citation_rate_pct"
        assert "results" in result, "Missing results"
        assert isinstance(result["results"], list), "results is not a list"
        assert result.get("total_queries", 0) >= 1, "total_queries < 1"
        for r in result["results"]:
            assert "query" in r, "Result missing query"
            assert "cited_kestrel" in r, "Result missing cited_kestrel"

        n = result["total_queries"]
        cited = result["kestrel_citations"]
        return True, f"{cited}/{n} queries cited Kestrel ({result['citation_rate_pct']}%)"
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, str(exc)


def test_content_brief_generator(sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    try:
        from tools.content_brief_generator import generate_brief  # type: ignore

        output_path = sandbox_dir / "test_brief.md"
        result = generate_brief(
            query="affordable satellite detection for insurance",
            dry_run=True,
            output_path=output_path,
        )

        assert isinstance(result, str), f"Result is {type(result)}, expected str"
        assert len(result) > 200, f"Brief too short: {len(result)} chars"
        assert output_path.exists(), "Brief file not written"

        has_structure = any(marker in result for marker in ["BLUF", "Kestrel", "#", "---"])
        assert has_structure, "Brief missing expected structure markers"

        return True, f"Brief generated ({len(result):,} chars, {len(result.split()):,} words)"
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, str(exc)


def test_prospect_signal_detector(sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    try:
        from tools.prospect_signal_detector import run_detector  # type: ignore

        result = run_detector(dry_run=True, output_path=sandbox_dir / "signals.json")

        assert isinstance(result, dict), "Result is not a dict"
        assert "signals" in result, "Missing signals key"
        assert isinstance(result["signals"], list), "signals is not a list"
        assert result.get("signals_found", 0) >= 1, "No signals found"

        for s in result["signals"]:
            assert "company" in s, "Signal missing company"
            assert "signal_type" in s, "Signal missing signal_type"
            assert "intent_score" in s, "Signal missing intent_score"
            assert "draft_outreach" in s, "Signal missing draft_outreach"
            assert len(s["draft_outreach"]) <= 310, "Draft outreach exceeds 300-char limit"

        return True, f"{result['signals_found']} signals, top intent={result['signals'][0]['intent_score']}"
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, str(exc)


def test_competitor_tracker(sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    try:
        from tools.competitor_tracker import run_tracker  # type: ignore

        result = run_tracker(dry_run=True, output_path=sandbox_dir / "competitor_snapshot.json")

        assert isinstance(result, dict), "Result is not a dict"
        assert "competitors" in result, "Missing competitors key"
        assert len(result["competitors"]) >= 1, "No competitors in snapshot"

        for name, data in result["competitors"].items():
            assert "prices_found" in data, f"{name} missing prices_found"
            assert isinstance(data["prices_found"], list), f"{name} prices_found not a list"
            assert "features_found" in data, f"{name} missing features_found"
            assert isinstance(data["features_found"], list), f"{name} features_found not a list"

        assert "kestrel_positioning" in result, "Missing kestrel_positioning"
        assert "price_advantage" in result["kestrel_positioning"], "Missing price_advantage"

        n_comps = len(result["competitors"])
        return True, f"{n_comps} competitors tracked, positioning computed"
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, str(exc)


def test_community_scanner(sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    try:
        from tools.community_scanner import run_scanner  # type: ignore

        result = run_scanner(dry_run=True, min_score=0.5, output_path=sandbox_dir / "community.json")

        assert isinstance(result, dict), "Result is not a dict"
        assert "threads" in result, "Missing threads key"
        assert isinstance(result["threads"], list), "threads is not a list"

        for t in result["threads"]:
            assert "platform" in t, "Thread missing platform"
            assert "relevance_score" in t, "Thread missing relevance_score"
            assert "title" in t, "Thread missing title"
            assert t["relevance_score"] >= 0.5, f"Thread below min_score: {t['relevance_score']}"

        high_relevance = [t for t in result["threads"] if t["relevance_score"] >= 0.7]
        for t in high_relevance:
            assert "draft_response" in t, f"High-relevance thread missing draft_response: {t['title'][:40]}"

        return True, f"{result['threads_surfaced']} threads, avg score={result['avg_relevance_score']}"
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, str(exc)


def test_customer_health_monitor(sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    try:
        from tools.customer_health_monitor import run_monitor  # type: ignore

        result = run_monitor(dry_run=True, output_path=sandbox_dir / "health.json")

        assert isinstance(result, dict), "Result is not a dict"
        assert "customers" in result, "Missing customers key"
        assert isinstance(result["customers"], list), "customers is not a list"

        for c in result["customers"]:
            assert "health_score" in c, "Customer missing health_score"
            assert 0 <= c["health_score"] <= 100, f"health_score out of range: {c['health_score']}"
            assert "signal" in c, "Customer missing signal"
            assert c["signal"] in ("upsell", "churn_risk", "expansion", "healthy"), \
                f"Unknown signal: {c['signal']}"
            # PII check — email must be redacted
            email = c.get("email", "")
            if "@" in email:
                local = email.split("@")[0]
                assert "***" in local or len(local) <= 2, \
                    f"Email not redacted: {email}"

        total_classified = (
            result["upsell_opportunities"] + result["churn_risks"] +
            result["expansion_candidates"] + result["healthy"]
        )
        assert total_classified == result["customers_analyzed"], \
            f"Signal counts don't sum to customers_analyzed: {total_classified} != {result['customers_analyzed']}"

        assert result["mrr_at_risk"] >= 0, "mrr_at_risk is negative"
        assert result["upsell_potential_mrr"] >= 0, "upsell_potential_mrr is negative"

        return True, (
            f"{result['customers_analyzed']} customers: "
            f"{result['upsell_opportunities']} upsell, "
            f"{result['churn_risks']} churn, "
            f"${result['mrr_at_risk']:.0f} MRR at risk"
        )
    except Exception as exc:
        if verbose:
            traceback.print_exc()
        return False, str(exc)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable[[Path, bool], tuple[bool, str]]] = {
    "llm_citation_monitor":    test_llm_citation_monitor,
    "content_brief_generator": test_content_brief_generator,
    "prospect_signal_detector": test_prospect_signal_detector,
    "competitor_tracker":       test_competitor_tracker,
    "community_scanner":        test_community_scanner,
    "customer_health_monitor":  test_customer_health_monitor,
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _make_sandbox() -> Path:
    ts = int(time.time())
    sandbox = Path(f"/tmp/kestrel_gtm_sandbox_{ts}")
    sandbox.mkdir(parents=True, exist_ok=True)
    return sandbox


def _run_test(name: str, sandbox_dir: Path, verbose: bool) -> tuple[bool, str]:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return False, f"No test registered for '{name}'"
    print(f"\n  Testing {name} ...", end="", flush=True)
    passed, reason = fn(sandbox_dir, verbose)
    marker = "OK" if passed else "!!"
    status = "PASS" if passed else "FAIL"
    print(f"\r  [{marker}] {name:<30} {status}  {reason}")
    return passed, reason


def run_all(tools: list[str], keep_data: bool, verbose: bool) -> None:
    sandbox_dir = _make_sandbox()
    print(f"\nSandbox: {sandbox_dir}")
    print(f"Testing {len(tools)} tool(s)\n")
    print("-" * 62)

    results: list[tuple[str, bool, str]] = []
    for name in tools:
        passed, reason = _run_test(name, sandbox_dir, verbose)
        results.append((name, passed, reason))

    print("-" * 62)
    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    failures = [(n, r) for n, p, r in results if not p]

    print(f"\nSummary: {passed_count}/{total} tools passed\n")

    if failures:
        print("Failures:")
        for name, reason in failures:
            print(f"  FAIL  {name}: {reason}")
        print()

    if keep_data:
        print(f"Sandbox preserved at: {sandbox_dir}")
    else:
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        print("Sandbox cleaned up.")

    if failures:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sandbox tester for all 6 Kestrel AI GTM tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sandbox/gtm_tool_tester.py --list
  python sandbox/gtm_tool_tester.py --all
  python sandbox/gtm_tool_tester.py --tool competitor_tracker
  python sandbox/gtm_tool_tester.py --all --verbose
  python sandbox/gtm_tool_tester.py --tool community_scanner --keep-data
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tool", metavar="NAME", help="Test a single tool")
    group.add_argument("--all", action="store_true", help="Test all tools")
    group.add_argument("--list", action="store_true", help="Print available tool names and exit")
    parser.add_argument("--verbose", action="store_true", help="Show full tracebacks on failure")
    parser.add_argument("--keep-data", action="store_true", help="Keep sandbox directory after tests")
    parser.add_argument("--live", action="store_true", help="[Reserved] Enable live API calls (not yet implemented)")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.live:
        print("⚠️  --live mode is reserved for future use. Running in dry-run mode.")

    if args.list:
        print("Available GTM tools:")
        for name in ALL_TOOLS:
            reg = "registered" if name in TOOL_REGISTRY else "NOT registered"
            print(f"  {name:<35} ({reg})")
        sys.exit(0)

    if args.all:
        run_all(ALL_TOOLS, keep_data=args.keep_data, verbose=args.verbose)
    elif args.tool:
        if args.tool not in TOOL_REGISTRY:
            print(f"❌ Unknown tool: '{args.tool}'")
            print(f"   Run --list to see available tools.")
            sys.exit(1)
        run_all([args.tool], keep_data=args.keep_data, verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
