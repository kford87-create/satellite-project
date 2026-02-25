"""
sandbox/gtm_tool_tester.py

Sandbox environment for testing all GTM agent tools before production use.
Every tool has a dedicated test suite that runs without touching real APIs,
external services, or Supabase. All tests use synthetic/mock data.

Think of this as a flight simulator — you can crash here safely.
No real Claude API calls. No real Reddit scraping. No real Supabase queries.

Usage:
    python sandbox/gtm_tool_tester.py --tool llm_citation_monitor
    python sandbox/gtm_tool_tester.py --all
    python sandbox/gtm_tool_tester.py --all --verbose
    python sandbox/gtm_tool_tester.py --tool competitor_tracker --keep-data
    python sandbox/gtm_tool_tester.py --list
"""

import os
import sys
import json
import time
import shutil
import traceback
import argparse
from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add tools directory to path
GTM_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(GTM_ROOT / "tools"))

SANDBOX_DIR = GTM_ROOT / "data" / "sandbox"


# ── Result Types ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    tool_name: str
    test_name: str
    passed: bool
    duration_seconds: float
    message: str = ""
    output_summary: Dict = field(default_factory=dict)
    error: str = ""
    tb: str = ""


@dataclass
class ToolReport:
    tool_name: str
    total_tests: int
    passed: int
    failed: int
    duration_seconds: float
    results: List[TestResult] = field(default_factory=list)
    sandbox_dir: str = ""

    @property
    def status(self):
        if self.failed == 0:
            return "✅ PASS"
        elif self.passed > 0:
            return "⚠️  PARTIAL"
        return "❌ FAIL"


# ── Test Runner ───────────────────────────────────────────────────────────────

class GTMToolTests:
    """One test method per GTM tool. All tests run in dry-run/mock mode."""

    def __init__(self, sandbox_dir: Path, verbose: bool = False):
        self.sandbox = sandbox_dir
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"     {msg}")

    def _run(self, test_name: str, fn: Callable) -> TestResult:
        start = time.time()
        try:
            output = fn()
            duration = time.time() - start
            return TestResult(
                tool_name="", test_name=test_name,
                passed=True, duration_seconds=round(duration, 3),
                output_summary=output if isinstance(output, dict) else {"result": str(output)[:200]}
            )
        except Exception as e:
            duration = time.time() - start
            return TestResult(
                tool_name="", test_name=test_name,
                passed=False, duration_seconds=round(duration, 3),
                error=str(e), tb=traceback.format_exc()
            )

    # ── Tool 1: LLM Citation Monitor ─────────────────────────────────────────

    def test_llm_citation_monitor(self) -> List[TestResult]:
        from llm_citation_monitor import (
            LLMCitationChecker, build_report, DEFAULT_QUERIES,
            BRAND_SIGNALS, COMPETITOR_SIGNALS
        )
        results = []

        def test_dry_run_query():
            checker = LLMCitationChecker(dry_run=True)
            result = checker.query("What is the best satellite object detection tool?", "claude")
            assert result.platform == "claude"
            assert isinstance(result.kestrel_cited, bool)
            assert isinstance(result.competitors_mentioned, list)
            assert result.latency_seconds >= 0
            assert result.error is None
            return {"kestrel_cited": result.kestrel_cited,
                    "competitors": len(result.competitors_mentioned)}

        def test_batch_queries():
            checker = LLMCitationChecker(dry_run=True)
            queries = DEFAULT_QUERIES[:5]
            batch_results = [checker.query(q, "claude") for q in queries]
            assert len(batch_results) == 5
            assert all(hasattr(r, "kestrel_cited") for r in batch_results)
            cited = sum(1 for r in batch_results if r.kestrel_cited)
            return {"total": len(batch_results), "cited": cited}

        def test_report_structure():
            checker = LLMCitationChecker(dry_run=True)
            query_results = [checker.query(q, "claude") for q in DEFAULT_QUERIES[:8]]
            report = build_report(query_results, "claude")
            assert report.total_queries == 8
            assert 0 <= report.citation_rate_pct <= 100
            assert report.platform == "claude"
            assert isinstance(report.sentiment_breakdown, dict)
            assert isinstance(report.top_competitor_mentions, dict)
            return {
                "citation_rate": report.citation_rate_pct,
                "sentiments": list(report.sentiment_breakdown.keys()),
            }

        def test_report_serializable():
            from dataclasses import asdict
            checker = LLMCitationChecker(dry_run=True)
            query_results = [checker.query(q, "claude") for q in DEFAULT_QUERIES[:3]]
            report = build_report(query_results, "claude")
            data = asdict(report)
            json_str = json.dumps(data)
            assert len(json_str) > 100
            return {"json_bytes": len(json_str)}

        def test_output_file():
            checker = LLMCitationChecker(dry_run=True)
            query_results = [checker.query(q, "test") for q in DEFAULT_QUERIES[:3]]
            report = build_report(query_results, "test")
            out_path = self.sandbox / "citation_report_test.json"
            from dataclasses import asdict
            out_path.write_text(json.dumps(asdict(report), indent=2))
            assert out_path.exists()
            assert out_path.stat().st_size > 500
            return {"file_size_bytes": out_path.stat().st_size}

        for name, fn in [
            ("dry_run_query", test_dry_run_query),
            ("batch_queries", test_batch_queries),
            ("report_structure", test_report_structure),
            ("report_serializable", test_report_serializable),
            ("output_file", test_output_file),
        ]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results

    # ── Tool 2: Content Brief Generator ──────────────────────────────────────

    def test_content_brief_generator(self) -> List[TestResult]:
        from content_brief_generator import (
            ContentResearcher, BriefGenerator, ContentBrief
        )
        results = []

        def test_mock_research():
            researcher = ContentResearcher(dry_run=True)
            results_data = researcher.get_top_results(
                "affordable satellite object detection for small business"
            )
            assert isinstance(results_data, list)
            assert len(results_data) >= 3
            assert all("title" in r and "url" in r for r in results_data)
            return {"n_results": len(results_data)}

        def test_brief_generation():
            researcher = ContentResearcher(dry_run=True)
            generator = BriefGenerator(dry_run=True)
            search_results = researcher.get_top_results("satellite imagery API for insurance")
            brief = generator.generate("satellite imagery API for insurance", search_results)
            assert isinstance(brief, ContentBrief)
            assert brief.query == "satellite imagery API for insurance"
            assert len(brief.faq_items) >= 4
            assert len(brief.citable_facts) >= 3
            assert len(brief.outline) >= 3
            assert brief.bluf_answer != ""
            assert 0 < brief.estimated_citation_score <= 1.0
            return {
                "faq_count": len(brief.faq_items),
                "fact_count": len(brief.citable_facts),
                "citation_score": brief.estimated_citation_score,
            }

        def test_markdown_output():
            researcher = ContentResearcher(dry_run=True)
            generator = BriefGenerator(dry_run=True)
            search_results = researcher.get_top_results("Picterra alternative")
            brief = generator.generate("Picterra alternative", search_results)
            md = brief.full_brief_markdown
            assert "# Content Brief" in md
            assert "BLUF" in md
            assert "FAQ" in md
            assert len(md) > 500
            return {"markdown_chars": len(md)}

        def test_file_save_json():
            from dataclasses import asdict
            researcher = ContentResearcher(dry_run=True)
            generator = BriefGenerator(dry_run=True)
            results_data = researcher.get_top_results("change detection satellite API")
            brief = generator.generate("change detection satellite API", results_data)
            out = self.sandbox / "test_brief.json"
            out.write_text(json.dumps(asdict(brief), indent=2))
            assert out.exists()
            loaded = json.loads(out.read_text())
            assert loaded["query"] == "change detection satellite API"
            return {"file_size_kb": round(out.stat().st_size / 1024, 1)}

        def test_file_save_markdown():
            researcher = ContentResearcher(dry_run=True)
            generator = BriefGenerator(dry_run=True)
            results_data = researcher.get_top_results("building damage detection insurance")
            brief = generator.generate("building damage detection insurance", results_data)
            out = self.sandbox / "test_brief.md"
            out.write_text(brief.full_brief_markdown)
            assert out.exists()
            content = out.read_text()
            assert "Kestrel" in content
            return {"file_size_kb": round(out.stat().st_size / 1024, 1)}

        for name, fn in [
            ("mock_research", test_mock_research),
            ("brief_generation", test_brief_generation),
            ("markdown_output", test_markdown_output),
            ("file_save_json", test_file_save_json),
            ("file_save_markdown", test_file_save_markdown),
        ]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results

    # ── Tool 3: Prospect Signal Detector ─────────────────────────────────────

    def test_prospect_signal_detector(self) -> List[TestResult]:
        from prospect_signal_detector import (
            ProspectScanner, build_report, score_intent,
            draft_outreach_message, ICP_SEGMENTS
        )
        results = []

        def test_intent_scoring():
            # High-value keywords should score higher
            score_high = score_intent("job_posting", ["satellite imagery", "geospatial"], 2)
            score_low = score_intent("conference", ["general tech"], 30)
            assert score_high > score_low
            assert 0 <= score_high <= 1.0
            assert 0 <= score_low <= 1.0
            return {"high_score": score_high, "low_score": score_low}

        def test_outreach_draft():
            hook, msg = draft_outreach_message(
                {"type": "job_posting", "title": "Geospatial Data Analyst"},
                "Acme Insurance", "insurance"
            )
            assert len(hook) > 20
            assert len(msg) > 100
            assert "Kestrel AI" in msg
            assert "$99" in msg
            return {"hook_len": len(hook), "msg_len": len(msg)}

        def test_mock_scan_all_segments():
            scanner = ProspectScanner(dry_run=True)
            signals = scanner.scan(list(ICP_SEGMENTS.keys()))
            assert isinstance(signals, list)
            assert len(signals) > 0
            assert all(hasattr(s, "intent_score") for s in signals)
            assert all(0 <= s.intent_score <= 1.0 for s in signals)
            return {"total_signals": len(signals)}

        def test_mock_scan_single_segment():
            scanner = ProspectScanner(dry_run=True)
            signals = scanner.scan(["insurance"])
            assert all(s.segment == "insurance" for s in signals)
            return {"insurance_signals": len(signals)}

        def test_report_structure():
            from dataclasses import asdict
            scanner = ProspectScanner(dry_run=True)
            signals = scanner.scan(list(ICP_SEGMENTS.keys()))
            report = build_report(signals, None)
            assert report.total_signals == len(signals)
            assert report.high_intent_count <= report.total_signals
            data = asdict(report)
            json.dumps(data)  # Must be serializable
            return {
                "total": report.total_signals,
                "high_intent": report.high_intent_count,
            }

        def test_output_file():
            from dataclasses import asdict
            scanner = ProspectScanner(dry_run=True)
            signals = scanner.scan(list(ICP_SEGMENTS.keys()))
            report = build_report(signals, None)
            out = self.sandbox / "signals_test.json"
            out.write_text(json.dumps(asdict(report), indent=2))
            assert out.exists()
            return {"file_size_bytes": out.stat().st_size}

        for name, fn in [
            ("intent_scoring", test_intent_scoring),
            ("outreach_draft", test_outreach_draft),
            ("mock_scan_all_segments", test_mock_scan_all_segments),
            ("mock_scan_single_segment", test_mock_scan_single_segment),
            ("report_structure", test_report_structure),
            ("output_file", test_output_file),
        ]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results

    # ── Tool 5: Competitor Tracker ────────────────────────────────────────────

    def test_competitor_tracker(self) -> List[TestResult]:
        from competitor_tracker import (
            CompetitorScraper, diff_snapshots, save_snapshot, COMPETITORS
        )
        results = []

        def test_mock_scrape_single():
            scraper = CompetitorScraper(dry_run=True)
            snap = scraper.scrape("picterra")
            assert snap.name == "Picterra"
            assert snap.scrape_success
            assert snap.pricing_text != ""
            assert len(snap.pricing_hash) == 12
            assert isinstance(snap.features_mentioned, list)
            return {
                "name": snap.name,
                "hash": snap.pricing_hash,
                "features": len(snap.features_mentioned),
            }

        def test_mock_scrape_all():
            scraper = CompetitorScraper(dry_run=True)
            snaps = [scraper.scrape(c) for c in COMPETITORS]
            assert len(snaps) == len(COMPETITORS)
            assert all(s.scrape_success for s in snaps)
            names = [s.name for s in snaps]
            assert "Picterra" in names
            assert "FlyPix AI" in names
            return {"scraped": len(snaps), "all_successful": True}

        def test_diff_no_change():
            scraper = CompetitorScraper(dry_run=True)
            snap1 = scraper.scrape("picterra")
            old_data = asdict(snap1)
            diff = diff_snapshots(old_data, snap1)
            assert not diff.pricing_changed
            assert diff.new_features == []
            return {"pricing_changed": diff.pricing_changed}

        def test_diff_detects_change():
            from competitor_tracker import CompetitorSnapshot
            import hashlib
            scraper = CompetitorScraper(dry_run=True)
            snap1 = scraper.scrape("picterra")
            old_data = asdict(snap1)

            # Simulate pricing change
            new_pricing = "Starter €75/month · NEW PRICING · Professional €600/month"
            snap2 = CompetitorSnapshot(
                competitor_id="picterra", name="Picterra",
                scraped_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                pricing_text=new_pricing,
                pricing_hash=hashlib.md5(new_pricing.encode()).hexdigest()[:12],
                features_mentioned=snap1.features_mentioned + ["new_feature"],
                recent_blog_titles=["Brand new post"] + snap1.recent_blog_titles[:2],
                press_mentions=[],
                page_word_count=1500,
                scrape_success=True,
            )
            diff = diff_snapshots(old_data, snap2)
            assert diff.pricing_changed
            assert diff.action_required
            assert "new_feature" in diff.new_features
            return {
                "pricing_changed": diff.pricing_changed,
                "new_features": diff.new_features,
                "new_posts": len(diff.new_blog_posts),
            }

        def test_snapshot_save_load():
            scraper = CompetitorScraper(dry_run=True)
            snap = scraper.scrape("flypix")

            # Override SNAPSHOT_DIR to sandbox
            import competitor_tracker
            original_dir = competitor_tracker.SNAPSHOT_DIR
            competitor_tracker.SNAPSHOT_DIR = self.sandbox / "competitor_snapshots"

            saved_path = save_snapshot(snap)
            assert saved_path.exists()

            loaded = json.loads(saved_path.read_text())
            assert loaded["name"] == "FlyPix AI"

            competitor_tracker.SNAPSHOT_DIR = original_dir
            return {"saved": True, "path": str(saved_path.name)}

        def test_report_serializable():
            from dataclasses import asdict as _asdict
            from competitor_tracker import TrackerReport
            scraper = CompetitorScraper(dry_run=True)
            snaps = [scraper.scrape(c) for c in list(COMPETITORS.keys())[:2]]
            report = TrackerReport(
                run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                competitors_tracked=len(snaps),
                changes_detected=0,
                snapshots=[_asdict(s) for s in snaps],
            )
            json_str = json.dumps(_asdict(report))
            assert len(json_str) > 200
            return {"json_bytes": len(json_str)}

        for name, fn in [
            ("mock_scrape_single", test_mock_scrape_single),
            ("mock_scrape_all", test_mock_scrape_all),
            ("diff_no_change", test_diff_no_change),
            ("diff_detects_change", test_diff_detects_change),
            ("snapshot_save_load", test_snapshot_save_load),
            ("report_serializable", test_report_serializable),
        ]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results

    # ── Tool 6: Community Scanner ─────────────────────────────────────────────

    def test_community_scanner(self) -> List[TestResult]:
        from community_scanner import (
            RedditScanner, HackerNewsScanner, build_report,
            score_relevance, draft_response
        )
        results = []

        def test_relevance_scoring_high():
            rel, opp, kws = score_relevance(
                "Best satellite object detection tool for insurance?",
                "Looking for an affordable alternative to Maxar for insurance property assessment.",
                score=45, age_hours=4
            )
            assert rel >= 0.4, f"Expected high relevance, got {rel}"
            assert opp in ("tool_recommendation", "technical_help", "general_discussion")
            assert len(kws) > 0
            return {"relevance": rel, "type": opp, "keywords": kws}

        def test_relevance_scoring_low():
            rel, opp, kws = score_relevance(
                "GPS tracking for delivery vehicles",
                "We need to track our fleet in real time.",
                score=5, age_hours=100
            )
            assert rel < 0.40, f"Expected low relevance for off-topic post, got {rel}"
            return {"relevance": rel}

        def test_draft_response():
            resp = draft_response(
                {"title": "Best satellite imagery API for small business?"},
                "tool_recommendation"
            )
            assert "Kestrel AI" in resp
            assert "kestrelai.com" in resp
            assert len(resp) > 50
            return {"response_len": len(resp)}

        def test_reddit_mock_scan():
            scanner = RedditScanner(dry_run=True)
            threads = scanner.scan()
            assert isinstance(threads, list)
            assert len(threads) > 0
            for t in threads:
                assert hasattr(t, "relevance_score")
                assert 0 <= t.relevance_score <= 1.0
                assert t.platform == "reddit"
                assert t.draft_response != ""
            return {"threads": len(threads), "platforms": list({t.platform for t in threads})}

        def test_hn_mock_scan():
            scanner = HackerNewsScanner(dry_run=True)
            threads = scanner.scan()
            assert isinstance(threads, list)
            assert len(threads) > 0
            assert all(t.platform == "hackernews" for t in threads)
            return {"hn_threads": len(threads)}

        def test_report_with_filters():
            reddit = RedditScanner(dry_run=True).scan()
            hn = HackerNewsScanner(dry_run=True).scan()
            all_threads = reddit + hn
            report = build_report(all_threads, platform_filter=None, min_score=0.30)
            assert report.total_threads_scanned == len(all_threads)
            assert report.relevant_threads <= report.total_threads_scanned
            assert report.high_opportunity_count <= report.relevant_threads
            return {
                "total_scanned": report.total_threads_scanned,
                "relevant": report.relevant_threads,
                "high_opp": report.high_opportunity_count,
            }

        def test_report_serializable():
            from dataclasses import asdict
            reddit = RedditScanner(dry_run=True).scan()
            report = build_report(reddit, None, 0.20)
            json_str = json.dumps(asdict(report))
            assert len(json_str) > 200
            return {"json_bytes": len(json_str)}

        for name, fn in [
            ("relevance_scoring_high", test_relevance_scoring_high),
            ("relevance_scoring_low", test_relevance_scoring_low),
            ("draft_response", test_draft_response),
            ("reddit_mock_scan", test_reddit_mock_scan),
            ("hn_mock_scan", test_hn_mock_scan),
            ("report_with_filters", test_report_with_filters),
            ("report_serializable", test_report_serializable),
        ]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results

    # ── Tool 7: Customer Health Monitor ──────────────────────────────────────

    def test_customer_health_monitor(self) -> List[TestResult]:
        from customer_health_monitor import (
            SupabaseHealthReader, detect_signals, build_report,
            draft_upsell_email, draft_churn_email, draft_expansion_email,
            CustomerMetrics, TIERS
        )
        results = []

        def make_metric(**overrides):
            defaults = dict(
                client_id="cli_test", company_name="Test Co",
                email="test@test.com", tier="starter",
                plan_price=99, monthly_limit=200,
                searches_this_month=100, searches_last_week=25,
                searches_week_before=22, last_active_date=datetime.now().strftime("%Y-%m-%d"),
                api_call_types={"detect": 95, "batch": 5},
                days_since_active=0, usage_pct=0.50,
            )
            defaults.update(overrides)
            return CustomerMetrics(**defaults)

        def test_upsell_detection():
            m = make_metric(searches_this_month=170, usage_pct=0.85)
            signals = detect_signals(m, None)
            upsell = [s for s in signals if s.signal_type == "upsell"]
            assert len(upsell) == 1
            assert upsell[0].severity in ("high", "medium")
            return {"upsell_signals": len(upsell), "severity": upsell[0].severity}

        def test_churn_detection_inactive():
            m = make_metric(days_since_active=20, searches_last_week=0,
                            searches_week_before=40, usage_pct=0.02)
            signals = detect_signals(m, None)
            churn = [s for s in signals if s.signal_type == "churn"]
            assert len(churn) == 1
            return {"churn_signals": len(churn), "days_inactive": 20}

        def test_churn_detection_drop():
            m = make_metric(searches_last_week=5, searches_week_before=50,
                            days_since_active=3, usage_pct=0.15)
            signals = detect_signals(m, None)
            churn = [s for s in signals if s.signal_type == "churn"]
            assert len(churn) == 1
            return {"churn_detected": True, "drop_pct": "90%"}

        def test_expansion_detection():
            m = make_metric(api_call_types={"detect": 140, "batch": 8},
                            tier="starter", days_since_active=0, usage_pct=0.75)
            signals = detect_signals(m, None)
            expansion = [s for s in signals if s.signal_type == "expansion"]
            assert len(expansion) == 1
            return {"expansion_signals": len(expansion)}

        def test_healthy_customer_no_signals():
            m = make_metric(searches_this_month=80, usage_pct=0.40,
                            searches_last_week=20, searches_week_before=18,
                            days_since_active=1, api_call_types={"detect": 80})
            signals = detect_signals(m, None)
            assert len(signals) == 0
            return {"signals": 0, "healthy": True}

        def test_email_drafts():
            m = make_metric(searches_this_month=180, usage_pct=0.90)
            subj, body = draft_upsell_email(m)
            assert "90%" in subj or "searches" in subj.lower()
            assert "Kestrel AI" in body or "kestrelai" in body.lower()
            assert "$399" in body  # Next tier price

            m2 = make_metric(days_since_active=20, searches_last_week=2)
            subj2, body2 = draft_churn_email(m2)
            assert len(subj2) > 10
            assert len(body2) > 100

            m3 = make_metric(api_call_types={"detect": 130, "batch": 10})
            subj3, body3 = draft_expansion_email(m3)
            assert "batch" in body3.lower() or "$399" in body3
            return {"upsell_email": len(body), "churn_email": len(body2),
                    "expansion_email": len(body3)}

        def test_mock_customer_load():
            reader = SupabaseHealthReader(dry_run=True)
            customers = reader.get_all_customers()
            assert len(customers) >= 3
            assert all(hasattr(c, "tier") for c in customers)
            assert all(c.tier in TIERS for c in customers)
            return {"customer_count": len(customers)}

        def test_full_report():
            from dataclasses import asdict
            reader = SupabaseHealthReader(dry_run=True)
            customers = reader.get_all_customers()
            all_signals = []
            for m in customers:
                all_signals.extend(detect_signals(m, None))
            report = build_report(all_signals, customers, None)
            assert report.total_customers == len(customers)
            assert report.upsell_opportunities + report.churn_risks + report.expansion_signals == len(all_signals)
            json_str = json.dumps(asdict(report))
            assert len(json_str) > 500
            return {
                "customers": report.total_customers,
                "upsell": report.upsell_opportunities,
                "churn": report.churn_risks,
                "expansion": report.expansion_signals,
                "at_risk_mrr": report.estimated_at_risk_mrr,
            }

        for name, fn in [
            ("upsell_detection", test_upsell_detection),
            ("churn_detection_inactive", test_churn_detection_inactive),
            ("churn_detection_drop", test_churn_detection_drop),
            ("expansion_detection", test_expansion_detection),
            ("healthy_customer_no_signals", test_healthy_customer_no_signals),
            ("email_drafts", test_email_drafts),
            ("mock_customer_load", test_mock_customer_load),
            ("full_report", test_full_report),
        ]:
            r = self._run(name, fn)
            self._log(f"{'✅' if r.passed else '❌'} {name}: {r.error[:80] if not r.passed else r.output_summary}")
            results.append(r)
        return results


# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY = {
    "llm_citation_monitor":    "test_llm_citation_monitor",
    "content_brief_generator": "test_content_brief_generator",
    "prospect_signal_detector":"test_prospect_signal_detector",
    "competitor_tracker":      "test_competitor_tracker",
    "community_scanner":       "test_community_scanner",
    "customer_health_monitor": "test_customer_health_monitor",
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_tool_tests(tool_name: str, verbose: bool = False, keep_data: bool = False) -> ToolReport:
    sandbox_dir = SANDBOX_DIR / f"{tool_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"🧪 SANDBOX TEST: {tool_name}")
    print(f"   Directory: {sandbox_dir}")
    print(f"{'='*60}")

    tester = GTMToolTests(sandbox_dir, verbose=verbose)
    method_name = TOOL_REGISTRY.get(tool_name)

    if not method_name:
        print(f"❌ Unknown tool: {tool_name}")
        print(f"   Available: {', '.join(TOOL_REGISTRY.keys())}")
        return ToolReport(tool_name=tool_name, total_tests=0, passed=0, failed=1,
                          duration_seconds=0)

    start = time.time()
    try:
        test_results = getattr(tester, method_name)()
    except Exception as e:
        print(f"❌ Test suite crashed: {e}")
        traceback.print_exc()
        test_results = [TestResult(tool_name=tool_name, test_name="suite_crash",
                                   passed=False, duration_seconds=0, error=str(e))]

    duration = time.time() - start
    passed = sum(1 for r in test_results if r.passed)
    failed = sum(1 for r in test_results if not r.passed)

    for r in test_results:
        r.tool_name = tool_name

    report = ToolReport(
        tool_name=tool_name, total_tests=len(test_results),
        passed=passed, failed=failed,
        duration_seconds=round(duration, 2),
        results=test_results,
        sandbox_dir=str(sandbox_dir),
    )

    print(f"\n📋 Results:")
    for r in test_results:
        icon = "✅" if r.passed else "❌"
        timing = f"{r.duration_seconds:.2f}s"
        err = f" — {r.error[:60]}" if not r.passed else ""
        print(f"   {icon} {r.test_name:<45} [{timing}]{err}")

    print(f"\n{'─'*60}")
    print(f"{report.status}  {passed}/{len(test_results)} tests passed in {duration:.2f}s")

    if failed > 0:
        print(f"\n🔍 Failure Details:")
        for r in test_results:
            if not r.passed:
                print(f"\n   ❌ {r.test_name}")
                print(f"   Error: {r.error}")
                if verbose and r.tb:
                    print(f"   Traceback:\n{r.tb}")

    report_path = sandbox_dir / "test_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2))
    print(f"\n📄 Full report: {report_path}")

    if not keep_data:
        shutil.rmtree(sandbox_dir)
        print(f"🧹 Sandbox cleaned up")
    else:
        print(f"📁 Sandbox data kept: {sandbox_dir}")

    return report


def run_all_tests(verbose: bool = False, keep_data: bool = False) -> Dict:
    print(f"\n{'='*60}")
    print(f"🧪 GTM TOOL FULL TEST SUITE")
    print(f"   Tools: {len(TOOL_REGISTRY)}")
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    all_reports = {}
    total_start = time.time()

    for tool_name in TOOL_REGISTRY:
        report = run_tool_tests(tool_name, verbose=verbose, keep_data=keep_data)
        all_reports[tool_name] = report

    total_duration = time.time() - total_start
    total_passed = sum(r.passed for r in all_reports.values())
    total_failed = sum(r.failed for r in all_reports.values())
    tools_passing = sum(1 for r in all_reports.values() if r.failed == 0)

    print(f"\n{'='*60}")
    print(f"📊 MASTER SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Tool':<35} {'Status':<15} {'Tests':<10} {'Time'}")
    print(f"{'─'*65}")
    for name, report in all_reports.items():
        print(f"{name:<35} {report.status:<15} {report.passed}/{report.total_tests:<8} {report.duration_seconds:.1f}s")

    print(f"\n{'─'*65}")
    print(f"{'TOTAL':<35} {'':15} {total_passed}/{total_passed+total_failed:<8} {total_duration:.1f}s")
    print(f"\nTools fully passing: {tools_passing}/{len(TOOL_REGISTRY)}")

    return all_reports


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTM tool sandbox tester")
    parser.add_argument("--tool", choices=list(TOOL_REGISTRY.keys()),
                        help="Test a specific tool")
    parser.add_argument("--all", action="store_true", help="Test all tools")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--keep-data", action="store_true",
                        help="Keep sandbox data after test")
    parser.add_argument("--list", action="store_true", help="List available tools")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable GTM tools:")
        for name in TOOL_REGISTRY:
            print(f"  {name}")
    elif args.all:
        run_all_tests(verbose=args.verbose, keep_data=args.keep_data)
    elif args.tool:
        run_tool_tests(args.tool, verbose=args.verbose, keep_data=args.keep_data)
    else:
        parser.print_help()
