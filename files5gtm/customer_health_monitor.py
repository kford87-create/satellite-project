"""
tools/customer_health_monitor.py  —  Tool 7: Customer Health Monitor

Reads Supabase api_requests and api_clients tables to compute per-customer
health scores. Identifies three signal types:
  - UPSELL:     approaching tier search limit (≥80% used this month)
  - CHURN RISK: usage dropped ≥50% week-over-week
  - EXPANSION:  using API patterns that suggest Premium features would help

Outputs a health report and draft outreach emails per signal.
This is the first tool run every morning — protecting revenue beats acquiring it.

Think of it like vital signs monitoring: you want to catch deterioration
early, not after the patient has already left.

Usage:
    python tools/customer_health_monitor.py --dry-run
    python tools/customer_health_monitor.py --output reports/health_2026_02.json
    python tools/customer_health_monitor.py --signal upsell
    python tools/customer_health_monitor.py --signal churn
    python tools/customer_health_monitor.py --signal expansion
"""

import json
import os
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field, asdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Tier Definitions ──────────────────────────────────────────────────────────

TIERS = {
    "starter": {
        "label": "Starter",
        "monthly_price": 99,
        "monthly_searches": 200,
        "next_tier": "professional",
    },
    "professional": {
        "label": "Professional",
        "monthly_price": 399,
        "monthly_searches": 1000,
        "next_tier": "enterprise",
    },
    "enterprise": {
        "label": "Enterprise",
        "monthly_price": 1499,
        "monthly_searches": 5000,
        "next_tier": None,
    },
}

# ── Thresholds ────────────────────────────────────────────────────────────────

UPSELL_THRESHOLD = 0.80         # 80% of monthly searches used → upsell signal
CHURN_DROP_THRESHOLD = 0.50     # 50% week-over-week drop → churn risk
EXPANSION_BATCH_THRESHOLD = 5   # 5+ batch API calls → wants batch tier
CHURN_INACTIVE_DAYS = 14        # No activity for 14 days → churn risk


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class CustomerMetrics:
    client_id: str
    company_name: str
    email: str
    tier: str
    plan_price: int
    monthly_limit: int
    searches_this_month: int
    searches_last_week: int
    searches_week_before: int
    last_active_date: str
    api_call_types: dict           # breakdown of call types e.g. {"detect": 45, "batch": 3}
    days_since_active: int
    usage_pct: float               # searches_this_month / monthly_limit


@dataclass
class HealthSignal:
    client_id: str
    company_name: str
    email: str
    tier: str
    signal_type: str              # upsell / churn / expansion
    severity: str                 # high / medium / low
    summary: str
    data_points: dict
    draft_subject: str
    draft_email: str


@dataclass
class HealthReport:
    run_date: str
    signal_filter: Optional[str]
    total_customers: int
    upsell_opportunities: int
    churn_risks: int
    expansion_signals: int
    estimated_at_risk_mrr: float  # MRR from churn-risk customers
    estimated_expansion_mrr: float  # Potential MRR from upsell customers
    signals: list = field(default_factory=list)
    healthy_customers: int = 0


# ── Email Drafts ──────────────────────────────────────────────────────────────

def draft_upsell_email(metrics: CustomerMetrics) -> tuple:
    """Draft upsell outreach when customer is near their search limit."""
    next_tier = TIERS[metrics.tier].get("next_tier", "enterprise")
    next_tier_info = TIERS.get(next_tier, {})
    next_limit = next_tier_info.get("monthly_searches", 5000)
    next_price = next_tier_info.get("monthly_price", 1499)

    subject = f"You've used {int(metrics.usage_pct*100)}% of your Kestrel AI searches this month"
    body = f"""Hi {metrics.company_name.split()[0] if metrics.company_name else 'there'},

Quick heads up — you've used {metrics.searches_this_month} of your {metrics.monthly_limit} \
monthly searches on the {TIERS[metrics.tier]['label']} plan \
({int(metrics.usage_pct*100)}% of your limit).

If you're running more detections than expected, upgrading to {next_tier_info.get('label', 'Professional')} \
gives you {next_limit:,} searches/month at ${next_price}/month — \
that's ${round(next_price/next_limit*100,1)}¢ per 100 searches.

Happy to walk through what's driving your usage if that would help you decide.

[Your name]
Kestrel AI | kestrelai.com/pricing"""

    return subject, body


def draft_churn_email(metrics: CustomerMetrics) -> tuple:
    """Draft re-engagement email when customer usage has dropped significantly."""
    if metrics.days_since_active > CHURN_INACTIVE_DAYS:
        reason = f"inactive for {metrics.days_since_active} days"
    else:
        drop_pct = round((1 - metrics.searches_last_week /
                          max(metrics.searches_week_before, 1)) * 100)
        reason = f"usage dropped {drop_pct}% last week"

    subject = f"Everything ok with your Kestrel AI account, {metrics.company_name.split()[0] if metrics.company_name else ''}?"
    body = f"""Hi {metrics.company_name.split()[0] if metrics.company_name else 'there'},

I noticed your Kestrel AI usage has been quieter lately ({reason}).

A few things I wanted to check:
- Is everything working as expected with the API?
- Are you hitting any limitations with your current {TIERS[metrics.tier]['label']} plan?
- Did your use case evolve in a direction we could better support?

Happy to jump on a quick call or help via email — no pressure either way.

[Your name]
Kestrel AI | kestrelai.com"""

    return subject, body


def draft_expansion_email(metrics: CustomerMetrics) -> tuple:
    """Draft expansion email when customer usage pattern suggests they'd benefit from upgrade."""
    batch_calls = metrics.api_call_types.get("batch", 0)
    subject = f"Looks like you're running batch jobs — we have a tier for that"
    body = f"""Hi {metrics.company_name.split()[0] if metrics.company_name else 'there'},

I noticed you've run {batch_calls} batch detection jobs this month — \
looks like you're processing larger portfolios.

The Professional plan ($399/month) is built for exactly this: \
1,000 searches/month, priority processing queue, and full batch API access \
without the per-request overhead.

If you're running property portfolios through the API regularly, \
the math usually works out to 60-70% lower cost per search versus Starter.

Worth a quick look at kestrelai.com/pricing — happy to run through the numbers if helpful.

[Your name]
Kestrel AI | kestrelai.com"""

    return subject, body


# ── Signal Detector ───────────────────────────────────────────────────────────

def detect_signals(metrics: CustomerMetrics, filter_type: Optional[str]) -> list:
    """Detect all health signals for a customer."""
    signals = []

    # Upsell signal
    if metrics.usage_pct >= UPSELL_THRESHOLD and metrics.tier != "enterprise":
        if not filter_type or filter_type == "upsell":
            subject, email = draft_upsell_email(metrics)
            severity = "high" if metrics.usage_pct >= 0.95 else "medium"
            signals.append(HealthSignal(
                client_id=metrics.client_id,
                company_name=metrics.company_name,
                email=metrics.email,
                tier=metrics.tier,
                signal_type="upsell",
                severity=severity,
                summary=f"{metrics.company_name} at {int(metrics.usage_pct*100)}% of {metrics.tier} limit",
                data_points={
                    "searches_this_month": metrics.searches_this_month,
                    "monthly_limit": metrics.monthly_limit,
                    "usage_pct": metrics.usage_pct,
                    "next_tier": TIERS[metrics.tier].get("next_tier"),
                },
                draft_subject=subject,
                draft_email=email,
            ))

    # Churn risk signals
    churn_triggered = False
    if metrics.days_since_active >= CHURN_INACTIVE_DAYS:
        churn_triggered = True
    elif (metrics.searches_week_before > 0 and
          metrics.searches_last_week / metrics.searches_week_before < (1 - CHURN_DROP_THRESHOLD)):
        churn_triggered = True

    if churn_triggered:
        if not filter_type or filter_type == "churn":
            subject, email = draft_churn_email(metrics)
            signals.append(HealthSignal(
                client_id=metrics.client_id,
                company_name=metrics.company_name,
                email=metrics.email,
                tier=metrics.tier,
                signal_type="churn",
                severity="high" if metrics.days_since_active >= CHURN_INACTIVE_DAYS else "medium",
                summary=f"{metrics.company_name} usage dropped — {metrics.days_since_active}d inactive",
                data_points={
                    "last_active_date": metrics.last_active_date,
                    "days_since_active": metrics.days_since_active,
                    "searches_last_week": metrics.searches_last_week,
                    "searches_week_before": metrics.searches_week_before,
                },
                draft_subject=subject,
                draft_email=email,
            ))

    # Expansion signal — batch usage on starter plan
    batch_calls = metrics.api_call_types.get("batch", 0)
    if (batch_calls >= EXPANSION_BATCH_THRESHOLD and
            metrics.tier == "starter" and not churn_triggered):
        if not filter_type or filter_type == "expansion":
            subject, email = draft_expansion_email(metrics)
            signals.append(HealthSignal(
                client_id=metrics.client_id,
                company_name=metrics.company_name,
                email=metrics.email,
                tier=metrics.tier,
                signal_type="expansion",
                severity="medium",
                summary=f"{metrics.company_name} running {batch_calls} batch jobs on Starter",
                data_points={
                    "batch_calls": batch_calls,
                    "api_call_types": metrics.api_call_types,
                    "current_tier": metrics.tier,
                    "recommended_tier": "professional",
                },
                draft_subject=subject,
                draft_email=email,
            ))

    return signals


# ── Data Layer ────────────────────────────────────────────────────────────────

class SupabaseHealthReader:
    """Reads customer usage data from Supabase. Falls back gracefully."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._client = None

        if not dry_run:
            try:
                from supabase import create_client
                url = os.getenv("SUPABASE_URL")
                key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                if url and key:
                    self._client = create_client(url, key)
                else:
                    print("⚠️  Supabase credentials not set — using dry-run mode")
                    self.dry_run = True
            except ImportError:
                print("⚠️  supabase-py not installed — using dry-run mode")
                self.dry_run = True

    def get_all_customers(self) -> list:
        """Return list of CustomerMetrics for all active customers."""
        if self.dry_run:
            return self._mock_customers()

        try:
            now = datetime.now()
            month_start = now.replace(day=1, hour=0, minute=0, second=0)
            week_ago = now - timedelta(days=7)
            two_weeks_ago = now - timedelta(days=14)

            clients = self._client.table("api_clients").select("*").execute()
            metrics_list = []

            for client in clients.data:
                client_id = client["id"]
                tier = client.get("tier", "starter")
                limit = TIERS.get(tier, TIERS["starter"])["monthly_searches"]

                # This month's usage
                this_month = self._client.table("api_requests") \
                    .select("id, created_at, endpoint") \
                    .eq("client_id", client_id) \
                    .gte("created_at", month_start.isoformat()) \
                    .execute()

                # Last week
                last_week = self._client.table("api_requests") \
                    .select("id") \
                    .eq("client_id", client_id) \
                    .gte("created_at", week_ago.isoformat()) \
                    .execute()

                # Week before that
                prev_week = self._client.table("api_requests") \
                    .select("id") \
                    .eq("client_id", client_id) \
                    .gte("created_at", two_weeks_ago.isoformat()) \
                    .lt("created_at", week_ago.isoformat()) \
                    .execute()

                # Most recent request
                recent = self._client.table("api_requests") \
                    .select("created_at") \
                    .eq("client_id", client_id) \
                    .order("created_at", desc=True) \
                    .limit(1) \
                    .execute()

                last_active = recent.data[0]["created_at"] if recent.data else None
                days_inactive = (
                    (now - datetime.fromisoformat(last_active.replace("Z", ""))).days
                    if last_active else 999
                )

                # API call type breakdown
                call_types = {}
                for req in this_month.data:
                    ep = req.get("endpoint", "detect")
                    call_types[ep] = call_types.get(ep, 0) + 1

                searches = len(this_month.data)
                metrics_list.append(CustomerMetrics(
                    client_id=client_id,
                    company_name=client.get("company_name", "Unknown"),
                    email=client.get("email", ""),
                    tier=tier,
                    plan_price=TIERS.get(tier, TIERS["starter"])["monthly_price"],
                    monthly_limit=limit,
                    searches_this_month=searches,
                    searches_last_week=len(last_week.data),
                    searches_week_before=len(prev_week.data),
                    last_active_date=last_active or "never",
                    api_call_types=call_types,
                    days_since_active=days_inactive,
                    usage_pct=round(searches / max(limit, 1), 3),
                ))

            return metrics_list

        except Exception as e:
            print(f"⚠️  Supabase read failed: {e} — using mock data")
            return self._mock_customers()

    def _mock_customers(self) -> list:
        """Generate synthetic customer data for testing."""
        import random
        random.seed(99)

        mock = [
            # Upsell candidate — near limit
            CustomerMetrics(
                client_id="cli_001", company_name="Acme Insurtech",
                email="data@acme-insurtech.com", tier="starter",
                plan_price=99, monthly_limit=200,
                searches_this_month=178, searches_last_week=45,
                searches_week_before=38, last_active_date=datetime.now().strftime("%Y-%m-%d"),
                api_call_types={"detect": 165, "batch": 13},
                days_since_active=0, usage_pct=0.89,
            ),
            # Healthy customer
            CustomerMetrics(
                client_id="cli_002", company_name="Regional P&C Group",
                email="tech@regionalpandc.com", tier="professional",
                plan_price=399, monthly_limit=1000,
                searches_this_month=412, searches_last_week=98,
                searches_week_before=87, last_active_date=datetime.now().strftime("%Y-%m-%d"),
                api_call_types={"detect": 400, "batch": 12},
                days_since_active=1, usage_pct=0.41,
            ),
            # Churn risk — dropped off
            CustomerMetrics(
                client_id="cli_003", company_name="StartupGIS LLC",
                email="founder@startupgis.io", tier="starter",
                plan_price=99, monthly_limit=200,
                searches_this_month=8, searches_last_week=2,
                searches_week_before=44, last_active_date=(datetime.now()-timedelta(days=18)).strftime("%Y-%m-%d"),
                api_call_types={"detect": 8},
                days_since_active=18, usage_pct=0.04,
            ),
            # Expansion — heavy batch user on starter
            CustomerMetrics(
                client_id="cli_004", company_name="MapTech Solutions",
                email="api@maptech.co", tier="starter",
                plan_price=99, monthly_limit=200,
                searches_this_month=167, searches_last_week=41,
                searches_week_before=39, last_active_date=datetime.now().strftime("%Y-%m-%d"),
                api_call_types={"detect": 145, "batch": 22},
                days_since_active=0, usage_pct=0.84,
            ),
            # Healthy — enterprise
            CustomerMetrics(
                client_id="cli_005", company_name="Defense Analytics Inc",
                email="contracts@defense-analytics.com", tier="enterprise",
                plan_price=1499, monthly_limit=5000,
                searches_this_month=2100, searches_last_week=520,
                searches_week_before=498, last_active_date=datetime.now().strftime("%Y-%m-%d"),
                api_call_types={"detect": 1800, "batch": 300},
                days_since_active=0, usage_pct=0.42,
            ),
        ]
        return mock


# ── Report Builder ────────────────────────────────────────────────────────────

def build_report(all_signals: list, all_metrics: list, filter_type: Optional[str]) -> HealthReport:
    upsell = [s for s in all_signals if s.signal_type == "upsell"]
    churn = [s for s in all_signals if s.signal_type == "churn"]
    expansion = [s for s in all_signals if s.signal_type == "expansion"]

    at_risk_mrr = sum(
        TIERS.get(m.tier, {}).get("monthly_price", 0)
        for m in all_metrics
        if any(s.client_id == m.client_id and s.signal_type == "churn" for s in all_signals)
    )
    expansion_mrr = sum(
        TIERS.get(TIERS.get(m.tier, {}).get("next_tier", ""), {}).get("monthly_price", 0) -
        TIERS.get(m.tier, {}).get("monthly_price", 0)
        for m in all_metrics
        if any(s.client_id == m.client_id and s.signal_type == "upsell" for s in all_signals)
    )

    healthy = len(all_metrics) - len({s.client_id for s in all_signals})

    return HealthReport(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        signal_filter=filter_type,
        total_customers=len(all_metrics),
        upsell_opportunities=len(upsell),
        churn_risks=len(churn),
        expansion_signals=len(expansion),
        estimated_at_risk_mrr=float(at_risk_mrr),
        estimated_expansion_mrr=float(expansion_mrr),
        signals=[asdict(s) for s in sorted(all_signals, key=lambda x: (
            {"high": 0, "medium": 1, "low": 2}.get(x.severity, 3)))],
        healthy_customers=healthy,
    )


def print_report(report: HealthReport):
    print(f"\n{'='*60}")
    print(f"💊 CUSTOMER HEALTH REPORT")
    print(f"   {report.run_date}")
    print(f"{'='*60}")
    print(f"\n   Total customers:       {report.total_customers}")
    print(f"   ✅ Healthy:            {report.healthy_customers}")
    print(f"   📈 Upsell opportunity: {report.upsell_opportunities}")
    print(f"   📉 Churn risks:        {report.churn_risks}")
    print(f"   🔧 Expansion signals:  {report.expansion_signals}")
    print(f"\n   💰 Est. at-risk MRR:   ${report.estimated_at_risk_mrr:,.0f}/mo")
    print(f"   💰 Est. upsell MRR:    +${report.estimated_expansion_mrr:,.0f}/mo if converted")

    if report.signals:
        print(f"\n🎯 Action Items (sorted by severity):")
        type_icons = {"upsell": "📈", "churn": "📉", "expansion": "🔧"}
        sev_icons = {"high": "🔴", "medium": "🟡", "low": "⚪"}
        for s in report.signals:
            icon = type_icons.get(s["signal_type"], "•")
            sev = sev_icons.get(s["severity"], "⚪")
            print(f"\n   {icon} {sev} [{s['signal_type'].upper()}] {s['company_name']} ({s['tier']})")
            print(f"      {s['summary']}")
            print(f"      Subject: {s['draft_subject']}")
    else:
        print("\n✅ No signals detected — all customers look healthy!")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Monitor customer health and surface churn/upsell signals"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock customer data (no Supabase connection)")
    parser.add_argument("--signal", choices=["upsell", "churn", "expansion"],
                        help="Filter to a specific signal type")
    parser.add_argument("--output", type=str,
                        help="Save report to this JSON path")
    args = parser.parse_args()

    print(f"🛰️  Kestrel AI — Customer Health Monitor")
    print(f"   Signal filter: {args.signal or 'all'}")
    print(f"   Mode: {'🧪 DRY RUN' if args.dry_run else '🔴 LIVE (Supabase)'}")

    reader = SupabaseHealthReader(dry_run=args.dry_run)
    print("\n   Loading customer data...")
    all_metrics = reader.get_all_customers()
    print(f"   Loaded {len(all_metrics)} customers")

    all_signals = []
    for m in all_metrics:
        signals = detect_signals(m, args.signal)
        all_signals.extend(signals)

    report = build_report(all_signals, all_metrics, args.signal)
    print_report(report)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2))
        print(f"\n✅ Report saved: {args.output}")


if __name__ == "__main__":
    main()
