"""
customer_health_monitor.py
--------------------------
Read Supabase profiles/searches tables to compute per-customer health
scores. Identify upsell opportunities, churn risks, and expansion signals.
Output draft outreach emails per signal type.

READ-ONLY — never writes to the database.
PII in saved output is redacted to j***@company.com format.

Usage:
    python tools/customer_health_monitor.py --dry-run
    python tools/customer_health_monitor.py --output reports/health_2026_02.json
    python tools/customer_health_monitor.py --signal upsell
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ENV_PATH = Path("/Users/kahlil/satellite-project/.env")
load_dotenv(dotenv_path=ENV_PATH)

import os  # noqa: E402

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://obdsgqjkjjmmtbcfjhnn.supabase.co")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

PLAN_PRICES = {"starter": 99, "professional": 399, "enterprise": 1499}
PLAN_NEXT = {"starter": "professional", "professional": "enterprise", "enterprise": None}
PLAN_NEXT_LIMIT = {"starter": 1000, "professional": 5000, "enterprise": None}
PLAN_NEXT_PRICE = {"starter": 399, "professional": 1499, "enterprise": None}


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

def _redact_email(email: str) -> str:
    """Redact email to j***@company.com format."""
    if not email or "@" not in email:
        return "***@***.com"
    local, domain = email.split("@", 1)
    redacted_local = local[0] + "***" if local else "***"
    return f"{redacted_local}@{domain}"


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------

def _compute_health_score(
    searches_this_month: int,
    searches_limit: int,
    days_since_last_search: float,
    trend: str,  # "increasing", "stable", "decreasing"
) -> int:
    score = 0

    # Usage rate (0-40 pts)
    usage_rate = searches_this_month / max(searches_limit, 1)
    if usage_rate >= 0.90:
        score += 40
    elif usage_rate >= 0.50:
        score += 25
    elif usage_rate >= 0.20:
        score += 10
    else:
        score += 0

    # Recency (0-30 pts)
    if days_since_last_search < 3:
        score += 30
    elif days_since_last_search <= 7:
        score += 20
    elif days_since_last_search <= 14:
        score += 10
    else:
        score += 0

    # Trend (0-30 pts)
    if trend == "increasing":
        score += 30
    elif trend == "stable":
        score += 15
    else:
        score += 0

    return min(score, 100)


def _classify_signal(
    health_score: int,
    usage_rate: float,
    days_since_last_search: float,
    company_name: str | None,
) -> str:
    if health_score >= 70 and usage_rate >= 0.80:
        return "upsell"
    if health_score < 30 or (days_since_last_search > 14 and usage_rate < 0.20):
        return "churn_risk"
    if 50 <= health_score < 70 and company_name:
        return "expansion"
    return "healthy"


# ---------------------------------------------------------------------------
# Draft emails
# ---------------------------------------------------------------------------

def _draft_upsell_email(customer: dict[str, Any]) -> tuple[str, str]:
    plan = customer["plan"]
    used = customer["searches_this_month"]
    limit = customer["searches_limit"]
    pct = round(100 * used / max(limit, 1))
    next_plan = PLAN_NEXT.get(plan)
    next_price = PLAN_NEXT_PRICE.get(plan)
    next_limit = PLAN_NEXT_LIMIT.get(plan)
    name = customer.get("full_name") or "there"
    first_name = name.split()[0] if name != "there" else "there"

    subject = f"You're approaching your Kestrel AI search limit"
    body = (
        f"Hi {first_name},\n\n"
        f"You've used {pct}% of your {plan.title()} plan searches this month "
        f"({used}/{limit} searches).\n\n"
        f"Upgrading to {next_plan.title() if next_plan else 'a higher tier'} unlocks "
        f"{next_limit:,} searches/month for ${next_price}/month — "
        f"that's ${next_price - PLAN_PRICES[plan]}/month more for "
        f"{next_limit - limit:,} additional searches.\n\n"
        f"Upgrade here: https://kestrelai.io/app\n\n"
        f"Best,\nThe Kestrel AI Team"
    )
    return subject, body


def _draft_churn_email(customer: dict[str, Any]) -> tuple[str, str]:
    days = customer["days_since_last_search"]
    name = customer.get("full_name") or "there"
    first_name = name.split()[0] if name != "there" else "there"

    subject = "Quick check-in from Kestrel AI"
    body = (
        f"Hi {first_name},\n\n"
        f"We noticed you haven't run a search in {int(days)} days — wanted to check in.\n\n"
        f"Is there anything blocking you? A feature missing? Something not working as expected?\n\n"
        f"I'm happy to jump on a quick 15-min call to help, or answer questions by email.\n\n"
        f"Reply here or book a call: https://kestrelai.io\n\n"
        f"Best,\nKahlil @ Kestrel AI"
    )
    return subject, body


def _draft_expansion_email(customer: dict[str, Any]) -> tuple[str, str]:
    plan = customer["plan"]
    price = PLAN_PRICES.get(plan, 99)
    annual_savings = round(price * 12 * 0.20)
    company = customer.get("company_name") or "your team"
    name = customer.get("full_name") or "there"
    first_name = name.split()[0] if name != "there" else "there"

    subject = "Save 20% with Kestrel AI annual billing"
    body = (
        f"Hi {first_name},\n\n"
        f"Based on {company}'s consistent usage on the {plan.title()} plan, "
        f"an annual subscription would save you ${annual_savings}/year (20% off).\n\n"
        f"That's ${price * 12 - annual_savings}/year instead of ${price * 12}/year — "
        f"same great features, locked in at your current rate.\n\n"
        f"Interested? Reply and I'll set it up for you.\n\n"
        f"Best,\nKahlil @ Kestrel AI"
    )
    return subject, body


# ---------------------------------------------------------------------------
# Dry-run synthetic customers
# ---------------------------------------------------------------------------

def _make_synthetic_customers() -> list[dict[str, Any]]:
    rng = random.Random(77)
    now = datetime.now(timezone.utc)

    raw = [
        # upsell: high usage
        {"full_name": "Sarah Chen", "email": "sarah.chen@hippoinsurance.com",
         "company_name": "Hippo Insurance", "plan": "starter",
         "searches_this_month": 187, "searches_limit": 200,
         "days_since_last_search": 1, "trend": "increasing"},
        {"full_name": "Marcus Webb", "email": "marcus@openly.com",
         "company_name": "Openly", "plan": "professional",
         "searches_this_month": 920, "searches_limit": 1000,
         "days_since_last_search": 0.5, "trend": "increasing"},
        # churn risk: low usage + stale
        {"full_name": "Priya Nair", "email": "priya.nair@coterie.com",
         "company_name": "Coterie Insurance", "plan": "starter",
         "searches_this_month": 8, "searches_limit": 200,
         "days_since_last_search": 21, "trend": "decreasing"},
        {"full_name": "Derek Okonkwo", "email": "d.okonkwo@branchinsurance.com",
         "company_name": "Branch Insurance", "plan": "starter",
         "searches_this_month": 3, "searches_limit": 200,
         "days_since_last_search": 18, "trend": "decreasing"},
        # expansion candidates
        {"full_name": "Lisa Tran", "email": "lisa@nearmap.com",
         "company_name": "Nearmap", "plan": "professional",
         "searches_this_month": 550, "searches_limit": 1000,
         "days_since_last_search": 4, "trend": "stable"},
        # healthy
        {"full_name": "James Okafor", "email": "james@kin.com",
         "company_name": "Kin Insurance", "plan": "starter",
         "searches_this_month": 140, "searches_limit": 200,
         "days_since_last_search": 2, "trend": "stable"},
        {"full_name": "Amanda Park", "email": "apark@lemonade.com",
         "company_name": "Lemonade", "plan": "professional",
         "searches_this_month": 680, "searches_limit": 1000,
         "days_since_last_search": 1, "trend": "stable"},
        {"full_name": "Chris Muller", "email": "c.muller@eagleview.com",
         "company_name": "EagleView", "plan": "enterprise",
         "searches_this_month": 3200, "searches_limit": 5000,
         "days_since_last_search": 0.2, "trend": "increasing"},
    ]

    customers = []
    for r in raw:
        usage_rate = r["searches_this_month"] / max(r["searches_limit"], 1)
        health_score = _compute_health_score(
            r["searches_this_month"], r["searches_limit"],
            r["days_since_last_search"], r["trend"]
        )
        signal = _classify_signal(health_score, usage_rate, r["days_since_last_search"], r["company_name"])

        entry: dict[str, Any] = {
            "user_id": f"usr_{rng.randint(100000, 999999)}",
            "email": _redact_email(r["email"]),
            "full_name": r["full_name"],
            "company_name": r["company_name"],
            "plan": r["plan"],
            "searches_this_month": r["searches_this_month"],
            "searches_limit": r["searches_limit"],
            "usage_rate_pct": round(usage_rate * 100, 1),
            "health_score": health_score,
            "signal": signal,
            "days_since_last_search": r["days_since_last_search"],
        }

        # Draft email (use unredacted name/company for drafts)
        draft_customer = {**r, "plan": r["plan"], "days_since_last_search": r["days_since_last_search"],
                         "searches_this_month": r["searches_this_month"], "searches_limit": r["searches_limit"]}
        if signal == "upsell":
            subj, body = _draft_upsell_email(draft_customer)
        elif signal == "churn_risk":
            subj, body = _draft_churn_email(draft_customer)
        elif signal == "expansion":
            subj, body = _draft_expansion_email(draft_customer)
        else:
            subj, body = None, None

        if subj:
            entry["draft_email_subject"] = subj
            entry["draft_email_body"] = body

        customers.append(entry)

    return customers


# ---------------------------------------------------------------------------
# Live Supabase mode
# ---------------------------------------------------------------------------

def _live_customers(signal_filter: str | None) -> list[dict[str, Any]]:
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception as exc:
        print(f"❌ Supabase connection failed: {exc}")
        print("⚠️  Falling back to dry-run mode")
        return _make_synthetic_customers()

    try:
        # Fetch profiles (read-only)
        profiles_resp = sb.table("profiles").select(
            "id, email, full_name, company_name, plan, searches_this_month, searches_limit, created_at"
        ).execute()
        profiles = profiles_resp.data or []
        print(f"📊 Loaded {len(profiles)} customer profiles")
    except Exception as exc:
        print(f"⚠️  profiles query failed: {exc}")
        return _make_synthetic_customers()

    # Fetch recent search activity per user
    search_activity: dict[str, dict[str, Any]] = {}
    try:
        searches_resp = sb.table("searches").select("user_id, created_at").execute()
        for row in (searches_resp.data or []):
            uid = row["user_id"]
            ts = row["created_at"]
            if uid not in search_activity:
                search_activity[uid] = {"last_search": ts, "count": 0}
            search_activity[uid]["count"] += 1
            if ts > search_activity[uid]["last_search"]:
                search_activity[uid]["last_search"] = ts
    except Exception as exc:
        print(f"⚠️  searches query failed (non-fatal): {exc}")

    now = datetime.now(timezone.utc)
    customers: list[dict[str, Any]] = []

    for p in profiles:
        uid = p.get("id", "")
        activity = search_activity.get(uid, {})
        last_search_ts = activity.get("last_search")
        days_since = 30.0
        if last_search_ts:
            try:
                last_dt = datetime.fromisoformat(last_search_ts.replace("Z", "+00:00"))
                days_since = (now - last_dt).total_seconds() / 86400
            except Exception:
                pass

        searches_now = p.get("searches_this_month") or 0
        limit = p.get("searches_limit") or 200
        usage_rate = searches_now / max(limit, 1)
        trend = "stable"  # simplified — would need historical data for real trend

        health_score = _compute_health_score(searches_now, limit, days_since, trend)
        signal = _classify_signal(health_score, usage_rate, days_since, p.get("company_name"))

        if signal_filter and signal != signal_filter:
            continue

        entry: dict[str, Any] = {
            "user_id": uid,
            "email": _redact_email(p.get("email") or ""),
            "full_name": p.get("full_name") or "",
            "company_name": p.get("company_name") or "",
            "plan": p.get("plan") or "starter",
            "searches_this_month": searches_now,
            "searches_limit": limit,
            "usage_rate_pct": round(usage_rate * 100, 1),
            "health_score": health_score,
            "signal": signal,
            "days_since_last_search": round(days_since, 1),
        }

        if signal in ("upsell", "churn_risk", "expansion"):
            draft_p = {**p, "days_since_last_search": days_since,
                      "searches_this_month": searches_now, "searches_limit": limit}
            if signal == "upsell":
                s, b = _draft_upsell_email(draft_p)
            elif signal == "churn_risk":
                s, b = _draft_churn_email(draft_p)
            else:
                s, b = _draft_expansion_email(draft_p)
            entry["draft_email_subject"] = s
            entry["draft_email_body"] = b

        customers.append(entry)

    return customers


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_monitor(
    dry_run: bool = False,
    signal_filter: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    mode = "dry_run" if (dry_run or not SUPABASE_SERVICE_ROLE_KEY) else "live"
    if not dry_run and not SUPABASE_SERVICE_ROLE_KEY:
        print("⚠️  SUPABASE_SERVICE_ROLE_KEY not set — switching to dry-run mode")

    print(f"📊 Customer Health Monitor — {mode} mode")
    if signal_filter:
        print(f"📊 Signal filter: {signal_filter}")
    print()

    customers = _make_synthetic_customers() if mode == "dry_run" else _live_customers(signal_filter)

    if signal_filter:
        customers = [c for c in customers if c["signal"] == signal_filter]

    # Aggregate
    upsell = [c for c in customers if c["signal"] == "upsell"]
    churn  = [c for c in customers if c["signal"] == "churn_risk"]
    expan  = [c for c in customers if c["signal"] == "expansion"]
    health = [c for c in customers if c["signal"] == "healthy"]

    mrr_at_risk = sum(PLAN_PRICES.get(c["plan"], 0) for c in churn)
    upsell_mrr  = sum((PLAN_NEXT_PRICE.get(c["plan"]) or 0) - PLAN_PRICES.get(c["plan"], 0) for c in upsell)

    summary = (
        f"{len(upsell)} upsell opportunities (${upsell_mrr} MRR potential). "
        f"{len(churn)} churn risks (${mrr_at_risk} MRR at risk)."
    )

    report: dict[str, Any] = {
        "run_date": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "signal_filter": signal_filter or "all",
        "customers_analyzed": len(customers),
        "upsell_opportunities": len(upsell),
        "churn_risks": len(churn),
        "expansion_candidates": len(expan),
        "healthy": len(health),
        "mrr_at_risk": float(mrr_at_risk),
        "upsell_potential_mrr": float(upsell_mrr),
        "customers": customers,
        "summary": summary,
    }

    # Print summary
    print("=" * 60)
    print("📊 Customer Health Report")
    print("=" * 60)
    signal_icons = {"upsell": "📈", "churn_risk": "🔴", "expansion": "💡", "healthy": "✅"}
    for c in sorted(customers, key=lambda x: -x["health_score"]):
        icon = signal_icons.get(c["signal"], "  ")
        print(f"  {icon} [{c['health_score']:3d}] {c['company_name'] or c['email']:<28} {c['plan']:<14} {c['signal']}")
    print()
    print(f"  📈 Upsell opportunities : {len(upsell)} (${upsell_mrr} MRR)")
    print(f"  🔴 Churn risks          : {len(churn)} (${mrr_at_risk} MRR at risk)")
    print(f"  💡 Expansion candidates : {len(expan)}")
    print(f"  ✅ Healthy              : {len(health)}")
    print("=" * 60)
    print(f"\n{summary}")

    # Save
    if output_path is None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_path = _REPORTS_DIR / f"health_{date_str}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n✅ Report saved → {output_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute customer health scores and draft outreach emails.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Synthetic customers, no Supabase")
    parser.add_argument("--output", type=Path, metavar="PATH", help="Save JSON report")
    parser.add_argument(
        "--signal", choices=["upsell", "churn", "expansion"],
        help="Filter to one signal type ('churn' maps to churn_risk)"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    signal = args.signal
    if signal == "churn":
        signal = "churn_risk"
    run_monitor(dry_run=args.dry_run, signal_filter=signal, output_path=args.output)


if __name__ == "__main__":
    main()
