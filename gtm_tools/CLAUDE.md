# Kestrel AI — GTM Agent Tools
# CLAUDE.md for Visual Studio Code + Claude Code

## What This Directory Is

7 autonomous agent tools that execute the Kestrel AI go-to-market strategy.
These tools are the operational layer between the GTM strategy document and
real-world traction — they monitor, research, alert, and draft so the founder
can focus on building and closing rather than watching and searching.

**Strategy source:** kestrel_ai_gtm_strategy.docx
**Pricing:** Starter $99/mo (200 searches) · Professional $399/mo (1,000 searches) · Enterprise $1,499/mo (5,000 searches)
**Beachhead:** Mid-market insurance & insurtech startups priced out of Maxar/Planet Labs
**Government path:** Dual-use, SBIR-eligible bootstrapping pipeline

---

## Tool Index

| # | Tool | File | Purpose |
|---|------|------|---------|
| 1 | LLM Citation Monitor | `tools/llm_citation_monitor.py` | Track when/how LLMs recommend Kestrel AI |
| 2 | Content Brief Generator | `tools/content_brief_generator.py` | Research-backed AEO content briefs |
| 3 | Prospect Signal Detector | `tools/prospect_signal_detector.py` | ICP buying-intent signals from LinkedIn/web |
| 5 | Competitor Price & Feature Tracker | `tools/competitor_tracker.py` | Watch Picterra, FlyPix, EOSDA, Geospatial Insight |
| 6 | Community Mention Scanner | `tools/community_scanner.py` | Surface Reddit/HN threads to answer |
| 7 | Customer Health Monitor | `tools/customer_health_monitor.py` | Supabase usage → churn/upsell signals |

**Tool 4 (SAM.gov / SBIR Watcher)** is tracked separately — see government sales playbook.

---

## Directory Structure

```
gtm_tools/
├── CLAUDE.md                          ← You are here
├── .env.example                       ← Copy to .env, never commit .env
├── requirements.txt
│
├── tools/
│   ├── llm_citation_monitor.py        ← Tool 1: LLM visibility tracking
│   ├── content_brief_generator.py     ← Tool 2: AEO content research
│   ├── prospect_signal_detector.py    ← Tool 3: Buying-intent detection
│   ├── competitor_tracker.py          ← Tool 5: Competitor surveillance
│   ├── community_scanner.py           ← Tool 6: Community opportunity finder
│   └── customer_health_monitor.py     ← Tool 7: Usage-based churn/upsell alerts
│
└── sandbox/
    └── gtm_tool_tester.py             ← TEST ALL TOOLS BEFORE PRODUCTION USE
```

---

## CRITICAL: Always Test Before Running

```bash
python sandbox/gtm_tool_tester.py --tool llm_citation_monitor
python sandbox/gtm_tool_tester.py --all
python sandbox/gtm_tool_tester.py --all --verbose
python sandbox/gtm_tool_tester.py --tool competitor_tracker --keep-data
python sandbox/gtm_tool_tester.py --list
```

---

## Recommended Run Order (Daily)

```
6:00am  → customer_health_monitor.py     # Protect revenue first
7:00am  → prospect_signal_detector.py    # Build pipeline
8:00am  → community_scanner.py           # Brand/community
```

## Recommended Run Order (Weekly)

```
Monday 6am   → llm_citation_monitor.py   # AEO feedback loop
Sunday 8pm   → competitor_tracker.py     # Competitive intel
On-demand    → content_brief_generator.py # Before each blog post
```

---

## Environment Variables

```bash
# Required for Tool 1 & 2
ANTHROPIC_API_KEY=sk-ant-...

# Required for Tool 7
SUPABASE_URL=https://obdsgqjkjjmmtbcfjhnn.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Optional — increases Reddit rate limits (Tool 6)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
```

Copy `.env.example` to `.env`. Never commit `.env`.

---

## Never Do

- ❌ Run any tool without `--dry-run` first on a new machine
- ❌ Post community responses directly — always review drafts before posting
- ❌ Send outreach emails directly — tool drafts only, human sends
- ❌ Run competitor_tracker more than once per day (respect robots.txt)
- ❌ Hard-code competitor URLs — keep them in `COMPETITOR_URLS` dict inside tool
- ❌ Use customer data from Tool 7 for anything other than internal alerting
- ❌ Store customer PII in any output files
