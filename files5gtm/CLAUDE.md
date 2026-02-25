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

The sandbox tester runs every tool against mock data without touching
any external API (Anthropic, Reddit, Supabase) unless you explicitly set
`--live` mode. Default is always offline/synthetic.

```bash
# Test one tool
python sandbox/gtm_tool_tester.py --tool llm_citation_monitor

# Test all tools
python sandbox/gtm_tool_tester.py --all

# Verbose (shows full output)
python sandbox/gtm_tool_tester.py --all --verbose

# Keep sandbox files for inspection
python sandbox/gtm_tool_tester.py --tool competitor_tracker --keep-data

# List available tools
python sandbox/gtm_tool_tester.py --list
```

---

## Tool Quick Reference

### Tool 1 — LLM Citation Monitor

Runs a standardized query battery across Claude, ChatGPT, Gemini, and
Perplexity. Tracks whether Kestrel AI is mentioned, at what rank, and
with what sentiment. This is the primary feedback loop for the AEO strategy.

```bash
# Dry run (no real API calls, uses mock responses)
python tools/llm_citation_monitor.py --dry-run

# Live run (requires ANTHROPIC_API_KEY)
python tools/llm_citation_monitor.py

# Custom query file
python tools/llm_citation_monitor.py --queries my_queries.json

# Export results
python tools/llm_citation_monitor.py --output reports/citations_2026_02.json
```

- **Frequency:** Weekly (cron Monday 6am)
- **Output:** `reports/citation_report_YYYYMMDD.json` + console summary
- **Key metric:** Citation rate % per platform, average rank when cited
- **Requires:** `ANTHROPIC_API_KEY` in .env (other platforms via web scraping)

---

### Tool 2 — Content Brief Generator

Takes a target query, scrapes top search results, identifies factual gaps
competitors miss, and produces a structured AEO content brief in BLUF format
(Bottom Line Up Front — maximizes LLM citation probability by 3.2×).

```bash
# Generate brief for a specific query
python tools/content_brief_generator.py \
  --query "affordable satellite object detection for small business"

# Generate briefs for all queries in a file
python tools/content_brief_generator.py --query-file queries.txt

# Save brief to file
python tools/content_brief_generator.py \
  --query "satellite imagery API for insurance" \
  --output briefs/insurance_brief.md
```

- **Frequency:** On-demand, before writing each blog post
- **Output:** Markdown content brief with title, BLUF answer, outline, citable facts
- **Key metric:** Estimated LLM citation score (based on structure analysis)
- **Requires:** `ANTHROPIC_API_KEY`, internet access for scraping

---

### Tool 3 — Prospect Signal Detector

Scans public web sources for buying-intent signals from ICP companies.
Looks for: job postings (geospatial analyst, insurtech engineer), funding
announcements, product launches in adjacent space, and conference registrations.
Outputs draft LinkedIn outreach messages with the signal as the hook.

```bash
# Run signal scan (dry-run by default, no external calls)
python tools/prospect_signal_detector.py --dry-run

# Live scan with output
python tools/prospect_signal_detector.py \
  --output signals/signals_2026_02.json

# Filter by ICP segment
python tools/prospect_signal_detector.py --segment insurance
python tools/prospect_signal_detector.py --segment insurtech
python tools/prospect_signal_detector.py --segment defense
```

- **Frequency:** Daily (cron 7am)
- **Output:** JSON signal report + draft outreach messages per signal
- **Key metric:** Signals detected, estimated intent score per company
- **Requires:** Internet access (scrapes public job boards, news, LinkedIn public)

---

### Tool 5 — Competitor Tracker

Monitors Picterra, FlyPix AI, EOSDA, and Geospatial Insight for pricing
changes, new feature announcements, and press releases. Keeps comparison
pages accurate (stale data kills LLM credibility). Alerts when action needed.

```bash
# Run competitive snapshot
python tools/competitor_tracker.py

# Compare to last snapshot (shows diffs)
python tools/competitor_tracker.py --diff

# Export for comparison page update
python tools/competitor_tracker.py --output reports/competitor_snapshot.json

# Watch specific competitor
python tools/competitor_tracker.py --competitor picterra
```

- **Frequency:** Weekly (cron Sunday 8pm)
- **Output:** `reports/competitor_snapshot_YYYYMMDD.json` + diff alerts
- **Key metric:** Pricing drift, new features, partnership announcements
- **Requires:** Internet access (public page scraping only)

---

### Tool 6 — Community Scanner

Watches Reddit (r/gis, r/remotesensing, r/MachineLearning, r/insurtech),
Hacker News, and dev.to for questions about satellite detection, geospatial AI,
and insurance imagery. Surfaces threads where a genuinely helpful answer would
also create natural brand awareness. Drafts response starters.

```bash
# Scan all communities (dry-run)
python tools/community_scanner.py --dry-run

# Live scan
python tools/community_scanner.py --output signals/community_2026_02.json

# Scan specific platform
python tools/community_scanner.py --platform reddit
python tools/community_scanner.py --platform hackernews

# Filter by relevance score threshold
python tools/community_scanner.py --min-score 0.7
```

- **Frequency:** Daily (cron 8am)
- **Output:** Ranked list of threads with relevance score + draft response starters
- **Key metric:** Threads found, avg relevance score, engagement opportunity score
- **Requires:** Internet access (Reddit public API, HN Algolia API — no auth needed)

---

### Tool 7 — Customer Health Monitor

Reads Supabase `api_requests` and `api_clients` tables to compute per-customer
health scores. Identifies: approaching tier limit (upsell), usage drop (churn
risk), feature gaps (expansion). Outputs draft outreach emails per signal type.

```bash
# Run health check (requires Supabase)
python tools/customer_health_monitor.py

# Dry-run with synthetic customers
python tools/customer_health_monitor.py --dry-run

# Export signals
python tools/customer_health_monitor.py --output reports/health_2026_02.json

# Focus on specific signal type
python tools/customer_health_monitor.py --signal upsell
python tools/customer_health_monitor.py --signal churn
python tools/customer_health_monitor.py --signal expansion
```

- **Frequency:** Daily (cron 6am — first thing every morning)
- **Output:** JSON health report + draft emails per at-risk/opportunity customer
- **Key metric:** Customers at risk, upsell opportunities, NRR forecast
- **Requires:** `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` in .env

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

## Supabase Tables Used (Tool 7)

These tables come from the main satellite_bootstrap pipeline:

| Table | Used For |
|-------|----------|
| `api_clients` | Customer accounts, tier, quota |
| `api_requests` | Per-request usage logs |

Customer health reads are **read-only** — this tool never writes to the DB.

---

## Environment Variables

```bash
# Required for Tool 1 (LLM Citation Monitor)
ANTHROPIC_API_KEY=sk-ant-...

# Required for Tool 2 (Content Brief Generator)
ANTHROPIC_API_KEY=sk-ant-...          # Same key

# Required for Tool 7 (Customer Health Monitor)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...      # Service role (read-only operations)

# Optional — used if set, gracefully skipped if not
REDDIT_CLIENT_ID=...                  # Tool 6 — increases Reddit rate limits
REDDIT_CLIENT_SECRET=...              # Tool 6 — increases Reddit rate limits
```

Copy `.env.example` to `.env`. Never commit `.env`.

---

## Coding Conventions (match main pipeline)

- All tools are standalone: `python tools/tool_name.py`
- Every tool has `--dry-run` mode that generates synthetic data — no external calls
- Every tool has `--output path` to save results to JSON or Markdown
- Use `pathlib.Path` not `os.path`
- Print ✅ success, ❌ error, ⚠️ warning, 📊 stats
- External API calls are always wrapped in try/except with graceful fallback
- Rate limit all scrapers: minimum 1.5s between requests
- Never commit API keys or credentials

---

## Never Do

- ❌ Run any tool without `--dry-run` first on a new machine
- ❌ Post community responses directly — always review drafts before posting
- ❌ Send outreach emails directly — tool drafts only, human sends
- ❌ Run competitor_tracker more than once per day (respect robots.txt)
- ❌ Hard-code competitor URLs — keep them in `COMPETITOR_URLS` dict inside tool
- ❌ Use customer data from Tool 7 for anything other than internal alerting
- ❌ Store customer PII in any output files

---

## Adding New Tools

Follow this checklist:
1. Create `tools/my_new_tool.py` with `--dry-run` and `--output` flags
2. Add test method `test_my_new_tool()` to `sandbox/gtm_tool_tester.py`
3. Add entry to `TOOL_REGISTRY` in tester
4. Add entry to this CLAUDE.md Tool Quick Reference section
5. Run `python sandbox/gtm_tool_tester.py --tool my_new_tool` — must pass before committing
6. Add cron schedule recommendation above
