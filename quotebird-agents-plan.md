# Quotebird · Agent Strategy

*A working plan for which specialist agents to build into your Claude Code setup (build-time) and into the Quotebird product (run-time). Read it end-to-end in 8–10 minutes. Skip to "Build This First" for the punch list.*

---

## How I applied the kill-test

For every agent I considered, I asked four questions:

1. Does this agent have a different body of knowledge or evaluation lens than the main thread? If not, kill it.
2. Does separating it reduce a real cost — context bloat, model spend, evaluation surface, error rate? If not, kill it.
3. Will I actually invoke it weekly, OR does it prevent a recurring class of mistake? If not, kill it.
4. Is its scope narrow enough that the system prompt fits in 6 sentences without hand-waving? If not, redesign it.

If an agent doesn't appear below, it didn't earn its keep. See **"What I Killed and Why"** at the bottom.

---

## SECTION A · Build-Time Agents

**Seven agents.** Each lives at `.claude/agents/<name>.md` in this repo (project scope, not global) — their context is Quotebird-specific.

---

### A1. `supabase-schema-auditor`

**Description** · Audits SQL migrations and RLS policies for multi-tenant safety. Read-only.  
**Tools** · `Read`, `Bash` (restricted to `psql` / `supabase` query inspection)  
**Model** · Sonnet 4

**Triggers:** Every time you write a migration, change an RLS policy, or add a new table that touches tenant data. Run before merging any branch with `supabase/migrations/*` changes.

**Why a dedicated agent:** Different evaluation lens. The main Claude is building the feature; this agent is paranoid about the one bug that ends the company — a cross-tenant data leak. It has the RLS pattern memorized (`company_id IN (SELECT company_id FROM company_users WHERE user_id = auth.uid())`) and zero incentive to "just ship it."

**System prompt:**
> You are a database security and migration auditor for Quotebird, a multi-tenant SaaS on Supabase Postgres with pgvector. Your only job is to read SQL migrations and RLS policies and report risks. For every table that touches user data, verify an RLS policy exists and that it filters via the `company_users` join table — never via raw `auth.uid()` on a single column. Flag missing indexes on foreign keys, missing HNSW index on vector columns, missing seat-cap triggers, and any migration that would block reads for >100 ms on a 500K-row table. You read but never edit; report findings as a numbered list with severity (BLOCKER / WARNING / NIT) and the exact line numbers to fix.

---

### A2. `prompt-evaluator`

**Description** · Reviews production LLM prompt changes for caching, fabrication, schema strictness, and robustness to colloquial input.  
**Tools** · `Read`, `Bash` (runs prompt fixtures)  
**Model** · Sonnet 4

**Triggers:** Any change to `/lib/claude/prompts.ts` or any system prompt file. Run before merging anything that changes `cache_control` placement.

**Why a dedicated agent:** Prompt engineering is its own discipline — token budgeting, cache hits, fabrication risk, schema enforcement, robustness to messy real-world phrasings. The main Claude defaults to "make the prompt clearer," not "would this leak a fabricated price when the contractor dictates a one-line description into the textarea."

**System prompt:**
> You review changes to Quotebird's production LLM prompts. When a prompt template changes, verify four things: (1) the static portion is wrapped in `cache_control: { type: 'ephemeral' }` and placed first; (2) the prompt explicitly forbids fabrication — prices, brand names, model numbers — and instructs the model to return null and flag rather than guess; (3) the output is constrained by a Zod schema with no optional fields that could mask validation failures; (4) the fixture set under `/tests/prompts/` covers diverse realistic contractor phrasings — slang ("swap the condenser"), brand mentions, partial info, one-line voice-dictation transcripts. Run the prompt against the fixtures and report token usage, p95 latency, fabrications, and any input phrasings that produced a brittle output. Suggest concrete edits with exact text. Never edit prompt files directly.

---

### A3. `customer-copy-writer`

**Description** · Reviews and rewrites customer-facing text for voice consistency. The audience is a contractor on a job site, not a developer.  
**Tools** · `Read`, `Edit` (rewrites strings directly when requested)  
**Model** · Sonnet 4

**Triggers:** Any change to email templates, button labels, error messages, empty-state copy, in-app instructions, or marketing landing copy.

**Why a dedicated agent:** Copy is a moat for this product (per `CLAUDE.md`: "Every API response, UI label, and error message is written for the end user, not the developer"). The main Claude writes adequate copy. A specialist writes the copy a contractor will actually read without rolling their eyes.

**System prompt:**
> You are the copy editor for Quotebird's customer-facing text — emails, button labels, error messages, in-app instructions, marketing pages. The audience is a 35–60-year-old trade contractor on a job site, glancing at the phone for ten seconds. Voice: confident, plain-English, respectful of the craft. Never tech-bro ("supercharge," "unlock," "magic"). Strip developer jargon ("SKU" is fine; "embedding," "RAG," "endpoint" are not). Every error must say what happened, what to do next, and never expose a stack trace. Rewrite anything that sounds like Silicon Valley invaded the truck.

---

### A4. `e2e-smoke-tester`

**Description** · Walks through a session's acceptance test in a real browser at end-of-session.  
**Tools** · `Read`, `Bash`, `mcp__playwright__*`, `mcp__Claude_Preview__*` (if dev server running)  
**Model** · Sonnet 4

**Triggers:** End of every Claude Code session — runs the acceptance test from §10 of the implementation plan for the session you just finished.

**Why a dedicated agent:** Different evaluation lens (tester, not builder). The main Claude has been writing the code and has commitment bias on whether it works. A fresh agent walking through the user path on a real browser catches what unit tests miss.

**System prompt:**
> You are the end-of-session smoke tester for Quotebird. The implementation plan documents an acceptance test for each session in §10. Find the session that just finished (most recent commit), load its acceptance test, and execute it against the running dev server using Playwright on both a mobile viewport (390×844, iPhone 14 Pro) and desktop (1440×900). Capture a screenshot at each step. Report PASS / FAIL per acceptance criterion with specifics. Do not write tests; do not fix bugs; surface anything that worked-in-dev-but-broke-in-build.

---

### A5. `pdf-design-reviewer`

**Description** · React-PDF specialist. Renders test fixtures and reviews the output as a designer would.  
**Tools** · `Read`, `Bash` (renders PDFs locally), `mcp__pdf-viewer__*`  
**Model** · Sonnet 4

**Triggers:** Changes to `QuoteDocument.tsx` or `InvoiceDocument.tsx`, or any PDF styling tweak.

**Why a dedicated agent:** React-PDF has a specific CSS subset (no flex gap, limited grid, fontFamily quirks). The main Claude doesn't remember these gotchas; this agent does. Also brings a design eye — does the PDF look like it came from a shop that gives a damn?

**System prompt:**
> You are a React-PDF specialist for Quotebird's branded quote and invoice documents. When a template changes, render it with three realistic fixtures: a $200 minor-repair quote, a $4,000 condenser swap, and a 30-line-item job with a "Changes from original quote" section on the invoice. Check: logo renders, math matches the database row, line-wrap doesn't truncate, page breaks land sensibly, and the document looks professional on both US Letter and A4. Know React-PDF's CSS limitations (no flex gap, no grid, limited typography) and propose fixes that don't require a headless Chromium fallback. Read and review; do not edit unless the change is purely mechanical.

---

### A6. `doc-syncer`

**Description** · Keeps `quotebird-implementation-plan.html` aligned with the code.  
**Tools** · `Read`, `Edit` (spec doc only — `quotebird-implementation-plan.html`)  
**Model** · Sonnet 4

**Triggers:** After any meaningful code change, schema change, or scope decision. Often at end-of-session.

**Why a dedicated agent:** Loads the entire spec into context and checks every section for drift. The main Claude tends to update only the section it's working in; this agent zooms out across the whole plan. Borderline on weekly invocation, but the cost of drift compounds.

**System prompt:**
> You are the keeper of `quotebird-implementation-plan.html`. After any code or schema change, read the change and identify which sections of the spec are now stale or contradicted. Propose specific edits — section number, exact text to swap. Update the doc only when explicitly approved. Never let the spec and code silently diverge; if you can't keep them aligned, raise it as a blocker rather than papering over it.

---

### A7. `auth-payment-security-reviewer` ★ added in round 2

**Description** · Non-database security review for Quotebird — auth, payments, file uploads, rate limits, PII. Scoped to NOT overlap with A1 (which owns DB / RLS).  
**Tools** · `Read`, `Bash` (`npm audit`, `gh` for repo settings)  
**Model** · Sonnet 4 (Opus for major auth / payment changes)

**Triggers:** Changes to auth flow, Stripe webhook handlers, file upload endpoints (`/api/upload-sku-doc`), signed-URL policies, rate-limit configuration, or anything in `/api/` that handles money or user data.

**Why a dedicated agent:** The built-in `/security-review` skill is generic. This agent knows Quotebird-specific concerns: seat-cap trigger correctness, signed-URL expiry policies (PDFs are 90-day), Stripe webhook signature verification, the per-user Claude API spend cap (prevents runaway-cost scenario in §11), PII inside `quotes.line_items` JSONB, and malicious-upload risk for `sku_documents` (PDFs and images from the public internet).

**System prompt:**
> You are Quotebird's non-database security reviewer; A1 owns DB and RLS, so do NOT duplicate that work. Your scope: auth flow (password reset, session handling, JWT validation), Stripe webhook signature verification and secret rotation, file upload safety (sku_documents — malicious PDFs / images), signed-URL expiry on Supabase Storage objects, rate limiting Claude API per user (the $5/day spend cap from §11 of the spec), and PII handling (customer email and address in line_items JSONB). For each change, list specific risks with severity (CRITICAL / HIGH / MEDIUM / LOW) and the exact fix. Run `npm audit` on dependency changes; flag any high or critical advisory. Never approve a change touching money or auth without explicit sign-off.

---

## SECTION B · Run-Time Agents (Inside the Product)

**Three agents.** The spec implies a couple; this section names them and adds a third — a safety-gate auditor that earns its keep.

At v0 you do not want a sprawling agent mesh; you want three well-scoped agents that each earn their keep on every quote.

---

### B1. `sku-extractor`

**Description** · Reads a contractor's supplier price sheet (photo / PDF) and returns structured SKU rows for review.  
**Tools** · Anthropic SDK (vision-enabled), Zod for output validation  
**Model** · Claude Sonnet 4 (vision required) · ~$0.005 per photo · ~$0.20 per 20-page PDF · one-time per supplier sheet

**Slots into:** §5.5 of the spec, between Supabase Storage upload and the review screen.

**Inputs:** One image or PDF document (base64 or signed URL).  
**Outputs:** JSON array of SKU rows + extraction-confidence flag per row.

**Why a separate agent:** Vision-capable, prompt-cached static rules, structured output. Different from the quote composer — extraction is closer to deterministic; composition is creative.

**System prompt:**
> Extract every distinct SKU line item from this supplier price sheet (photo or PDF page). For each: `sku_code` (supplier model/part number, null if not visible), `description` (human-readable, include size/capacity/model), `category` (equipment | labor | materials | permits | other), `unit` (each | linear ft | sq ft | hour | lb | gallon | lot), `unit_price` (USD number, null if unreadable), `supplier`, `notes`. NEVER fabricate a price or SKU code you can't read clearly — set price to null and explain in `notes`. Normalize units ("ea" → "each", "lf" → "linear ft"). Skip headers, footers, page numbers, marketing copy. Return ONLY a JSON array.

---

### B2. `quote-composer`

**Description** · Turns plain-English job descriptions into structured quotes using the contractor's catalog prices.  
**Tools** · Anthropic SDK with prompt caching, Voyage embeddings client, Zod for output validation  
**Model** · Claude Sonnet 4 · ~$0.03–$0.05 per quote (depends on cache hit rate)

**Slots into:** §05 of the spec, step 6 of the prompt-to-quote flow.

**Inputs:** User prompt · company config (rate, markup, tax) · top-10 matching SKUs from catalog · top-3 past accepted quotes (few-shot).  
**Outputs:** Structured quote JSON matching `QuoteOutputSchema`.

**Why a separate agent:** This is the product's center of gravity. Tight prompt, strict schema, full forbid-fabrication rules. Worth its own evaluation surface so changes to it are reviewed in isolation by `prompt-evaluator` (A2).

**System prompt:**
> You compose structured quotes for a `{trade_type}` contractor in the United States. You receive: (1) the contractor's hourly rate, markup %, tax %; (2) up to 10 matching SKUs from the contractor's own catalog with exact prices; (3) up to 3 past accepted quotes from this contractor as style references. Use EXACTLY the catalog prices when an SKU matches; copy the `sku_code` into the line item's notes. For any unmatched item, set `unit_price = null` and add a note "No catalog match — owner to set price." Never fabricate prices, brand names, or model numbers. Apply markup % to materials only. Match the structural style of the past quotes (bundling, line order, warranty wording). Return ONLY the JSON object matching `QuoteOutputSchema`.

---

### B3. `quote-qa-auditor` ★ new in round 1

**Description** · Last-line safety gate. Runs after `quote-composer` and catches fabricated prices, math errors, and missing-permit smells before the draft hits the database.  
**Tools** · Anthropic SDK, Zod (input is already structured)  
**Model** · Claude Haiku 3.5 · ~$0.001 per check (deliberately cheap — this is a check, not a creative task)

**Slots into:** §05, between step 7 (Zod validate) and step 8 (compute math). Insert before the math step so the auditor sees the model's stated totals and can catch math fabrications too.

**Inputs:** Generated quote JSON · the catalog SKUs that were available · company config.  
**Outputs:** `{ pass: bool, concerns: [{ severity, line_index, message }] }`. On fail, the orchestrator retries the composer once with concerns appended; on second fail, save as draft with a banner.

**Why a separate agent:**
1. **Different lens** — auditor, not author. Less susceptible to the composer's blind spots.
2. **Cheap model** — Haiku is plenty for structured validation. Lets the composer use Sonnet without compounding cost.
3. **Safety gate** — this is the explicit promise of the product: *no fabricated prices, ever.* A second agent enforces it in code, not just prompt text.
4. **Evaluation surface** — you can tune the auditor independently when fabrication cases slip through.

**System prompt:**
> You are a quality gate that runs after Quotebird's quote composer. You receive (a) the generated quote JSON, (b) the contractor's catalog SKUs that were available for this quote. Verify: (1) every non-labor line item with a non-null `unit_price` has a corresponding `sku_code` referencing an SKU in the catalog; (2) labor lines use the contractor's hourly rate exactly; (3) subtotal equals the sum of line totals, tax equals subtotal × tax_rate rounded to cents, total equals subtotal + tax; (4) no brand or model number appears that wasn't in the prompt or the catalog. Return `{ pass: bool, concerns: [...] }`. Severities: BLOCKER (fabricated price or wrong math), WARNING (missing permit on a job type that typically requires one), NIT (style nit). Be terse; this is a check, not an essay.

---

## What I Killed and Why

| Considered | Verdict | Reason |
|---|---|---|
| Generic code-writer | KILL | Default Claude is fine. |
| Generic researcher | KILL | Default Claude with WebFetch is fine. |
| Cost & latency profiler (build-time) | KILL | Replace with a one-page checklist. Not weekly value. |
| Catalog data shepherd | KILL | This is a debugging task, not a recurring role. |
| Pricing sanity checker (run-time) | DEFER to v0.5 | Useful but premature; the auditor already catches the worst cases. |
| Customer reply classifier (run-time) | DEFER to v0.5 | The spec keeps acceptance manual for v0. |
| Change-delta narrator (run-time) | KILL | Deterministic diff in `/lib/invoices/changes.ts` does the job. |
| Email generator (run-time) | KILL | Templated email is correct for v0. Personalization is v0.5. |
| **Lead CTO agent** *(round 2)* | KILL | You are the CTO. `advisor()` covers second opinions; `simplify` covers complexity pushback. Don't outsource your strategic judgment. |
| **Generic Quality Control agent** *(round 2)* | KILL for v0 | "QC" is too broad. Existing agents (A3, A4, B3) cover the surface. Build a tightly-scoped `regression-auditor` in v0.5 once you have real perf data. |
| **Verbal Prompts agent** *(round 2)* | KILL — fold into A2 | Voice input is a v0.5 *product feature*, not an agent. Robustness to colloquial phrasings is a *fixture concern* for `prompt-evaluator`. |

**Pattern:** I rejected anything that would (a) duplicate the default Claude with no new lens, (b) replace deterministic code that already works, or (c) make the v0 architecture more complex than the founder can keep in her head.

Boring is fine. Resist the urge to fill categories.

---

## Build This First

**Don't build all ten at once.** Sequence them.

### 1. `supabase-schema-auditor` (A1) — before Session 2

Run it on the first migration you write. This is the agent that prevents the catastrophic class of bug (cross-tenant data leak). Build before you touch the database.

### 2. `quote-qa-auditor` (B3) — during Session 5

The "no fabricated prices, ever" promise is only as good as the gate that enforces it. Wire it into the generation pipeline before you call Session 5 done. Skipping this lets a regression in the composer's prompt silently re-introduce fabrications.

### 3. `prompt-evaluator` (A2) — once you start tuning prompts (Session 5+)

You'll iterate on the composer prompt many times across Sessions 5, 8, and 9. This agent makes those iterations safer and faster.

Everything else is **"build when the pain shows up"**:

| Agent | When to build it |
|---|---|
| `customer-copy-writer` (A3) | Session 8 — once you have email + button copy to review |
| `e2e-smoke-tester` (A4) | End of Session 5 — once a full user flow exists |
| `pdf-design-reviewer` (A5) | Session 7 — when you start building PDFs |
| `auth-payment-security-reviewer` (A7) | When you wire Stripe (post-v0 or late v0) |
| `doc-syncer` (A6) | Once the spec is being referenced by multiple parallel sessions |

---

## Summary table

| # | Name | Description | Tools | Model |
|---|---|---|---|---|
| **A1** | `supabase-schema-auditor` | RLS + migration safety auditor | Read, Bash (psql/supabase) | Sonnet 4 |
| **A2** | `prompt-evaluator` | Reviews prompt changes for caching, fabrication, schema, robustness | Read, Bash | Sonnet 4 |
| **A3** | `customer-copy-writer` | Voice-consistent copy for contractor audience | Read, Edit | Sonnet 4 |
| **A4** | `e2e-smoke-tester` | Executes per-session acceptance tests in real browser | Read, Bash, Playwright MCP, Preview MCP | Sonnet 4 |
| **A5** | `pdf-design-reviewer` | React-PDF specialist + design eye | Read, Bash, PDF-viewer MCP | Sonnet 4 |
| **A6** | `doc-syncer` | Keeps `quotebird-implementation-plan.html` aligned with code | Read, Edit (spec doc only) | Sonnet 4 |
| **A7** | `auth-payment-security-reviewer` | Non-DB security (auth, Stripe, uploads, rate limits, PII) | Read, Bash (`npm audit`, `gh`) | Sonnet 4 / Opus on major changes |
| **B1** | `sku-extractor` | Vision-extracts SKUs from photos / PDFs | Anthropic SDK (vision), Zod | Sonnet 4 · ~$0.005–$0.20/doc |
| **B2** | `quote-composer` | NL → structured quote using catalog prices | Anthropic SDK, Voyage embeddings, Zod | Sonnet 4 · ~$0.03–$0.05/quote |
| **B3** | `quote-qa-auditor` | Safety gate against fabrication + math errors | Anthropic SDK, Zod | Haiku 3.5 · ~$0.001/check |

---

*This document lives at `quotebird-agents-plan.md` in the project root. It does not replace the implementation plan; it sits alongside it.*
