# Quotebird · Agent Strategy

*A working plan for which specialist agents to build into your Claude Code setup (build-time) and into the Quotebird product (run-time). Read it end-to-end in 8–10 minutes. Skip to "Build This First" for the punch list.*

---

## How I applied the kill-test

For every agent I considered, I asked the four criteria from the prompt. I'm being honest about the ones I rejected — see **What I Killed and Why** at the bottom. If an agent doesn't appear below, it didn't earn its keep.

---

## SECTION A · Build-Time Agents

Six agents. Each lives at `.claude/agents/<name>.md` in this repo (project scope) — not global — because their context is Quotebird-specific.

---

### A1. `supabase-schema-auditor`

**Description (one-liner):** Audits SQL migrations and RLS policies for multi-tenant safety. Read-only.

**Triggers:** Every time you write a migration, change an RLS policy, or add a new table that touches tenant data. Also run before merging any branch with `supabase/migrations/*` changes.

**Why a dedicated agent:** Different evaluation lens. The main Claude is building the feature; this agent is paranoid about the one bug that ends the company — a cross-tenant data leak. It comes with the RLS pattern memorized (`company_id IN (SELECT company_id FROM company_users WHERE user_id = auth.uid())`), and it has zero incentive to "ship the feature."

**Tools:** `Read`, `Bash` (restricted to `psql`/`supabase` invocations for query inspection). **No** `Write`, **no** `Edit`.

**System prompt:**
> You are a database security and migration auditor for Quotebird, a multi-tenant SaaS on Supabase Postgres with pgvector. Your only job is to read SQL migrations and RLS policies and report risks. For every table that touches user data, verify an RLS policy exists and that it filters via the `company_users` join table — never via raw `auth.uid()` on a single column. Flag missing indexes on foreign keys, missing HNSW index on vector columns, missing seat-cap triggers, and any migration that would block reads for >100 ms on a 500K-row table. You read but never edit; report findings as a numbered list with severity (BLOCKER / WARNING / NIT) and the exact line numbers to fix.

---

### A2. `prompt-evaluator`

**Description (one-liner):** Reviews changes to production LLM prompts. Checks caching, fabrication rules, schema strictness, and cost.

**Triggers:** Any change to `/lib/claude/prompts.ts` or any system prompt file. Also run before merging anything that changes `cache_control` placement.

**Why a dedicated agent:** Prompt engineering is its own evaluation lens — token budgeting, cache hits, fabrication risk, schema enforcement. The main Claude defaults to "make the prompt clearer," not "would this leak a fabricated price under temperature spike."

**Tools:** `Read`, `Bash` (for running prompt fixture tests if present). No direct `Edit` — suggests changes for you to accept.

**System prompt:**
> You review changes to Quotebird's production LLM prompts. When a prompt template changes, verify three things: (1) the static portion is wrapped in `cache_control: { type: 'ephemeral' }` and placed first; (2) the prompt explicitly forbids fabrication — prices, brand names, model numbers — and instructs the model to return null and flag rather than guess; (3) the output is constrained by a Zod schema with no optional fields that could mask validation failures. Run the prompt against any fixtures under `/tests/prompts/` and report token usage, p95 latency, and fabrications. Suggest concrete edits with the exact text. Never edit prompt files directly.

---

### A3. `customer-copy-writer`

**Description (one-liner):** Reviews and rewrites customer-facing text for voice consistency.

**Triggers:** Any change to email templates, button labels, error messages, empty-state copy, in-app instructions, or marketing landing copy.

**Why a dedicated agent:** Copy quality is a moat for this product (per `CLAUDE.md`: "Every API response, UI label, and error message is written for the end user, not the developer"). The main Claude writes adequate copy. A specialist writes the copy a contractor would actually read without rolling their eyes.

**Tools:** `Read`, `Edit` (for rewriting strings directly when requested).

**System prompt:**
> You are the copy editor for Quotebird's customer-facing text — emails, button labels, error messages, in-app instructions, marketing pages. The audience is a 35–60-year-old trade contractor on a job site, glancing at the phone for ten seconds. Voice: confident, plain-English, respectful of the craft. Never tech-bro ("supercharge," "unlock," "magic"). Strip developer jargon ("SKU" is fine; "embedding," "RAG," "endpoint" are not). Every error must say what happened, what to do next, and never expose a stack trace. Rewrite anything that sounds like Silicon Valley invaded the truck.

---

### A4. `e2e-smoke-tester`

**Description (one-liner):** Walks through a session's acceptance test in a real browser at end-of-session.

**Triggers:** End of every Claude Code session — runs the acceptance test from §10 of the implementation plan for the session you just finished.

**Why a dedicated agent:** Different evaluation lens (tester, not builder). The main Claude has been writing the code; it has commitment bias on whether it works. A fresh agent walking through the user path on a real browser catches what unit tests miss.

**Tools:** `Read`, `Bash`, `mcp__playwright__*` for headless browser, `mcp__Claude_Preview__*` if the local server is running. **No** `Edit` / `Write` — reports findings, doesn't fix them.

**System prompt:**
> You are the end-of-session smoke tester for Quotebird. The implementation plan documents an acceptance test for each session in §10. Find the session that just finished (most recent commit), load its acceptance test, and execute it against the running dev server using Playwright on both a mobile viewport (390×844, iPhone 14 Pro) and desktop (1440×900). Capture a screenshot at each step. Report PASS / FAIL per acceptance criterion with specifics. Do not write tests; do not fix bugs; surface anything that worked-in-dev-but-broke-in-build.

---

### A5. `pdf-design-reviewer`

**Description (one-liner):** React-PDF specialist; renders test fixtures and reviews output as a designer would.

**Triggers:** Changes to `QuoteDocument.tsx` or `InvoiceDocument.tsx`, or any time you tweak PDF styling.

**Why a dedicated agent:** React-PDF has a specific CSS subset (no flex gap, limited grid, fontFamily quirks). The main Claude doesn't remember these gotchas; this agent does. Also brings a design eye — does it look like it came from a shop that gives a damn?

**Tools:** `Read`, `Bash` (renders the PDF locally), `mcp__pdf-viewer__*` to inspect the result visually.

**System prompt:**
> You are a React-PDF specialist for Quotebird's branded quote and invoice documents. When a template changes, render it with three realistic fixtures: a $200 minor-repair quote, a $4,000 condenser swap, and a 30-line-item job with a "Changes from original quote" section on the invoice. Check: logo renders, math matches the database row, line-wrap doesn't truncate, page breaks land sensibly, and the document looks professional on both US Letter and A4. Know React-PDF's CSS limitations (no flex gap, no grid, limited typography) and propose fixes that don't require a headless Chromium fallback. Read and review; do not edit unless the change is purely mechanical.

---

### A6. `doc-syncer`

**Description (one-liner):** Keeps `quotebird-implementation-plan.html` aligned with the code.

**Triggers:** After any meaningful code change, schema change, or scope decision. Often runs at end-of-session.

**Why a dedicated agent:** This is the borderline one. The main Claude can update docs, but it tends to update only the section it's working in. This agent loads the entire spec into context and checks every section for drift. Worth it because the spec is the source of truth for future sessions — drift compounds.

**Tools:** `Read`, `Edit` (for spec doc only). No `Bash` / `Write`.

**System prompt:**
> You are the keeper of `quotebird-implementation-plan.html`. After any code or schema change, read the change and identify which sections of the spec are now stale or contradicted. Propose specific edits — section number, exact text to swap. Update the doc only when explicitly approved. Never let the spec and code silently diverge; if you can't keep them aligned, raise it as a blocker rather than papering over it.

---

## SECTION B · Run-Time Agents (Inside the Product)

Only three. The spec implies a couple; this section names them and adds a third — a safety-gate auditor that earns its keep.

I considered six others and killed them (see bottom). At v0 you do not want a sprawling agent mesh; you want three well-scoped agents that earn their keep on every quote.

---

### B1. `sku-extractor`

**User-facing job:** Read a contractor's supplier price sheet (photo or PDF) and return structured SKU rows for the contractor to review before they enter the catalog.

**Inputs:** One image or PDF document (base64 or signed URL).
**Outputs:** JSON array of SKU rows + extraction confidence per row.

**Model:** Claude Sonnet 4 (vision required).
**Cost:** ~$0.005 per photo · ~$0.20 per 20-page PDF. One-time per supplier sheet.

**Why a separate agent:** Vision-capable, prompt-cached static rules, structured output. Different from the quote composer — extraction is deterministic-ish; composition is creative.

**Slots into:** §5.5 of the spec, between Supabase Storage upload and the review screen.

**System prompt:**
> Extract every distinct SKU line item from this supplier price sheet (photo or PDF page). For each: `sku_code` (supplier model/part number, null if not visible), `description` (human-readable, include size/capacity/model), `category` (equipment | labor | materials | permits | other), `unit` (each | linear ft | sq ft | hour | lb | gallon | lot), `unit_price` (USD number, null if unreadable), `supplier`, `notes`. NEVER fabricate a price or SKU code you can't read clearly — set price to null and explain in `notes`. Normalize units ("ea" → "each", "lf" → "linear ft"). Skip headers, footers, page numbers, marketing copy. Return ONLY a JSON array.

---

### B2. `quote-composer`

**User-facing job:** Turn the contractor's plain-English job description into a structured quote using their catalog prices.

**Inputs:** User prompt · company config (rate, markup, tax) · top-10 matching SKUs from catalog · top-3 past accepted quotes (few-shot).
**Outputs:** Structured quote JSON matching `QuoteOutputSchema`.

**Model:** Claude Sonnet 4.
**Cost:** ~$0.03–$0.05 per quote (depends on prompt cache hit rate).

**Why a separate agent:** This is the product's center of gravity. Tight prompt, strict schema, full forbid-fabrication rules. Worth its own evaluation surface.

**Slots into:** §05 of the spec, step 6 of the prompt-to-quote flow.

**System prompt:**
> You compose structured quotes for a `{trade_type}` contractor in the United States. You receive: (1) the contractor's hourly rate, markup %, tax %; (2) up to 10 matching SKUs from the contractor's own catalog with exact prices; (3) up to 3 past accepted quotes from this contractor as style references. Use EXACTLY the catalog prices when an SKU matches; copy the `sku_code` into the line item's notes. For any unmatched item, set `unit_price = null` and add a note "No catalog match — owner to set price." Never fabricate prices, brand names, or model numbers. Apply markup % to materials only. Match the structural style of the past quotes (bundling, line order, warranty wording). Return ONLY the JSON object matching `QuoteOutputSchema`.

---

### B3. `quote-qa-auditor` ★ new

**User-facing job:** Last-line safety gate. Runs after `quote-composer` and before the draft hits the database. Catches fabricated prices, math errors, and missing-permit smells.

**Inputs:** Generated quote JSON · the catalog SKUs that were available · company config.
**Outputs:** `{ pass: bool, concerns: [{ severity, line_index, message }] }`. On fail, the orchestrator retries the composer once with the concerns appended; on second fail, the quote is saved as a draft with a banner.

**Model:** Claude Haiku 3.5 (cheap, deterministic — this is a check, not a creative task).
**Cost:** ~$0.001 per check.

**Why a separate agent:**
1. **Different lens** — auditor, not author. Less susceptible to the composer's own blind spots.
2. **Cheap model** — Haiku is plenty for structured validation. Lets the composer use Sonnet without compounding cost.
3. **Safety gate** — this is the explicit promise of the SKU-driven product: *no fabricated prices, ever.* A second agent enforces it.
4. **Evaluation surface** — you can tune the auditor independently when fabrication cases slip through.

**Slots into:** §05, between step 7 (Zod validate) and step 8 (compute math). Insert before the math step so the auditor sees the model's stated totals and can catch math fabrications too.

**System prompt:**
> You are a quality gate that runs after Quotebird's quote composer. You receive (a) the generated quote JSON, (b) the contractor's catalog SKUs that were available for this quote. Verify: (1) every non-labor line item with a non-null `unit_price` has a corresponding `sku_code` referencing an SKU in the catalog; (2) labor lines use the contractor's hourly rate exactly; (3) subtotal equals the sum of line totals, tax equals subtotal × tax_rate rounded to cents, total equals subtotal + tax; (4) no brand or model number appears that wasn't in the prompt or the catalog. Return `{ pass: bool, concerns: [...] }`. Severities: BLOCKER (fabricated price or wrong math), WARNING (missing permit on a job type that typically requires one), NIT (style nit). Be terse; this is a check, not an essay.

---

## What I Killed and Why

Eight agents I considered, then rejected against the kill-test:

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

**Pattern:** I rejected anything that would either (a) duplicate the default Claude with no new lens, (b) replace deterministic code that already works, or (c) make the v0 architecture more complex than the founder can keep in her head. Boring is fine.

---

## Build This First

**Don't build all nine at once.** Sequence them. The right order:

### 1. `supabase-schema-auditor` (build-time, A1) — before Session 2

Run it on the first migration you write. This is the agent that prevents the catastrophic class of bug (cross-tenant leak). Build before you touch the database.

### 2. `quote-qa-auditor` (run-time, B3) — during Session 5

The "no fabricated prices, ever" promise is only as good as the gate that enforces it. Wire it into the generation pipeline before you call Session 5 done. Skipping this lets a regression in the composer's prompt silently introduce fabrications.

### 3. `prompt-evaluator` (build-time, A2) — once you start tuning prompts (Session 5+)

You'll iterate on the composer prompt many times across Sessions 5, 8, and 9. This agent makes those iterations safer and faster.

Everything else is "build when the pain shows up":
- `customer-copy-writer` once you have customer-facing copy to review (Session 8: email send).
- `e2e-smoke-tester` once you have a full user flow to test (end of Session 5).
- `pdf-design-reviewer` during Session 7 (PDF generation).
- `doc-syncer` once the implementation plan is being referenced in multiple parallel sessions.

---

*This document lives at `quotebird-agents-plan.md` in the project root. It does not replace the implementation plan; it sits alongside it.*
