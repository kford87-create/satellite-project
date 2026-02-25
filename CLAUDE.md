# Kestrel AI — Project Guidelines for Claude

## Core Principle: Keep It Simple

Everything built in this project must be simple and deliverable to customers
with minimal to no training required.

**Before writing any code, ask:**
- Can a non-technical user understand what this does?
- Can a customer use this without reading a manual?
- Is there a simpler way to achieve the same outcome?

## Design Rules

- **No over-engineering.** Solve the problem in front of you, not hypothetical future problems.
- **No unnecessary abstractions.** Three clear lines of code beat a clever utility function.
- **No feature creep.** Build what was asked. Stop there.
- **Interfaces must be obvious.** If it needs a tooltip or explanation, redesign it.
- **Defaults must work out of the box.** Configuration is optional, not required.
- **Errors must be human-readable.** No stack traces facing end users.

## What "Customer-Ready" Means

- A new user can get value within 5 minutes of signing up
- Support tickets should be rare — if users keep asking the same question, fix the UX
- Every API response, UI label, and error message is written for the end user, not the developer

## What to Avoid

- Complex setup steps (Docker-only deploys, manual config file editing, CLI-only workflows)
- Jargon in user-facing text (mAP50, OBB, YOLO are internal terms — never customer-facing)
- Fragile systems that require babysitting
- Building for scale you don't have yet
