# Implementation Plan: Guardrails Middleware & Configurable Monitoring Metrics

**Branch**: `001-guardrails-monitoring-middlewares` | **Date**: 2026-06-02 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-guardrails-monitoring-middlewares/spec.md`

## Summary

Add a new **"Guardrails"** middleware type and make the existing **Monitoring** middleware's emitted metrics selectable — both within the existing Middlewares framework (`Middleware` entity with `middleware_type` + JSON `config`, the `MiddlewareForm` UI, and runtime wiring in `agentTools.create_agent`).

- **Guardrails**: a new `MiddlewareType` whose JSON `config` holds per-protection boolean flags grouped into `input` and `output` sets (all default `true`) plus a pre-filled `custom_prompt`. At runtime a custom `AgentMiddleware` subclass injects a composed guardrail `SystemMessage` (built from the enabled protections + custom prompt) before the model call, steering the agent to reject malicious/jailbreak input and to avoid PII leakage, toxic/biased language, and off-topic answers. Enforcement is prompt/instruction-based per the spec assumptions.
- **Monitoring metrics**: extend the Monitoring `config` with a `metrics` map of boolean flags (input_tokens, output_tokens, total_tokens, models, llm_calls), all default `true`. The streaming service's `[Monitoring]` emission reads the flags and prints/stores only the selected metrics. Absent config => all metrics (backward compatible).

No DB migration is required: `middleware_type` is stored as a string column and configuration lives in the existing JSON `config` column.

## Technical Context

**Language/Version**: Python 3.11+ (backend), TypeScript / React 18 (frontend)

**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic, LangChain/LangGraph (`langchain==1.2.10`, `langchain.agents.middleware.AgentMiddleware`); React 18 + Vite + Tailwind CSS

**Storage**: PostgreSQL — reuses existing `Middleware.config` JSON column; no schema change

**Testing**: pytest (`tests/unit`, `tests/integration`); frontend has no test suite yet (lint only)

**Target Platform**: Linux server (Docker), shared backend consumed by all frontend clients

**Project Type**: Web application (backend + frontend library)

**Performance Goals**: Guardrail injection adds a single static SystemMessage per turn — negligible latency; no extra LLM call

**Constraints**: Must not change behavior of existing middleware types or pre-existing Monitoring middlewares (all-metrics default); enforcement is prompt-based (no external moderation service in v1)

**Scale/Scope**: ~6 files changed across backend + frontend; one new runtime middleware class; no new tables/endpoints

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is an unpopulated template — no ratified principles define binding gates. Advisory engineering norms from `CLAUDE.md`/repo conventions are applied:

- **Simplicity (YAGNI)**: PASS — Reuses the existing `Middleware` entity, JSON `config`, form, and runtime wiring. No new table, endpoint, or subsystem. New type added to the existing enum; new runtime path mirrors the existing PII/HITL branches.
- **Layering (router -> service -> repository)**: PASS — No new business logic in routers; validation stays in `MiddlewareService`; runtime wiring stays in `tools/agentTools.py`.
- **No behavior change to existing features**: PASS — Backward-compatible config defaults (absent => all-on).
- **Test coverage**: PASS — Unit tests for guardrail-prompt composition and monitoring metric filtering; integration test for create/edit/persist round-trip.

**Result**: PASS (no violations; Complexity Tracking not required).

## Project Structure

### Documentation (this feature)

```text
specs/001-guardrails-monitoring-middlewares/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── middleware-config.md   # Config-shape contract for guardrails + monitoring
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
backend/
├── models/
│   └── middleware.py                 # Add GUARDRAILS to MiddlewareType enum
├── tools/
│   ├── agentTools.py                 # Wire 'guardrails' branch -> GuardrailsMiddleware
│   └── middleware/
│       └── guardrails.py             # NEW: GuardrailsMiddleware + prompt composition + defaults
├── services/
│   ├── middleware_service.py         # Validate new type (enum already covers it)
│   └── agent_streaming_service.py    # Filter [Monitoring] output by config.metrics flags
└── schemas/
    └── middleware_schemas.py         # (no shape change — config stays Dict[str, Any])

frontend/src/
├── components/forms/
│   └── MiddlewareForm.tsx            # Guardrails type entry, Input/Output checkbox sections,
│                                     #   Custom Prompt textarea, Monitoring metrics checkboxes,
│                                     #   default-config + submit handling
└── pages/settings/
    └── MiddlewaresPage.tsx           # Add 'guardrails' to MIDDLEWARE_TYPE_LABELS

tests/
├── unit/
│   └── tools/test_guardrails_middleware.py    # NEW: prompt composition + metric filtering
└── integration/
    └── test_middlewares_*.py                  # create/edit/persist round-trip (extend if exists)
```

**Structure Decision**: Web application (Option 2). The feature is a vertical slice across the existing `backend/` (models, tools, services) and `frontend/` (form + page) using the established Middlewares pattern. A new `backend/tools/middleware/guardrails.py` module houses the custom runtime middleware and the canonical default protections + custom prompt, keeping `agentTools.py` thin.

## Complexity Tracking

> No constitution violations — section intentionally empty.
