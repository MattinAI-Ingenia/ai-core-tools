---
description: "Task list for Guardrails Middleware & Configurable Monitoring Metrics"
---

# Tasks: Guardrails Middleware & Configurable Monitoring Metrics

**Input**: Design documents from `specs/001-guardrails-monitoring-middlewares/`

**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/ ✅ quickstart.md ✅

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no pending dependencies)
- **[Story]**: Which user story this task belongs to
- Exact file paths in every description

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create new module directory; no existing files changed yet.

- [ ] T001 Create `backend/tools/middleware/` package with `backend/tools/middleware/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add `GUARDRAILS` to the enum so the backend accepts the new type. Must complete before any other phase.

**⚠️ CRITICAL**: The enum value in `backend/models/middleware.py` is the single shared foundation block for Phases 3 and 4. Phase 5 (monitoring) has no dependency on this.

- [ ] T002 Add `GUARDRAILS = "guardrails"` to `MiddlewareType` enum in `backend/models/middleware.py`

**Checkpoint**: Foundation ready — backend now recognises `middleware_type = "guardrails"`.

---

## Phase 3: User Story 1 — Guardrails middleware with default protections (Priority: P1) 🎯 MVP

**Goal**: A fully-working Guardrails middleware type that an editor can create, save, and attach to an agent. All protections are on by default; the runtime injects the composed guardrail instruction before every model call.

**Independent Test**: Create a `guardrails` middleware via the UI accepting all defaults, attach to an agent, send a known jailbreak prompt → agent refuses; send a normal prompt → answered normally. Round-trip: reopen middleware → all checkboxes still checked.

### Backend: runtime middleware

- [ ] T003 [P] [US1] Implement `GuardrailsMiddleware(AgentMiddleware)` with `before_model` hook that composes and injects a guardrail `SystemMessage` from enabled flags in `backend/tools/middleware/guardrails.py` (include `GUARDRAILS_DEFAULT_CONFIG`, `GUARDRAILS_DEFAULT_CUSTOM_PROMPT`, and `compose_guardrail_message` helper)
- [ ] T004 [P] [US1] Write unit tests for `compose_guardrail_message` covering: all flags on, subset of flags on, no flags + empty prompt (no-op), custom_prompt appended in `tests/unit/tools/test_guardrails_middleware.py`

### Backend: runtime wiring

- [ ] T005 [US1] Wire `'guardrails'` branch in `agentTools.create_agent` in `backend/tools/agentTools.py`: read `config`, instantiate `GuardrailsMiddleware`, append to `middleware` list (depends on T002, T003)

### Frontend: form UI

- [ ] T006 [P] [US1] Add `guardrails` entry to `MIDDLEWARE_TYPES` array in `frontend/src/components/forms/MiddlewareForm.tsx` (value, label, description, hooks `['before_model']`)
- [ ] T007 [P] [US1] Add `guardrails` default-config branch to `handleTypeSelect` in `frontend/src/components/forms/MiddlewareForm.tsx` — sets `config` to `GUARDRAILS_DEFAULT_CONFIG` shape with all flags `true` (mirrors `backend/tools/middleware/guardrails.py` constants)
- [ ] T008 [US1] Add Input Guardrails checkbox section (2 protections, all checked by default) inside the `formData.middleware_type === 'guardrails'` conditional block in `frontend/src/components/forms/MiddlewareForm.tsx` (depends on T006, T007)
- [ ] T009 [US1] Add Output Guardrails checkbox section (3 protections, all checked by default) in the same conditional block in `frontend/src/components/forms/MiddlewareForm.tsx` (depends on T008)
- [ ] T010 [US1] Add edit-mode hydration for guardrails `config` (restore checkbox states and custom prompt from `middleware.config` on load) in `frontend/src/components/forms/MiddlewareForm.tsx` (depends on T009)

### Frontend: page label

- [ ] T011 [P] [US1] Add `guardrails: 'Guardrails'` to `MIDDLEWARE_TYPE_LABELS` in `frontend/src/pages/settings/MiddlewaresPage.tsx`

**Checkpoint**: US1 fully functional — create, save, attach, and jailbreak-test the Guardrails middleware independently.

---

## Phase 4: User Story 2 — Custom Prompt section (Priority: P2)

**Goal**: The Guardrails form exposes a pre-filled, editable Custom Prompt textarea. Custom rules persist and are applied by the runtime in addition to the checkbox protections.

**Independent Test**: Edit a Guardrails middleware, confirm the Custom Prompt field is pre-populated with `GUARDRAILS_DEFAULT_CUSTOM_PROMPT`, append a custom rule, save, reopen → rule still there. Attach to agent, ask something that violates the custom rule → agent declines per the rule.

**Dependency**: Depends on Phase 3 (US1 form block and runtime middleware already in place).

- [ ] T012 [US2] Add Custom Prompt `<textarea>` sub-section inside the `guardrails` form block in `frontend/src/components/forms/MiddlewareForm.tsx` — pre-fill from `GUARDRAILS_DEFAULT_CUSTOM_PROMPT` constant on type select; restore from `middleware.config.custom_prompt` on edit (depends on T010)
- [ ] T013 [US2] Verify `GuardrailsMiddleware.compose_guardrail_message` appends `custom_prompt` verbatim after the protection bullets in `backend/tools/middleware/guardrails.py` — add/extend unit test in `tests/unit/tools/test_guardrails_middleware.py` for non-empty and empty custom_prompt cases (depends on T003, T004)

**Checkpoint**: US2 functional — custom rules persist and are honored by the agent.

---

## Phase 5: User Story 3 — Configurable Monitoring metrics (Priority: P2)

**Goal**: Monitoring middleware configuration shows metric-selection checkboxes. Only selected metrics are printed/stored per turn. Pre-existing Monitoring middlewares are unaffected.

**Independent Test**: Open a Monitoring middleware, uncheck "output tokens", save. Run an agent turn. Inspect `[Monitoring] …` log line — includes input tokens, total tokens, models, llm_calls; omits output tokens. Open an existing Monitoring middleware without a `metrics` config → all five metrics still emit.

**Dependency**: Independent from US1/US2 (no dependency on the Guardrails work). Can be done in parallel with Phase 3/4 if multiple contributors.

### Backend: metric-filtered emission

- [ ] T014 [P] [US3] Extract a `_emit_monitoring_log` helper in `backend/services/agent_streaming_service.py` that reads `metrics` flags from the monitoring middleware config and builds the `[Monitoring] …` log line with only enabled metrics (absent config/flag ⇒ emit); replace both existing `[Monitoring]` log blocks with calls to this helper
- [ ] T015 [P] [US3] Write unit tests for `_emit_monitoring_log` covering: all metrics enabled, subset enabled, no config (all-on fallback), all flags false (empty output) in `tests/unit/tools/test_guardrails_middleware.py` (or `test_monitoring_metrics.py` if kept separate)

### Frontend: metric-selection UI

- [ ] T016 [P] [US3] Add `monitoring` default-config branch to `handleTypeSelect` in `frontend/src/components/forms/MiddlewareForm.tsx` — sets `config.metrics` with all five flags `true` (only affects newly created Monitoring middlewares; existing ones without `metrics` key remain backward compatible)
- [ ] T017 [US3] Add Metrics checkbox section rendered when `formData.middleware_type === 'monitoring'` in `frontend/src/components/forms/MiddlewareForm.tsx` — 5 checkboxes (input tokens, output tokens, total tokens, models, LLM calls), all checked by default, each toggling the corresponding `config.metrics.*` boolean (depends on T016)
- [ ] T018 [US3] Add edit-mode hydration for monitoring `metrics` config (restore checkbox states from `middleware.config.metrics` on load; treat absent flags as `true`) in `frontend/src/components/forms/MiddlewareForm.tsx` (depends on T017)

**Checkpoint**: US3 functional — metric selection persists and only selected metrics are emitted; pre-existing middlewares unchanged.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Cleanup, validation, and end-to-end verification.

- [ ] T019 [P] Run quickstart.md validation steps (Parts A and B) and confirm all acceptance scenarios pass
- [ ] T020 [P] Run existing unit + integration test suite to confirm no regressions: `pytest tests/unit/ tests/integration/ -v --tb=short`
- [ ] T021 [P] Run frontend lint: `cd frontend && npm run lint`
- [ ] T022 Add UI warning when all guardrail protections are unchecked and custom prompt is empty in `frontend/src/components/forms/MiddlewareForm.tsx` (edge case from spec)
- [ ] T023 Add UI notice when all monitoring metrics are unchecked in `frontend/src/components/forms/MiddlewareForm.tsx` (edge case from spec)
- [ ] T024 [P] Update `MIDDLEWARE_TYPES` description for `monitoring` in `frontend/src/components/forms/MiddlewareForm.tsx` to mention configurable metrics
- [ ] T025 [P] Update `backend/tools/middleware/__init__.py` to re-export `GuardrailsMiddleware` for clean imports

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **blocks Phases 3 and 4** (needs enum value)
- **Phase 3 (US1)**: Depends on Phase 2 — backend enum + `guardrails.py` + `agentTools.py` + form UI
- **Phase 4 (US2)**: Depends on Phase 3 — extends the form block and runtime helper already in place
- **Phase 5 (US3)**: **Independent** — no dependency on Phases 3/4; can run in parallel
- **Phase 6 (Polish)**: Depends on all prior phases complete

### User Story Dependencies

- **US1 (P1)**: After Phase 2 — no dependency on US2/US3
- **US2 (P2)**: After US1 — extends the US1 form block and runtime (tight coupling by design)
- **US3 (P2)**: After Phase 1 only — fully independent; can run in parallel with US1+US2

### Within Each Phase

- `[P]`-marked tasks touch different files and can be done concurrently
- Backend and frontend tasks for US1 (T003–T011) can be split across backend/frontend developers
- T005 (agentTools wiring) depends on T002 (enum) and T003 (class)
- T008–T010 (form sections) chain sequentially within the same file
- T014 (monitoring emission) and T016–T018 (monitoring UI) are fully parallel

---

## Parallel Execution Examples

### Phase 3 (US1) — Backend dev + Frontend dev simultaneously

```bash
# Backend developer:
Task T003: Implement GuardrailsMiddleware in backend/tools/middleware/guardrails.py
Task T004: Write unit tests in tests/unit/tools/test_guardrails_middleware.py

# Frontend developer (in parallel):
Task T006: Add 'guardrails' to MIDDLEWARE_TYPES in MiddlewareForm.tsx
Task T007: Add default-config branch in handleTypeSelect in MiddlewareForm.tsx
Task T011: Add label in MiddlewaresPage.tsx

# After T002 + T003 complete:
Task T005: Wire guardrails branch in agentTools.py

# After T006 + T007 complete:
Task T008 → T009 → T010: Form sections (sequential within MiddlewareForm.tsx)
```

### Phase 5 (US3) — Fully parallel with Phase 3

```bash
# Can run simultaneously with all of Phase 3:
Task T014: _emit_monitoring_log helper in agent_streaming_service.py
Task T015: Unit tests for metric filtering
Task T016: Monitoring default-config in handleTypeSelect
# Then:
Task T017 → T018: Monitoring metric checkboxes + hydration
```

---

## Implementation Strategy

### MVP (User Story 1 only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002)
3. Complete Phase 3: US1 — T003 through T011
4. **STOP and VALIDATE**: quickstart.md Part A steps 1–8 (exclude custom-prompt step)
5. Demo to stakeholder — a working Guardrails middleware with all-default protections

### Incremental Delivery

1. Setup + Foundational → enum recognised ✓
2. Phase 3 (US1) → Guardrails middleware with checkbox defaults ✓ (MVP!)
3. Phase 4 (US2) → Custom Prompt added ✓
4. Phase 5 (US3) → Monitoring metric selection ✓ (can deliver independently at any point)
5. Phase 6 → Polish, edge-case UI warnings, full test run ✓

### Parallel Team Strategy

With two developers:
1. Both complete Phase 1 + Phase 2 together
2. **Dev A**: Phase 3 → Phase 4 (Guardrails, US1 → US2)
3. **Dev B**: Phase 5 (Monitoring US3, fully parallel)
4. Both converge on Phase 6

---

## Notes

- `[P]` = different files, no incomplete dependencies — safe to run concurrently
- `[Story]` label maps each task to its user story for traceability
- No Alembic migration needed — `middleware_type` is a string column; `config` is already JSON
- The canonical default config/prompt lives in `backend/tools/middleware/guardrails.py`; the frontend mirrors those values as constants to avoid drift
- Avoid modifying `api.ts`, `middleware_schemas.py`, or any router file — no contract changes
- Each phase produces an independently testable and demonstrable increment
