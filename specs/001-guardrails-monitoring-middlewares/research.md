# Phase 0 Research: Guardrails Middleware & Configurable Monitoring Metrics

All Technical Context items are known from the existing codebase; no `NEEDS CLARIFICATION` remained. This document records the key design decisions and the evidence behind them.

## R1 — How to implement the Guardrails runtime middleware

**Decision**: Implement a custom `GuardrailsMiddleware(AgentMiddleware)` in `backend/tools/middleware/guardrails.py`. In its `before_model` hook it composes a single guardrail `SystemMessage` from (a) the enabled input/output protection flags and (b) the `custom_prompt`, and injects it ahead of the conversation so the LLM is steered to refuse malicious/jailbreak input and to avoid PII leakage, toxic/biased language, and off-topic answers.

**Rationale**:
- `langchain==1.2.10` exposes no dedicated guardrail middleware. Inspecting `langchain.agents.middleware` yields: `AgentMiddleware`, `PIIMiddleware`, `SummarizationMiddleware`, `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware`, `HumanInTheLoopMiddleware`, plus model retry/fallback/selector helpers — none for general guardrails.
- `AgentMiddleware` provides `before_model` / `after_model` / `wrap_model_call` hooks. The repo already subclasses `AgentMiddleware` for the `_PIILogMiddleware` after-PII logger (`agentTools.py`), confirming the pattern is idiomatic here.
- The spec's Assumptions explicitly scope v1 enforcement as **prompt/instruction-based**, matching how the platform already steers behavior (system prompt, skills). A single injected SystemMessage covers both input and output guardrails with no extra LLM call and negligible latency.

**Alternatives considered**:
- *Dedicated external moderation API (e.g. content-safety service)*: rejected for v1 — out of scope per assumptions, adds a dependency, latency, and cost.
- *`after_model` output validation/regex + re-prompt loop*: rejected for v1 — adds complexity and potential extra model calls; prompt-based output rules are sufficient for the MVP. Left as a documented future enhancement.
- *Reusing `PIIMiddleware` for the PII-leakage output protection*: deferred — would compose cleanly but mixes two config models; v1 keeps PII-leakage as a guardrail instruction. Agents needing hard redaction can still attach the standalone PII middleware (the two compose without error).

## R2 — Storage model for the new type and configs

**Decision**: Add `GUARDRAILS = "guardrails"` to the `MiddlewareType` enum. Store guardrail settings and monitoring metric selections inside the existing `Middleware.config` JSON column. **No Alembic migration.**

**Rationale**:
- The `Middleware.middleware_type` DB column was created as `sa.String()` (see `alembic/versions/mw001_…`), and the SQLAlchemy `Enum(MiddlewareType)` maps by value. Existing types (`pii`, `human_in_the_loop`) were added the same way without per-type migrations.
- `config` is already `JSON` and is the established home for per-type settings (PII types, HITL `interrupt_on`, summarization params). Adding `input`/`output`/`custom_prompt` and `metrics` keys requires no schema change.
- `MiddlewareService.create_or_update_middleware` already validates via `MiddlewareType(data.middleware_type)` and falls back to MONITORING on `ValueError`; adding the enum member makes `guardrails` valid automatically.

**Alternatives considered**:
- *Dedicated columns/tables for guardrail flags*: rejected — over-engineered; JSON config is the existing convention and keeps the change surgical.

## R3 — Backward-compatible monitoring metric selection

**Decision**: Extend Monitoring `config` with a `metrics` object: `{input_tokens, output_tokens, total_tokens, models, llm_calls}` booleans, all default `true`. In `agent_streaming_service.py`, the two `[Monitoring]` emission blocks read these flags and include only enabled metrics. When `config` or `metrics` is absent, treat all as enabled (current behavior preserved).

**Rationale**:
- Monitoring currently always logs `models`, `input_tokens`, `output_tokens`, `total_tokens`, `llm_calls` (two identical blocks in `agent_streaming_service.py`). These five map 1:1 to user-facing checkboxes.
- Default-on + absent-means-all guarantees pre-existing Monitoring middlewares and current tests are unaffected (FR-014, SC-006).

**Alternatives considered**:
- *Per-metric enum list instead of boolean map*: rejected — boolean map mirrors the PII `apply_to_*` pattern already in the form, easing UI reuse.

## R4 — Frontend integration pattern

**Decision**: Extend `MiddlewareForm.tsx` following the existing per-type conditional-render pattern (as used for `pii`, `summarization`, `human_in_the_loop`): add a `guardrails` entry to `MIDDLEWARE_TYPES`, a default-config branch in `handleTypeSelect`, two checkbox sections (Input/Output) plus a Custom Prompt `<textarea>`, and a Monitoring `metrics` checkbox block rendered when `middleware_type === 'monitoring'`. Add `guardrails` to `MIDDLEWARE_TYPE_LABELS` in `MiddlewaresPage.tsx`. No `api.ts` changes — the generic `config` object already round-trips.

**Rationale**:
- The form already drives everything off `formData.config` and a free-form `Record<string, any>`, and `api.ts` `createMiddleware`/`updateMiddleware` send `config` verbatim. The PII checkbox group (lines ~743–820) is a direct template for the guardrail/metric checkbox groups.
- `MiddlewareListItemSchema`/`MiddlewareDetailSchema` already expose `config: Optional[Dict[str, Any]]`, so detail/edit round-trips work with no schema edits.

**Alternatives considered**:
- *New dedicated form component for guardrails*: rejected — duplicates plumbing; conditional sections keep one cohesive form consistent with the other types.

## Summary of decisions

| # | Decision | Migration? | New endpoint? |
|---|----------|-----------|---------------|
| R1 | Custom `GuardrailsMiddleware` injects a composed guardrail SystemMessage in `before_model` | No | No |
| R2 | Add `GUARDRAILS` enum value; settings in JSON `config` | No | No |
| R3 | Monitoring `config.metrics` boolean map, default/absent => all-on | No | No |
| R4 | Extend existing `MiddlewareForm` + `MiddlewaresPage` labels | No | No |

All `NEEDS CLARIFICATION`: none.
