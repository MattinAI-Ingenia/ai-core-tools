# Phase 1 Data Model: Guardrails Middleware & Configurable Monitoring Metrics

No new tables or columns. The feature reuses the existing `Middleware` entity and its JSON `config` column. This document defines the **logical shape** of the `config` payloads for the affected middleware types, plus validation rules.

## Entity: Middleware (existing)

| Field | Type | Notes |
|-------|------|-------|
| `middleware_id` | int (PK) | unchanged |
| `name` | str | unchanged |
| `description` | str | unchanged |
| `middleware_type` | str (enum value) | **+ new value `"guardrails"`** |
| `config` | JSON | type-specific payload (see below) |
| `app_id` | int (FK App) | unchanged; tenant scope |
| relationships | `agent_associations`, `mcp_associations` | unchanged |

### Enum change

`MiddlewareType` (`backend/models/middleware.py`) gains:

```python
GUARDRAILS = "guardrails"
```

Stored as a string (column is `VARCHAR`); no migration required.

## Config shape: `guardrails`

```jsonc
{
  "input": {
    "block_malicious_prompts": true,   // detect/refuse malicious instructions
    "block_jailbreak": true            // resist jailbreak / prompt-injection
  },
  "output": {
    "prevent_pii_leakage": true,       // do not reveal PII
    "block_toxic_biased": true,        // no toxic or biased language
    "enforce_business_facts": true     // stay within defined business facts/logic
  },
  "custom_prompt": "You are protected by guardrails. ..."  // pre-filled, editable
}
```

**Defaults (new middleware)**: every protection flag `true`; `custom_prompt` pre-filled with a working default instruction (canonical text lives in `backend/tools/middleware/guardrails.py` and is mirrored as the form's initial value).

**Validation rules**:
- Each flag is a boolean; missing flag ⇒ treated as `true` (default-on) at runtime.
- `custom_prompt` is an optional string; empty string is allowed (no custom rules).
- All flags `false` **and** empty `custom_prompt` ⇒ middleware saves but applies no restrictions (edge case; UI signals the "no protection" state).
- Safer-wins composition: a `custom_prompt` cannot silently weaken an enabled checkbox protection; the injected instruction states that the most restrictive rule applies.

**Runtime mapping** (`GuardrailsMiddleware.before_model`):
- Compose a `SystemMessage` containing: a header, one bullet per **enabled** input protection, one bullet per **enabled** output protection, then the `custom_prompt` appended verbatim.
- Inject the composed message ahead of the model call. If no flags enabled and no custom prompt, inject nothing (no-op).

## Config shape: `monitoring` (extended)

```jsonc
{
  "metrics": {
    "input_tokens": true,
    "output_tokens": true,
    "total_tokens": true,
    "models": true,
    "llm_calls": true
  }
}
```

**Defaults (new middleware)**: all `true`.

**Backward compatibility**: `config` absent, `metrics` absent, or any flag missing ⇒ that metric is **enabled** (all-on). This guarantees pre-existing Monitoring middlewares behave exactly as today.

**Validation rules**:
- Each flag is a boolean.
- All flags `true` ⇒ monitoring emits everything
- All flags `false`-> monitoring emits nothing (edge case; UI signals "no output" state).

**Runtime mapping** (`agent_streaming_service.py`, both `[Monitoring]` blocks):
- Read `metrics` from the monitoring middleware's config (default `{}` ⇒ all-on).
- Build the `[Monitoring] …` line including only the enabled metrics; omit disabled ones.

## Config shape: other middleware types

Unchanged. `summarization`, `model_call_limit`, `tool_call_limit`, `pii`, `human_in_the_loop` keep their current config shapes and behavior.

## State / lifecycle

No new states. Guardrails and Monitoring middlewares follow the existing create → edit → attach-to-agent → (delete) lifecycle. Config is read fresh per agent build (`agentTools.create_agent`) and per streamed turn (`agent_streaming_service`).
