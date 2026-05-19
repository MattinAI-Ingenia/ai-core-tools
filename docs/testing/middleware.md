# Middleware Testing - Mattin AI Agent Orchestrator

## Objective

This document shows how different middlewares were tested in the **Mattin AI** agent orchestrator:

1. **Monitoring Middleware**
2. **Human-in-the-Loop Middleware**
3. **PII Detection / Redaction Middleware**
4. **Summarization Middleware**

The test uses an agent configured to force specific behaviours and verify whether the middlewares correctly intercept tool calls, execution logs, and sensitive information.

---

## 1. Test Agent

### Configured system prompt

```text
You are a test assistant. Respond directly to the user WITHOUT calling tools unless explicitly asked.

STRICT RULES:
- Use the "Greeting Agent" tool when you are going to send a greeting.
- Only use "anonymize_text" if the user explicitly asks to anonymize text.
- For any other question, respond directly without using tools.
```

### Test intention

The prompt forces three behaviours:

| Case | Expected behaviour |
|---|---|
| User greeting | The agent must call `Greeting Agent` |
| Explicit anonymization request | The agent must call `anonymize_text` |
| Any other question | The agent must respond directly, without tools |

---

## 2. Monitoring Middleware

### Tested case

The agent is triggered and the middleware is expected to log execution metrics.

### Generated log

```text
mattin-backend  | 2026-05-19 12:20:41,996 - services.agent_streaming_service - INFO - [Monitoring] agent_id=3 | models=['gpt-5.2-2025-12-11'] | input_tokens=3309 | output_tokens=818 | total_tokens=4127 | llm_calls=1
```

### Expected result

The middleware correctly captures:

| Metric | Value |
|---|---:|
| `agent_id` | `3` |
| Model used | `gpt-5.2-2025-12-11` |
| Input tokens | `3309` |
| Output tokens | `818` |
| Total tokens | `4127` |
| LLM calls | `1` |

### Validation

The **Monitoring Middleware** works correctly because it logs the model, token consumption, and number of LLM calls during the agent execution.

---

## 3. Human-in-the-Loop Middleware - Greeting Agent

### Tested case

The user sends a greeting:

```text
Good morning!
```

According to the system prompt, the agent must use the `Greeting Agent` tool.

Since the HITL middleware is active, execution is paused before running the tool and human approval is requested.

### Recreated conversation

```text
User
14:31
Good morning!
```

```text
⏸️ Execution paused - waiting for human approval.
14:31

Greeting Agent
```

### Approval required

```text
⚠️ Approval required

🔧 Greeting_Agent

{
  "query": "Good morning!",
  "args": null,
  "kwargs": null
}

✓ Approve
✗ Reject
```

### Expected result

The user must be able to:

| Action | Result |
|---|---|
| Approve | `Greeting_Agent` is executed |
| Reject | The tool execution is blocked |

### Validation

The **Human-in-the-Loop Middleware** works correctly because it intercepts the call to `Greeting_Agent` before execution and requires explicit user approval.

---

## 4. Human-in-the-Loop Middleware - anonymize_text

### Tested case

The user explicitly asks to anonymize text containing sensitive information:

```text
Anonymize this: usuario@mattin.de, password 1234_mattin
```

According to the system prompt, the agent may use `anonymize_text` because the user explicitly requested anonymization.

### Recreated conversation

```text
User
14:36
Anonymize this: usuario@mattin.de, password 1234_mattin
```

```text
⏸️ Execution paused - waiting for human approval.
14:36

anonymize text
```

### Approval required

```text
⚠️ Approval required

🔧 anonymize_text

{
  "text": "usuario@mattin.de, password 1234_mattin",
  "model_family": "spaCy",
  "model_name": "en_core_web_lg",
  "threshold": 0.4
}

✓ Approve
✗ Reject
```

### Response after approval

```text
Here is the anonymized text:

[EMAIL_ADDRESS], password [PASSWORD]

Detected:

EMAIL_ADDRESS: usuario@mattin.de
```

### Expected result

The HITL middleware pauses execution before calling `anonymize_text`.

After approval, the tool returns the anonymized text:

```text
[EMAIL_ADDRESS], password [PASSWORD]
```

### Validation

The **Human-in-the-Loop Middleware** works correctly because:

1. It detects a tool call.
2. It pauses execution.
3. It displays the exact arguments that will be sent.
4. It allows the user to approve or reject execution.
5. It continues correctly after approval.

---

## 5. PII Detection / Redaction Middleware

### Tested case

The user sends a message containing personal data:

```text
My email is pedro_perez@gmail.com and my IP address is 194.22.23.3
```

### Message received by the LLM

Before reaching the model, the middleware replaces sensitive data with placeholders.

```text
My email is [REDACTED_EMAIL] and my IP address is [REDACTED_IP]
```

### Generated log

```text
mattin-backend  | 2026-05-19 12:53:08,258 - tools.agentTools - INFO - [PII] Message after redaction: My email is [REDACTED_EMAIL] and my IP address is [REDACTED_IP]
```

### Expected result

| Original data | Redacted data |
|---|---|
| `pedro_perez@gmail.com` | `[REDACTED_EMAIL]` |
| `194.22.23.3` | `[REDACTED_IP]` |

### Validation

The **PII Detection Middleware** works correctly because it redacts sensitive information before the message reaches the LLM and before it is stored in the conversation.

---

## 6. Model Call Limit Middleware

### Configuration

The middleware was configured with:

```text
Max LLM calls per run = 1
```

### Tested case

```text
Anonymize this: usuario@mattin.de, password 1234_mattin
15:26
Model call limits exceeded: run limit (1/1)
```

### Why this happens

When a tool is involved (for example `anonymize_text`), the run usually needs **2 model calls**:

| LLM call | Purpose |
|---|---|
| Call 1 | The model decides to invoke the tool and emits the tool call |
| Call 2 | After tool output is available, the model composes the final user-facing answer |

With `run_limit = 1`, the first call is allowed, but the second call is blocked.

### Expected result

The middleware stops the run and returns:

```text
Model call limits exceeded: run limit (1/1)
```

### Validation

The **Model Call Limit Middleware** works correctly because it enforces the per-run LLM cap and prevents the second model call required to finish a tool-based answer.

---

## 7. Summarization Middleware

### Configuration

The middleware was configured with:

```text
trigger=('tokens', 500)
keep=('messages', 2)
trim_tokens_to_summarize=500
```

### What this middleware does

When the conversation history exceeds the trigger threshold, the middleware summarizes older messages and keeps only the most recent context (plus the generated summary), reducing memory size while preserving continuity.

### Activation evidence

```text
mattin-backend  | 2026-05-19 14:26:51,174 - tools.agentTools - INFO - [Summarization] abefore_model: agent=3, messages=5, approx_tokens=2396, trigger=('tokens', 500)
mattin-backend  | 2026-05-19 14:26:55 - INFO - [Summarization] TRIGGERED for agent 3: reduced to 4 messages (summary generated)
mattin-backend  | 2026-05-19 14:26:55,516 - tools.agentTools - INFO - [Summarization] TRIGGERED for agent 3: reduced to 4 messages (summary generated)
```

### Validation

The **Summarization Middleware** works correctly because:

1. It detects when the token threshold is exceeded (`approx_tokens=2396 > 500`).
2. It triggers summarization before the model call.
3. It rewrites conversation memory with a compact summary and reduced message window.

---

## 8. Results Summary

| Middleware | Tested case | Result |
|---|---|---|
| Monitoring | Logging tokens, model, and LLM calls | ✅ Correct |
| HITL | Call to `Greeting_Agent` | ✅ Correct |
| HITL | Call to `anonymize_text` | ✅ Correct |
| PII Detection | Email and IP redaction | ✅ Correct |
| Model Call Limit (`run_limit=1`) | Tool flow with `anonymize_text` | ✅ Correct (blocked at second LLM call) |
| Summarization (`trigger=('tokens', 500)`) | Long conversation memory compaction | ✅ Correct (triggered and summary generated) |

---

## 9. Conclusion

The tested middlewares work as expected:

- **Monitoring** logs execution metrics.
- **HITL** intercepts tool calls and requires human approval.
- **PII Detection** redacts sensitive data before it reaches the LLM and before it is persisted in the conversation.
- **Model Call Limit** enforces per-run limits and can intentionally block tool-based flows when the cap is too low (e.g., `1`).
- **Summarization** compacts long conversation history when token thresholds are exceeded, preserving recent context while reducing memory size.

The architecture provides stronger control over agent behaviour, especially for sensitive operations such as tool execution or handling personal data.
