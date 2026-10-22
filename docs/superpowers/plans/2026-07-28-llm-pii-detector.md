# Custom LLM PII Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, additive LLM-based PII detector to the existing `pii` middleware, so agents can catch PII that regex can't (names, addresses, custom entity types) alongside the existing regex `PIIMiddleware` detectors.

**Architecture:** A new `LLMPIIMiddleware(AgentMiddleware)` class with real async hooks (`abefore_model`/`aafter_model`) calls a configurable LLM via structured output to find PII, then locates each reported value in the original text by literal search to build `PIIMatch` offsets, then reuses LangChain's own `apply_strategy` to redact/mask/hash/block — same output shapes as the regex path. It is wired into `agentTools.py`'s existing `pii` branch as one more middleware instance appended alongside the regex ones, never replacing them.

**Tech Stack:** Python 3.11+, LangChain 1.x (`langchain.agents.middleware`), Pydantic, pytest/pytest-asyncio (backend); React 19 + TypeScript (frontend, `MiddlewareForm.tsx`).

**Design doc:** `docs/superpowers/specs/2026-07-28-llm-pii-detector-design.md`

## Global Constraints

- Detection mode is **additive only** — the LLM detector runs *alongside* the regex `PIIMiddleware` instances built from `pii_types`, never replacing them. There is no "regex vs LLM" exclusive toggle.
- `strategy`, `apply_to_input`, `apply_to_output`, `apply_to_tool_results` are shared between the regex and LLM detectors — one policy object, not two independent configs.
- `llm_detector.ai_service` uses the string encoding `"agent_llm"` (default) or `"ai_service:<id>"`, exactly like the existing `summarization_model` field (`backend/tools/agentTools.py:262-338`).
- `LLMPIIMiddleware` must only implement the **async** hooks (`abefore_model`/`aafter_model`) — this app only ever invokes agents via `ainvoke()`/`astream()`. Do not add sync `before_model`/`after_model` overrides; the inherited no-op defaults are correct and sufficient.
- A finding whose reported `value` cannot be located verbatim (case-insensitive) in the original content must be **skipped and logged as a warning**, never raised as an error and never crash the run.
- Reuse `apply_strategy`, `PIIMatch` from `langchain.agents.middleware._redaction` — do not reimplement redact/mask/hash/block logic.
- Reuse `_build_summarization_llm_from_service` (`backend/tools/agentTools.py:133-154`) for AI-Service lookup — do not duplicate that logic.

---

### Task 1: `LLMPIIMiddleware` — detection and redaction logic

**Files:**
- Create: `backend/tools/middleware/llm_pii.py`
- Test: `tests/unit/tools/test_llm_pii_middleware.py`

**Interfaces:**
- Consumes: `langchain.agents.middleware._redaction.{PIIMatch, apply_strategy}`, `langchain.agents.middleware.types.AgentMiddleware`, `langchain_core.messages.{HumanMessage, AIMessage, ToolMessage}`, `pydantic.BaseModel`, `utils.logger.get_logger`.
- Produces (used by Task 2): `LLMPIIMiddleware(llm, entities: list[str], strategy: str = "redact", apply_to_input: bool = True, apply_to_output: bool = True, apply_to_tool_results: bool = True)` — a class whose constructor takes any object exposing `.with_structured_output(schema)` (e.g. any LangChain `BaseChatModel`) as `llm`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/tools/test_llm_pii_middleware.py`:

```python
"""Unit tests for LLMPIIMiddleware (LLM-based PII detection, additive to regex)."""
import pytest

from langchain.agents.middleware._redaction import PIIDetectionError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tools.middleware.llm_pii import LLMPIIMiddleware, _PIIDetectionResult, _PIIFinding


class _FakeStructuredLLM:
    def __init__(self, findings):
        self._findings = findings

    async def ainvoke(self, prompt):
        return _PIIDetectionResult(findings=self._findings)


class _FakeLLM:
    def __init__(self, findings):
        self._findings = findings

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._findings)


class _FakeRuntime:
    pass


@pytest.mark.asyncio
async def test_redacts_detected_entity_in_input():
    llm = _FakeLLM([_PIIFinding(type="person", value="John Smith")])
    mw = LLMPIIMiddleware(llm=llm, entities=["person"], strategy="redact")
    state = {"messages": [HumanMessage(content="My name is John Smith.")]}

    result = await mw.abefore_model(state, _FakeRuntime())

    assert result is not None
    assert result["messages"][0].content == "My name is [REDACTED_PERSON]."


@pytest.mark.asyncio
async def test_no_findings_returns_none():
    llm = _FakeLLM([])
    mw = LLMPIIMiddleware(llm=llm, entities=["person"], strategy="redact")
    state = {"messages": [HumanMessage(content="Nothing sensitive here.")]}

    result = await mw.abefore_model(state, _FakeRuntime())

    assert result is None


@pytest.mark.asyncio
async def test_value_not_found_verbatim_is_skipped_not_raised():
    llm = _FakeLLM([_PIIFinding(type="person", value="Jonathan Smithsonian")])
    mw = LLMPIIMiddleware(llm=llm, entities=["person"], strategy="redact")
    state = {"messages": [HumanMessage(content="My name is John Smith.")]}

    result = await mw.abefore_model(state, _FakeRuntime())

    assert result is None


@pytest.mark.asyncio
async def test_block_strategy_raises_on_match():
    llm = _FakeLLM([_PIIFinding(type="person", value="John Smith")])
    mw = LLMPIIMiddleware(llm=llm, entities=["person"], strategy="block")
    state = {"messages": [HumanMessage(content="My name is John Smith.")]}

    with pytest.raises(PIIDetectionError):
        await mw.abefore_model(state, _FakeRuntime())


@pytest.mark.asyncio
async def test_apply_to_output_redacts_ai_message():
    llm = _FakeLLM([_PIIFinding(type="email", value="jane@example.com")])
    mw = LLMPIIMiddleware(llm=llm, entities=["email"], strategy="mask", apply_to_output=True)
    state = {"messages": [AIMessage(content="Contact jane@example.com for details.")]}

    result = await mw.aafter_model(state, _FakeRuntime())

    assert result is not None
    assert "jane@example.com" not in result["messages"][0].content


@pytest.mark.asyncio
async def test_apply_to_tool_results_redacts_tool_message():
    llm = _FakeLLM([_PIIFinding(type="ip", value="10.0.0.5")])
    mw = LLMPIIMiddleware(
        llm=llm, entities=["ip"], strategy="hash",
        apply_to_input=False, apply_to_tool_results=True,
    )
    state = {
        "messages": [
            AIMessage(content="calling tool", tool_calls=[]),
            ToolMessage(content="Server at 10.0.0.5 responded", tool_call_id="call_1"),
        ]
    }

    result = await mw.abefore_model(state, _FakeRuntime())

    assert result is not None
    assert "10.0.0.5" not in result["messages"][1].content


@pytest.mark.asyncio
async def test_empty_entities_list_skips_detection_entirely():
    llm = _FakeLLM([_PIIFinding(type="person", value="John Smith")])
    mw = LLMPIIMiddleware(llm=llm, entities=[], strategy="redact")
    state = {"messages": [HumanMessage(content="My name is John Smith.")]}

    result = await mw.abefore_model(state, _FakeRuntime())

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/unit/tools/test_llm_pii_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.middleware.llm_pii'` (the module doesn't exist yet).

- [ ] **Step 3: Write the implementation**

Create `backend/tools/middleware/llm_pii.py`:

```python
"""LLM-based PII detection middleware — additive to the regex-based PIIMiddleware.

Unlike LangChain's built-in ``PIIMiddleware``, this uses an LLM call to find PII
that doesn't follow a fixed pattern (names, addresses, custom entity types).
This needs its own async hooks: LangChain's built-in ``PIIMiddleware.detector``
param is a synchronous callable only (``abefore_model`` just calls the sync
``before_model`` directly), so an LLM call — a network request — can't be driven
through it without blocking the event loop this app's agents run on.
"""
from __future__ import annotations

import re

from langchain.agents.middleware._redaction import PIIMatch, apply_strategy
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from utils.logger import get_logger

logger = get_logger(__name__)


class _PIIFinding(BaseModel):
    type: str
    value: str


class _PIIDetectionResult(BaseModel):
    findings: list[_PIIFinding]


_DETECTION_PROMPT_TEMPLATE = (
    "You are a PII detection engine. Scan the TEXT below and find every instance "
    "of these entity types: {entities}.\n\n"
    "Return only entities that literally appear in the text — do not paraphrase, "
    "normalize, or translate the value; return the exact substring as it appears. "
    "If none are found, return an empty list.\n\n"
    "TEXT:\n\"\"\"\n{content}\n\"\"\""
)


def _build_matches(content: str, findings: list[_PIIFinding]) -> list[PIIMatch]:
    matches: list[PIIMatch] = []
    for finding in findings:
        if not finding.value:
            continue
        pattern = re.compile(re.escape(finding.value), re.IGNORECASE)
        found_any = False
        for m in pattern.finditer(content):
            found_any = True
            matches.append(PIIMatch(type=finding.type, value=m.group(), start=m.start(), end=m.end()))
        if not found_any:
            logger.warning(
                f"[LLMPIIMiddleware] LLM-reported value not found verbatim in content, "
                f"skipped: type={finding.type!r}"
            )
    return matches


class LLMPIIMiddleware(AgentMiddleware):
    """Detect PII using an LLM, in addition to (not instead of) regex detection.

    Runs alongside the built-in ``PIIMiddleware`` instances built from the
    regex ``pii_types`` selection — this middleware only adds a second,
    LLM-driven detection pass over the same messages.
    """

    def __init__(
        self,
        llm,
        entities: list[str],
        strategy: str = "redact",
        apply_to_input: bool = True,
        apply_to_output: bool = True,
        apply_to_tool_results: bool = True,
    ) -> None:
        super().__init__()
        self.entities = entities
        self.strategy = strategy
        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results
        self._structured_llm = llm.with_structured_output(_PIIDetectionResult)

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}[{','.join(self.entities)}]"

    async def _detect(self, content: str) -> list[PIIMatch]:
        if not content or not self.entities:
            return []
        prompt = _DETECTION_PROMPT_TEMPLATE.format(entities=", ".join(self.entities), content=content)
        result = await self._structured_llm.ainvoke(prompt)
        return _build_matches(content, result.findings)

    async def _process_content(self, content: str) -> tuple[str, list[PIIMatch]]:
        matches = await self._detect(content)
        if not matches:
            return content, []
        sanitized = apply_strategy(content, matches, self.strategy)
        return sanitized, matches

    async def abefore_model(self, state, runtime) -> dict | None:
        """Check user input and tool results for PII before the model is called."""
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages = list(messages)
        any_modified = False

        if self.apply_to_input:
            last_user_msg = None
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_user_msg = messages[i]
                    last_user_idx = i
                    break

            if last_user_idx is not None and last_user_msg and last_user_msg.content:
                content = str(last_user_msg.content)
                new_content, matches = await self._process_content(content)
                if matches:
                    new_messages[last_user_idx] = HumanMessage(
                        content=new_content, id=last_user_msg.id, name=last_user_msg.name,
                    )
                    any_modified = True

        if self.apply_to_tool_results:
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is not None:
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMessage) and msg.content:
                        content = str(msg.content)
                        new_content, matches = await self._process_content(content)
                        if matches:
                            new_messages[i] = ToolMessage(
                                content=new_content, id=msg.id, name=msg.name, tool_call_id=msg.tool_call_id,
                            )
                            any_modified = True

        if any_modified:
            return {"messages": new_messages}
        return None

    async def aafter_model(self, state, runtime) -> dict | None:
        """Check the AI's response for PII after the model is called."""
        if not self.apply_to_output:
            return None

        messages = state["messages"]
        if not messages:
            return None

        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_idx = i
                break

        if last_ai_idx is None:
            return None

        ai_msg = messages[last_ai_idx]
        if not ai_msg.content:
            return None

        content = str(ai_msg.content)
        new_content, matches = await self._process_content(content)
        if not matches:
            return None

        new_messages = list(messages)
        new_messages[last_ai_idx] = AIMessage(
            content=new_content, id=ai_msg.id, name=ai_msg.name, tool_calls=ai_msg.tool_calls,
        )
        return {"messages": new_messages}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/unit/tools/test_llm_pii_middleware.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/tools/middleware/llm_pii.py tests/unit/tools/test_llm_pii_middleware.py
git commit -m "feat(middlewares): add LLM-based PII detector middleware"
```

---

### Task 2: Wire `LLMPIIMiddleware` into `agentTools.py`'s `pii` branch

**Files:**
- Modify: `backend/tools/agentTools.py:397-423` (the `elif mw_type == 'pii':` branch)
- Test: `tests/integration/tools/test_agent_middleware_chain_integration.py`

**Interfaces:**
- Consumes: `LLMPIIMiddleware` from Task 1 (`tools.middleware.llm_pii`); `_build_summarization_llm_from_service(agent, ai_service_id: int)` (already exists, `agentTools.py:133-154`); `llm` (the agent's own resolved LLM, already in scope at `agentTools.py:166` within `create_agent`).
- Produces: the `pii` middleware config gains a `llm_detector` sub-object (`{"enabled": bool, "ai_service": str, "extra_entities": list[str]}`) that Task 3's frontend writes.

- [ ] **Step 1: Write the failing integration test**

Add to `tests/integration/tools/test_agent_middleware_chain_integration.py` (new class, after `TestGuardrailsMiddlewareEndToEnd`):

```python
class TestLLMPIIMiddlewareEndToEnd:
    @pytest.mark.asyncio
    async def test_llm_pii_middleware_redacts_alongside_regex(self, db, fake_app, fake_agent):
        """An LLM PII detector attached via config must run ADDITIVELY alongside
        the regex PIIMiddleware instances, redacting what regex alone can't catch."""
        from tools.middleware.llm_pii import _PIIDetectionResult, _PIIFinding

        class _FakeStructuredDetector:
            async def ainvoke(self, prompt):
                return _PIIDetectionResult(findings=[_PIIFinding(type="person", value="John Smith")])

        class _FakeDetectorLLM:
            def with_structured_output(self, schema):
                return _FakeStructuredDetector()

        middleware = Middleware(
            name="Test PII",
            middleware_type=MiddlewareType.PII,
            config={
                "pii_types": ["email"],
                "strategy": "redact",
                "apply_to_input": True,
                "apply_to_output": False,
                "apply_to_tool_results": False,
                "llm_detector": {
                    "enabled": True,
                    "ai_service": "ai_service:999",
                    "extra_entities": ["person"],
                },
            },
            app_id=fake_app.app_id,
        )
        db.add(middleware)
        db.flush()
        db.add(AgentMiddleware(agent_id=fake_agent.agent_id, middleware_id=middleware.middleware_id, order=0))
        db.flush()
        db.refresh(fake_agent)

        fake_llm = FakeListChatModel(responses=["Hello from the test LLM"])
        with (
            patch("tools.agentTools.get_llm", return_value=fake_llm),
            patch("tools.agentTools._build_summarization_llm_from_service", return_value=_FakeDetectorLLM()),
        ):
            agent_chain, _ = await create_agent(fake_agent)
            result = await agent_chain.ainvoke(
                {"messages": [HumanMessage(content="I'm John Smith, email me at test@example.com")]},
                config={},
            )

        human_messages = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        content = human_messages[0].content
        assert "[REDACTED_PERSON]" in content  # from the LLM detector (extra_entities)
        assert "[REDACTED_EMAIL]" in content    # from the regex detector (pii_types), unaffected
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./scripts/test.sh -m integration -k test_llm_pii_middleware_redacts_alongside_regex -v`
Expected: FAIL — the redaction doesn't happen because `agentTools.py` doesn't read `llm_detector` yet (content still contains `"John Smith"` and/or `"test@example.com"` unredacted, or a `KeyError`/no-op since the config key is simply ignored).

- [ ] **Step 3: Wire it into `agentTools.py`**

In `backend/tools/agentTools.py`, add the import near the other middleware imports (top of file, alongside `from langchain.agents.middleware.pii import PIIMiddleware`):

```python
from tools.middleware.llm_pii import LLMPIIMiddleware
```

Then in the `elif mw_type == 'pii':` branch (`agentTools.py:397-423`), immediately after the existing `middleware.append(_PIILogMiddleware())` / `logger.info(...)` lines that close the regex loop, add:

```python
                llm_detector_cfg = mw_config.get('llm_detector') or {}
                if llm_detector_cfg.get('enabled'):
                    ai_service_value = llm_detector_cfg.get('ai_service', 'agent_llm')
                    detector_llm = llm
                    if ai_service_value and ai_service_value != 'agent_llm':
                        try:
                            service_id = int(ai_service_value.split(':', 1)[1])
                            detector_llm = _build_summarization_llm_from_service(agent, service_id) or llm
                        except (ValueError, IndexError):
                            logger.warning(
                                f"Invalid ai_service format '{ai_service_value}' for LLM PII detector "
                                f"— using agent LLM"
                            )

                    pii_type_labels = {
                        "email": "email addresses",
                        "credit_card": "credit card numbers",
                        "ip": "IP addresses",
                        "mac_address": "MAC addresses",
                        "url": "URLs",
                    }
                    llm_entities = [pii_type_labels.get(t, t) for t in pii_types] + list(
                        llm_detector_cfg.get('extra_entities') or []
                    )

                    middleware.append(LLMPIIMiddleware(
                        llm=detector_llm,
                        entities=llm_entities,
                        strategy=strategy,
                        apply_to_input=apply_to_input,
                        apply_to_output=apply_to_output,
                        apply_to_tool_results=apply_to_tool_results,
                    ))
                    logger.info(
                        f"LLMPIIMiddleware enabled for agent {agent.agent_id} "
                        f"(ai_service={ai_service_value}, entities={llm_entities})"
                    )
```

This sits inside the same `elif mw_type == 'pii':` block, so `pii_types`, `strategy`, `apply_to_input`, `apply_to_output`, and `apply_to_tool_results` (already parsed a few lines above at `agentTools.py:398-402`) are all in scope.

- [ ] **Step 4: Run the test to verify it passes**

Run: `./scripts/test.sh -m integration -k test_llm_pii_middleware_redacts_alongside_regex -v`
Expected: PASS.

- [ ] **Step 5: Run the full middleware integration file to check for regressions**

Run: `./scripts/test.sh -m integration -k test_agent_middleware_chain_integration -v`
Expected: PASS (all tests in the file, including the pre-existing guardrails/monitoring/no-middleware ones).

- [ ] **Step 6: Commit**

```bash
git add backend/tools/agentTools.py tests/integration/tools/test_agent_middleware_chain_integration.py
git commit -m "feat(middlewares): wire LLM PII detector into the pii middleware branch"
```

---

### Task 3: Frontend — LLM detector config UI in `MiddlewareForm.tsx`

**Files:**
- Modify: `frontend/src/components/forms/MiddlewareForm.tsx`

**Interfaces:**
- Consumes: existing `aiServices: AIServiceOption[]` state (already fetched unconditionally whenever `appId` is set, `MiddlewareForm.tsx:166-181`) — fields used: `service_id`, `name`, `provider`, `model_name`.
- Produces: on submit, `formData.config.llm_detector` matches the backend shape from Task 2: `{ enabled: boolean, ai_service: string, extra_entities: string[] }`.

- [ ] **Step 1: Add local state for the raw extra-entities text**

Near the other per-field local state declarations (`MiddlewareForm.tsx:156-166`), add:

```typescript
    const [extraEntitiesText, setExtraEntitiesText] = useState('');
```

*(Rationale: binding a text input directly to a derived `array.join(', ')` value causes the classic controlled-input bug — typing a trailing comma/space gets stripped on every keystroke because the array is re-parsed and re-joined immediately. Keeping the raw string in its own state and only parsing into an array at submit time avoids this, matching how `summarizationModel`/`triggerTokens`/etc. are already handled in this same file.)*

- [ ] **Step 2: Populate `extraEntitiesText` when editing an existing middleware**

In the `useEffect` that hydrates `formData` from an existing `middleware` (`MiddlewareForm.tsx:183-216`), add after the `trim_tokens` block:

```typescript
            if (middleware.config?.llm_detector?.extra_entities) {
                setExtraEntitiesText(middleware.config.llm_detector.extra_entities.join(', '));
            }
```

- [ ] **Step 3: Default `llm_detector` config when switching to the `pii` type**

In `handleTypeSelect` (`MiddlewareForm.tsx:317-319`), change:

```typescript
        if (typeValue === 'pii') {
            newConfig = { pii_types: ['email', 'credit_card', 'ip', 'mac_address', 'url'], strategy: 'redact', apply_to_input: true, apply_to_output: true, apply_to_tool_results: true };
        }
```

to:

```typescript
        if (typeValue === 'pii') {
            newConfig = {
                pii_types: ['email', 'credit_card', 'ip', 'mac_address', 'url'],
                strategy: 'redact',
                apply_to_input: true,
                apply_to_output: true,
                apply_to_tool_results: true,
                llm_detector: { enabled: false, ai_service: 'agent_llm', extra_entities: [] },
            };
            setExtraEntitiesText('');
        }
```

- [ ] **Step 4: Add the LLM detector section to the PII config block**

In the PII config block (`MiddlewareForm.tsx:766-845`), insert this new `<div>` right after the "Apply to" `<div>` closes and before the block's own closing `</div>` (i.e. immediately before line 844's `</div>` which closes the `space-y-4` wrapper):

```tsx
                    <div>
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={formData.config?.llm_detector?.enabled ?? false}
                                disabled={isSubmitting}
                                onChange={(e) => setFormData(prev => ({
                                    ...prev,
                                    config: {
                                        ...prev.config,
                                        llm_detector: {
                                            ...(prev.config?.llm_detector ?? { ai_service: 'agent_llm', extra_entities: [] }),
                                            enabled: e.target.checked,
                                        },
                                    },
                                }))}
                                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                            />
                            <span className="text-sm font-medium text-gray-700">
                                Also run an LLM-based detector (in addition to the types above)
                            </span>
                        </label>

                        {formData.config?.llm_detector?.enabled && (
                            <div className="mt-3 space-y-3 pl-6">
                                <div>
                                    <label htmlFor="llm_detector_ai_service" className="block text-sm font-medium text-gray-700 mb-1">
                                        AI Service
                                    </label>
                                    <select
                                        id="llm_detector_ai_service"
                                        value={formData.config?.llm_detector?.ai_service ?? 'agent_llm'}
                                        disabled={isSubmitting}
                                        onChange={(e) => setFormData(prev => ({
                                            ...prev,
                                            config: {
                                                ...prev.config,
                                                llm_detector: { ...(prev.config?.llm_detector ?? {}), ai_service: e.target.value },
                                            },
                                        }))}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                                    >
                                        <option value="agent_llm">Agent's LLM (default)</option>
                                        {aiServices.map((svc) => (
                                            <option key={svc.service_id} value={`ai_service:${svc.service_id}`}>
                                                {svc.name} ({svc.provider} · {svc.model_name})
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label htmlFor="llm_detector_extra_entities" className="block text-sm font-medium text-gray-700 mb-1">
                                        Extra entities <span className="text-gray-400 font-normal">(optional)</span>
                                    </label>
                                    <input
                                        type="text"
                                        id="llm_detector_extra_entities"
                                        value={extraEntitiesText}
                                        disabled={isSubmitting}
                                        onChange={(e) => setExtraEntitiesText(e.target.value)}
                                        className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
                                        placeholder="person, last name, passport number"
                                    />
                                    <p className="mt-1 text-xs text-gray-500">
                                        Comma-separated list of extra entity types to look for, e.g. person, last name, passport number.
                                    </p>
                                </div>
                            </div>
                        )}
                    </div>
```

- [ ] **Step 5: Parse `extraEntitiesText` into the submitted config**

In `handleSubmit` (`MiddlewareForm.tsx:395-454`), the function already builds a type-specific `submitData` for `summarization` and `human_in_the_loop` before calling `onSubmit`. Add a matching branch for `pii`, right after the `human_in_the_loop` branch and before `await onSubmit(submitData);`:

```typescript
            if (formData.middleware_type === 'pii' && formData.config?.llm_detector?.enabled) {
                submitData = {
                    ...formData,
                    config: {
                        ...formData.config,
                        llm_detector: {
                            ...formData.config.llm_detector,
                            extra_entities: extraEntitiesText.split(',').map((s: string) => s.trim()).filter(Boolean),
                        },
                    },
                };
            }
```

- [ ] **Step 6: Type-check and lint the frontend**

Run: `cd frontend && npx tsc -b --noEmit && npm run lint`
Expected: no new type errors, no new lint errors.

- [ ] **Step 7: Manual smoke test**

Run: `cd frontend && npm run dev` (with the backend running per `CLAUDE.md`'s backend dev command). In the app's Middlewares settings page, create/edit a `pii` middleware, tick "Also run an LLM-based detector", pick an AI Service, type `person, passport number` into the extra-entities field (verify a trailing comma/space doesn't get stripped while typing), save, then reopen it and confirm the checkbox, AI Service selection, and extra-entities text all persisted correctly.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/forms/MiddlewareForm.tsx
git commit -m "feat(frontend): add LLM PII detector config UI to MiddlewareForm"
```

---

## Self-Review Notes

- **Spec coverage:** config shape (Task 2/3), backend detector + strategy application (Task 1), async-only hooks constraint (Task 1's Global Constraint + implementation), `agentTools.py` wiring reusing `_build_summarization_llm_from_service` (Task 2), frontend UI incl. AI Service dropdown defaulting to the agent's own LLM + extra-entities field with helper text (Task 3), unit + integration testing (Tasks 1 & 2) — all covered.
- **Type consistency:** `LLMPIIMiddleware(llm, entities, strategy, apply_to_input, apply_to_output, apply_to_tool_results)` constructor signature is identical between Task 1 (definition) and Task 2 (call site). `_PIIFinding`/`_PIIDetectionResult` names match between Task 1's implementation and both Task 1's and Task 2's tests. Config key `llm_detector: { enabled, ai_service, extra_entities }` is identical across the design doc, Task 2's backend read, and Task 3's frontend write.
- **No placeholders:** every step above has concrete, runnable code — no "similar to Task N" or "add appropriate handling" left unresolved.
