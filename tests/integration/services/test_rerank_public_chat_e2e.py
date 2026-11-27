"""AC-14 end to end: an agent chat through the PUBLIC chat endpoint with a rerank stage (rerank-services step_013).

REAL: the public router, API-key auth, AgentExecutionService, the LangGraph agent chain, the real retriever tool
(get_retriever_tool), plan building, RetrievalPipeline, RerankRetriever and the rerank HTTP client.
MOCKED: the LLM (a fake chat model that calls the retriever tool once, then answers), the first-stage search
method (ordered fake documents), the reranker transport (httpx.MockTransport: ok / 500 / invalid / hang) and the
health recorder. The fixtures come from test_rerank_retrieval (same world).
"""
import time
from datetime import datetime

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from models.agent import Agent
from models.api_key import APIKey
from tests.integration.services.test_rerank_retrieval import (  # noqa: F401  (fixtures)
    _Recorder,
    transport,
    world,
)

pytestmark = pytest.mark.integration

TOTAL_TIMEOUT_S = 0.4  # patched by the world fixture
MARGIN_S = 1.0


SEEN: list = []  # module level: pydantic copies list fields, the product may copy the model
TOOLS: list = []


class _CallsRetrieverOnce(BaseChatModel):

    @property
    def _llm_type(self) -> str:
        return "rerank-e2e-fake"

    def bind_tools(self, tools, **kwargs):
        TOOLS[:] = [getattr(t, "name", None) or t.get("name") or t["function"]["name"] for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        SEEN.append(list(messages))
        if any(isinstance(m, ToolMessage) for m in messages):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="final answer"))])
        name = next(n for n in TOOLS if "retriev" in n.lower() or "silo" in n.lower())
        call = {"name": name, "args": {"query": "top-secret-question"}, "id": "call-1", "type": "tool_call"}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])


@pytest.fixture
def chat(world, db, fake_user, monkeypatch):
    from models.ai_service import AIService

    ai = AIService(name="E2E LLM", provider="OpenAI", api_key="sk-test-key", app_id=world.app.app_id)  # pragma: allowlist secret
    db.add(ai)
    db.flush()
    agent = Agent(name="Rerank agent", description="d", system_prompt="Use the tool.", app_id=world.app.app_id,
                  service_id=ai.service_id, silo_id=world.silo.silo_id, has_memory=False, temperature=0,
                  rag_k=8, rag_rerank_enabled=True, rag_rerank_top_n=5)
    key = APIKey(key="rerank-e2e-api-key-0123456789", name="k", app_id=world.app.app_id,  # pragma: allowlist secret
                 user_id=fake_user.user_id, is_active=True, created_at=datetime.now())
    db.add_all([agent, key])
    db.flush()
    SEEN.clear()
    TOOLS.clear()
    seen = SEEN
    monkeypatch.setattr("tools.aiServiceTools.create_llm_from_service", lambda *a, **k: _CallsRetrieverOnce())

    async def _no_metrics(payload):
        return None

    monkeypatch.setattr("services.agent_metrics_recorder.persist_metrics_payload", _no_metrics)
    return type("Chat", (), {"agent": agent, "key": key, "seen": seen, "world": world})


def _call(client, chat):
    return client.post(
        f"/public/v1/app/{chat.world.app.app_id}/chat/{chat.agent.agent_id}/call",
        data={"message": "top-secret-question"}, headers={"X-API-KEY": chat.key.key},
    )


def _tool_text(chat):
    msgs = [m for m in chat.seen[-1] if isinstance(m, ToolMessage)]
    assert msgs, "the retriever tool never ran"
    return str(msgs[-1].content)


def test_public_chat_with_a_healthy_reranker_returns_reranked_docs(client, chat):
    response = _call(client, chat)

    assert response.status_code == 200, response.text
    assert response.json()["response"] == "final answer"
    text = _tool_text(chat)
    assert text.index("doc-7") < text.index("doc-6")  # reranker order (last candidate best), top N = 5
    assert "doc-3" in text and "doc-2" not in text
    assert _Recorder.failures == []


@pytest.mark.parametrize("mode", ["500", "invalid", "hang"])
def test_public_chat_survives_a_failing_reranker_within_the_budget(client, chat, caplog, mode):
    start = time.monotonic()
    baseline = _call(client, chat)
    baseline_s = time.monotonic() - start
    assert baseline.status_code == 200, baseline.text

    chat.world.transport.mode = mode
    chat.seen.clear()
    with caplog.at_level("INFO"):
        start = time.monotonic()
        response = _call(client, chat)
        elapsed = time.monotonic() - start

    assert response.status_code == 200, response.text
    assert response.json()["response"] == "final answer"
    assert elapsed <= baseline_s + TOTAL_TIMEOUT_S + MARGIN_S
    text = _tool_text(chat)  # first-stage order truncated to top N
    assert [text.index(f"doc-{i}") for i in range(5)] == sorted(text.index(f"doc-{i}") for i in range(5))
    assert "doc-5" not in text
    assert [sid for sid, _ in _Recorder.failures] == [chat.world.a.service_id]
    assert "top-secret-question" not in caplog.text  # AC-24 across the whole chat pipeline
    assert "doc-0" not in "\n".join(r.getMessage() for r in caplog.records if "rerank" in r.getMessage().lower())
