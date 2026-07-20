"""Integration tests for the middleware chain built by tools.agentTools.create_agent.

Unlike tests/unit/tools/test_guardrails_middleware.py (which exercises
GuardrailsMiddleware in isolation) and the chat-router integration tests
(which mock AgentExecutionService/AgentStreamingService wholesale), these
tests run the real DB-backed Agent/Middleware/AgentMiddleware association
through the real create_agent() dispatch logic and the real LangGraph
agent, with only the LLM call itself faked. This is the missing coverage
for "does an agent with a middleware attached actually apply it at
execution time" end-to-end.
"""
import pytest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from models.middleware import Middleware, MiddlewareType, AgentMiddleware
from tools.agentTools import create_agent

pytestmark = pytest.mark.integration


async def _run_agent(agent, response_text="Hello from the test LLM"):
    """Build the real middleware chain for `agent` and invoke it with a fake LLM.

    Mirrors how agent_execution_service/agent_streaming_service attach the
    monitoring callback handler — create_agent() only builds it, the caller
    is responsible for wiring it into the invocation's callbacks.
    """
    fake_llm = FakeListChatModel(responses=[response_text])
    with patch("tools.agentTools.get_llm", return_value=fake_llm):
        agent_chain, mcp_client, monitoring_handler = await create_agent(agent)

    config = {}
    if monitoring_handler is not None:
        config.setdefault("callbacks", []).append(monitoring_handler)

    result = await agent_chain.ainvoke({"messages": [HumanMessage(content="Hi there")]}, config=config)
    return result, monitoring_handler


class TestGuardrailsMiddlewareEndToEnd:
    @pytest.mark.asyncio
    async def test_guardrails_middleware_injects_system_messages(self, db, fake_app, fake_agent):
        """A guardrails middleware attached via AgentMiddleware must actually
        wrap the real agent invocation with input/output SystemMessages."""
        middleware = Middleware(
            name="Test Guardrails",
            middleware_type=MiddlewareType.GUARDRAILS,
            config={
                "input": {"block_jailbreak": True},
                "output": {"prevent_pii_leakage": True},
                "custom_prompt": "",
            },
            app_id=fake_app.app_id,
        )
        db.add(middleware)
        db.flush()
        db.add(AgentMiddleware(agent_id=fake_agent.agent_id, middleware_id=middleware.middleware_id, order=0))
        db.flush()
        db.refresh(fake_agent)

        result, _ = await _run_agent(fake_agent)

        messages = result["messages"]
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_messages) == 2
        assert "INPUT GUARDRAIL" in system_messages[0].content
        assert "OUTPUT GUARDRAIL" in system_messages[1].content
        # The guardrail SystemMessages must not replace the actual conversation.
        assert any(isinstance(m, HumanMessage) for m in messages)
        assert any(isinstance(m, AIMessage) and m.content == "Hello from the test LLM" for m in messages)


class TestMonitoringMiddlewareEndToEnd:
    @pytest.mark.asyncio
    async def test_monitoring_middleware_counts_real_llm_call(self, db, fake_app, fake_agent):
        """A monitoring middleware attached via AgentMiddleware must produce a
        callback handler that actually observed the real chain invocation."""
        middleware = Middleware(
            name="Test Monitoring",
            middleware_type=MiddlewareType.MONITORING,
            config={},
            app_id=fake_app.app_id,
        )
        db.add(middleware)
        db.flush()
        db.add(AgentMiddleware(agent_id=fake_agent.agent_id, middleware_id=middleware.middleware_id, order=0))
        db.flush()
        db.refresh(fake_agent)

        _, monitoring_handler = await _run_agent(fake_agent)

        assert monitoring_handler is not None
        assert monitoring_handler.call_count == 1


class TestNoMiddlewareEndToEnd:
    @pytest.mark.asyncio
    async def test_agent_without_middlewares_runs_without_extra_messages(self, db, fake_app, fake_agent):
        """Baseline: an agent with no middleware associations gets no monitoring
        handler and no injected SystemMessages, confirming the two tests above
        are actually attributable to the attached middleware."""
        result, monitoring_handler = await _run_agent(fake_agent)

        assert monitoring_handler is None
        assert not any(isinstance(m, SystemMessage) for m in result["messages"])
