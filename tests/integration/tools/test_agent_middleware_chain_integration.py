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
import logging

import pytest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from models.middleware import Middleware, MiddlewareType, AgentMiddleware
from tools.agentTools import create_agent
from tools.middleware import monitoring as monitoring_module

pytestmark = pytest.mark.integration


async def _run_agent(agent, response_text="Hello from the test LLM"):
    """Build the real middleware chain for `agent` and invoke it with a fake LLM.

    Middleware (guardrails, monitoring, ...) is baked into the chain itself by
    create_agent() — there's no external wiring left for the caller to do.
    """
    fake_llm = FakeListChatModel(responses=[response_text])
    with patch("tools.agentTools.get_llm", return_value=fake_llm):
        agent_chain, _ = await create_agent(agent)

    result = await agent_chain.ainvoke({"messages": [HumanMessage(content="Hi there")]}, config={})
    return result


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

        result = await _run_agent(fake_agent)

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
    async def test_monitoring_middleware_logs_real_llm_call(self, db, fake_app, fake_agent):
        """A monitoring middleware attached via AgentMiddleware must actually
        observe the real chain invocation and log its metrics."""
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

        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = ListHandler()
        monitoring_module.logger.addHandler(handler)
        try:
            await _run_agent(fake_agent)
        finally:
            monitoring_module.logger.removeHandler(handler)

        monitoring_lines = [r for r in records if r.startswith("[Monitoring] agent_id=")]
        assert len(monitoring_lines) == 1
        assert "llm_calls=1" in monitoring_lines[0]


class TestNoMiddlewareEndToEnd:
    @pytest.mark.asyncio
    async def test_agent_without_middlewares_runs_without_extra_messages(self, db, fake_app, fake_agent):
        """Baseline: an agent with no middleware associations gets no injected
        SystemMessages and no monitoring log, confirming the two tests above
        are actually attributable to the attached middleware."""
        records: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = ListHandler()
        monitoring_module.logger.addHandler(handler)
        try:
            result = await _run_agent(fake_agent)
        finally:
            monitoring_module.logger.removeHandler(handler)

        assert not any(isinstance(m, SystemMessage) for m in result["messages"])
        assert not any(r.startswith("[Monitoring] agent_id=") for r in records)
