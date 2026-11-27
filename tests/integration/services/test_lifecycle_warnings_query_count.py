"""Query-count test: lifecycle warnings must not add queries per item (NFR-3, AD-8)."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import event

from models.agent import Agent
from models.ai_service import AIService
from models.embedding_service import EmbeddingService
from models.middleware import AgentMiddleware, Middleware, MiddlewareType
from models.ocr_agent import OCRAgent
from models.rerank_service import RerankService
from models.silo import Silo
from services.agent_service import AgentService

pytestmark = pytest.mark.integration


def _count_statements(db, fn):
    connection = db.get_bind()
    statements = []

    def _on_execute(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(connection, "before_cursor_execute", _on_execute)
    try:
        db.expire_all()
        result = fn()
    finally:
        event.remove(connection, "before_cursor_execute", _on_execute)
    return len(statements), result


def _make_agents(db, app_id, count, llm, silo, middleware):
    for i in range(count):
        cls = OCRAgent if i % 3 == 0 else Agent
        extra = {"vision_service_id": llm.service_id} if cls is OCRAgent else {}
        agent = cls(
            name=f"qc-agent-{count}-{i}", description="d", system_prompt="s", app_id=app_id,
            service_id=llm.service_id, silo_id=silo.silo_id, temperature=0.5,
            rag_rerank_enabled=i % 2 == 1, **extra,
        )
        db.add(agent)
        db.flush()
        if i % 2 == 0:
            db.add(AgentMiddleware(agent_id=agent.agent_id, middleware_id=middleware.middleware_id, order=0))
    db.flush()


class TestAgentListQueryCount:
    def test_statement_count_is_constant_and_warnings_are_present(self, db, fake_app):
        soon = datetime.now(timezone.utc) + timedelta(days=5)
        past = datetime.now(timezone.utc) - timedelta(days=1)
        llm = AIService(name="qc-llm", provider="OpenAI", api_key="k", app_id=fake_app.app_id,  # pragma: allowlist secret
                        end_of_service_at=soon)
        emb = EmbeddingService(name="qc-emb", provider="OpenAI", app_id=fake_app.app_id, end_of_service_at=past)
        rr = RerankService(name="qc-rr", provider="CohereCompatible", model_name="m", endpoint="https://rr.example.com",
                           app_id=fake_app.app_id, end_of_service_at=past)
        db.add_all([llm, emb, rr])
        db.flush()
        fake_app.default_rerank_service_id = rr.service_id
        silo = Silo(name="qc-silo", app_id=fake_app.app_id, embedding_service_id=emb.service_id)
        middleware = Middleware(
            name="qc-mw", middleware_type=MiddlewareType.SUMMARIZATION, app_id=fake_app.app_id,
            config={"summarization_model": f"ai_service:{llm.service_id}"},
        )
        db.add_all([silo, middleware])
        db.flush()

        service = AgentService()
        _make_agents(db, fake_app.app_id, 2, llm, silo, middleware)
        small_count, small = _count_statements(db, lambda: service.get_agents_list(db, fake_app.app_id))
        assert len(small) == 2

        _make_agents(db, fake_app.app_id, 18, llm, silo, middleware)
        big_count, big = _count_statements(db, lambda: service.get_agents_list(db, fake_app.app_id))
        assert len(big) == 20

        assert big_count == small_count
        for item in big:
            roles = {(w.role, w.lifecycle_status) for w in item.lifecycle_warnings}
            assert ("silo_embedding", "retired") in roles
            assert ("llm", "deprecated") in roles
        # agents with reranking on and no explicit service report the retired app default (same query count)
        assert any(("rerank", "retired") in {(w.role, w.lifecycle_status) for w in i.lifecycle_warnings} for i in big)
