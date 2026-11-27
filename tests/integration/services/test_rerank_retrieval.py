"""Rerank stage wired into agent retrieval and the silo playground (rerank-services step_008).

Everything runs for real (resolution query, plan building, RetrievalPipeline, RerankRetriever, rerank HTTP client)
except: the first-stage search methods (a fake retriever returning ordered documents), the reranker transport
(httpx.MockTransport) and the health recorder (captures calls; the DB writer uses its own session, which cannot
see the test transaction).

The agent path is exercised through the real ``get_retriever_tool`` coroutine (what the LLM's tool call runs).
"""
import asyncio
import time
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config
from models.rerank_service import RerankService
from services.silo_service import SiloService, resolve_search_params
from tests.factories import EmbeddingServiceFactory, configure_factories
from tools.rerank import http_clients
from tools.retrieval import rerank_transformer as rt
from tools.retrieval.retrieval_component import SearchMethod
from utils.error_handlers import ValidationError

from models.silo import Silo

N_CANDIDATES = 12
HOST_A = "https://rerank-a.example.test"
HOST_B = "https://rerank-b.example.test"


class _FirstStage(BaseRetriever):
    k: int

    def _get_relevant_documents(self, query, *, run_manager) -> List[Document]:
        return [Document(page_content=f"doc-{i}", metadata={"i": i}) for i in range(N_CANDIDATES)][: self.k]


class _FakeMethod(SearchMethod):
    def build(self, ctx):
        return _FirstStage(k=ctx.search_kwargs.get("k", 30))


class _Recorder:
    successes: list = []
    failures: list = []

    def record_success(self, target):
        type(self).successes.append(target.service_id)

    def record_failure(self, target, message):
        type(self).failures.append((target.service_id, message))


class _SessionProxy:
    """Hands the test session to code that opens its own ``SessionLocal()`` and closes it."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        pass


class _Transport:
    """Records requests and answers according to ``mode``."""

    def __init__(self):
        self.mode = "ok"
        self.hosts: list[str] = []
        self.docs_sent: list[int] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        import json
        body = json.loads(request.content)
        self.hosts.append(f"{request.url.scheme}://{request.url.host}")
        self.docs_sent.append(len(body["documents"]))
        if self.mode == "500":
            return httpx.Response(500, json={"error": "boom"})
        if self.mode == "invalid":
            return httpx.Response(200, json={"unexpected": True})
        if self.mode == "hang":
            await asyncio.sleep(10)
        n = len(body["documents"])
        order = list(reversed(range(n)))[: body["top_n"]]  # the last candidate is the best
        return httpx.Response(200, json={"results": [{"index": i, "relevance_score": i / 100} for i in order]})


@pytest.fixture
def transport():
    t = _Transport()
    client = httpx.AsyncClient(transport=httpx.MockTransport(t))
    http_clients._set_clients_for_tests(client)
    yield t
    http_clients._set_clients_for_tests(None)


@pytest.fixture
def world(db, fake_app, transport, monkeypatch):
    _Recorder.successes, _Recorder.failures = [], []
    rt._failure_last.clear()
    configure_factories(db)
    emb = EmbeddingServiceFactory(app=fake_app)
    silo = Silo(name="Rerank silo", silo_type="REPO", app_id=fake_app.app_id, vector_db_type="PGVECTOR",
                embedding_service_id=emb.service_id)
    db.add(silo)
    svc_a = RerankService(name="A", provider="CohereCompatible", model_name="m", endpoint=HOST_A, app_id=fake_app.app_id)
    svc_b = RerankService(name="B", provider="CohereCompatible", model_name="m", endpoint=HOST_B, app_id=fake_app.app_id)
    db.add_all([svc_a, svc_b])
    db.flush()
    fake_app.default_rerank_service_id = svc_a.service_id
    db.flush()
    monkeypatch.setattr(config, "RERANK_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(config, "RERANK_CONNECT_TIMEOUT_SECONDS", 0.2)
    proxy = _SessionProxy(db)
    with patch("services.silo_service.SessionLocal", lambda: proxy), \
         patch("services.silo_service._get_vector_store", return_value=MagicMock()), \
         patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=True), \
         patch("services.rerank_health.DbRerankHealthRecorder", _Recorder), \
         patch("tools.retrieval.search_methods.search_method_factory.SearchMethodFactory.get_search_method",
               return_value=_FakeMethod()):
        yield SimpleNamespace(db=db, app=fake_app, silo=silo, a=svc_a, b=svc_b, transport=transport)


def _agent(**kw):
    base = dict(rag_k=8, rag_search_type="similarity", rag_score_threshold=None, rag_fixed_filters=None,
                rag_max_retrieval_calls=None, rag_search_method=["dense"], rag_rerank_enabled=True,
                rag_rerank_service_id=None, rag_rerank_top_n=5, silo=None)
    base.update(kw)
    return SimpleNamespace(**base)


async def _agent_retrieve(world, agent, caller=None):
    sp, _ = resolve_search_params(agent, caller or {})
    retriever = await asyncio.to_thread(SiloService.get_silo_retriever, world.silo.silo_id, sp)
    return await retriever.ainvoke("question")


def _agent_tool(world, agent):
    from tools.agentTools import get_retriever_tool
    sp, pinned = resolve_search_params(agent, {})
    return get_retriever_tool(world.silo, sp, pinned_filter=pinned)


# ---------------------------------------------------------------- AC-9
@pytest.mark.asyncio
@pytest.mark.parametrize("methods", [["dense"], ["bm25"], ["dense", "bm25"]])
async def test_agent_retrieval_returns_top_n_reranked_docs(world, methods):
    docs = await _agent_retrieve(world, _agent(rag_search_method=methods))

    assert len(docs) == 5  # min(top_n, candidates)
    assert docs[0].page_content == "doc-7"  # candidates = max(rag_k, top_n) = 8; last one scored best
    assert world.transport.docs_sent == [8]


@pytest.mark.asyncio
async def test_top_n_larger_than_candidates_returns_all_candidates(world):
    docs = await _agent_retrieve(world, _agent(rag_k=3, rag_rerank_top_n=50))

    assert len(docs) == N_CANDIDATES  # candidates = max(3, 50) = 50, first stage only has 12
    assert world.transport.docs_sent == [N_CANDIDATES]


@pytest.mark.asyncio
async def test_rerank_disabled_agent_never_calls_the_reranker(world):
    docs = await _agent_retrieve(world, _agent(rag_rerank_enabled=False, rag_k=8))

    assert len(docs) == 8
    assert world.transport.hosts == []


# ---------------------------------------------------------------- AC-11
@pytest.mark.asyncio
async def test_changing_the_app_default_applies_to_the_next_retrieval(world):
    agent = _agent()
    await _agent_retrieve(world, agent)
    world.app.default_rerank_service_id = world.b.service_id
    world.db.flush()
    await _agent_retrieve(world, agent)

    assert world.transport.hosts == [HOST_A, HOST_B]


# ---------------------------------------------------------------- playground
def _search(world, **kw):
    return SiloService.search_silo_documents_router(world.silo.silo_id, "question", db=world.db, **kw)


def test_playground_reranked_search_reports_service_and_scores(world):
    result = _search(world, rerank=True, rerank_top_n=4, limit=10)

    assert result["rerank"]["requested"] is True
    assert result["rerank"]["applied"] is True
    assert result["rerank"]["service_id"] == world.a.service_id
    assert result["rerank"]["service_name"] == "A"
    assert result["total_results"] == 4
    assert [r["rerank_score"] for r in result["results"]] == sorted(
        [r["rerank_score"] for r in result["results"]], reverse=True
    )
    assert result["results"][0]["page_content"] == "doc-9"  # candidates = max(10, 4)


def test_playground_panels_with_different_settings_are_independent(world):
    first = _search(world, rerank=True, rerank_top_n=3, limit=10)
    second = _search(world, rerank=True, rerank_service_id=world.b.service_id, rerank_top_n=6, limit=10)
    third = _search(world, rerank=False, limit=10)

    assert (first["total_results"], first["rerank"]["service_id"]) == (3, world.a.service_id)
    assert (second["total_results"], second["rerank"]["service_id"]) == (6, world.b.service_id)
    assert "rerank" not in third
    assert all(r["rerank_score"] is None for r in third["results"])
    assert world.transport.hosts == [HOST_A, HOST_B]


def test_playground_unknown_service_is_a_validation_error(world):
    with pytest.raises(ValidationError):
        _search(world, rerank=True, rerank_service_id=999999)
    assert world.transport.hosts == []


def test_playground_service_of_another_app_is_not_visible(world, db):
    other_app = type(world.app)(name="Other", slug="other-rerank-app", owner_id=world.app.owner_id,
                                agent_rate_limit=0, max_file_size_mb=10)
    db.add(other_app)
    db.flush()
    foreign = RerankService(name="F", provider="CohereCompatible", model_name="m", endpoint=HOST_B,
                            app_id=other_app.app_id)
    db.add(foreign)
    db.flush()

    with pytest.raises(ValidationError):
        _search(world, rerank=True, rerank_service_id=foreign.service_id)


# ---------------------------------------------------------------- AC-14
FAILURES = [("500", "service_error"), ("invalid", "service_error"), ("hang", "timeout")]
BUDGET_S = 0.4 + 1.5  # patched total timeout + margin


@pytest.mark.parametrize("mode,reason", FAILURES)
def test_playground_survives_a_failing_reranker(world, mode, reason):
    world.transport.mode = mode
    start = time.monotonic()
    result = _search(world, rerank=True, rerank_top_n=5, limit=10)
    elapsed = time.monotonic() - start

    assert elapsed < BUDGET_S
    assert result["rerank"]["applied"] is False
    assert result["rerank"]["skipped_reason"] == reason
    assert [r["page_content"] for r in result["results"]] == [f"doc-{i}" for i in range(5)]  # first-stage order
    assert all(r["rerank_score"] is None for r in result["results"])
    assert [sid for sid, _ in _Recorder.failures] == [world.a.service_id]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,reason", FAILURES)
async def test_agent_tool_survives_a_failing_reranker(world, mode, reason):
    world.transport.mode = mode
    tool = _agent_tool(world, _agent())
    start = time.monotonic()
    content, docs = await tool.coroutine("question")
    elapsed = time.monotonic() - start

    assert elapsed < BUDGET_S
    assert [d.page_content for d in docs] == [f"doc-{i}" for i in range(5)]  # first-stage order cut to top N
    assert content.startswith("5 results")
    assert [sid for sid, _ in _Recorder.failures] == [world.a.service_id]


# ---------------------------------------------------------------- AC-15
@pytest.mark.asyncio
async def test_retired_explicit_service_is_not_substituted(world, caplog):
    from datetime import datetime, timedelta, timezone
    world.b.end_of_service_at = datetime.now(timezone.utc) - timedelta(days=1)
    world.db.flush()

    with caplog.at_level("WARNING"):
        docs = await _agent_retrieve(world, _agent(rag_rerank_service_id=world.b.service_id))

    assert world.transport.hosts == []  # neither B (retired) nor the app default A was called
    assert [d.page_content for d in docs] == [f"doc-{i}" for i in range(5)]
    assert any("service_retired" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------- review-board fixes
def test_playground_returns_top_n_even_when_limit_is_smaller(world):
    result = _search(world, rerank=True, rerank_top_n=5, limit=2)

    assert result["total_results"] == 5
    assert result["rerank"]["applied"] is True
    assert world.transport.docs_sent == [5]  # candidates = max(limit, top_n)


def test_playground_without_limit_does_not_request_default_candidates(world):
    _search(world, rerank=True, rerank_top_n=3)

    assert world.transport.docs_sent == [3]


@pytest.mark.asyncio
async def test_candidate_count_is_clamped_to_the_server_cap(world, monkeypatch):
    monkeypatch.setattr(config, "RERANK_MAX_CANDIDATES", 6)
    docs = await _agent_retrieve(world, _agent(rag_k=8, rag_rerank_top_n=5))

    assert world.transport.docs_sent == [6]
    assert len(docs) == 5


def test_resolution_failure_never_fails_the_search(world, caplog):
    with patch("services.rerank_resolution_service.RerankResolutionService.resolve",
               side_effect=RuntimeError("db exploded secret-query")), caplog.at_level("WARNING"):
        result = _search(world, rerank=True, rerank_top_n=4, limit=10)

    assert [r["page_content"] for r in result["results"]] == [f"doc-{i}" for i in range(4)]
    assert result["rerank"]["applied"] is False
    assert result["rerank"]["skipped_reason"] == "service_error"
    assert world.transport.hosts == []
    assert "secret-query" not in caplog.text


def test_missing_collection_still_reports_the_rerank_status(world):
    with patch("services.silo_service.SiloService.check_silo_collection_exists", return_value=False):
        result = _search(world, rerank=True)

    assert result["results"] == []
    assert result["rerank"] == {
        "requested": True, "applied": False, "service_id": None, "service_name": None,
        "source": None, "skipped_reason": "no_candidates", "latency_ms": None,
    }


# ---------------------------------------------------------------- router level (422, no query in logs)
def _post(client, world, headers, **body):
    return client.post(
        f"/internal/apps/{world.app.app_id}/silos/{world.silo.silo_id}/search",
        json={"query": "top-secret-question", "rerank": True, **body}, headers=headers,
    )


def test_router_unknown_rerank_service_is_422(world, client, app_viewer_headers, caplog):
    with caplog.at_level("INFO"):
        response = _post(client, world, app_viewer_headers, rerank_service_id=999999)

    assert response.status_code == 422
    assert response.json()["detail"] == "Rerank service 999999 not found"
    assert "top-secret-question" not in caplog.text


def test_router_foreign_rerank_service_is_422_with_the_same_message(world, client, app_viewer_headers, db):
    other = type(world.app)(name="Other", slug="other-rerank-router", owner_id=world.app.owner_id,
                            agent_rate_limit=0, max_file_size_mb=10)
    db.add(other)
    db.flush()
    foreign = RerankService(name="F", provider="CohereCompatible", model_name="m", endpoint=HOST_B,
                            app_id=other.app_id)
    db.add(foreign)
    db.flush()

    response = _post(client, world, app_viewer_headers, rerank_service_id=foreign.service_id)

    assert response.status_code == 422
    assert response.json()["detail"] == f"Rerank service {foreign.service_id} not found"


def test_router_reranked_search_logs_no_query_text(world, client, app_viewer_headers, caplog):
    with caplog.at_level("INFO"):
        response = _post(client, world, app_viewer_headers, rerank_top_n=3, filter_metadata={"secret_field_value": "x"})

    assert response.status_code == 200
    assert response.json()["rerank"]["applied"] is True
    assert response.json()["total_results"] == 3
    assert "top-secret-question" not in caplog.text
