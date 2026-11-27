"""Unit tests for the rerank stage (RerankRetriever)."""
import asyncio
import logging
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tools.rerank.errors import (
    InvalidRerankResponseError, RerankBusyError, RerankHTTPError, RerankTimeoutError,
    UnknownRerankProviderError,
)
from tools.rerank.types import RerankResult, RerankTarget
from tools.retrieval import rerank_transformer as rt
from tools.retrieval.rerank_transformer import RerankPlan, RerankRetriever
from tools.retrieval.retrieval_context import RetrievalContext
from tools.retrieval.retrieval_pipeline import RetrievalPipeline

MOD = "tools.retrieval.rerank_transformer"
SECRET_QUERY = "secret-query-text"
SECRET_DOC = "secret-doc-text"


class _Fake(BaseRetriever):
    docs: List[Document]

    def _get_relevant_documents(self, query, *, run_manager):
        return list(self.docs)


def _docs(n, prefix="d"):
    return [Document(page_content=f"{SECRET_DOC} {prefix}{i}", metadata={"i": i}) for i in range(n)]


def _target(last_error="prev"):
    return RerankTarget(service_id=5, app_id=9, name="svc", provider="cohere",
                        endpoint="https://x.example", model="m", api_key="KEY", last_error=last_error)


@pytest.fixture(autouse=True)
def _clear_throttle():
    rt._failure_last.clear()
    yield
    rt._failure_last.clear()



def _reverse(target, query, documents, top_n):
    n = len(documents)
    return [RerankResult(index=i, score=float(i)) for i in reversed(range(n))][:top_n]


def _plan(top_n=3, limit=None, recorder=None, target="default"):
    return RerankPlan(top_n=top_n, candidate_limit=limit,
                      target=_target() if target == "default" else target,
                      source="agent", health_recorder=recorder)


def _fused(n):
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
    a, b = _Fake(docs=_docs(n, "a")), _Fake(docs=_docs(n, "b"))
    return EnsembleRetriever(retrievers=[a, b], weights=[0.5, 0.5])


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("n_cand,top_n", [(8, 3), (2, 5)])
def test_reranked_order_and_count_sync(method, n_cand, top_n):
    base = _fused(n_cand) if method == "hybrid" else _Fake(docs=_docs(n_cand, method))
    plan = _plan(top_n=top_n, limit=None)
    r = RerankRetriever(base_retriever=base, plan=plan)
    with patch(f"{MOD}.rerank_sync", side_effect=_reverse):
        out = r.invoke(SECRET_QUERY)
    first = base.invoke(SECRET_QUERY)
    assert len(out) == min(top_n, len(first))
    assert out == list(reversed(first))[:top_n]
    assert plan.outcome.applied and plan.outcome.service_id == 5 and plan.outcome.source == "agent"
    assert len(plan.outcome.scores) == len(out)


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
async def test_reranked_order_async(method):
    base = _fused(6) if method == "hybrid" else _Fake(docs=_docs(6))
    plan = _plan(top_n=4)
    r = RerankRetriever(base_retriever=base, plan=plan)
    with patch(f"{MOD}.rerank_async", new=AsyncMock(side_effect=_reverse)):
        out = await r.ainvoke("q")
    first = await base.ainvoke("q")
    assert out == list(reversed(first))[:4]


def test_candidates_truncated_to_candidate_limit_for_hybrid():
    seen = {}

    def spy(target, query, documents, top_n):
        seen["n"] = len(documents)
        return _reverse(target, query, documents, top_n)

    r = RerankRetriever(base_retriever=_fused(10), plan=_plan(top_n=3, limit=7))
    with patch(f"{MOD}.rerank_sync", side_effect=spy):
        assert len(r.invoke("q")) == 3
    assert seen["n"] == 7


async def test_candidates_truncated_async():
    seen = {}

    async def spy(target, query, documents, top_n):
        seen["n"] = len(documents)
        return _reverse(target, query, documents, top_n)

    r = RerankRetriever(base_retriever=_fused(10), plan=_plan(top_n=3, limit=7))
    with patch(f"{MOD}.rerank_async", new=spy):
        await r.ainvoke("q")
    assert seen["n"] == 7


ERRORS = [
    (RerankHTTPError("HTTP 500", status_code=500), "service_error"),
    (RerankTimeoutError("timed out"), "timeout"),
    (InvalidRerankResponseError("bad body"), "service_error"),
    (UnknownRerankProviderError("unknown provider"), "service_error"),
]


@pytest.mark.parametrize("exc", [RuntimeError("boom " + SECRET_QUERY), ValueError("bug")])
def test_non_rerank_exception_falls_back_without_touching_health(exc, caplog):
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    base = _Fake(docs=_docs(5))
    with patch(f"{MOD}.rerank_sync", side_effect=exc), caplog.at_level(logging.DEBUG, logger=MOD):
        out = RerankRetriever(base_retriever=base, plan=plan).invoke(SECRET_QUERY)
    assert out == base.docs[:2]
    assert not plan.outcome.applied and plan.outcome.skipped_reason == "service_error"
    rec.record_failure.assert_not_called()
    assert SECRET_QUERY not in caplog.text


def test_service_failure_rerank_error_still_records():
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    err = RerankHTTPError("HTTP 500", status_code=500)
    assert err.service_failure is True
    with patch(f"{MOD}.rerank_sync", side_effect=err):
        RerankRetriever(base_retriever=_Fake(docs=_docs(5)), plan=plan).invoke("q")
    rec.record_failure.assert_called_once()


@pytest.mark.parametrize("exc,reason", ERRORS)
def test_failure_falls_back_sync(exc, reason, caplog):
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    base = _Fake(docs=_docs(5))
    with patch(f"{MOD}.rerank_sync", side_effect=exc), caplog.at_level(logging.DEBUG, logger=MOD):
        out = RerankRetriever(base_retriever=base, plan=plan).invoke(SECRET_QUERY)
    assert out == base.docs[:2]
    assert not plan.outcome.applied and plan.outcome.skipped_reason == reason
    rec.record_failure.assert_called_once()
    rec.record_success.assert_not_called()
    warnings = [x for x in caplog.records if x.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = caplog.text
    assert SECRET_QUERY not in text and SECRET_DOC not in text and "KEY" not in text.replace("KEY_", "")
    assert type(exc).__name__ in warnings[0].getMessage()


@pytest.mark.parametrize("exc,reason", ERRORS)
async def test_failure_falls_back_async(exc, reason):
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    base = _Fake(docs=_docs(5))
    with patch(f"{MOD}.rerank_async", new=AsyncMock(side_effect=exc)):
        out = await RerankRetriever(base_retriever=base, plan=plan).ainvoke("q")
    assert out == base.docs[:2]
    assert plan.outcome.skipped_reason == reason
    rec.record_failure.assert_called_once()


def test_busy_error_does_not_record_health():
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    with patch(f"{MOD}.rerank_sync", side_effect=RerankBusyError("pool busy")):
        out = RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).invoke("q")
    assert len(out) == 2 and plan.outcome.skipped_reason == "busy"
    rec.record_failure.assert_not_called()


def test_success_records_health_and_recorder_failure_is_swallowed():
    rec = MagicMock()
    rec.record_success.side_effect = RuntimeError("db down")
    plan = _plan(top_n=2, recorder=rec)
    with patch(f"{MOD}.rerank_sync", side_effect=_reverse):
        out = RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).invoke("q")
    assert len(out) == 2 and plan.outcome.applied
    rec.record_success.assert_called_once()


def test_recorder_failure_in_fallback_is_swallowed():
    rec = MagicMock()
    rec.record_failure.side_effect = RuntimeError("db down")
    plan = _plan(top_n=2, recorder=rec)
    with patch(f"{MOD}.rerank_sync", side_effect=RerankHTTPError("HTTP 500", status_code=500)):
        out = RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).invoke("q")
    assert len(out) == 2


@pytest.mark.parametrize("reason", ["not_configured", "service_unavailable", "service_retired"])
def test_skip_plan_makes_no_call(reason, caplog):
    plan = _plan(top_n=2, target=None)
    plan.skip_reason = reason
    base = _Fake(docs=_docs(4))
    with patch(f"{MOD}.rerank_sync") as call, patch(f"{MOD}.rerank_async") as acall, \
         caplog.at_level(logging.INFO, logger=MOD):
        out = RerankRetriever(base_retriever=base, plan=plan).invoke("q")
    call.assert_not_called()
    acall.assert_not_called()
    assert out == base.docs[:2] and plan.outcome.skipped_reason == reason
    levels = {x.levelno for x in caplog.records}
    assert (logging.WARNING in levels) == (reason != "not_configured")


def test_no_candidates():
    plan = _plan()
    with patch(f"{MOD}.rerank_sync") as call:
        out = RerankRetriever(base_retriever=_Fake(docs=[]), plan=plan).invoke("q")
    call.assert_not_called()
    assert out == [] and plan.outcome.skipped_reason == "no_candidates"


def test_cancelled_error_propagates():
    plan = _plan(recorder=MagicMock())
    with patch(f"{MOD}.rerank_sync", side_effect=asyncio.CancelledError()):
        with pytest.raises(asyncio.CancelledError):
            RerankRetriever(base_retriever=_Fake(docs=_docs(3)), plan=plan).invoke("q")
    plan.health_recorder.record_failure.assert_not_called()


async def test_cancelled_error_propagates_async():
    plan = _plan(recorder=MagicMock())
    with patch(f"{MOD}.rerank_async", new=AsyncMock(side_effect=asyncio.CancelledError())):
        with pytest.raises(asyncio.CancelledError):
            await RerankRetriever(base_retriever=_Fake(docs=_docs(3)), plan=plan).ainvoke("q")
    plan.health_recorder.record_failure.assert_not_called()


def test_pipeline_builds_rerank_retriever():
    base = _Fake(docs=_docs(4))
    m = MagicMock()
    m.build.return_value = base
    ctx = RetrievalContext(rerank=_plan(top_n=2))
    with patch("tools.retrieval.retrieval_pipeline.SearchMethodFactory.get_search_method", return_value=m):
        r = RetrievalPipeline.build(ctx, ["dense"])
    with patch(f"{MOD}.rerank_sync", side_effect=_reverse):
        assert r.invoke("q") == list(reversed(base.docs))[:2]


def test_outcome_reset_between_calls():
    plan = _plan(top_n=2)
    base = _Fake(docs=_docs(4))
    r = RerankRetriever(base_retriever=base, plan=plan)
    with patch(f"{MOD}.rerank_sync", side_effect=_reverse):
        r.invoke("q")
    assert plan.outcome.scores and plan.outcome.applied
    base.docs = []
    r.invoke("q")
    assert plan.outcome.scores == [] and plan.outcome.service_id is None
    assert plan.outcome.latency_ms is None and plan.outcome.skipped_reason == "no_candidates"


def test_latency_set_on_success_and_timeout():
    plan = _plan(top_n=2)
    r = RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan)
    with patch(f"{MOD}.rerank_sync", side_effect=_reverse):
        r.invoke("q")
    assert plan.outcome.latency_ms is not None
    with patch(f"{MOD}.rerank_sync", side_effect=RerankTimeoutError("t")):
        r.invoke("q")
    assert plan.outcome.latency_ms is not None and plan.outcome.skipped_reason == "timeout"


def test_success_not_recorded_when_service_already_healthy():
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec, target=_target(last_error=None))
    with patch(f"{MOD}.rerank_sync", side_effect=_reverse):
        RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).invoke("q")
    rec.record_success.assert_not_called()


def test_failure_recording_is_throttled_per_service():
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    r = RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan)
    with patch(f"{MOD}.rerank_sync", side_effect=RerankHTTPError("HTTP 500", status_code=500)):
        for _ in range(3):
            r.invoke("q")
    assert rec.record_failure.call_count == 1


def test_fallback_warning_has_ad11_fields(caplog):
    plan = _plan(top_n=2)
    with patch(f"{MOD}.rerank_sync", side_effect=RerankTimeoutError("t")), caplog.at_level(logging.WARNING, logger=MOD):
        RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).invoke(SECRET_QUERY)
    msg = [x for x in caplog.records if x.levelno == logging.WARNING][0].getMessage()
    for field in ("service_id=5", "app_id=9", "provider=cohere", "error_class=RerankTimeoutError", "latency_ms=", "candidates=4"):
        assert field in msg


def test_skip_log_carries_ids(caplog):
    plan = _plan(top_n=2, target=None)
    plan.skip_reason, plan.skip_service_id, plan.app_id = "service_retired", 77, 9
    with caplog.at_level(logging.INFO, logger=MOD):
        RerankRetriever(base_retriever=_Fake(docs=_docs(3)), plan=plan).invoke("q")
    assert "service_id=77" in caplog.text and "app_id=9" in caplog.text


async def test_busy_does_not_record_health_async():
    rec = MagicMock()
    plan = _plan(top_n=2, recorder=rec)
    with patch(f"{MOD}.rerank_async", new=AsyncMock(side_effect=RerankBusyError("pool busy"))):
        out = await RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).ainvoke("q")
    assert len(out) == 2 and plan.outcome.skipped_reason == "busy"
    rec.record_failure.assert_not_called()


async def test_recorder_failure_in_fallback_is_swallowed_async():
    rec = MagicMock()
    rec.record_failure.side_effect = RuntimeError("db down")
    plan = _plan(top_n=2, recorder=rec)
    with patch(f"{MOD}.rerank_async", new=AsyncMock(side_effect=RerankHTTPError("HTTP 500", status_code=500))):
        out = await RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).ainvoke("q")
    assert len(out) == 2 and plan.outcome.skipped_reason == "service_error"


async def test_success_recorder_failure_is_swallowed_async():
    rec = MagicMock()
    rec.record_success.side_effect = RuntimeError("db down")
    plan = _plan(top_n=2, recorder=rec)
    with patch(f"{MOD}.rerank_async", new=AsyncMock(side_effect=_reverse)):
        out = await RerankRetriever(base_retriever=_Fake(docs=_docs(4)), plan=plan).ainvoke("q")
    assert len(out) == 2 and plan.outcome.applied
    rec.record_success.assert_called_once()
