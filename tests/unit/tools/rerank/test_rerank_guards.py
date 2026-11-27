"""Gate follow-ups: per-target in-flight cap and rerank_sync called from a running event loop."""
import asyncio
import logging

import pytest

from tools.rerank import client
from unittest.mock import patch

from tools.rerank.errors import RerankBusyError, RerankLocalError
from tools.rerank.types import RerankTarget


def _target(service_id=1):
    return RerankTarget(service_id=service_id, app_id=None, name="s", provider="CohereCompatible",
                        endpoint="https://x.example.test", model="m")


def test_inflight_cap_rejects_with_busy_and_releases_slots():
    target = _target()
    slots = []
    for _ in range(client.MAX_INFLIGHT_PER_TARGET):
        cm = client._inflight_slot(target)
        cm.__enter__()
        slots.append(cm)

    with pytest.raises(RerankBusyError) as exc:
        with client._inflight_slot(target):
            pass
    assert exc.value.service_failure is False  # a local limit never marks the service unhealthy

    with client._inflight_slot(_target(service_id=2)):  # other targets are unaffected
        pass

    for cm in slots:
        cm.__exit__(None, None, None)
    with client._inflight_slot(target):  # capacity is back
        pass
    assert not client._inflight


@pytest.mark.asyncio
async def test_rerank_sync_from_running_loop_warns(caplog, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("stop before any network")

    monkeypatch.setattr(client.asyncio, "run_coroutine_threadsafe", _boom)
    client._warned_sites.clear()
    with caplog.at_level(logging.WARNING, logger=client.logger.name):
        for _ in range(3):  # same call site: logged once
            with pytest.raises(RuntimeError):
                client.rerank_sync(_target(), "q", ["a"], 1)

    assert len([r for r in caplog.records if "running event loop" in r.getMessage()]) == 1


def _system_target(service_id=1):
    return RerankTarget(service_id=service_id, app_id=None, name="s", provider="CohereCompatible",
                        endpoint="https://x.example.test", model="m", system=True)


def test_shared_service_has_a_per_app_sub_limit():
    target = _system_target()
    held = []
    for _ in range(client._SUB_LIMIT):
        cm = client._inflight_slot(target, caller_app_id=1)
        cm.__enter__()
        held.append(cm)

    with pytest.raises(RerankBusyError):  # app 1 is at its sub-limit
        with client._inflight_slot(target, caller_app_id=1):
            pass
    with client._inflight_slot(target, caller_app_id=2):  # app 2 is not blocked
        pass

    for cm in held:
        cm.__exit__(None, None, None)
    assert not client._inflight


def test_unmapped_exception_is_a_local_error():
    mapped = client._map_exc(ValueError("x"))
    assert isinstance(mapped, RerankLocalError) and mapped.service_failure is False


def test_local_errors_never_reach_the_health_recorder():
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from tools.retrieval import rerank_transformer as rt

    class _First(BaseRetriever):
        def _get_relevant_documents(self, query, *, run_manager):
            return [Document(page_content="a"), Document(page_content="b")]

    class _Rec:
        failures: list = []

        def record_success(self, target):
            pass

        def record_failure(self, target, message):
            self.failures.append(message)

    rec = _Rec()
    for exc in (RerankLocalError("misuse"), RerankLocalError("Rerank call was cancelled")):
        rt._failure_last.clear()
        plan = rt.RerankPlan(top_n=1, target=_target(), health_recorder=rec)
        with patch.object(rt, "rerank_sync", side_effect=exc):
            docs = rt.RerankRetriever(base_retriever=_First(), plan=plan).invoke("q")
        assert [d.page_content for d in docs] == ["a"]
        assert plan.outcome.applied is False

    assert rec.failures == []
