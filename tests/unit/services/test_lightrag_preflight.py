"""Pre-flight test calls to a LightRAG silo's extract LLM and embedding service."""

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from services.silo_service import ensure_lightrag_services_reachable
from utils.service_lifecycle import ServiceRetiredError


class _AuthError(Exception):
    status_code = 401


def _silo(extract="llm", embedding="emb", db_type="LIGHTRAG"):
    svc = lambda name: name and SimpleNamespace(service_id=1, name=name, last_error=None)  # noqa: E731
    return SimpleNamespace(silo_id=7, vector_db_type=db_type, extract_service=svc(extract),
                           indexing_service=None, embedding_service=svc(embedding))


def _check(silo, llm=None, embed=None):
    """Run the check with the LLM/embedding calls mocked; ``llm``/``embed`` are side effects."""
    with patch("tools.aiServiceTools.create_llm_from_service") as build, \
            patch("tools.aiServiceTools.get_embedding", side_effect=embed) as embedding, \
            patch("services.silo_service.ServiceHealthService") as health:
        build.return_value.invoke.side_effect = llm
        ensure_lightrag_services_reachable(silo)
        return build, embedding, health


def test_healthy_services_pass_and_each_is_called_once():
    build, embedding, health = _check(_silo())
    build.return_value.invoke.assert_called_once()
    embedding.assert_called_once()
    assert health.record_success.call_count == 2


@pytest.mark.parametrize("silo", [None, _silo(db_type="PGVECTOR")])
def test_non_lightrag_silos_are_not_tested(silo):
    build, embedding, _ = _check(silo)
    build.assert_not_called()
    embedding.assert_not_called()


def test_rejected_llm_key_is_503_naming_the_service_and_recorded():
    with pytest.raises(HTTPException) as exc:
        _check(_silo(), llm=_AuthError("bad key sk-secret"))
    assert exc.value.status_code == 503
    assert "Extract AI service 'llm'" in exc.value.detail and "credentials" in exc.value.detail
    assert "sk-secret" not in exc.value.detail


def test_failure_is_recorded_on_the_service():
    with patch("tools.aiServiceTools.create_llm_from_service") as build, \
            patch("tools.aiServiceTools.get_embedding"), \
            patch("services.silo_service.ServiceHealthService") as health:
        build.return_value.invoke.side_effect = _AuthError()
        with pytest.raises(HTTPException):
            ensure_lightrag_services_reachable(_silo())
    health.record_failure.assert_called_once()


def test_embedding_failure_is_503():
    with pytest.raises(HTTPException) as exc:
        _check(_silo(), embed=_AuthError())
    assert exc.value.status_code == 503 and "Embedding service 'emb'" in exc.value.detail


def test_missing_extract_service_is_503_without_calls():
    with patch("tools.aiServiceTools.create_llm_from_service") as build, \
            patch("tools.aiServiceTools.get_embedding") as embedding:
        with pytest.raises(HTTPException) as exc:
            ensure_lightrag_services_reachable(_silo(extract=None))
    assert exc.value.status_code == 503 and "no Extract AI service" in exc.value.detail
    build.assert_not_called()
    embedding.assert_not_called()


def test_hung_service_times_out_as_503_and_is_not_recorded():
    """A provider that accepts the connection but never answers: the probe times out and
    raises 503 — deliberately NOT recorded on the service (a timeout is transient, its
    fix is not a service-config change). The sleep keeps the worker thread alive past the
    probe deadline, then the pool is abandoned (wait=False) as in production."""
    with patch("tools.aiServiceTools.create_llm_from_service") as build, \
            patch("tools.aiServiceTools.get_embedding") as embedding, \
            patch("services.silo_service.ServiceHealthService") as health, \
            patch("services.silo_service.LIGHTRAG_PREFLIGHT_TIMEOUT_SECONDS", 0.01):
        build.return_value.invoke.side_effect = lambda _s: time.sleep(0.3)
        with pytest.raises(HTTPException) as exc:
            ensure_lightrag_services_reachable(_silo())
    assert exc.value.status_code == 503
    assert "did not answer within" in exc.value.detail
    health.record_failure.assert_not_called()


def test_retired_service_propagates_for_the_409_handler():
    with pytest.raises(ServiceRetiredError):
        _check(_silo(), llm=ServiceRetiredError(
            SimpleNamespace(name="llm", end_of_service_at=datetime(2020, 1, 1, tzinfo=timezone.utc)), "ai"))
