"""Direct tests of the LightRAG partial-delete guard."""

import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import services.silo_service as silo_service_module
from services.silo_service import (
    LIGHTRAG_UNSUPPORTED_DELETE_DETAIL,
    LIGHTRAG_UNSUPPORTED_REINDEX_DETAIL,
    ensure_partial_delete_supported,
    ensure_reindex_supported,
)


@pytest.fixture(autouse=True)
def _silo_service_caplog(caplog):
    """The project logger does not propagate to root, so attach caplog's handler directly."""
    silo_service_module.logger.addHandler(caplog.handler)
    yield
    silo_service_module.logger.removeHandler(caplog.handler)


@pytest.mark.parametrize("silo", [None, SimpleNamespace(silo_id=1, vector_db_type="PGVECTOR"),
                                  SimpleNamespace(silo_id=1, vector_db_type="QDRANT")])
def test_supported_silos_do_not_raise(silo):
    ensure_partial_delete_supported(silo)


def test_lightrag_silo_raises_409_and_warns(caplog):
    silo = SimpleNamespace(silo_id=7, vector_db_type="LIGHTRAG")
    with caplog.at_level(logging.WARNING), pytest.raises(HTTPException) as exc:
        ensure_partial_delete_supported(silo)
    assert exc.value.status_code == 409
    assert exc.value.detail == LIGHTRAG_UNSUPPORTED_DELETE_DETAIL
    assert any(r.levelno == logging.WARNING and "LightRAG silo 7" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("silo", [None, SimpleNamespace(silo_id=1, vector_db_type="PGVECTOR"),
                                  SimpleNamespace(silo_id=1, vector_db_type="QDRANT")])
def test_reindex_is_supported_outside_lightrag(silo):
    ensure_reindex_supported(silo)


def test_reindex_on_a_lightrag_silo_raises_409():
    with pytest.raises(HTTPException) as exc:
        ensure_reindex_supported(SimpleNamespace(silo_id=7, vector_db_type="LIGHTRAG"))
    assert exc.value.status_code == 409
    assert exc.value.detail == LIGHTRAG_UNSUPPORTED_REINDEX_DETAIL
