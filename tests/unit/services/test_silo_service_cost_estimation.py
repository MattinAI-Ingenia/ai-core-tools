"""Unit tests for ``SiloService.estimate_indexing_cost``."""

from unittest.mock import MagicMock, patch

import pytest

from services.silo_service import SiloService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lightrag_silo(chunk_size=1200, model_name="gpt-4o"):
    silo = MagicMock()
    silo.vector_db_type = "LIGHTRAG"
    silo.lightrag_chunk_token_size = chunk_size
    silo.indexing_service = MagicMock()
    silo.indexing_service.model_name = model_name
    silo.embedding_service = MagicMock()
    silo.embedding_service.model_name = "text-embedding-3-small"
    return silo


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_estimate_cost_basic_calculation():
    silo = _make_lightrag_silo(chunk_size=1200)
    db = MagicMock()

    # 4800 chars → 1200 tokens → 1 chunk (1200 / 1200)
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 1
    assert result["chunk_token_size"] == 1200
    assert result["estimated_llm_calls"] == 2        # chunks * 2
    assert result["estimated_embedding_calls"] == 1   # chunks * 1
    assert result["estimated_input_tokens"] == 1200   # chunks * chunk_size
    assert result["estimated_output_tokens"] == 500   # chunks * 500
    assert result["warnings"] == []


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_estimate_cost_silo_not_found():
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=None):
        with pytest.raises(LookupError, match="not found"):
            SiloService.estimate_indexing_cost(999, [], db)


def test_estimate_cost_not_lightrag_silo():
    silo = MagicMock()
    silo.vector_db_type = "PGVECTOR"
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        with pytest.raises(ValueError, match="LightRAG"):
            SiloService.estimate_indexing_cost(1, [], db)


def test_estimate_cost_no_indexing_service():
    silo = _make_lightrag_silo()
    silo.indexing_service = None
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        with pytest.raises(ValueError, match="indexing"):
            SiloService.estimate_indexing_cost(1, [], db)


def test_estimate_cost_no_embedding_service():
    silo = _make_lightrag_silo()
    silo.embedding_service = None
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        with pytest.raises(ValueError, match="embedding"):
            SiloService.estimate_indexing_cost(1, [], db)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def test_estimate_cost_warns_on_small_model():
    silo = _make_lightrag_silo(model_name="gpt-4o-mini")
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert len(result["warnings"]) > 0
    assert "gpt-4o-mini" in result["warnings"][0]


def test_estimate_cost_no_warning_on_large_model():
    silo = _make_lightrag_silo(model_name="gpt-4o")
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["warnings"] == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_estimate_cost_default_chunk_size():
    silo = _make_lightrag_silo()
    silo.lightrag_chunk_token_size = None
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["chunk_token_size"] == 1200


def test_estimate_cost_empty_documents():
    silo = _make_lightrag_silo()
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, [], db)

    # max(1, 0) = 1
    assert result["total_chunks"] == 1
