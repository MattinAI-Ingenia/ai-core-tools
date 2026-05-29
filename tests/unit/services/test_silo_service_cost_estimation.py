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
    silo.lightrag_chunk_overlap_token_size = 0
    silo.lightrag_chunk_strategy = None
    silo.llm_model_max_sync = 4
    silo.embedding_model_max_sync = 8
    # Legacy indexing service — used as fallback for extract_service.
    silo.indexing_service = MagicMock()
    silo.indexing_service.description = None
    silo.indexing_service.model_name = model_name
    silo.indexing_service.name = model_name
    silo.embedding_service = MagicMock()
    silo.embedding_service.description = None
    silo.embedding_service.model_name = "text-embedding-3-small"
    silo.embedding_service.name = "text-embedding-3-small"
    # Role-specific services — None means "not configured, use fallback".
    silo.extract_service = None
    silo.query_service = None
    silo.keywords_service = None
    silo.vlm_service = None
    return silo


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_estimate_cost_basic_calculation():
    silo = _make_lightrag_silo(chunk_size=1200)
    db = MagicMock()

    # 4800 chars → (4800/4)*1.25 = 1500 tokens → ceil(1500/1200) = 2 chunks
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 2
    assert result["chunk_token_size"] == 1200
    assert result["estimated_llm_calls"] == 4           # chunks * 2
    assert result["estimated_embedding_calls"] == 2     # chunks * 1
    assert result["estimated_input_tokens"] == 4800     # 4 calls * chunk_size
    assert result["estimated_output_tokens"] == 1000    # 4 calls * 250 avg
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


def test_estimate_cost_warns_on_insufficient_model():
    # mistral-small has only 10B params (below 32B minimum for LightRAG)
    silo = _make_lightrag_silo(model_name="mistral-small")
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert len(result["warnings"]) > 0
    assert "mistral-small" in result["warnings"][0]


def test_estimate_cost_no_warning_on_large_model():
    silo = _make_lightrag_silo(model_name="gpt-4o")
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["warnings"] == []


def test_estimate_cost_no_warning_on_gpt_4o_mini_for_extract():
    # gpt-4o-mini: 128K context, ~15B params. EXTRACT threshold is 12B/32K
    # so 15B >= 12B — no warning expected for extract-only usage.
    silo = _make_lightrag_silo(model_name="gpt-4o-mini")
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["warnings"] == []


def test_estimate_cost_warns_on_small_model_for_extract():
    # mistral-small: 32K context, ~10B params. EXTRACT threshold is 12B
    # so 10B < 12B — should warn.
    silo = _make_lightrag_silo(model_name="mistral-small")
    db = MagicMock()
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert len(result["warnings"]) > 0
    assert "EXTRACT" in result["warnings"][0]
    assert "mistral-small" in result["warnings"][0]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_estimate_cost_multiple_documents():
    """Verify chunks are calculated per document, not globally."""
    silo = _make_lightrag_silo(chunk_size=1200)
    db = MagicMock()

    # Doc1: 2400 chars → (2400/4)*1.25 = 750 tokens → ceil(750/1200) = 1 chunk
    # Doc2: 4800 chars → (4800/4)*1.25 = 1500 tokens → ceil(1500/1200) = 2 chunks
    # Total: 3 chunks (per-doc sum)
    documents = [
        {"content": "x" * 2400},
        {"content": "x" * 4800},
    ]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 3  # 1 + 2
    assert result["estimated_llm_calls"] == 6  # 3 chunks * 2


def test_estimate_cost_per_doc_with_overlap():
    """Verify overlap is correctly handled per document."""
    silo = _make_lightrag_silo(chunk_size=1200)
    silo.lightrag_chunk_overlap_token_size = 200
    db = MagicMock()

    # Single doc: 2400 tokens with 200 overlap
    # stride = 1200 - 200 = 1000
    # chunks = 1 + ceil((2400 - 1200) / 1000) = 1 + ceil(1.2) = 1 + 2 = 3
    documents = [{"content": "x" * (2400 * 4)}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 3

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

    # No documents → 0 chunks
    assert result["total_chunks"] == 0
