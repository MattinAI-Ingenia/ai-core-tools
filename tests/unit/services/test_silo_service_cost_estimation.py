"""Unit tests for ``SiloService.estimate_indexing_cost``."""

from unittest.mock import MagicMock, patch

import pytest

from services.silo_service import SiloService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _deterministic_token_count():
    """Pin token counting to chars/4 so chunk/cost math is deterministic.

    The real estimator uses tiktoken (matching LightRAG's chunker); these tests
    cover the windowing + cost formulas, not the tokenizer, so we mock the
    token-count boundary to a simple chars/4 rule.
    """
    with patch("services.silo_service._count_tokens", side_effect=lambda t: len(t) // 4):
        yield


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
    silo.keywords_service = None
    silo.vlm_service = None
    return silo


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_estimate_cost_basic_calculation():
    silo = _make_lightrag_silo(chunk_size=1200)
    db = MagicMock()

    # 4800 chars → 4800/4 = 1200 tokens → ceil(1200/1200) = 1 chunk
    documents = [{"content": "x" * 4800}]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 1
    assert result["chunk_token_size"] == 1200
    assert result["estimated_llm_calls"] == 1           # chunks * (1 + 0 gleaning)
    assert result["estimated_embedding_calls"] == 1     # chunks * 1
    # 1 call * (avg_chunk_tokens 1200 + 3800 overhead) = 5000
    assert result["estimated_input_tokens"] == 5000
    assert result["estimated_output_tokens"] == 2000    # 1 call * 2000 avg
    # chunk embeddings (content 1200) + graph embeddings (≈ output 2000) = 3200
    assert result["estimated_embedding_tokens"] == 3200
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

    # Doc1: 5200 chars → 1300 tokens → ceil(1300/1200) = 2 chunks
    # Doc2: 2800 chars → 700 tokens → 1 chunk
    # Per-doc sum: 3 chunks. (A global 2000-token chunking would give only 2,
    # so total_chunks == 3 proves the per-document calculation.)
    documents = [
        {"content": "x" * 5200},
        {"content": "x" * 2800},
    ]

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 3  # 2 + 1
    assert result["estimated_llm_calls"] == 3  # 3 chunks * (1 + 0 gleaning)


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


def test_estimate_cost_with_gleaning():
    """Verify that ENTITY_EXTRACT_MAX_GLEANING scales the mathematical estimates correctly."""
    silo = _make_lightrag_silo(chunk_size=1200)
    db = MagicMock()
    documents = [{"content": "x" * 5200}]  # 1300 tokens → 2 chunks

    # Patch ENTITY_EXTRACT_MAX_GLEANING to 2
    with patch("config.ENTITY_EXTRACT_MAX_GLEANING", 2):
        with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
            result = SiloService.estimate_indexing_cost(1, documents, db)

    assert result["total_chunks"] == 2
    assert result["chunk_token_size"] == 1200
    # 2 chunks * (1 + 2 gleaning) = 6 calls
    assert result["estimated_llm_calls"] == 6
    assert result["estimated_embedding_calls"] == 2
    # 6 calls * (avg_chunk_tokens 650 + 3800 overhead) = 26700 input tokens
    assert result["estimated_input_tokens"] == 26700
    assert result["warnings"] == []

# ---------------------------------------------------------------------------
# Self-hosted providers
# ---------------------------------------------------------------------------


def _priced(monkeypatch, *, llm=(1.0, 2.0), emb=0.02):
    """Patch the pricing catalog so the cost formula is deterministic."""
    from services import pricing_service

    monkeypatch.setattr(pricing_service.PricingService, "get_llm_pricing",
                        staticmethod(lambda db, model: llm))
    monkeypatch.setattr(pricing_service.PricingService, "get_embedding_pricing",
                        staticmethod(lambda db, model: emb))
    monkeypatch.setattr("services.silo_service._ensure_pricing_catalog",
                        lambda db: None)


def test_self_hosted_llm_still_charges_for_embeddings(monkeypatch):
    """A local vLLM/Ollama extraction model must not void the whole estimate.

    The embeddings are still billed by OpenAI, so the estimate is the embedding
    cost alone instead of the previous "Unavailable".
    """
    silo = _make_lightrag_silo(chunk_size=1200)
    silo.indexing_service.provider = "Custom"
    silo.embedding_service.provider = "OpenAI"
    _priced(monkeypatch)
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, [{"content": "x" * 4800}], db)

    # 3200 embedding tokens * 0.02 / 1e6, LLM side counted as 0.
    expected = round(3200 * 0.02 / 1_000_000, 4)
    assert result["estimated_cost_min"] == expected
    assert result["estimated_cost_max"] == expected
    assert result["warnings"] == []


def test_self_hosted_embedding_still_charges_for_llm(monkeypatch):
    silo = _make_lightrag_silo(chunk_size=1200)
    silo.indexing_service.provider = "OpenAI"
    silo.embedding_service.provider = "Ollama"
    _priced(monkeypatch)
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, [{"content": "x" * 4800}], db)

    # input 5000 * 1.0 + output (1000..3000) * 2.0, embeddings counted as 0.
    assert result["estimated_cost_min"] == round((5000 * 1.0 + 1000 * 2.0) / 1_000_000, 4)
    assert result["estimated_cost_max"] == round((5000 * 1.0 + 3000 * 2.0) / 1_000_000, 4)
    assert result["warnings"] == []


def test_both_self_hosted_reports_unavailable(monkeypatch):
    """Nothing is billed per token, and GPU/power cost is not knowable here.

    Reporting 0.00 would read as "free", so the estimate stays None and the
    reason is surfaced as a warning.
    """
    silo = _make_lightrag_silo(chunk_size=1200)
    silo.indexing_service.provider = "Custom"
    silo.embedding_service.provider = "Custom"
    _priced(monkeypatch)
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, [{"content": "x" * 4800}], db)

    assert result["estimated_cost_min"] is None
    assert result["estimated_cost_max"] is None
    assert any("self-hosted" in w for w in result["warnings"]), result["warnings"]
    # Token counts are still reported — only the money is unknown.
    assert result["estimated_embedding_tokens"] == 3200


def test_unpriced_cloud_model_stays_unavailable(monkeypatch):
    """An unknown cloud model is not the same as a self-hosted one: it IS billed,
    we just do not know the rate, so guessing 0 would understate the cost."""
    silo = _make_lightrag_silo(chunk_size=1200)
    silo.indexing_service.provider = "OpenAI"
    silo.embedding_service.provider = "OpenAI"
    _priced(monkeypatch, llm=None)
    db = MagicMock()

    with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
        result = SiloService.estimate_indexing_cost(1, [{"content": "x" * 4800}], db)

    assert result["estimated_cost_min"] is None
    assert any("No pricing found" in w for w in result["warnings"]), result["warnings"]
