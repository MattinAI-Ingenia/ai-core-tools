"""Unit tests for ``SiloService.estimate_indexing_cost``."""

from unittest.mock import MagicMock, patch

import pytest

from services.silo_service import SiloService, _llm_throughput_bounds, _model_params_b


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


# ---------------------------------------------------------------------------
# Time estimate reacts to real concurrency
# ---------------------------------------------------------------------------


class TestTimeEstimateFollowsConcurrencyEnvVars:
    """llm_workers/embedding_workers used to read Silo.llm_model_max_sync /
    Silo.embedding_model_max_sync — columns that never existed on the model,
    so getattr's default silently won every time in production. In these
    tests the silo is a MagicMock, which auto-creates any attribute you
    haven't explicitly blanked, so the same code accidentally read a real
    number here and the bug never showed up in this suite. Raising
    MAX_ASYNC_LLM/MAX_PARALLEL_INSERT made indexing genuinely faster while the
    estimate stayed exactly the same, no matter what was set."""

    @staticmethod
    def _estimate(env):
        silo = _make_lightrag_silo(chunk_size=1200)
        db = MagicMock()
        documents = [{"content": "x" * 48000}]  # enough chunks for workers to matter
        with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo), \
             patch.dict("os.environ", env, clear=False):
            return SiloService.estimate_indexing_cost(1, documents, db)

    def test_more_concurrent_llm_calls_lowers_the_estimate(self):
        slow = self._estimate({"MAX_ASYNC_LLM": "2", "MAX_PARALLEL_INSERT": "2"})
        fast = self._estimate({"MAX_ASYNC_LLM": "20", "MAX_PARALLEL_INSERT": "20"})

        assert fast["estimated_indexing_time_max"] < slow["estimated_indexing_time_max"]
        assert fast["estimated_indexing_time_min"] < slow["estimated_indexing_time_min"]

    def test_llm_concurrency_is_capped_by_the_smaller_of_the_two_knobs(self):
        """A chunk's LLM call cannot start before it has a feeder slot —
        MAX_PARALLEL_INSERT=2 caps real throughput even if MAX_ASYNC_LLM=20."""
        bottlenecked = self._estimate({"MAX_ASYNC_LLM": "20", "MAX_PARALLEL_INSERT": "2"})
        both_low = self._estimate({"MAX_ASYNC_LLM": "2", "MAX_PARALLEL_INSERT": "2"})

        assert bottlenecked["estimated_indexing_time_max"] == both_low["estimated_indexing_time_max"]

    def test_more_embedding_concurrency_lowers_the_estimate(self):
        slow = self._estimate({"EMBEDDING_FUNC_MAX_ASYNC": "1"})
        fast = self._estimate({"EMBEDDING_FUNC_MAX_ASYNC": "32"})

        assert fast["estimated_indexing_time_max"] <= slow["estimated_indexing_time_max"]


# ---------------------------------------------------------------------------
# Time estimate follows model size and hosting (cloud API vs self-hosted)
# ---------------------------------------------------------------------------


class TestModelParamsB:
    def test_reads_the_curated_table_for_a_known_commercial_model(self):
        # Not gpt-4o-mini: _lookup_model_specs' substring matching hits the
        # shorter "gpt-4o" entry first for that name (pre-existing, unrelated
        # to this change) — pick a name with no such collision.
        assert _model_params_b("mistral-small") == 10

    def test_falls_back_to_the_size_in_a_self_hosted_models_own_name(self):
        assert _model_params_b("Qwen3-30B-A3B-Instruct") == 30
        assert _model_params_b("Qwen3-4B-Instruct") == 4

    def test_unknown_name_with_no_size_hint_yields_none(self):
        assert _model_params_b("my-custom-finetune") is None


class TestLlmThroughputBounds:
    def test_cloud_api_is_faster_than_the_self_hosted_baseline(self):
        cloud = _llm_throughput_bounds("gpt-4o", self_hosted=False)
        self_hosted_big = _llm_throughput_bounds("Qwen3-30B-Instruct", self_hosted=True)
        assert cloud[0] > self_hosted_big[0]
        assert cloud[1] > self_hosted_big[1]

    def test_a_small_self_hosted_model_is_faster_than_a_big_one(self):
        small = _llm_throughput_bounds("Qwen3-4B-Instruct", self_hosted=True)
        big = _llm_throughput_bounds("Qwen3-30B-Instruct", self_hosted=True)
        assert small[0] > big[0]
        assert small[1] > big[1]

    def test_self_hosted_with_no_size_hint_keeps_the_calibrated_baseline(self):
        """Unknown size must not guess "small" and overpromise — the baseline
        was measured against this deployment's own ~30B model."""
        assert _llm_throughput_bounds("my-custom-finetune", self_hosted=True) == (180.0, 70.0)


class TestTimeEstimateEndToEnd:
    """Through estimate_indexing_cost, not just the bounds helper — confirms
    the wiring (model_name/llm_self_hosted actually reach it) and not only
    the formula in isolation."""

    @staticmethod
    def _estimate(model_name, provider):
        silo = _make_lightrag_silo(chunk_size=1200, model_name=model_name)
        silo.indexing_service.provider = provider
        silo.embedding_service.provider = provider
        db = MagicMock()
        documents = [{"content": "x" * 48000}]
        with patch("services.silo_service.SiloRepository.get_by_id", return_value=silo):
            return SiloService.estimate_indexing_cost(1, documents, db)

    def test_cloud_model_estimates_faster_than_self_hosted(self):
        cloud = self._estimate("gpt-4o", "OpenAI")
        self_hosted = self._estimate("Qwen3-30B-A3B-Instruct", "Custom")
        assert cloud["estimated_indexing_time_max"] < self_hosted["estimated_indexing_time_max"]

    def test_small_self_hosted_model_estimates_faster_than_a_big_one(self):
        small = self._estimate("Qwen3-4B-Instruct", "Custom")
        big = self._estimate("Qwen3-30B-A3B-Instruct", "Custom")
        assert small["estimated_indexing_time_max"] < big["estimated_indexing_time_max"]
