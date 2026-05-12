"""Unit tests for RAG retrieval configuration."""

import pytest
from pydantic import ValidationError

from schemas.agent_schemas import RetrievalConfig


# ==================== RetrievalConfig validation ====================

class TestRetrievalConfigDefaults:
    def test_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.search_type == "similarity"
        assert cfg.k == 30
        assert cfg.fetch_k == 100
        assert cfg.lambda_mult == 0.5
        assert cfg.score_threshold is None

    def test_mmr_valid(self):
        cfg = RetrievalConfig(search_type="mmr", k=10, fetch_k=50, lambda_mult=0.3)
        assert cfg.search_type == "mmr"
        assert cfg.k == 10

    def test_score_threshold_valid(self):
        cfg = RetrievalConfig(search_type="similarity_score_threshold", score_threshold=0.7)
        assert cfg.score_threshold == 0.7

    def test_score_threshold_required_when_type_is_threshold(self):
        with pytest.raises(ValidationError, match="score_threshold is required"):
            RetrievalConfig(search_type="similarity_score_threshold")

    def test_k_too_low(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(k=0)

    def test_k_too_high(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(k=201)

    def test_k_boundary_valid(self):
        assert RetrievalConfig(k=1).k == 1
        assert RetrievalConfig(k=200).k == 200

    def test_fetch_k_too_low(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(fetch_k=0)

    def test_lambda_mult_out_of_range_low(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(lambda_mult=-0.1)

    def test_lambda_mult_out_of_range_high(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(lambda_mult=1.1)

    def test_lambda_mult_boundary_valid(self):
        assert RetrievalConfig(lambda_mult=0.0).lambda_mult == 0.0
        assert RetrievalConfig(lambda_mult=1.0).lambda_mult == 1.0

    def test_score_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(
                search_type="similarity_score_threshold",
                score_threshold=1.5,
            )

    def test_invalid_search_type(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(search_type="invalid_type")

    def test_serialization(self):
        cfg = RetrievalConfig(search_type="mmr", k=5, fetch_k=20, lambda_mult=0.6)
        d = cfg.model_dump()
        assert d["search_type"] == "mmr"
        assert d["k"] == 5
        assert d["fetch_k"] == 20
        assert d["lambda_mult"] == 0.6


# ==================== get_silo_retriever merge logic ====================

class TestSiloRetrieverMerge:
    """Test the 3-layer merge: defaults < retrieval_config < search_params."""

    def _merged(self, retrieval_config=None, search_params=None):
        """Replicate the merge logic from SiloService.get_silo_retriever()."""
        known_params = {"k", "filter", "score_threshold", "fetch_k", "lambda_mult", "search_type"}

        merged: dict = {"k": 30}  # Layer 1: defaults

        if retrieval_config:  # Layer 2: agent config
            for key, value in retrieval_config.items():
                if value is not None and key in known_params:
                    merged[key] = value

        if search_params:  # Layer 3: runtime overrides
            filter_fields = {}
            direct_params = {}
            for key, value in search_params.items():
                if key in known_params:
                    direct_params[key] = value
                else:
                    filter_fields[key] = value
            merged.update(direct_params)
            if filter_fields:
                if "filter" in merged:
                    merged["filter"].update(filter_fields)
                else:
                    merged["filter"] = filter_fields

        return merged

    def test_no_config_uses_defaults(self):
        result = self._merged()
        assert result == {"k": 30}

    def test_retrieval_config_overrides_defaults(self):
        result = self._merged(retrieval_config={"search_type": "mmr", "k": 10, "fetch_k": 50})
        assert result["k"] == 10
        assert result["search_type"] == "mmr"
        assert result["fetch_k"] == 50

    def test_search_params_override_retrieval_config(self):
        result = self._merged(
            retrieval_config={"k": 10, "search_type": "mmr"},
            search_params={"k": 5},
        )
        assert result["k"] == 5
        assert result["search_type"] == "mmr"  # preserved from config

    def test_search_params_filter_wrapped(self):
        result = self._merged(search_params={"custom_field": "value"})
        assert result["filter"] == {"custom_field": "value"}

    def test_search_params_known_and_unknown_mixed(self):
        result = self._merged(search_params={"k": 7, "category": "docs"})
        assert result["k"] == 7
        assert result["filter"] == {"category": "docs"}

    def test_retrieval_config_none_values_ignored(self):
        result = self._merged(retrieval_config={"k": None, "search_type": "mmr"})
        assert result["k"] == 30  # None not applied — default preserved
        assert result["search_type"] == "mmr"

    def test_full_priority_chain(self):
        result = self._merged(
            retrieval_config={"k": 20, "search_type": "mmr", "fetch_k": 80},
            search_params={"k": 3, "score_threshold": 0.5},
        )
        assert result["k"] == 3              # runtime wins
        assert result["search_type"] == "mmr"  # from config
        assert result["fetch_k"] == 80         # from config
        assert result["score_threshold"] == 0.5  # runtime adds
