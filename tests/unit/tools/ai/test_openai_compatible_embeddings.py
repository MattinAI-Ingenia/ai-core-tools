"""Pins the OpenAI-compatible self-hosted embedding provider.

Every step of this path failed silently rather than loudly when first wired up,
which is why each gets a test:

* the model id arrives HuggingFace-style ("BAAI/bge-m3"), and the embedding
  family patterns only matched the bare name — the wizard listed **zero models
  with no error**, which reads as "the endpoint is empty";
* rerankers share the naming of embedding families ("bge-reranker-v2-m3"
  matches ^bge-) and would have been offered as embedding models, producing
  garbage vectors instead of an error;
* OpenAIEmbeddings tokenizes client-side by default and posts token IDs, which
  such servers reject with a 422.
"""

import pytest

from tools.ai.model_catalog import (
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_COMPATIBLE,
    enrich,
    heuristic_capabilities_from_id,
    is_chat_model,
    is_embedding_model,
)


class TestHuggingFaceStyleIds:
    """Self-hosted servers report the HF repo id, prefix included."""

    @pytest.mark.parametrize("model_id", ["BAAI/bge-m3", "baai/bge-m3", "bge-m3"])
    def test_prefixed_embedding_model_is_recognised(self, model_id):
        caps = heuristic_capabilities_from_id(PROVIDER_OPENAI_COMPATIBLE, model_id)
        assert caps.embedding, f"{model_id} must classify as an embedding model"

    def test_the_prefix_is_only_stripped_for_providers_that_use_one(self):
        """Plain OpenAI never sends slash-prefixed ids; stripping there could
        only mask a malformed id."""
        caps = heuristic_capabilities_from_id(PROVIDER_OPENAI, "BAAI/bge-m3")
        assert not caps.embedding


class TestRerankersAreNotEmbeddingModels:
    @pytest.mark.parametrize(
        "model_id",
        ["BAAI/bge-reranker-v2-m3", "bge-reranker-base", "cross-encoder/ms-marco"],
    )
    def test_reranker_offers_no_capability(self, model_id):
        info = enrich(PROVIDER_OPENAI_COMPATIBLE, model_id)
        assert not is_embedding_model(info), "a cross-encoder returns no vector"
        assert not is_chat_model(info), "nor is it a chat model"

    def test_the_embedding_sibling_still_passes(self):
        """The veto must not be so broad it takes the real model with it."""
        assert is_embedding_model(enrich(PROVIDER_OPENAI_COMPATIBLE, "BAAI/bge-m3"))


class TestBuilderTargetsTheEndpoint:
    """`endpoint` must reach OpenAIEmbeddings as base_url, and client-side
    tokenization must be off for custom endpoints only."""

    @staticmethod
    def _build(endpoint):
        from types import SimpleNamespace
        from unittest.mock import patch

        service = SimpleNamespace(
            provider="OpenAICompatible",
            description="BAAI/bge-m3",
            endpoint=endpoint,
            api_key="k",
            api_version=None,
        )
        with patch("tools.embeddingTools.OpenAIEmbeddings") as ctor:
            from tools.embeddingTools import _build_openai_embeddings

            _build_openai_embeddings(service, "BAAI/bge-m3")
        return ctor.call_args.kwargs

    def test_custom_endpoint_is_used_and_tokenization_disabled(self):
        kwargs = self._build("http://reranker:7998")
        assert kwargs["base_url"] == "http://reranker:7998"
        assert kwargs["check_embedding_ctx_length"] is False

    @pytest.mark.parametrize("empty", ["", None])
    def test_real_openai_is_untouched(self, empty):
        """Existing OpenAI services store '' in that column, not NULL — both
        must keep the default API and its client-side chunking."""
        kwargs = self._build(empty)
        assert kwargs["base_url"] is None
        assert kwargs["check_embedding_ctx_length"] is True
