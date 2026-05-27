"""Unit tests for ``tools.vector_stores.lightrag.adapters``.

These tests mock the underlying LangChain LLM / embeddings layer so they run
without network access and without requiring ``lightrag-hku`` for the
``is_lightrag_available`` flag branch.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tools.vector_stores.lightrag import adapters

# pytest.ini does not enable pytest-asyncio's auto mode, so mark every
# async test in this module explicitly.
pytestmark = pytest.mark.asyncio


def _make_ai_service(provider: str = "OpenAI") -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        name="test-llm",
        description="gpt-test",
        api_key="sk-test",
        endpoint=None,
    )


def _make_embedding_service(
    *, provider: str = "OpenAI", description: str = "text-embedding-3-small"
) -> SimpleNamespace:
    return SimpleNamespace(
        provider=provider,
        name="test-embed",
        description=description,
        api_key="sk-test",
        endpoint=None,
        api_version=None,
    )


# ---------------------------------------------------------------------------
# is_lightrag_available
# ---------------------------------------------------------------------------


class TestIsLightragAvailable:
    def test_returns_false_when_flag_off(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "LIGHTRAG_ENABLED", False, raising=False)
        assert adapters.is_lightrag_available() is False

    def test_returns_true_when_flag_on_and_package_present(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "LIGHTRAG_ENABLED", True, raising=False)
        # Install a stub ``lightrag`` module so import succeeds even if the
        # real package is not installed in this environment.
        monkeypatch.setitem(sys.modules, "lightrag", MagicMock(name="lightrag-stub"))
        assert adapters.is_lightrag_available() is True

    def test_returns_false_when_package_missing(self, monkeypatch):
        import config

        monkeypatch.setattr(config, "LIGHTRAG_ENABLED", True, raising=False)
        # Ensure import fails by replacing the loader entry with one that
        # raises on import.
        monkeypatch.delitem(sys.modules, "lightrag", raising=False)

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name == "lightrag" or name.startswith("lightrag."):
                raise ImportError("forced for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        assert adapters.is_lightrag_available() is False


# ---------------------------------------------------------------------------
# build_llm_model_func
# ---------------------------------------------------------------------------


class TestBuildLlmModelFunc:
    async def test_round_trips_prompt_through_langchain_llm(self):
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="hello-world"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ) as create_llm:
            llm_func = adapters.build_llm_model_func(ai_service)
            result = await llm_func(
                "what's up",
                system_prompt="be brief",
                history_messages=[{"role": "user", "content": "hi"}],
            )

        create_llm.assert_called_once()
        assert result == "hello-world"

        # Verify the messages composed for the LLM are in the right order:
        # system -> history -> current prompt.
        sent_messages = fake_llm.ainvoke.await_args.args[0]
        assert sent_messages[0] == {"role": "system", "content": "be brief"}
        assert sent_messages[1] == {"role": "user", "content": "hi"}
        assert sent_messages[-1] == {"role": "user", "content": "what's up"}

    async def test_tolerates_missing_system_prompt_and_history(self):
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service)
            result = await llm_func("just a prompt")

        assert result == "ok"
        sent_messages = fake_llm.ainvoke.await_args.args[0]
        assert sent_messages == [{"role": "user", "content": "just a prompt"}]

    async def test_handles_string_content_response(self):
        """Some custom wrappers return a bare string instead of AIMessage."""
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value="bare-string")

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service)
            result = await llm_func("hi")

        assert result == "bare-string"

    def test_raises_on_missing_ai_service(self):
        with pytest.raises(ValueError):
            adapters.build_llm_model_func(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_embedding_func
# ---------------------------------------------------------------------------


class _StubEmbeddingFunc:
    """Minimal stand-in for ``lightrag.utils.EmbeddingFunc`` for tests.

    The real class is a dataclass with extra fields; for unit tests we only
    need attribute storage and equality of the values we pass in.
    """

    def __init__(self, embedding_dim, max_token_size, func, model_name=None, **_kw):
        self.embedding_dim = embedding_dim
        self.max_token_size = max_token_size
        self.func = func
        self.model_name = model_name


def _patch_lightrag_utils(monkeypatch):
    """Install a stub ``lightrag.utils`` module exposing ``EmbeddingFunc``."""
    stub_utils = SimpleNamespace(EmbeddingFunc=_StubEmbeddingFunc)
    stub_pkg = MagicMock(name="lightrag-stub")
    stub_pkg.utils = stub_utils
    monkeypatch.setitem(sys.modules, "lightrag", stub_pkg)
    monkeypatch.setitem(sys.modules, "lightrag.utils", stub_utils)


class TestBuildEmbeddingFunc:
    async def test_returns_embedding_func_with_correct_shape(self, monkeypatch):
        _patch_lightrag_utils(monkeypatch)

        embedding_service = _make_embedding_service()

        fake_embeddings = MagicMock()
        # ``embed_query`` is used once to detect embedding_dim.
        fake_embeddings.embed_query.return_value = [0.0] * 1536
        # No ``aembed_documents`` attribute => fall back to sync path.
        del fake_embeddings.aembed_documents
        fake_embeddings.embed_documents.return_value = [
            [0.1] * 1536,
            [0.2] * 1536,
            [0.3] * 1536,
        ]

        with patch(
            "tools.embeddingTools.get_embeddings_model",
            return_value=fake_embeddings,
        ):
            embedding_func = adapters.build_embedding_func(embedding_service)

        assert embedding_func.embedding_dim == 1536
        assert embedding_func.max_token_size == 8191  # text-embedding-3-small
        assert embedding_func.model_name == "text-embedding-3-small"

        result = await embedding_func.func(["a", "b", "c"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1536)
        assert result.dtype == np.float32

    async def test_empty_input_returns_empty_array(self, monkeypatch):
        _patch_lightrag_utils(monkeypatch)

        embedding_service = _make_embedding_service()
        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.0] * 8
        del fake_embeddings.aembed_documents

        with patch(
            "tools.embeddingTools.get_embeddings_model",
            return_value=fake_embeddings,
        ):
            embedding_func = adapters.build_embedding_func(embedding_service)

        result = await embedding_func.func([])
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 8)
        fake_embeddings.embed_documents.assert_not_called()

    async def test_uses_async_embed_when_available(self, monkeypatch):
        _patch_lightrag_utils(monkeypatch)

        embedding_service = _make_embedding_service(description="unknown-model")
        fake_embeddings = MagicMock()
        fake_embeddings.embed_query.return_value = [0.0] * 4
        fake_embeddings.aembed_documents = AsyncMock(
            return_value=[[1.0, 2.0, 3.0, 4.0]]
        )

        with patch(
            "tools.embeddingTools.get_embeddings_model",
            return_value=fake_embeddings,
        ):
            embedding_func = adapters.build_embedding_func(embedding_service)
            result = await embedding_func.func(["x"])

        # Unknown model => default max_token_size.
        assert embedding_func.max_token_size == adapters.DEFAULT_EMBEDDING_MAX_TOKENS
        assert result.shape == (1, 4)
        fake_embeddings.aembed_documents.assert_awaited_once_with(["x"])

    def test_raises_on_missing_service(self):
        with pytest.raises(ValueError):
            adapters.build_embedding_func(None)  # type: ignore[arg-type]
