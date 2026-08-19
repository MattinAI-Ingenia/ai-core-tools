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

    async def test_max_tokens_is_forwarded_when_set(self):
        """Uncapped, vLLM defaults max_tokens to the rest of the context window."""
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service, max_tokens=8192)
            await llm_func("hi")

        assert fake_llm.ainvoke.await_args.kwargs["max_tokens"] == 8192

    async def test_guided_json_only_on_extraction_calls(self):
        """The extract role is reused for plain-text summaries when merging.

        Constraining those to the entity/relationship schema would corrupt every
        entity description, so the schema rides only on calls whose system
        prompt is LightRAG's JSON extraction prompt.
        """
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(
                ai_service, json_marker="Return one valid JSON object"
            )

            # Extraction: system prompt carries LightRAG's JSON marker.
            await llm_func(
                "chunk text",
                system_prompt="...Return one valid JSON object with entities...",
            )
            extraction_kwargs = fake_llm.ainvoke.await_args.kwargs

            # Summary (operate.py:436): same role func, no system prompt.
            await llm_func("Summarize these descriptions")
            summary_kwargs = fake_llm.ainvoke.await_args.kwargs

        schema = extraction_kwargs["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]) == {"entities", "relationships"}
        assert "response_format" not in summary_kwargs

    async def test_no_marker_sends_no_schema(self):
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service, json_marker=None)
            await llm_func("x", system_prompt="Return one valid JSON object ...")

        assert "response_format" not in fake_llm.ainvoke.await_args.kwargs

    async def test_max_tokens_is_omitted_when_unset(self):
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service)
            await llm_func("hi")

        assert "max_tokens" not in fake_llm.ainvoke.await_args.kwargs


class TestLengthLimitSalvage:
    """A capped, schema-constrained call that hits the limit must not lose the chunk.

    With response_format set, the OpenAI SDK raises LengthFinishReasonError instead
    of returning truncated text — measured in the benchmark as 2 of 17 chunks lost
    on a technical manual. The partial JSON travels inside the exception.
    """

    def _length_error(self, content, tokens=(100, 64)):
        class LengthFinishReasonError(Exception):
            pass

        usage = SimpleNamespace(prompt_tokens=tokens[0], completion_tokens=tokens[1],
                                total_tokens=sum(tokens))
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )
        exc = LengthFinishReasonError("length limit reached")
        exc.completion = completion
        return exc

    async def test_partial_json_is_returned_instead_of_raising(self):
        partial = '{"entities": [{"name": "Ada", "type": "Person", "description": "x"}, {"name'
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=self._length_error(partial))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service, max_tokens=64)
            out = await llm_func("hi")

        assert out == partial
        # json_repair (what LightRAG uses) must recover the complete record.
        import json_repair

        assert json_repair.loads(out)["entities"][0]["name"] == "Ada"

    async def test_other_errors_still_propagate(self):
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=RuntimeError("connection reset"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service, max_tokens=64)
            with pytest.raises(RuntimeError, match="connection reset"):
                await llm_func("hi")

    @pytest.mark.parametrize("empty", ["", "   \n"])
    async def test_length_error_with_empty_partial_propagates(self, empty):
        """Zero records produced → surface the failure, do not report success."""
        ai_service = _make_ai_service()
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=self._length_error(empty))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service, max_tokens=64)
            with pytest.raises(Exception, match="length limit"):
                await llm_func("hi")

    async def test_length_error_without_choices_propagates(self):
        """Nothing to salvage → the error must not be swallowed silently."""
        ai_service = _make_ai_service()
        exc = self._length_error("")
        exc.completion.choices = []
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=exc)

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            llm_func = adapters.build_llm_model_func(ai_service, max_tokens=64)
            with pytest.raises(Exception, match="length limit"):
                await llm_func("hi")


# ---------------------------------------------------------------------------
# _derive_json_extraction_marker
# ---------------------------------------------------------------------------


class TestDeriveJsonExtractionMarker:
    """The marker is read from the installed prompts, not hardcoded.

    A hardcoded sentence stops matching the day upstream rewords it, silently
    disabling guided JSON; deriving it means the reword is picked up instead.
    """

    def _stub_prompts(self, monkeypatch, extraction, summary):
        stub = SimpleNamespace(
            PROMPTS={
                "entity_extraction_json_system_prompt": extraction,
                "summarize_entity_descriptions": summary,
            }
        )
        stub_pkg = MagicMock(name="lightrag-stub")
        monkeypatch.setitem(sys.modules, "lightrag", stub_pkg)
        monkeypatch.setitem(sys.modules, "lightrag.prompt", stub)

    def test_picks_longest_line_absent_from_summary(self, monkeypatch):
        self._stub_prompts(
            monkeypatch,
            extraction=(
                "short\n"
                "Summarise the descriptions below into one paragraph of text.\n"
                "Return one valid JSON object with entities and relationships arrays.\n"
            ),
            summary="Summarise the descriptions below into one paragraph of text.\n",
        )
        marker = adapters._derive_json_extraction_marker()
        assert marker == (
            "Return one valid JSON object with entities and relationships arrays."
        )

    def test_skips_lines_with_placeholders_or_json_braces(self, monkeypatch):
        self._stub_prompts(
            monkeypatch,
            extraction=(
                "Output at most {max_total_records} records across the two arrays.\n"
                '  "entities": [ { "name": "<entity_name>" } ]\n'
                "Extract entities and relationships as one JSON object please.\n"
            ),
            summary="",
        )
        marker = adapters._derive_json_extraction_marker()
        assert marker == (
            "Extract entities and relationships as one JSON object please."
        )

    def test_returns_none_when_every_candidate_collides(self, monkeypatch):
        """No unique sentence → no schema, rather than risking the summary calls."""
        shared = "This very same long sentence appears in both prompts here.\n"
        self._stub_prompts(monkeypatch, extraction=shared, summary=shared)
        assert adapters._derive_json_extraction_marker() is None

    def test_returns_none_when_prompt_key_is_gone(self, monkeypatch):
        self._stub_prompts(monkeypatch, extraction="", summary="")
        assert adapters._derive_json_extraction_marker() is None


# ---------------------------------------------------------------------------
# build_role_llm_configs
# ---------------------------------------------------------------------------


class TestBuildRoleLlmConfigs:
    async def test_only_extract_role_gets_cap_and_schema(self, monkeypatch):
        """Both knobs target indexing-time extraction, not query-time roles."""
        import config as app_config

        monkeypatch.setattr(app_config, "LIGHTRAG_EXTRACT_MAX_TOKENS", 8192)

        stub_roles = SimpleNamespace(
            RoleLLMConfig=lambda func, metadata=None, **_kw: SimpleNamespace(
                func=func, metadata=metadata
            )
        )
        marker = "Return one valid JSON object with both arrays and nothing else."
        stub_prompt = SimpleNamespace(
            PROMPTS={
                "entity_extraction_json_system_prompt": marker + "\n",
                "summarize_entity_descriptions": "Summarise these descriptions.\n",
            }
        )
        stub_pkg = MagicMock(name="lightrag-stub")
        monkeypatch.setitem(sys.modules, "lightrag", stub_pkg)
        monkeypatch.setitem(sys.modules, "lightrag.llm_roles", stub_roles)
        monkeypatch.setitem(sys.modules, "lightrag.prompt", stub_prompt)

        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))

        with patch(
            "tools.aiServiceTools.create_llm_from_service",
            return_value=fake_llm,
        ):
            configs = adapters.build_role_llm_configs(
                extract_service=_make_ai_service(),
                keywords_service=_make_ai_service(),
            )
            await configs['extract'].func("hi", system_prompt=marker)
            extract_kwargs = fake_llm.ainvoke.await_args.kwargs
            await configs['keyword'].func("hi", system_prompt=marker)
            keyword_kwargs = fake_llm.ainvoke.await_args.kwargs

        assert extract_kwargs["max_tokens"] == 8192
        assert "max_tokens" not in keyword_kwargs
        # The schema must ride on extraction only: the keyword role answers
        # queries with a short list, and forcing the shape would break it.
        assert "response_format" in extract_kwargs
        assert "response_format" not in keyword_kwargs


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
