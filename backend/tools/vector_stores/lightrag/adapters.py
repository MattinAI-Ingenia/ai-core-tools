"""Adapter layer bridging our ``AIService`` / ``EmbeddingService`` to LightRAG.

LightRAG (``lightrag-hku==1.5.5rc1``) expects:

* ``llm_model_func``: ``async def(prompt, system_prompt=None,
  history_messages=None, **kwargs) -> str``  — base/fallback LLM callable.
* ``role_llm_configs``: ``dict[str, RoleLLMConfig]`` — per-role LLM overrides
  keyed by ``"extract" | "keyword" | "query" | "vlm"``.
* ``embedding_func``: an ``EmbeddingFunc`` object with ``embedding_dim``,
  ``max_token_size`` and an async ``func(texts: list[str]) -> np.ndarray``.

This module produces those objects from the project's existing service
configuration so LightRAG reuses whichever LLM/embedding provider an
``AIService`` / ``EmbeddingService`` already points at. It contains **no**
business logic and never instantiates a ``LightRAG`` object — that wiring is
left to later steps.

All imports of ``lightrag`` are local to the function that needs them so this
module can be imported even when ``lightrag-hku`` is not installed or when
``LIGHTRAG_ENABLED=false``.
"""

from __future__ import annotations

import contextvars
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Optional

from tools.vector_stores.lightrag.token_accumulator import IndexingTokenAccumulator

# contextvars slot: holds the active accumulator during an indexing run,
# or None when not in an indexing context (e.g. during query).
_active_accumulator: contextvars.ContextVar[Optional[IndexingTokenAccumulator]] = (
    contextvars.ContextVar("_lightrag_active_accumulator", default=None)
)

import numpy as np

from models.ai_service import AIService
from models.embedding_service import EmbeddingService

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from lightrag.llm_roles import RoleLLMConfig
    from lightrag.utils import EmbeddingFunc

logger = logging.getLogger(__name__)

# Per-provider / per-model context window for embeddings. Used to populate
# ``EmbeddingFunc.max_token_size`` which LightRAG uses for input chunking.
# Conservative defaults are intentionally chosen — when unknown, fall back
# to ``DEFAULT_EMBEDDING_MAX_TOKENS``.
DEFAULT_EMBEDDING_MAX_TOKENS = 8192

_EMBEDDING_MAX_TOKENS_BY_MODEL: dict[str, int] = {
    # OpenAI
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
    # MistralAI
    "mistral-embed": 8192,
}


_LIGHTRAG_INSTALL_HINT = (
    "lightrag-hku is not installed. Install the optional extra "
    "(`pip install 'lightrag-hku[offline-storage]==1.5.5rc1'`) and set "
    "LIGHTRAG_ENABLED=true to enable the LightRAG integration."
)


def _import_lightrag_utils():
    """Lazy import of ``lightrag.utils`` with a clear error message."""
    try:
        from lightrag import utils as _utils  # noqa: WPS433 - intentional local import
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise RuntimeError(_LIGHTRAG_INSTALL_HINT) from exc
    return _utils


def is_lightrag_available() -> bool:
    """Return ``True`` only if both conditions hold:

    * ``settings.LIGHTRAG_ENABLED`` is true.
    * The ``lightrag`` package can be imported.

    Used as a feature flag by callers so they can degrade gracefully when the
    optional dependency is missing or the operator hasn't opted in.
    """
    import config  # local import: avoid any import-time coupling

    if not getattr(config, "LIGHTRAG_ENABLED", False):
        return False

    try:
        import lightrag  # noqa: F401, WPS433 - probe only
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------


def build_role_llm_configs(
    *,
    extract_service: AIService,
    keywords_service: Optional[AIService] = None,
    vlm_service: Optional[AIService] = None,
    temperature: float = 0.0,
) -> dict[str, "RoleLLMConfig"]:
    """Build a ``role_llm_configs`` dict for the ``LightRAG`` constructor.

    Returns a mapping from LightRAG role name (``"extract" | "keyword" | "vlm"``)
    to a :class:`RoleLLMConfig` containing the ``llm_model_func`` callable for
    that role.

    ``extract_service`` is mandatory — it is also used as the base
    ``llm_model_func`` fallback. The other roles default to ``None``
    (omitted from the dict) which tells LightRAG to reuse the base
    ``llm_model_func`` automatically. The query role is intentionally omitted
    because ``only_need_context=True`` is always used and LightRAG falls back
    to the base LLM for context assembly.

    LightRAG 1.5.5rc1 expects role keys in **lowercase** and the keyword
    role as singular ``"keyword"`` (not ``"keywords"``).
    """
    from lightrag.llm_roles import RoleLLMConfig  # noqa: WPS433

    if extract_service is None:
        raise ValueError("extract_service is required to build LightRAG role LLMs")

    def _role_config(service: "AIService") -> "RoleLLMConfig":
        return RoleLLMConfig(
            func=build_llm_model_func(service, temperature=temperature),
            metadata={
                "binding": getattr(service, "provider", None),
                "model": getattr(service, "description", None),
            },
        )

    configs: dict[str, RoleLLMConfig] = {
        'extract': _role_config(extract_service),
    }

    # LightRAG uses "keyword" (singular), not "keywords".
    if keywords_service is not None:
        configs['keyword'] = _role_config(keywords_service)

    # VLM is optional — omitting the key is cleaner than an entry with
    # func=None, because LightRAG's resolver treats a missing key as
    # "fall back to base" and a None func as an error.
    if vlm_service is not None:
        configs['vlm'] = _role_config(vlm_service)

    return configs


def build_llm_model_func(
    ai_service: AIService,
    *,
    temperature: float = 0.0,
) -> Callable[..., Awaitable[str]]:
    """Build a LightRAG-shaped ``llm_model_func`` from an :class:`AIService`.

    Reuses :func:`tools.aiServiceTools.create_llm_from_service` so all
    provider-switching logic stays in one place.

    The returned coroutine accepts LightRAG's call signature:
        ``async def(prompt, system_prompt=None, history_messages=None, **kwargs) -> str``

    History messages are passed through as already-formatted LangChain-style
    dicts (``{"role": "user"|"assistant"|"system", "content": "..."}``). When
    ``system_prompt`` is provided it is prepended as a system message.
    """
    if ai_service is None:
        raise ValueError("ai_service is required to build an LLM adapter")

    # Local import to avoid importing the LLM stack at module import time.
    from tools.aiServiceTools import create_llm_from_service

    llm = create_llm_from_service(ai_service, temperature=temperature)

    async def llm_model_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict]] = None,
        **_kwargs: Any,
    ) -> str:
        messages: List[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # lc_source tag → _INTERNAL_LC_SOURCES filter drops these tokens;
        # untagged, the keyword-extraction JSON leaks into the chat stream.
        response = await llm.ainvoke(
            messages, config={"metadata": {"lc_source": "lightrag"}}
        )
        # LangChain chat models return an AIMessage; fall back to str() for
        # custom wrappers that might return a plain string.
        content = getattr(response, "content", response)
        if isinstance(content, list):
            # Some providers return content as a list of parts; concatenate
            # text parts.
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )

        # --- Token usage capture ---
        acc = _active_accumulator.get()
        if acc is not None:
            prompt_toks: int = 0
            completion_toks: int = 0
            source = "estimated"

            # 1. Try LangChain usage_metadata (OpenAI, Anthropic, Mistral …)
            usage_meta = getattr(response, "usage_metadata", None)
            if usage_meta and isinstance(usage_meta, dict):
                prompt_toks = usage_meta.get("input_tokens", 0) or 0
                completion_toks = usage_meta.get("output_tokens", 0) or 0
                if prompt_toks or completion_toks:
                    source = "provider"

            # 2. Fallback: response_metadata["token_usage"] (some providers)
            if source == "estimated":
                resp_meta = getattr(response, "response_metadata", {}) or {}
                token_usage = resp_meta.get("token_usage") or {}
                if isinstance(token_usage, dict):
                    prompt_toks = token_usage.get("prompt_tokens", 0) or 0
                    completion_toks = token_usage.get("completion_tokens", 0) or 0
                    if prompt_toks or completion_toks:
                        source = "provider"

            # 3. Tiktoken fallback: estimate from prompt text + response content
            if source == "estimated":
                try:
                    import tiktoken  # noqa: WPS433
                    enc = tiktoken.get_encoding("cl100k_base")
                    prompt_toks = sum(len(enc.encode(m["content"])) for m in messages if isinstance(m.get("content"), str))
                    completion_toks = len(enc.encode(str(content)))
                except Exception:
                    prompt_toks = len(" ".join(m.get("content", "") for m in messages)) // 4
                    completion_toks = len(str(content)) // 4

            acc.add_llm_usage(prompt=prompt_toks, completion=completion_toks, source=source)

        return str(content)

    return llm_model_func


def set_active_accumulator(acc: Optional[IndexingTokenAccumulator]) -> Any:
    """Set the active token accumulator for the current context.

    Returns a ``contextvars.Token`` that can be passed to :func:`reset_active_accumulator`
    to restore the previous value.
    """
    return _active_accumulator.set(acc)


def reset_active_accumulator(token: Any) -> None:
    """Restore the accumulator context var to its previous value."""
    _active_accumulator.reset(token)


def get_active_accumulator() -> Optional[IndexingTokenAccumulator]:
    """Return the active accumulator, or ``None`` when not in an indexing run."""
    return _active_accumulator.get()


# ---------------------------------------------------------------------------
# Embedding adapter
# ---------------------------------------------------------------------------


def _resolve_max_token_size(embedding_service: EmbeddingService) -> int:
    """Best-effort lookup of the embedding model's max input tokens."""
    model_id = getattr(embedding_service, "description", None)
    if model_id and model_id in _EMBEDDING_MAX_TOKENS_BY_MODEL:
        return _EMBEDDING_MAX_TOKENS_BY_MODEL[model_id]
    return DEFAULT_EMBEDDING_MAX_TOKENS


async def _embed_batch(embeddings_model: Any, texts: List[str]) -> List[List[float]]:
    """Call ``embed_documents`` from sync or async embedding clients."""
    # LangChain embeddings expose ``aembed_documents`` when async is supported.
    aembed = getattr(embeddings_model, "aembed_documents", None)
    if aembed is not None:
        return await aembed(texts)

    # Fall back to the sync API in a worker thread so we don't block the loop.
    import asyncio

    return await asyncio.to_thread(embeddings_model.embed_documents, texts)


def build_embedding_func(embedding_service: EmbeddingService) -> "EmbeddingFunc":
    """Build a LightRAG ``EmbeddingFunc`` from an :class:`EmbeddingService`.

    Reuses :func:`tools.embeddingTools.get_embeddings_model` so provider
    selection stays centralised. ``embedding_dim`` is detected by issuing one
    test embedding call against ``"."`` (cheap, single token) and cached on
    the returned ``EmbeddingFunc``.
    """
    if embedding_service is None:
        raise ValueError("embedding_service is required to build an embedding adapter")

    EmbeddingFunc = _import_lightrag_utils().EmbeddingFunc  # noqa: N806

    # Local import to keep the LangChain embedding stack out of import time.
    from tools.embeddingTools import get_embeddings_model

    embeddings_model = get_embeddings_model(embedding_service)

    # Detect embedding dimensionality with a single probe. Cached for the
    # lifetime of the returned EmbeddingFunc — LightRAG only reads it once.
    probe_vector = embeddings_model.embed_query(".")
    embedding_dim = len(probe_vector)
    max_token_size = _resolve_max_token_size(embedding_service)

    provider = getattr(embedding_service, "provider", None)
    if hasattr(provider, "value"):
        provider = provider.value
    model_name = (
        getattr(embedding_service, "description", None)
        or f"{provider or 'unknown'}-embedding"
    )

    async def func(texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, embedding_dim), dtype=np.float32)
        vectors = await _embed_batch(embeddings_model, list(texts))

        # Report embedding token usage to the active accumulator (if any).
        # We estimate tokens via character count / 4 (standard approximation).
        acc = _active_accumulator.get()
        if acc is not None:
            estimated_tokens = sum(len(t) for t in texts) // 4
            acc.add_embedding_usage(estimated_tokens)

        return np.asarray(vectors, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        func=func,
        model_name=model_name,
    )


__all__ = [
    "DEFAULT_EMBEDDING_MAX_TOKENS",
    "build_embedding_func",
    "build_llm_model_func",
    "build_role_llm_configs",
    "is_lightrag_available",
]
