"""Adapter layer bridging our ``AIService`` / ``EmbeddingService`` to LightRAG.

LightRAG (``lightrag-hku==1.4.16``) expects two callables when building a
``LightRAG`` instance:

* ``llm_model_func``: ``async def(prompt, system_prompt=None,
  history_messages=None, **kwargs) -> str``
* ``embedding_func``: an ``EmbeddingFunc`` object with ``embedding_dim``,
  ``max_token_size`` and an async ``func(texts: list[str]) -> np.ndarray``.

This module produces those callables from the project's existing service
configuration so LightRAG reuses whichever LLM/embedding provider an
``AIService`` / ``EmbeddingService`` already points at. It contains **no**
business logic and never instantiates a ``LightRAG`` object — that wiring is
left to later steps.

All imports of ``lightrag`` are local to the function that needs them so this
module can be imported even when ``lightrag-hku`` is not installed or when
``LIGHTRAG_ENABLED=false``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, List, Optional

import numpy as np

from models.ai_service import AIService
from models.embedding_service import EmbeddingService

if TYPE_CHECKING:  # pragma: no cover - type-checking only
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
    "(`pip install 'lightrag-hku[offline-storage]==1.4.16'`) and set "
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

        response = await llm.ainvoke(messages)
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
        return str(content)

    return llm_model_func


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
    "is_lightrag_available",
]
