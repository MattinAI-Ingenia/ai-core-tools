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
from types import SimpleNamespace
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


# JSON Schema mirroring the contract LightRAG's JSON extraction parser reads
# (``_process_json_extraction_result``, ``lightrag/operate.py:722``): entity
# objects keyed name/type/description, relationship objects keyed
# source/target/keywords/description. Sent as ``response_format`` so the server
# constrains decoding instead of merely being asked to produce JSON.
_EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["name", "type", "description"],
                "additionalProperties": False,
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "keywords": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["source", "target", "keywords", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}

# Providers whose API accepts OpenAI-style ``response_format={"type":
# "json_schema"}``. Anthropic / Mistral / Google use different mechanisms, so
# they keep prompt-only JSON instead of erroring on an unknown parameter.
_GUIDED_JSON_PROVIDERS = frozenset({"Custom", "OpenAI", "Azure", "OpenRouter"})

def _derive_json_extraction_marker() -> Optional[str]:
    """Return a sentence that appears **only** in LightRAG's JSON extraction prompt.

    The ``extract`` role func is not used for extraction alone:
    ``_handle_entity_relation_summary`` (``lightrag/operate.py:436``) reuses it to
    summarise entity descriptions as **plain text**, and constraining those to
    :data:`_EXTRACTION_JSON_SCHEMA` would corrupt every description in the graph.
    LightRAG consumes ``_priority`` before calling (``utils.py:2080``), so the
    call carries no explicit marker and the prompt is the only discriminator.

    The sentence is read from the installed library instead of being hardcoded:
    upstream rewording then moves the marker with it, rather than silently
    disabling guided JSON. Lines containing braces are skipped — they are
    placeholders or escaped JSON literals that do not survive templating.

    Returns ``None`` if the prompt is missing or every candidate also appears in
    the summary prompt; guided JSON is then left off rather than applied blindly.
    """
    from lightrag.prompt import PROMPTS  # noqa: WPS433

    template = PROMPTS.get("entity_extraction_json_system_prompt") or ""
    summary = PROMPTS.get("summarize_entity_descriptions") or ""
    if not template:
        return None

    candidates = [
        stripped
        for line in template.splitlines()
        if "{" not in line and "}" not in line
        for stripped in (line.strip(),)
        if len(stripped) >= 40  # corta encabezados y frases sueltas
    ]
    # Longest first: the most distinctive sentence is the least likely to
    # collide with another prompt.
    for candidate in sorted(candidates, key=len, reverse=True):
        if candidate not in summary:
            return candidate
    return None


def _salvage_length_limit(exc: Exception):
    """Recover the partial completion when a capped, schema-constrained call hits the limit.

    With ``response_format`` set, the OpenAI SDK does not hand back a truncated
    string: it raises ``LengthFinishReasonError``, which would lose the whole
    chunk. The partial JSON is carried inside the exception, and LightRAG parses
    extraction output with ``json_repair`` (``operate.py:746``), so returning it
    keeps every complete record produced before the cut instead of nothing.

    Returns a response-shaped stand-in, or ``None`` if *exc* is a different error
    (which must keep propagating).
    """
    if type(exc).__name__ != "LengthFinishReasonError":
        return None
    completion = getattr(exc, "completion", None)
    if completion is None or not getattr(completion, "choices", None):
        return None
    content = getattr(completion.choices[0].message, "content", None) or ""
    if not content.strip():
        # Nothing was produced before the cut: swallowing the error here would
        # report a successful extraction of zero records, hiding the failure.
        return None
    usage = getattr(completion, "usage", None)
    usage_metadata = None
    if usage is not None:
        usage_metadata = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
    # Mimics the LangChain message shape completely: consumers read
    # `.response_metadata` directly (scripts/compare_extraction.py), not only
    # through getattr with a default.
    return SimpleNamespace(
        content=content, usage_metadata=usage_metadata, response_metadata={}
    )


def _resolve_json_marker(service: "AIService") -> Optional[str]:
    """Marker enabling schema-constrained extraction for *service*, or ``None``.

    ``None`` means "send no schema" — LightRAG's prompt-level JSON still applies.
    The decision is logged: a silent downgrade would otherwise only surface much
    later, as runaway generation.
    """
    import config  # local import to avoid coupling at module import time

    provider = getattr(service, "provider", None)
    if not config.LIGHTRAG_EXTRACT_GUIDED_JSON:
        logger.info("LightRAG guided JSON off: disabled by configuration")
        return None
    if provider not in _GUIDED_JSON_PROVIDERS:
        logger.info(
            "LightRAG guided JSON off: provider %s has no OpenAI-style json_schema",
            provider,
        )
        return None

    marker = _derive_json_extraction_marker()
    if marker is None:
        logger.warning(
            "LightRAG guided JSON off: no sentence unique to the JSON extraction "
            "prompt could be derived — falling back to prompt-only JSON"
        )
    else:
        logger.info("LightRAG guided JSON on (marker: %.60s)", marker)
    return marker


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
    import config  # local import to avoid coupling at module import time
    from lightrag.llm_roles import RoleLLMConfig  # noqa: WPS433

    if extract_service is None:
        raise ValueError("extract_service is required to build LightRAG role LLMs")

    def _role_config(
        service: "AIService",
        *,
        max_tokens: Optional[int] = None,
        json_marker: Optional[str] = None,
    ) -> "RoleLLMConfig":
        return RoleLLMConfig(
            func=build_llm_model_func(
                service,
                temperature=temperature,
                max_tokens=max_tokens,
                json_marker=json_marker,
            ),
            metadata={
                "binding": getattr(service, "provider", None),
                "model": getattr(service, "description", None),
            },
        )

    # Only `extract` is capped: it is the role that generates long JSON records
    # per chunk during indexing. `keyword` answers queries with a short list and
    # `vlm` describes images, neither of which runs away.
    configs: dict[str, RoleLLMConfig] = {
        'extract': _role_config(
            extract_service,
            max_tokens=config.LIGHTRAG_EXTRACT_MAX_TOKENS,
            json_marker=_resolve_json_marker(extract_service),
        ),
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
    max_tokens: Optional[int] = None,
    json_marker: Optional[str] = None,
) -> Callable[..., Awaitable[str]]:
    """Build a LightRAG-shaped ``llm_model_func`` from an :class:`AIService`.

    Reuses :func:`tools.aiServiceTools.create_llm_from_service` so all
    provider-switching logic stays in one place.

    The returned coroutine accepts LightRAG's call signature:
        ``async def(prompt, system_prompt=None, history_messages=None, **kwargs) -> str``

    History messages are passed through as already-formatted LangChain-style
    dicts (``{"role": "user"|"assistant"|"system", "content": "..."}``). When
    ``system_prompt`` is provided it is prepended as a system message.

    ``max_tokens`` caps the completion length. Left unset, an OpenAI-compatible
    server fills the remaining context window (~30k on vLLM), so a model that
    starts inventing relationships runs for minutes before stopping.

``json_marker``, when given, sends :data:`_EXTRACTION_JSON_SCHEMA` as
    ``response_format`` on the calls whose ``system_prompt`` contains that
    sentence, so the server constrains decoding to the schema. It is the
    discriminator built by :func:`_resolve_json_marker`; passing ``None`` keeps
    LightRAG's prompt-only JSON, where nothing prevents malformed output.
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
        invoke_kwargs: dict[str, Any] = {}
        if max_tokens:
            invoke_kwargs["max_tokens"] = max_tokens
        if json_marker and system_prompt and json_marker in system_prompt:
            invoke_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "lightrag_entity_extraction",
                    "schema": _EXTRACTION_JSON_SCHEMA,
                },
            }
        try:
            response = await llm.ainvoke(
                messages,
                config={"metadata": {"lc_source": "lightrag"}},
                **invoke_kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised unless salvageable
            response = _salvage_length_limit(exc)
            if response is None:
                raise
            logger.warning(
                "LightRAG extraction hit max_tokens=%s; keeping the %d characters "
                "produced before the cut (json_repair recovers the complete records)",
                max_tokens, len(response.content),
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
