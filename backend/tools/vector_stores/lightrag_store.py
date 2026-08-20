"""LightRAG implementation of the vector store interface.

This module provides a LightRAG-backed implementation of
:class:`VectorStoreInterface`, wrapping ``lightrag-hku==1.5.6``'s embedded
Python API behind the same abstract interface used by PGVectorStore and
QdrantStore.

All ``lightrag`` imports are **lazy** (inside methods) so this file is
importable even when ``lightrag-hku`` is not installed.

Note: Requires ``lightrag-hku`` and its storage extras:
    pip install 'lightrag-hku[offline-storage]==1.5.6'
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from tools.vector_stores.vector_store_interface import VectorStoreInterface
from utils.logger import get_logger

logger = get_logger(__name__)


class _DropUnconfiguredRoleLog(logging.Filter):
    """Hide LightRAG's per-role config line for the ``query`` role.

    We deliberately never configure the ``query`` role — we use
    ``only_need_context=True`` so the agent's own LLM produces the answer — so
    its config line renders as noise (``- query: None/None, host=None, ...``).
    We drop only that role's line and keep every other role (including
    ``vlm``, which should show whenever it is configured).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if (
            isinstance(args, tuple)
            and len(args) == 6
            and isinstance(record.msg, str)
            and record.msg.lstrip().startswith("- %s: %s/%s")
            and args[0] == "query"
        ):
            return False  # suppress only the unconfigurable query role
        return True


# Installed once at import; LightRAG's logger is the module-level "lightrag".
logging.getLogger("lightrag").addFilter(_DropUnconfiguredRoleLog())

# Valid LightRAG query modes — used for pass-through validation.
_LIGHTRAG_MODES = frozenset({"local", "global", "hybrid", "naive", "mix", "bypass"})

# Maps a silo's chunking strategy to LightRAG's native ``process_options``
# selector char (see lightrag.constants.PROCESS_OPTION_CHUNK_*). Unknown or
# legacy values (e.g. the old "token_window") fall back to fixed-token "F".
_CHUNK_STRATEGY_OPTION = {
    "fixed_token": "F",
    "recursive_character": "R",
    "semantic_vector": "V",
    "paragraph_semantic": "P",
}

# Spanish translation of LightRAG's own English default entity-type list
# (Person, Creature, Organization, Location, Event, Concept, Method, Content,
# Data, Artifact, NaturalObject). Used only when a silo's language is Spanish
# and no explicit lightrag_entity_types was configured — LightRAG does not
# localize this list itself, so this project maintains the translation.
# Keep in sync with the frontend copy in
# frontend/src/components/forms/LightRAGAdvancedSettings.tsx.
_SPANISH_DEFAULT_ENTITY_TYPES = [
    "Persona", "Criatura", "Organización", "Lugar", "Evento",
    "Concepto", "Método", "Contenido", "Datos", "Artefacto", "ObjetoNatural",
]


def _parse_entity_types(raw: Optional[str]) -> List[str]:
    """Split a comma-separated entity-types string into a clean list.

    Trims whitespace, drops empty items, and drops case-insensitive
    duplicates (keeping the first occurrence).
    """
    if not raw:
        return []
    seen: set[str] = set()
    result: List[str] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _build_entity_types_guidance(entity_types: List[str]) -> str:
    """Render a type list into LightRAG's ``entity_types_guidance`` shape.

    LightRAG 1.5.6 does not accept a bare type list — its extraction
    prompt only reads ``addon_params['entity_types_guidance']``, a
    multi-line instruction string (see ``lightrag/prompt.py``'s
    ``default_entity_types_guidance``). This mirrors that shape (header
    sentence + one bullet per type) without per-type descriptions, since
    this project only collects type names from the user, not descriptions.
    """
    bullets = "\n".join(f"- {entity_type}" for entity_type in entity_types)
    return (
        "Classify each entity using one of the following types. "
        "If no type fits, use `Other`.\n\n" + bullets
    )


def _source_label_from_metadata(metadata: dict) -> str:
    """Build a human-readable source label for a chunk from its document metadata.

    Prefers the original file name and appends the page number when available
    (PDFs carry ``page``). Falls back through other known location keys, and
    finally to LightRAG's own ``"unknown_source"`` sentinel so the UI can detect
    and label truly source-less chunks consistently.
    """
    name = (
        metadata.get("name")
        or metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("ref")
    )
    if not name:
        return "unknown_source"

    # Keep just the file name, not the full path.
    label = str(name).replace("\\", "/").split("/")[-1]

    page = metadata.get("page")
    if page is not None:
        label = f"{label} (p. {page})"
    return label


# Matches a chunk id LightRAG derives from an id _lightrag_doc_id returned
# (f"{doc_id}-chunk-{n}"): resource id and page, straight out of the id
# instead of re-parsing the human-readable file_path label.
CHUNK_ID_RESOURCE_PAGE_RE = re.compile(r"^res(\d+)-p(\d+)-chunk-\d+$")


def _lightrag_doc_id(metadata: dict, content: str) -> str:
    """Stable LightRAG document id for one page, so a chunk's own id can be
    parsed straight back to (resource_id, page) — see CHUNK_ID_RESOURCE_PAGE_RE
    — instead of guessing from the display-only file_path label.

    Content-hash duplicate detection (lightrag/pipeline.py,
    get_existing_doc_by_content_hash) is independent of the id we supply, so
    this changes nothing about resume/dedup behaviour.

    Falls back to LightRAG's own default (content hash) for content with no
    resource/page — non-PDF sources such as crawled Domain pages.
    """
    resource_id = metadata.get("resource_id")
    page = metadata.get("page")
    if resource_id is not None and page is not None:
        return f"res{resource_id}-p{page}"
    from lightrag.utils import compute_mdhash_id  # noqa: WPS433

    return compute_mdhash_id(content, prefix="doc-")


async def _ainsert(rag, texts, file_paths=None, ids=None, process_options="F"):
    """Insert documents through LightRAG's modern (non-legacy) chunking router.

    The plain ``rag.ainsert`` enqueues without a chunking selector, which makes
    LightRAG fall back to its legacy 6-arg ``chunking_func`` path (logged as
    ``Chunking F(legacy)``). Enqueuing with ``process_options="F"`` instead
    selects the fixed-token chunker explicitly, so the new file-chunker contract
    is used (logged as ``Chunking F``). The chunk size/overlap still come from
    the silo config via ``addon_params['chunker']``.
    """
    from lightrag.parser.routing import resolve_chunk_options

    chunk_opts = resolve_chunk_options(
        rag.addon_params, process_options=process_options
    )
    await rag.apipeline_enqueue_documents(
        texts,
        ids=ids,
        file_paths=file_paths,
        process_options=process_options,
        chunk_options=chunk_opts,
    )
    await rag.apipeline_process_enqueue_documents()


async def _ainsert_with_progress(rag, texts, progress_callback=None, file_paths=None, ids=None, process_options="F"):
    """Run the insert with optional sub-document progress reporting.

    Polls the workspace's doc_status counts every 0.5s while the insert runs.

    ``file_paths`` (one per text) is forwarded to LightRAG for citation so the
    retrieved chunks carry a human-readable source instead of "unknown_source".

    ``progress_callback`` is called as ``(done, total)`` where both are counted
    in LightRAG documents (one per page for PDFs) — the same unit, so the
    caller can render a percentage without mixing it with its own estimates.

    ``done`` counts documents that actually **finished**.  The obvious counter,
    ``pipeline_status["cur_batch"]``, is incremented when a document *enters*
    the ``max_parallel_insert`` semaphore (``lightrag/pipeline.py:2155-2178``),
    so the first 3 pages complete "instantly" and the bar stays exactly
    ``max_parallel_insert`` ahead of reality — which also biased the ETA, since
    it is extrapolated from the first samples.  doc_status is the only source
    of true completions: pipeline_status has no such counter.
    """
    if progress_callback is None:
        await _ainsert(rag, texts, file_paths=file_paths, ids=ids, process_options=process_options)
        return

    total = len(texts)

    async def _finished_docs():
        """Documents of this workspace in a terminal state, as one GROUP BY."""
        counts = await rag.doc_status.get_status_counts()
        return counts.get("processed", 0) + counts.get("failed", 0)

    # Earlier resources indexed into the same silo are already 'processed', so
    # only the delta belongs to this run.  Read before the insert starts: a
    # cache hit can finish a document inside the first poll interval.
    # doc_status is always PostgreSQL in this project (see storage_config.py).
    try:
        baseline = await _finished_docs()
    except Exception:
        baseline = 0

    async def _real_total() -> int:
        """How many documents this run will actually process.

        On a resumed file most pages are already ``PROCESSED`` and LightRAG skips
        them, so ``len(texts)`` overstates the work left and the bar would sit at
        3/10 and snap to 100%. LightRAG publishes the real figure as
        ``pipeline_status["docs"] = len(to_process_docs)`` (``pipeline.py:1085``).

        Guarded because that value is per-workspace state: during the first ticks
        the enqueue may not have run yet and it still holds the previous run's
        number. Outside ``1..len(texts)`` it is ignored.
        """
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            status = await get_namespace_data(
                "pipeline_status", workspace=getattr(rag, "workspace", None)
            )
            docs = int(status.get("docs") or 0)
        except Exception:
            return total
        return docs if 0 < docs <= total else total

    # Last known real total, so the final 100% call uses the same denominator
    # the bar has been showing instead of jumping back to len(texts).
    run_total = [total]

    async def _poll():
        while True:
            await asyncio.sleep(0.5)
            try:
                done = await _finished_docs() - baseline
                run_total[0] = await _real_total()
                progress_callback(max(0, min(run_total[0] - 1, done)), run_total[0])
            except Exception:
                pass  # polling is best-effort

    poll_task = asyncio.create_task(_poll())
    try:
        await _ainsert(rag, texts, file_paths=file_paths, ids=ids, process_options=process_options)
    finally:
        poll_task.cancel()
        await asyncio.gather(poll_task, return_exceptions=True)

    progress_callback(run_total[0], run_total[0])


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    Reuses the current thread's event loop when available, so that Neo4j and
    other async clients that bind futures to a specific loop stay on the same
    loop across multiple calls (avoids "Future attached to a different loop").

    If an event loop is already running (e.g. inside an async FastAPI handler),
    offloads to a fresh worker thread instead to avoid blocking.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # Synchronous context (e.g. a background threading.Thread).
        # Prefer the thread's own persistent loop so all coroutines within this
        # thread share one loop — keeps Neo4j connections consistent.
        try:
            thread_loop = asyncio.get_event_loop()
            if thread_loop.is_closed():
                raise RuntimeError("loop is closed")
        except RuntimeError:
            thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(thread_loop)

        return thread_loop.run_until_complete(coro)

    # An event loop is already running — offload to a new thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class LightRAGRetriever(BaseRetriever):
    """LangChain retriever backed by a LightRAG instance.

    This retriever wraps ``LightRAG.aquery`` so it can be plugged into any
    LangChain chain that expects a :class:`~langchain_core.retrievers.BaseRetriever`.

    The retriever always calls LightRAG with ``only_need_context=True``,
    skipping LightRAG's own LLM generation and returning the raw graph context
    instead. The agent's LLM handles query synthesis. The ``raw_data``
    (entities, relationships, chunks, references) is stored in the Document
    metadata under the ``lightrag_raw_data`` key so the streaming layer can
    surface it as a knowledge-graph bubble in the playground UI.

    ``store`` + ``collection_name`` are used instead of a pre-built
    ``rag_instance`` so that the async path can call ``store._aget_rag_instance``
    and ensure Neo4j is initialised in the *same* event loop that runs
    ``aquery()``.  Using a pre-built instance would bind the Neo4j driver to
    a different loop (the one that ran ``initialize_storages``), causing
    "Future attached to a different loop" errors on subsequent requests.
    """

    store: Any = Field(exclude=True)
    collection_name: str
    query_mode: str = "hybrid"
    top_k: int = 5

    class Config:  # noqa: D106
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        from lightrag.base import QueryParam  # noqa: WPS433

        rag = self.store._get_rag_instance(self.collection_name)
        param = QueryParam(
            mode=self.query_mode,
            top_k=self.top_k,
            only_need_context=True,
        )
        # aquery_llm returns a dict with both the context string
        # (llm_response.content) and the structured graph data (data.*).
        # The legacy aquery() wrapper discards raw_data, so we can't use it.
        response = _run_async(rag.aquery_llm(query, param=param))
        docs = _wrap_query_response(response, self.query_mode)
        logger.debug("[LightRAG retriever] query=%r mode=%s top_k=%d → %d doc(s)", query, self.query_mode, self.top_k, len(docs))
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        from lightrag.base import QueryParam  # noqa: WPS433

        # _aget_rag_instance initialises Neo4j in this event loop, ensuring
        # the driver and aquery() share the same loop on every call.
        rag = await self.store._aget_rag_instance(self.collection_name)
        param = QueryParam(
            mode=self.query_mode,
            top_k=self.top_k,
            only_need_context=True,
        )
        # aquery_llm returns both context (llm_response.content) and graph
        # data (data.*); the legacy aquery() wrapper discards raw_data.
        response = await rag.aquery_llm(query, param=param)
        docs = _wrap_query_response(response, self.query_mode)
        logger.debug("[LightRAG retriever] query=%r mode=%s top_k=%d → %d doc(s)", query, self.query_mode, self.top_k, len(docs))
        return docs


_LIGHTRAG_KEYWORDS_RE = re.compile(
    r'\{[^{}]*"high_level_keywords"[^{}]*\}', re.DOTALL
)
# LightRAG global-mode context embeds community-report citation tags like
# [Data: Reports (1, 2, 3)] or [Data: Entities (4)].  The outer LLM reads
# these IDs and echoes them as "[N]" citations — meaningless to users.
_LIGHTRAG_DATA_TAG_RE = re.compile(r'\[Data:[^\]]+\]')


def _extract_lightrag_keywords(text: str) -> tuple[str, dict]:
    """Split LightRAG context string into (clean_content, keywords_dict).

    Strips two kinds of LightRAG metadata from page_content so the outer LLM
    never sees them:
    - The query-keyword JSON block (high_level_keywords / low_level_keywords)
    - [Data: Reports (N, N)] citation tags from global-mode community reports
    Both are internal LightRAG artefacts, not retrieved knowledge.
    """
    import json as _json
    m = _LIGHTRAG_KEYWORDS_RE.search(text)
    keywords = {}
    if m:
        try:
            keywords = _json.loads(m.group())
        except Exception:
            pass
        text = _LIGHTRAG_KEYWORDS_RE.sub("", text).strip()
    text = _LIGHTRAG_DATA_TAG_RE.sub("", text).strip()
    return text, keywords


def _wrap_response(response: str, query_mode: str) -> List[Document]:
    """Wrap a plain string LightRAG response into a LangChain Document."""
    if not response or not str(response).strip():
        return []
    content, keywords = _extract_lightrag_keywords(str(response))
    return [
        Document(
            page_content=content,
            metadata={"source": "lightrag", "query_mode": query_mode, "lightrag_keywords": keywords},
        )
    ]


def _wrap_query_response(response: Any, query_mode: str) -> List[Document]:
    """Wrap a LightRAG ``aquery_llm`` response into LangChain Documents.

    *response* is the dict returned by ``rag.aquery_llm`` — it carries the
    context string under ``llm_response.content`` (because we always query with
    ``only_need_context=True``) and the structured graph data under ``data``.
    That graph data is stored in Document metadata under ``lightrag_raw_data``
    so the streaming layer can surface it as a knowledge-graph bubble in the
    playground UI. A plain ``QueryResult``/string is still accepted as a
    fallback for backward compatibility.
    """
    if not response:
        return []

    if isinstance(response, dict):
        content = (response.get("llm_response") or {}).get("content") or ""
        raw_data = response
    else:
        llm_r = getattr(response, "llm_response", None)
        content = getattr(llm_r, "content", None) if llm_r else None
        content = content or getattr(response, "content", None) or str(response)
        raw_data = getattr(response, "raw_data", None) or {}
    raw_content = str(content)
    content, keywords = _extract_lightrag_keywords(raw_content)
    if keywords:
        logger.info("[LightRAG] stripped keywords from content (found=%s)", list(keywords.keys()))
    if "high_level_keywords" in raw_content and not keywords:
        logger.warning("[LightRAG] keyword JSON in content but regex did NOT strip it; tail=%r", raw_content[-200:])
    graph_data = _normalize_lightrag_graph(raw_data)
    graph = graph_data.get("data", {})
    has_graph = bool(graph.get("entities") or graph.get("chunks"))
    if not content:
        logger.warning("[LightRAG] content empty after strip for mode=%s (has_graph=%s)", query_mode, has_graph)
        if not has_graph:
            return []
        # Return a doc with a minimal placeholder so the artifact carries graph data
        # even when LightRAG produces no context string (e.g. empty corpus, global mode).
        content = "(no context retrieved)"
    logger.info("[LightRAG] graph_data entities=%d relationships=%d chunks=%d",
                len(graph_data.get("data", {}).get("entities", [])),
                len(graph_data.get("data", {}).get("relationships", [])),
                len(graph_data.get("data", {}).get("chunks", [])))
    return [
        Document(
            page_content=str(content),
            metadata={
                "source": "lightrag",
                "query_mode": query_mode,
                "lightrag_raw_data": graph_data,
                "lightrag_keywords": keywords,
            },
        )
    ]


def _normalize_lightrag_graph(raw_data: dict) -> dict:
    """Normalize LightRAG raw_data to the shape expected by the frontend LightRAGGraphData type.

    LightRAG returns entity_name/entity_type/description and entity1/entity2 for
    relationships. The frontend expects id/name for entities and id/source/target
    for relationships.
    """
    if not isinstance(raw_data, dict):
        return {}
    data = raw_data.get("data", {})
    if not isinstance(data, dict):
        try:
            data = vars(data)
        except TypeError:
            logger.warning("[LightRAG] _normalize: cannot coerce %s to dict", type(data).__name__)
            return {}
    entities = []
    known = set()
    for e in data.get("entities", []):
        name = e.get("entity_name") or e.get("name") or ""
        if name:
            known.add(name)
        entities.append({
            "id": name,
            "name": name,
            "entity_type": e.get("entity_type", ""),
            "description": e.get("description", ""),
            "file_path": e.get("file_path", ""),
            # <SEP>-joined chunk ids this entity was extracted from — lets the
            # frontend highlight entities that came from cited chunks.
            "source_id": e.get("source_id", ""),
        })

    relationships = []
    for i, r in enumerate(data.get("relationships", [])):
        src = r.get("src_id") or r.get("entity1") or r.get("source") or ""
        tgt = r.get("tgt_id") or r.get("entity2") or r.get("target") or ""
        relationships.append({
            "id": r.get("id") or f"rel_{i}",
            "source": src,
            "target": tgt,
            "description": r.get("description", ""),
            "keywords": r.get("keywords", ""),
            "source_id": r.get("source_id", ""),
        })
        # LightRAG truncates entities and relationships by independent token
        # budgets, so a relationship endpoint may not appear in the entity list.
        # Add it as a minimal "partial" node so the edge can still be drawn;
        # the frontend renders these in a muted color.
        for endpoint in (src, tgt):
            if endpoint and endpoint not in known:
                known.add(endpoint)
                entities.append({
                    "id": endpoint,
                    "name": endpoint,
                    "entity_type": "",
                    "description": "",
                    "file_path": "",
                    "partial": True,
                })

    chunks = []
    for c in data.get("chunks", []):
        chunk_id = c.get("chunk_id") or c.get("id") or ""
        entry = {
            "id": chunk_id,
            "content": c.get("content") or c.get("text") or "",
            "file_path": c.get("file_path", ""),
        }
        match = CHUNK_ID_RESOURCE_PAGE_RE.match(chunk_id)
        if match:
            entry["resource_id"] = int(match.group(1))
            entry["page"] = int(match.group(2))
        chunks.append(entry)

    return {
        "data": {
            "entities": entities,
            "relationships": relationships,
            "chunks": chunks,
            "references": data.get("references", []),
        }
    }


def _resolve_query_mode(search_type: str) -> str:
    """Map a ``search_type`` string to a LightRAG query mode.

    Recognises the standard ``VectorStoreInterface`` values (``similarity``,
    ``mmr``) and also passes through native LightRAG modes unchanged.
    """
    if search_type in _LIGHTRAG_MODES:
        return search_type
    mapping = {
        "similarity": "hybrid",
        "similarity_score_threshold": "hybrid",
        "mmr": "mix",
    }
    return mapping.get(search_type, "hybrid")


class LightRAGStore(VectorStoreInterface):
    """LightRAG implementation of the vector store interface.

    Each *collection* (``silo_{id}``) maps to a separate ``LightRAG``
    workspace.  Instances are created lazily on first use and cached for the
    lifetime of this store.
    """

    def __init__(
        self,
        db,
        ai_service,
        embedding_service,
        workspace_prefix: str = "silo",
        *,
        keywords_service=None,
        vlm_service=None,
        lightrag_vector_db_type: Optional[str] = None,
        lightrag_chunk_token_size: Optional[int] = None,
        lightrag_chunk_overlap_token_size: Optional[int] = None,
        lightrag_chunk_strategy: Optional[str] = None,
        lightrag_language: Optional[str] = None,
        lightrag_entity_extract_max_gleaning: Optional[int] = None,
        lightrag_max_source_ids_per_entity: Optional[int] = None,
        lightrag_max_source_ids_per_relation: Optional[int] = None,
        lightrag_entity_types: Optional[str] = None,
    ):
        self.db = db
        # ``ai_service`` is the legacy single-LLM parameter — it acts as the
        # EXTRACT role and as a fallback for KEYWORDS/VLM when those are
        # not configured. The new role-specific services let callers
        # override each role independently.
        self.ai_service = ai_service
        self.extract_service = ai_service
        self.keywords_service = keywords_service
        self.vlm_service = vlm_service
        self.lightrag_vector_db_type = (lightrag_vector_db_type or "QDRANT").upper()
        self._chunk_token_size = lightrag_chunk_token_size
        self._chunk_overlap_token_size = lightrag_chunk_overlap_token_size
        self._chunk_process_option = _CHUNK_STRATEGY_OPTION.get(
            lightrag_chunk_strategy, "F"
        )
        self.language = lightrag_language
        self.entity_extract_max_gleaning = lightrag_entity_extract_max_gleaning
        self.max_source_ids_per_entity = lightrag_max_source_ids_per_entity
        self.max_source_ids_per_relation = lightrag_max_source_ids_per_relation
        self.entity_types = lightrag_entity_types
        self.embedding_service = embedding_service
        self.workspace_prefix = workspace_prefix
        self._rag_instances: Dict[str, Any] = {}
        # Per-collection asyncio locks to prevent concurrent initializations.
        self._init_locks: Dict[str, asyncio.Lock] = {}
        # Shared temp directory so LightRAG's working_dir requirement is met
        # without polluting the project tree.  The directory is only used for
        # ancillary local caches (e.g. tiktoken); actual data lives in
        # Neo4j / Qdrant / PostgreSQL.
        self._working_dir = tempfile.mkdtemp(prefix="lightrag_")

    def _get_rag_instance(self, collection_name: str):
        """Return a cached ``LightRAG`` instance for *collection_name*.

        Used by synchronous callers (indexing, background threads). Async
        callers (query path) should use ``_aget_rag_instance`` instead so that
        Neo4j is initialised in the same event loop that will run ``aquery()``.
        """
        if collection_name in self._rag_instances:
            return self._rag_instances[collection_name]

        rag = self._build_rag(collection_name)
        _run_async(rag.initialize_storages())
        self._rag_instances[collection_name] = rag
        logger.info("Created LightRAG instance for workspace '%s'", collection_name)
        return rag

    def _build_rag(self, collection_name: str):
        """Build a fresh (uninitialised) LightRAG instance for *collection_name*."""
        from lightrag import LightRAG  # noqa: WPS433

        from tools.vector_stores.lightrag.adapters import (
            build_embedding_func,
            build_llm_model_func,
            build_role_llm_configs,
        )
        from tools.vector_stores.lightrag.storage_config import build_storage_config

        import config  # noqa: WPS433

        storage_cfg = build_storage_config(vector_db_type=self.lightrag_vector_db_type)
        base_llm_func = build_llm_model_func(self.extract_service)
        role_configs = build_role_llm_configs(
            extract_service=self.extract_service,
            keywords_service=self.keywords_service,
            vlm_service=self.vlm_service,
        )
        emb_func = build_embedding_func(self.embedding_service)

        # Forward the silo's chunking/source-id-cap config to LightRAG.
        # Omit each key when unset so LightRAG keeps its own defaults.
        extra_kwargs: Dict[str, Any] = {}

        # Force JSON-structured extraction unconditionally — LightRAG's own
        # default (env var ENTITY_EXTRACTION_USE_JSON, false unless set) causes
        # silent, severe data loss with the delimited-text format on several
        # LLMs (open-source and cloud alike): runaway/repetitive generation
        # that never reaches the completion marker, and record-prefix/newline
        # collapse on repetitive content that the parser can't recover from.
        # JSON mode can't have either failure by construction (schema-validated
        # objects, not free text). See docs/testing/lightrag_extraction_benchmark_corpus.md.
        # Passed as a kwarg (not the env var) so no deployment can silently
        # regress to the broken default.
        extra_kwargs["entity_extraction_use_json"] = True

        if self._chunk_token_size:
            extra_kwargs["chunk_token_size"] = self._chunk_token_size
        if self._chunk_overlap_token_size is not None:
            extra_kwargs["chunk_overlap_token_size"] = self._chunk_overlap_token_size
        if self.max_source_ids_per_entity:
            extra_kwargs["max_source_ids_per_entity"] = self.max_source_ids_per_entity
        if self.max_source_ids_per_relation:
            extra_kwargs["max_source_ids_per_relation"] = self.max_source_ids_per_relation

        gleaning = (
            self.entity_extract_max_gleaning
            if self.entity_extract_max_gleaning is not None
            else config.ENTITY_EXTRACT_MAX_GLEANING
        )

        rag = LightRAG(
            working_dir=self._working_dir,
            workspace=collection_name,
            llm_model_func=base_llm_func,
            role_llm_configs=role_configs,
            embedding_func=emb_func,
            graph_storage=storage_cfg["graph_storage"],
            vector_storage=storage_cfg["vector_storage"],
            kv_storage=storage_cfg["kv_storage"],
            doc_status_storage=storage_cfg["doc_status_storage"],
            entity_extract_max_gleaning=gleaning,
            **extra_kwargs,
        )
        # Mutate in place (not a constructor kwarg) so LightRAG's own
        # addon_params defaults (e.g. entity_types) aren't clobbered.
        # Shared by both entity extraction (indexing) and keyword extraction
        # (query) — LightRAG has no per-role language override.
        if self.language:
            rag.addon_params["language"] = self.language

        # entity_types resolution — LightRAG 1.5.6 reads
        # addon_params['entity_types_guidance'] (a guidance string), not a
        # bare type list. LightRAG does not localize its own default
        # guidance, so a Spanish silo left blank gets this project's Spanish
        # translation instead; an English/unset silo left blank keeps
        # LightRAG's own built-in English default guidance untouched.
        explicit_entity_types = _parse_entity_types(self.entity_types)
        if explicit_entity_types:
            rag.addon_params["entity_types_guidance"] = _build_entity_types_guidance(
                explicit_entity_types
            )
        elif self.language == "Spanish":
            rag.addon_params["entity_types_guidance"] = _build_entity_types_guidance(
                _SPANISH_DEFAULT_ENTITY_TYPES
            )
        return rag

    async def _aget_rag_instance(self, collection_name: str):
        """Async version of _get_rag_instance.

        Initialises LightRAG storages (including the Neo4j driver) in the
        *caller's* event loop so that subsequent ``aquery()`` calls reuse the
        same loop. Using _run_async from an async context would offload to a
        worker thread with a fresh loop, binding the Neo4j driver to a loop
        that is gone by the next request — triggering "Future attached to a
        different loop" errors.
        """
        if collection_name in self._rag_instances:
            return self._rag_instances[collection_name]

        if collection_name not in self._init_locks:
            self._init_locks[collection_name] = asyncio.Lock()

        async with self._init_locks[collection_name]:
            # Re-check after acquiring lock (another coroutine may have initialized it).
            if collection_name in self._rag_instances:
                return self._rag_instances[collection_name]

            rag = self._build_rag(collection_name)
            await rag.initialize_storages()
            self._rag_instances[collection_name] = rag
            logger.info("Created LightRAG instance (async) for workspace '%s'", collection_name)
            return rag

    # ------------------------------------------------------------------
    # VectorStoreInterface implementation
    # ------------------------------------------------------------------

    def index_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embedding_service=None,
        progress_callback=None,
    ) -> Optional[Dict]:
        """Index documents and return token/timing metrics dict, or None when skipped."""
        if not documents:
            return None

        logger.info(
            "Indexing %d documents into LightRAG workspace '%s'",
            len(documents),
            collection_name,
        )

        rag = self._get_rag_instance(collection_name)
        # Keep texts, source labels and ids aligned (skip empty-content docs).
        texts: List[str] = []
        file_paths: List[str] = []
        ids: List[str] = []
        for doc in documents:
            if not doc.page_content:
                continue
            texts.append(doc.page_content)
            file_paths.append(_source_label_from_metadata(doc.metadata or {}))
            ids.append(_lightrag_doc_id(doc.metadata or {}, doc.page_content))

        if not texts:
            logger.debug("No non-empty texts to index; skipping.")
            return None

        from tools.vector_stores.lightrag.token_accumulator import IndexingTokenAccumulator
        from tools.vector_stores.lightrag.adapters import (
            set_active_accumulator,
            reset_active_accumulator,
        )

        accumulator = IndexingTokenAccumulator()
        ctx_token = set_active_accumulator(accumulator)
        t_start = time.perf_counter()
        try:
            _run_async(_ainsert_with_progress(rag, texts, progress_callback, file_paths=file_paths, ids=ids, process_options=self._chunk_process_option))
        finally:
            reset_active_accumulator(ctx_token)

        duration = time.perf_counter() - t_start
        totals = accumulator.totals()
        totals["duration_seconds"] = round(duration, 3)

        logger.info(
            "Successfully indexed %d texts into workspace '%s' "
            "in %.1fs (%d tokens, %d LLM calls)",
            len(texts),
            collection_name,
            duration,
            totals["total_tokens"],
            totals["llm_calls"],
        )
        return totals

    def delete_documents(
        self,
        collection_name: str,
        ids,
        embedding_service=None,
    ) -> None:
        # ponytail: no-op by design. Per-document delete in LightRAG means
        # unwinding the doc's entities/relations from the knowledge graph, which
        # is not supported yet. Whole-silo teardown still works via
        # delete_collection(). Upgrade path: wire adelete_by_doc_id + graph
        # entity/relation cleanup (followup #2).
        logger.warning(
            "LightRAGStore.delete_documents is a no-op (per-document graph "
            "deletion unsupported); data retained for workspace '%s'. Delete the "
            "whole silo to purge.",
            collection_name,
        )

    def delete_documents_excluding(
        self,
        collection_name: str,
        filter_metadata: Dict[str, Any],
        exclude: Dict[str, Any],
        embedding_service=None,
    ) -> None:
        # ponytail: no-op — LightRAG deletes by doc_id, not metadata filter, so
        # it can't run the index-then-swap stale-chunk purge. Re-inserting the
        # same content is idempotent (doc_id = content hash), so reindex still
        # refreshes; orphaned chunks from removed content are the known ceiling.
        # Upgrade path: map resource_id -> doc_ids and delete_by_doc_id (followup #2/#3).
        logger.warning(
            "LightRAGStore.delete_documents_excluding is a no-op (metadata-filter "
            "deletion unsupported); stale chunks not purged for workspace '%s'.",
            collection_name,
        )

    def delete_collection(
        self,
        collection_name: str,
        embedding_service=None,
    ) -> None:
        logger.info("Deleting LightRAG workspace '%s'", collection_name)

        # Remove cached instance first.
        self._rag_instances.pop(collection_name, None)

        # Best-effort cleanup of the underlying storage backends.
        # LightRAG does not expose a single "drop workspace" API so we reach
        # into each backend directly.
        self._cleanup_neo4j(collection_name)
        if self.lightrag_vector_db_type == "QDRANT":
            self._cleanup_qdrant(collection_name)
        self._cleanup_postgres(collection_name)

    # -- Backend-specific cleanup helpers ---------------------------------

    def _cleanup_neo4j(self, collection_name: str) -> None:
        """Drop all Neo4j nodes/edges scoped to *collection_name*."""
        try:
            import config  # noqa: WPS433

            uri = getattr(config, "NEO4J_URI", None)
            password = getattr(config, "NEO4J_PASSWORD", None)
            username = getattr(config, "NEO4J_USERNAME", None) or "neo4j"
            if not uri or not password:
                return

            from neo4j import GraphDatabase  # noqa: WPS433

            driver = GraphDatabase.driver(uri, auth=(username, password))
            # LightRAG stores the workspace as a node *label*, not a property.
            # Use label-based matching for reliable deletion.
            with driver.session() as session:
                session.run(
                    f"MATCH (n:`{collection_name}`) DETACH DELETE n",
                )
            driver.close()
            logger.info("Cleaned up Neo4j data for workspace '%s'", collection_name)
        except Exception as exc:
            logger.warning("Neo4j cleanup for '%s' failed: %s", collection_name, exc)

    def _cleanup_qdrant(self, collection_name: str) -> None:
        """Drop Qdrant collections prefixed with ``lightrag_{collection_name}``."""
        try:
            import config  # noqa: WPS433
            from qdrant_client import QdrantClient  # noqa: WPS433

            url = getattr(config, "QDRANT_URL", None) or "http://localhost:6333"
            api_key = getattr(config, "QDRANT_API_KEY", None)
            client = QdrantClient(url=url, api_key=api_key)

            prefix = f"lightrag_{collection_name}_"
            for col in client.get_collections().collections:
                if col.name.startswith(prefix):
                    client.delete_collection(col.name)
                    logger.debug("Deleted Qdrant collection '%s'", col.name)

            client.close()
            logger.info("Cleaned up Qdrant collections for workspace '%s'", collection_name)
        except Exception as exc:
            logger.warning("Qdrant cleanup for '%s' failed: %s", collection_name, exc)

    def _cleanup_postgres(self, collection_name: str) -> None:
        """Remove KV / vector / doc-status rows scoped to *collection_name*."""
        try:
            from sqlalchemy import text  # noqa: WPS433

            # LightRAG's Postgres backend names its tables ``LIGHTRAG_*`` (see
            # NAMESPACE_TABLE_MAP), not ``kv_store_*``. It creates them with
            # unquoted DDL, so Postgres folds the identifiers to lower case —
            # we must delete using the lower-cased names (a quoted upper-case
            # name would not match). KV + doc-status live in Postgres for every
            # LightRAG silo; the vector tables (lightrag_vdb_*) only when
            # lightrag_vector_db_type=PGVECTOR. Deleting them all is harmless
            # otherwise. Best-effort — tables might not exist yet.
            from lightrag.kg.postgres_impl import NAMESPACE_TABLE_MAP  # noqa: WPS433

            tables = {t.lower() for t in NAMESPACE_TABLE_MAP.values()}
            for table in tables:
                try:
                    self.db.execute(
                        text(f'DELETE FROM {table} WHERE workspace = :ws'),
                        {"ws": collection_name},
                    )
                except Exception:
                    pass  # Table may not exist — that's fine.
            self.db.commit()
            logger.info("Cleaned up PostgreSQL data for workspace '%s'", collection_name)
        except Exception as exc:
            logger.warning("PostgreSQL cleanup for '%s' failed: %s", collection_name, exc)

    # ------------------------------------------------------------------

    def search_similar_documents(
        self,
        collection_name: str,
        query: str,
        embedding_service=None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        k: int = 5,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ) -> List[Document]:
        from lightrag.base import QueryParam  # noqa: WPS433

        if filter_metadata:
            logger.debug(
                "LightRAGStore does not support filter_metadata; ignoring."
            )
        if score_threshold is not None:
            logger.debug(
                "LightRAGStore does not support score_threshold; ignoring."
            )
        if fetch_k is not None:
            logger.debug(
                "LightRAGStore does not support fetch_k; ignoring."
            )
        if lambda_mult is not None:
            logger.debug(
                "LightRAGStore does not support lambda_mult; ignoring."
            )

        mode = _resolve_query_mode(search_type)

        if mode == "bypass":
            return []

        rag = self._get_rag_instance(collection_name)
        param = QueryParam(mode=mode, top_k=k)

        response = _run_async(rag.aquery(query, param=param))
        return _wrap_response(response, mode)

    def get_retriever(
        self,
        collection_name: str,
        embedding_service=None,
        search_params: Optional[Dict[str, Any]] = None,
        use_async: bool = False,
        search_type: str = "similarity",
        **kwargs,
    ):
        search_params = search_params or {}

        # Allow callers to override the LightRAG query mode directly.
        mode = search_params.pop("lightrag_query_mode", None) or _resolve_query_mode(
            search_type
        )
        top_k = search_params.get("k", 5)

        return LightRAGRetriever(
            store=self,
            collection_name=collection_name,
            query_mode=mode,
            top_k=top_k,
        )

    async def aretrieve_graph_context(
        self,
        collection_name: str,
        query: str,
        mode: str,
        top_k: int,
    ) -> dict:
        """Async: call aquery_llm with only_need_context=True and return normalized graph data.

        Uses _aget_rag_instance so Neo4j is initialised in the *same* event loop
        that runs aquery_llm — avoids "Future attached to a different loop" errors.
        Unlike _get_relevant_documents, does NOT discard the result when the LLM
        context string is empty, so entities/chunks are returned even when LightRAG
        finds no strong keyword match.
        """
        from lightrag.base import QueryParam  # noqa: WPS433

        rag = await self._aget_rag_instance(collection_name)
        param = QueryParam(mode=mode, top_k=top_k, only_need_context=True)
        response = await rag.aquery_llm(query, param=param)

        if isinstance(response, dict):
            raw_data = response
        else:
            raw_data = getattr(response, "raw_data", None) or {}

        return _normalize_lightrag_graph(raw_data)

    def collection_exists(self, collection_name: str) -> bool:
        try:
            rag = self._get_rag_instance(collection_name)
            # If we can instantiate and initialise without error the workspace
            # is valid.  We cannot cheaply check for document count without a
            # backend-specific query — treat a successfully initialised
            # instance as "exists".
            return rag is not None
        except Exception as exc:
            logger.debug(
                "LightRAG collection_exists check failed for '%s': %s",
                collection_name,
                exc,
            )
            return False

    def count_documents(
        self,
        collection_name: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_content_length: Optional[int] = None,
        max_content_length: Optional[int] = None,
    ) -> int:
        if filter_metadata:
            logger.debug("LightRAGStore.count_documents ignores filter_metadata.")
        if min_content_length is not None or max_content_length is not None:
            logger.debug(
                "LightRAGStore.count_documents ignores content-length filters."
            )

        # Query LightRAG's doc-status table for this workspace. The Postgres
        # backend names it ``LIGHTRAG_DOC_STATUS`` via unquoted DDL, so the
        # real (lower-cased) table is ``lightrag_doc_status``.
        try:
            from sqlalchemy import text  # noqa: WPS433

            result = self.db.execute(
                text(
                    'SELECT COUNT(*) FROM lightrag_doc_status WHERE workspace = :ws'
                ),
                {"ws": collection_name},
            )
            row = result.scalar()
            return int(row) if row else 0
        except Exception as exc:
            logger.debug(
                "LightRAG count_documents failed for '%s': %s",
                collection_name,
                exc,
            )
            return 0

    def update_documents_metadata(
        self,
        collection_name: str,
        filter_metadata: Dict[str, Any],
        metadata_updates: Dict[str, Any],
        replace: bool = False,
    ) -> int:
        logger.warning(
            "LightRAGStore does not support metadata updates. "
            "Re-index documents to apply changes."
        )
        return 0

    def get_distinct_metadata_values(
        self,
        collection_name: str,
        field: str,
        prefix: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        logger.debug(
            "LightRAGStore does not support get_distinct_metadata_values; "
            "returning empty list."
        )
        return []
