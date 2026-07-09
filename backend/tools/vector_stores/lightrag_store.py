"""LightRAG implementation of the vector store interface.

This module provides a LightRAG-backed implementation of
:class:`VectorStoreInterface`, wrapping ``lightrag-hku==1.5.0rc3``'s embedded
Python API behind the same abstract interface used by PGVectorStore and
QdrantStore.

All ``lightrag`` imports are **lazy** (inside methods) so this file is
importable even when ``lightrag-hku`` is not installed.

Note: Requires ``lightrag-hku`` and its storage extras:
    pip install 'lightrag-hku[offline-storage]==1.5.0rc3'
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


async def _ainsert(rag, texts, file_paths=None, process_options="F"):
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
        file_paths=file_paths,
        process_options=process_options,
        chunk_options=chunk_opts,
    )
    await rag.apipeline_process_enqueue_documents()


async def _ainsert_with_progress(rag, texts, progress_callback=None, file_paths=None, process_options="F"):
    """Run the insert with optional sub-document progress reporting.

    Polls pipeline_status every 0.5s while the insert runs.  Falls back
    gracefully if the namespace isn't available.

    ``file_paths`` (one per text) is forwarded to LightRAG for citation so the
    retrieved chunks carry a human-readable source instead of "unknown_source".
    """
    if progress_callback is None:
        await _ainsert(rag, texts, file_paths=file_paths, process_options=process_options)
        return

    total = len(texts)

    async def _poll():
        while True:
            await asyncio.sleep(0.5)
            try:
                from lightrag.kg.shared_storage import get_namespace_data
                workspace = getattr(rag, 'workspace', None)
                status = await get_namespace_data("pipeline_status", workspace=workspace)
                cur = status.get("cur_batch", 0)
                batches = max(1, status.get("batchs", 1))
                progress_callback(min(total - 1, int(total * cur / batches)))
            except Exception:
                pass  # polling is best-effort

    poll_task = asyncio.create_task(_poll())
    try:
        await _ainsert(rag, texts, file_paths=file_paths, process_options=process_options)
    finally:
        poll_task.cancel()
        await asyncio.gather(poll_task, return_exceptions=True)

    progress_callback(total)


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
        chunks.append({
            "id": c.get("chunk_id") or c.get("id") or "",
            "content": c.get("content") or c.get("text") or "",
            "file_path": c.get("file_path", ""),
        })

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

        # Forward the silo's chunking config so LightRAG chunks the (now
        # un-pre-split) page/file text by token size instead of using its
        # own defaults. Omit when None so LightRAG keeps its defaults.
        chunk_kwargs: Dict[str, Any] = {}
        if self._chunk_token_size:
            chunk_kwargs["chunk_token_size"] = self._chunk_token_size
        if self._chunk_overlap_token_size is not None:
            chunk_kwargs["chunk_overlap_token_size"] = self._chunk_overlap_token_size

        return LightRAG(
            working_dir=self._working_dir,
            workspace=collection_name,
            llm_model_func=base_llm_func,
            role_llm_configs=role_configs,
            embedding_func=emb_func,
            graph_storage=storage_cfg["graph_storage"],
            vector_storage=storage_cfg["vector_storage"],
            kv_storage=storage_cfg["kv_storage"],
            doc_status_storage=storage_cfg["doc_status_storage"],
            entity_extract_max_gleaning=config.ENTITY_EXTRACT_MAX_GLEANING,
            **chunk_kwargs,
        )

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
        # Keep texts and their source labels aligned (skip empty-content docs).
        texts: List[str] = []
        file_paths: List[str] = []
        for doc in documents:
            if not doc.page_content:
                continue
            texts.append(doc.page_content)
            file_paths.append(_source_label_from_metadata(doc.metadata or {}))

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
            _run_async(_ainsert_with_progress(rag, texts, progress_callback, file_paths=file_paths, process_options=self._chunk_process_option))
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
        """Remove KV / doc-status rows scoped to *collection_name*."""
        try:
            from sqlalchemy import text  # noqa: WPS433

            # LightRAG PGKVStorage uses tables like ``kv_store_full_docs``,
            # ``kv_store_text_chunks``, etc., each with a ``workspace``
            # column.  Best-effort delete — tables might not exist yet.
            kv_tables = [
                "kv_store_full_docs",
                "kv_store_text_chunks",
                "kv_store_full_entities",
                "kv_store_full_relations",
                "kv_store_llm_response_cache",
            ]
            doc_status_table = "doc_status"

            for table in [*kv_tables, doc_status_table]:
                try:
                    self.db.execute(
                        text(f'DELETE FROM "{table}" WHERE workspace = :ws'),
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

        # Query LightRAG's doc-status table for this workspace.
        try:
            from sqlalchemy import text  # noqa: WPS433

            result = self.db.execute(
                text(
                    'SELECT COUNT(*) FROM "doc_status" WHERE workspace = :ws'
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
