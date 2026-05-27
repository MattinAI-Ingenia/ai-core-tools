"""LightRAG implementation of the vector store interface.

This module provides a LightRAG-backed implementation of
:class:`VectorStoreInterface`, wrapping ``lightrag-hku==1.4.16``'s embedded
Python API behind the same abstract interface used by PGVectorStore and
QdrantStore.

All ``lightrag`` imports are **lazy** (inside methods) so this file is
importable even when ``lightrag-hku`` is not installed.

Note: Requires ``lightrag-hku`` and its storage extras:
    pip install 'lightrag-hku[offline-storage]==1.4.16'
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from tools.vector_stores.vector_store_interface import VectorStoreInterface

logger = logging.getLogger(__name__)

# Valid LightRAG query modes — used for pass-through validation.
_LIGHTRAG_MODES = frozenset({"local", "global", "hybrid", "naive", "mix", "bypass"})


def _run_async(coro):
    """Run an async coroutine from synchronous code.

    If an event loop is already running (e.g. inside Jupyter or an async
    framework), this falls back to ``asyncio.ensure_future`` + a new thread
    so we never block the running loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # An event loop is already running — offload to a new thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class LightRAGRetriever(BaseRetriever):
    """LangChain retriever backed by a LightRAG instance.

    This retriever wraps ``LightRAG.aquery`` / ``LightRAG.query`` so it can
    be plugged into any LangChain chain that expects a
    :class:`~langchain_core.retrievers.BaseRetriever`.
    """

    rag_instance: Any = Field(exclude=True)
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

        param = QueryParam(mode=self.query_mode, top_k=self.top_k)
        response = _run_async(self.rag_instance.aquery(query, param=param))
        return _wrap_response(response, self.query_mode)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        from lightrag.base import QueryParam  # noqa: WPS433

        param = QueryParam(mode=self.query_mode, top_k=self.top_k)
        response = await self.rag_instance.aquery(query, param=param)
        return _wrap_response(response, self.query_mode)


def _wrap_response(response: str, query_mode: str) -> List[Document]:
    """Wrap LightRAG's string response into a list of LangChain Documents."""
    if not response or not str(response).strip():
        return []
    return [
        Document(
            page_content=str(response),
            metadata={"source": "lightrag", "query_mode": query_mode},
        )
    ]


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
    ):
        self.db = db
        self.ai_service = ai_service
        self.embedding_service = embedding_service
        self.workspace_prefix = workspace_prefix
        self._rag_instances: Dict[str, Any] = {}
        # Shared temp directory so LightRAG's working_dir requirement is met
        # without polluting the project tree.  The directory is only used for
        # ancillary local caches (e.g. tiktoken); actual data lives in
        # Neo4j / Qdrant / PostgreSQL.
        self._working_dir = tempfile.mkdtemp(prefix="lightrag_")

    def _get_rag_instance(self, collection_name: str):
        """Return a cached ``LightRAG`` instance for *collection_name*.

        Creates and initialises one on the first call for a given name.
        """
        if collection_name in self._rag_instances:
            return self._rag_instances[collection_name]

        from lightrag import LightRAG  # noqa: WPS433

        from tools.vector_stores.lightrag.adapters import (
            build_embedding_func,
            build_llm_model_func,
        )
        from tools.vector_stores.lightrag.storage_config import (
            build_storage_config,
        )

        storage_cfg = build_storage_config()
        llm_func = build_llm_model_func(self.ai_service)
        emb_func = build_embedding_func(self.embedding_service)

        rag = LightRAG(
            working_dir=self._working_dir,
            workspace=collection_name,
            llm_model_func=llm_func,
            embedding_func=emb_func,
            graph_storage=storage_cfg["graph_storage"],
            vector_storage=storage_cfg["vector_storage"],
            kv_storage=storage_cfg["kv_storage"],
            doc_status_storage=storage_cfg["doc_status_storage"],
        )

        # Storages must be initialised before the instance is usable.
        _run_async(rag.initialize_storages())

        self._rag_instances[collection_name] = rag
        logger.info("Created LightRAG instance for workspace '%s'", collection_name)
        return rag

    # ------------------------------------------------------------------
    # VectorStoreInterface implementation
    # ------------------------------------------------------------------

    def index_documents(
        self,
        collection_name: str,
        documents: List[Document],
        embedding_service=None,
    ) -> None:
        if not documents:
            return

        logger.info(
            "Indexing %d documents into LightRAG workspace '%s'",
            len(documents),
            collection_name,
        )

        rag = self._get_rag_instance(collection_name)
        texts = [doc.page_content for doc in documents if doc.page_content]

        if not texts:
            logger.debug("No non-empty texts to index; skipping.")
            return

        _run_async(rag.ainsert(texts))
        logger.info(
            "Successfully indexed %d texts into workspace '%s'",
            len(texts),
            collection_name,
        )

    def delete_documents(
        self,
        collection_name: str,
        ids,
        embedding_service=None,
    ) -> None:
        rag = self._get_rag_instance(collection_name)

        if isinstance(ids, list):
            for doc_id in ids:
                try:
                    _run_async(rag.adelete_by_doc_id(str(doc_id)))
                except Exception as exc:
                    logger.warning(
                        "LightRAG delete_by_doc_id failed for '%s': %s",
                        doc_id,
                        exc,
                    )
        else:
            logger.warning(
                "LightRAGStore.delete_documents only supports list-of-id "
                "deletion.  Metadata-filter deletion is not supported; "
                "ignoring request for workspace '%s'.",
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
            # LightRAG stores the workspace in a ``workspace`` property on
            # every node/relationship.  Deleting by that value is the safest
            # approach.
            with driver.session() as session:
                session.run(
                    "MATCH (n) WHERE n.workspace = $ws DETACH DELETE n",
                    ws=collection_name,
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
        rag = self._get_rag_instance(collection_name)
        search_params = search_params or {}

        # Allow callers to override the LightRAG query mode directly.
        mode = search_params.pop("lightrag_query_mode", None) or _resolve_query_mode(
            search_type
        )
        top_k = search_params.get("k", 5)

        return LightRAGRetriever(
            rag_instance=rag,
            query_mode=mode,
            top_k=top_k,
        )

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
