"""
Shared context object for the retrieval pipeline.

A single :class:`RetrievalContext` is built once per retrieval request and
passed to every component (sources and transformers) in the pipeline. It
carries everything a component might need to build or wrap a retriever, so the
component interfaces stay small and uniform.

This is what lets the architecture admit BOTH new search methods (sources, e.g.
dense / BM25 / sparse — which need the vector store, collection and embeddings
to *produce* a retriever) and post-processing strategies (transformers, e.g.
rerank / ensemble — which only *wrap* already-built retrievers).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RetrievalContext:
    """Everything a retrieval component needs for one request.

    Attributes:
        vector_store: The resolved vector store backend for the silo. Sources
            use it to produce a retriever; transformers usually ignore it.
        collection_name: The silo's collection name (e.g. ``silo_42``).
        embedding_service: The silo's embedding service ORM object. Needed by
            sources that embed queries and by transformers that re-score
            candidates (e.g. the embeddings rerank).
        search_kwargs: Backend-agnostic search parameters shared by sources
            (``k``, ``filter``, ``score_threshold`` ...).
        params: Component-specific parameters (e.g. ``search_type`` for the
            dense source, ``top_n`` / ``similarity_threshold`` for rerank).
    """

    vector_store: Any = None
    collection_name: Optional[str] = None
    embedding_service: Any = None
    search_kwargs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
