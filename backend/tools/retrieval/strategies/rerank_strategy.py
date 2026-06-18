"""
Embeddings-based rerank retrieval strategy.

This is the first "real" retrieval strategy. It wraps the base retriever in a
``ContextualCompressionRetriever`` so that, after the vector store returns its
candidate set (Phase 1 — search_type decides *how* the DB is queried), the
candidates are re-scored against the query embedding and truncated to the most
relevant ones (Phase 2 — the strategy operates only on the already-retrieved
candidates, with no extra DB access).

It uses the silo's existing embedding service (via ``get_embeddings_model``), so
it adds **no new dependencies**: the reranking model is the same embedding model
the silo already relies on.

Parameters (read from ``params``):
    top_n: How many documents to keep after reranking. Defaults to 5.
    similarity_threshold: Optional float in [0, 1]; documents scoring below it
        are dropped. When set, it complements ``top_n``.
"""

from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

from tools.embeddingTools import get_embeddings_model
from tools.retrieval.retrieval_component import RetrievalTransformer
from tools.retrieval.retrieval_context import RetrievalContext
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_N = 5


class RerankStrategy(RetrievalTransformer):
    """Re-score and truncate the candidate set using the silo's embeddings."""

    def apply(
        self,
        retrievers: List[BaseRetriever],
        ctx: RetrievalContext,
    ) -> BaseRetriever:
        base_retriever = retrievers[0]
        embedding_service = ctx.embedding_service

        if embedding_service is None:
            raise ValueError(
                "RerankStrategy requires the silo's embedding_service to rerank "
                "candidates, but none was provided."
            )

        params = ctx.params or {}
        top_n = params.get("top_n", DEFAULT_TOP_N)
        similarity_threshold = params.get("similarity_threshold")

        embeddings = get_embeddings_model(embedding_service)

        compressor = EmbeddingsFilter(
            embeddings=embeddings,
            k=top_n,
            similarity_threshold=similarity_threshold,
        )

        logger.debug(
            "RerankStrategy: wrapping base retriever with EmbeddingsFilter "
            "(top_n=%s, similarity_threshold=%s)",
            top_n,
            similarity_threshold,
        )

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
