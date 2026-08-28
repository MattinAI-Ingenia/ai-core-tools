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
