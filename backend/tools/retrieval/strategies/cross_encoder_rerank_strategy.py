from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import config
from tools.retrieval.retrieval_component import RetrievalTransformer
from tools.retrieval.retrieval_context import RetrievalContext
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TOP_N = 5


class CrossEncoderRerankStrategy(RetrievalTransformer):
    """Re-score and truncate the candidate set using a dedicated cross-encoder model."""

    def __init__(self) -> None:
        logger.info(
            "Loading cross-encoder rerank model %s (device=%s)",
            config.CROSS_ENCODER_RERANK_MODEL,
            config.CROSS_ENCODER_RERANK_DEVICE,
        )
        self._cross_encoder = HuggingFaceCrossEncoder(
            model_name=config.CROSS_ENCODER_RERANK_MODEL,
            model_kwargs={"device": config.CROSS_ENCODER_RERANK_DEVICE},
        )

    def apply(
        self,
        retrievers: List[BaseRetriever],
        ctx: RetrievalContext,
    ) -> BaseRetriever:
        base_retriever = retrievers[0]

        params = ctx.params or {}
        top_n = params.get("top_n", DEFAULT_TOP_N)

        compressor = CrossEncoderReranker(model=self._cross_encoder, top_n=top_n)

        logger.debug(
            "CrossEncoderRerankStrategy: wrapping base retriever with CrossEncoderReranker (top_n=%s)",
            top_n,
        )

        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )
