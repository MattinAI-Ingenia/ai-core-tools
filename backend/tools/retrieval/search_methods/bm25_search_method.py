import re
from typing import List

import snowballstemmer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_component import SearchMethod
from tools.retrieval.retrieval_context import RetrievalContext
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_K = 30
DEFAULT_MAX_DOCS = 5000

_WORD_RE = re.compile(r"\w+", re.UNICODE)
_stemmer = snowballstemmer.stemmer("english")


def _preprocess(text: str) -> List[str]:
    """Lowercase, tokenize on word boundaries, and stem — see module docstring."""
    tokens = _WORD_RE.findall(text.lower())
    return _stemmer.stemWords(tokens)


class BM25SearchMethod(SearchMethod):
    """Produce an in-memory BM25 (lexical) retriever over the silo's corpus."""

    def build(self, ctx: RetrievalContext) -> BaseRetriever:
        if ctx.vector_store is None:
            raise ValueError("BM25SearchMethod requires a vector_store in the context.")
        if not ctx.collection_name:
            raise ValueError("BM25SearchMethod requires a collection_name in the context.")

        # Lazy import so the rank_bm25 dependency is only required when BM25 is used.
        from langchain_community.retrievers import BM25Retriever

        k = ctx.search_kwargs.get("k", DEFAULT_K)
        filter_metadata = ctx.search_kwargs.get("filter")
        max_docs = ctx.params.get("bm25_max_docs", DEFAULT_MAX_DOCS)

        documents: List[Document] = ctx.vector_store.get_all_documents(
            ctx.collection_name,
            filter_metadata=filter_metadata,
            limit=max_docs,
        )

        if not documents:
            raise ValueError(
                f"BM25SearchMethod: collection '{ctx.collection_name}' has no documents to index."
            )

        logger.debug(
            "BM25SearchMethod: building index for collection=%s (docs=%d, k=%s)",
            ctx.collection_name,
            len(documents),
            k,
        )

        retriever = BM25Retriever.from_documents(documents, preprocess_func=_preprocess)
        retriever.k = k
        return retriever
