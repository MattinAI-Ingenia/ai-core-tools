"""
BM25 lexical search method.

Unlike the dense search method, BM25 is a **sparse / lexical** retriever: it
ranks documents by exact term overlap (term frequency x inverse document
frequency) rather than by embedding similarity. It needs the whole corpus to
build an in-memory index, so this search method fetches every document from the
silo's collection (via ``vector_store.get_all_documents``) and feeds them to
LangChain's ``BM25Retriever``.

Because the index is built in memory on every call, a configurable cap
(``bm25_max_docs``) bounds how many documents are pulled from large collections.

``BM25Retriever``'s default tokenizer is a bare ``text.split()`` — no
lowercasing, no punctuation stripping, no stemming. That means a query like
"plan" would NOT match a document containing "plans." (plural, with trailing
punctuation still attached to the token): the two are different tokens as far
as bare BM25 is concerned, even though a human (or a dense/embeddings search)
would consider them an obvious match. ``_preprocess`` fixes this by lowercasing,
tokenizing on word boundaries (dropping punctuation), and stemming each token
with a Snowball stemmer, so plural/singular and simple verb-conjugation
differences no longer prevent a lexical match.
"""

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
