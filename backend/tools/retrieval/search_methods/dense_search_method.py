"""
Dense vector search method.

This is the default search method: it asks the silo's vector store backend for a
LangChain retriever configured with the requested ``search_type``
(similarity / mmr / similarity_score_threshold). "Dense" refers to the fact that
documents and queries are represented as dense embedding vectors and matched by
vector similarity — as opposed to sparse/lexical methods such as BM25.

It preserves the exact behaviour the application had before the retrieval
pipeline refactor: it is a thin wrapper over ``vector_store.get_retriever``.
"""

from langchain_core.retrievers import BaseRetriever

from tools.retrieval.retrieval_component import SearchMethod
from tools.retrieval.retrieval_context import RetrievalContext
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEARCH_TYPE = "similarity"


class DenseSearchMethod(SearchMethod):
    """Produce a dense-vector retriever from the silo's vector store."""

    def build(self, ctx: RetrievalContext) -> BaseRetriever:
        if ctx.vector_store is None:
            raise ValueError("DenseSearchMethod requires a vector_store in the context.")

        search_type = ctx.params.get("search_type", DEFAULT_SEARCH_TYPE)
        # Defaults to True (async psycopg engine) to preserve the LangGraph agent
        # path unchanged. Sync callers (e.g. the Silo Playground, which invokes the
        # retriever with .invoke() rather than .ainvoke()) must pass
        # params={"use_async": False, ...} — an async-mode retriever raises
        # "This method must be called without async_mode" when invoked synchronously.
        use_async = ctx.params.get("use_async", True)

        logger.debug(
            "DenseSearchMethod: building retriever for collection=%s (search_type=%s, use_async=%s)",
            ctx.collection_name,
            search_type,
            use_async,
        )

        return ctx.vector_store.get_retriever(
            ctx.collection_name,
            ctx.embedding_service,
            ctx.search_kwargs,
            search_type=search_type,
            use_async=use_async,
        )
