from typing import Dict, List, Optional

from tools.retrieval.retrieval_component import SearchMethod
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SEARCH_METHOD = "dense"


class SearchMethodFactory:

    # Supported retrieval search methods (including future planned support)
    SUPPORTED_SEARCH_METHODS = {
        "dense": "Dense vector search (embeddings similarity)",
        "bm25": "BM25 lexical search (keyword matching)",
        "hybrid": "Hybrid dense + lexical search (RRF fusion)",
        "sparse": "Sparse / SPLADE search (future support)",
    }

    # Search methods that are currently implemented and can be selected by users
    IMPLEMENTED_SEARCH_METHODS = ("dense", "bm25", "hybrid")

    # Atomic methods selectable in an agent's multiselect rag_search_method list.
    # "hybrid" is excluded here: combining dense+bm25 is now expressed by selecting
    # both atomic methods (RetrievalPipeline fuses them via RRF), not by the
    # "hybrid" string. "hybrid" itself remains valid as a single-value override
    # (e.g. the Silo Playground) via IMPLEMENTED_SEARCH_METHODS above.
    MULTISELECT_SEARCH_METHODS = tuple(m for m in IMPLEMENTED_SEARCH_METHODS if m != "hybrid")

    _instances: Dict[str, SearchMethod] = {}

    @staticmethod
    def get_search_method(search_method_name: Optional[str] = None) -> SearchMethod:
        """Return a cached search method instance for the requested name."""

        resolved_name = (search_method_name or DEFAULT_SEARCH_METHOD).lower()

        if resolved_name not in SearchMethodFactory.SUPPORTED_SEARCH_METHODS:
            supported = ", ".join(SearchMethodFactory.SUPPORTED_SEARCH_METHODS.keys())
            raise ValueError(
                f"Unsupported retrieval search method: {resolved_name}. Supported search methods: {supported}"
            )

        if resolved_name not in SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS:
            raise NotImplementedError(
                f"{resolved_name} search method is planned but not yet implemented. Currently available: "
                f"{', '.join(SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS)}"
            )

        if resolved_name in SearchMethodFactory._instances:
            return SearchMethodFactory._instances[resolved_name]

        logger.info("Initializing retrieval search method: %s", resolved_name)

        if resolved_name == "dense":
            instance = SearchMethodFactory._create_dense_search_method()
        elif resolved_name == "bm25":
            instance = SearchMethodFactory._create_bm25_search_method()
        elif resolved_name == "hybrid":
            instance = SearchMethodFactory._create_hybrid_search_method()
        else:
            # Guard clause for future implementations
            raise NotImplementedError(f"Retrieval search method {resolved_name} is not implemented yet")

        SearchMethodFactory._instances[resolved_name] = instance
        return instance

    @staticmethod
    def get_available_search_method_options() -> List[Dict[str, str]]:
        """Expose implemented search method choices with human-friendly labels."""

        options: List[Dict[str, str]] = []
        for key in SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS:
            label = SearchMethodFactory.SUPPORTED_SEARCH_METHODS.get(key, key)
            options.append({"code": key, "label": label})
        return options

    @staticmethod
    def get_available_multiselect_options() -> List[Dict[str, str]]:
        """Expose the atomic methods selectable in an agent's multiselect config."""

        options: List[Dict[str, str]] = []
        for key in SearchMethodFactory.MULTISELECT_SEARCH_METHODS:
            label = SearchMethodFactory.SUPPORTED_SEARCH_METHODS.get(key, key)
            options.append({"code": key, "label": label})
        return options

    @staticmethod
    def _create_dense_search_method() -> SearchMethod:
        """Create a DenseSearchMethod instance."""

        from tools.retrieval.search_methods.dense_search_method import DenseSearchMethod

        return DenseSearchMethod()

    @staticmethod
    def _create_bm25_search_method() -> SearchMethod:
        """Create a BM25SearchMethod instance."""

        from tools.retrieval.search_methods.bm25_search_method import BM25SearchMethod

        return BM25SearchMethod()

    @staticmethod
    def _create_hybrid_search_method() -> SearchMethod:
        """Create a HybridSearchMethod instance."""

        from tools.retrieval.search_methods.hybrid_search_method import HybridSearchMethod

        return HybridSearchMethod()
