from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from tools.retrieval.rerank_transformer import RerankPlan


@dataclass
class RetrievalContext:
    """Everything a retrieval component needs for one request.

    Attributes:
        vector_store: The resolved vector store backend for the silo. Sources
            use it to produce a retriever; transformers usually ignore it.
        collection_name: The silo's collection name (e.g. ``silo_42``).
        embedding_service: The silo's embedding service ORM object. Needed by
            sources that embed queries.
        search_kwargs: Backend-agnostic search parameters shared by sources
            (``k``, ``filter``, ``score_threshold`` ...).
        params: Component-specific parameters (e.g. ``search_type`` for the
            dense source).
        rerank: Optional rerank-stage plan (top N, candidate limit, resolved service
            or skip reason). When set, the pipeline appends the rerank stage.
    """

    vector_store: Any = None
    collection_name: Optional[str] = None
    embedding_service: Any = None
    search_kwargs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    rerank: Optional["RerankPlan"] = None
