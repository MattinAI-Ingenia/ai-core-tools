from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from schemas.embedding_service_schemas import EmbeddingServiceOptionSchema
from schemas.service_lifecycle_schemas import ServiceLifecycleWarningSchema
from schemas.indexing_schemas import IndexingOptionsSchema, IndexingSettingsInput


class SiloListItemSchema(BaseModel):
    """Schema for silo list items"""
    silo_id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[datetime] = None
    docs_count: int
    vector_db_type: Optional[str] = None
    is_frozen: bool = False
    lifecycle_warnings: List[ServiceLifecycleWarningSchema] = []

    model_config = ConfigDict(from_attributes=True)


class SiloDetailSchema(BaseModel):
    """Schema for detailed silo information"""
    silo_id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[datetime] = None
    docs_count: int
    vector_db_type: Optional[str] = None
    # Current values for editing
    metadata_definition_id: Optional[int] = None
    embedding_service_id: Optional[int] = None
    # Media (video/audio) indexing services
    transcription_service_id: Optional[int] = None
    video_ai_service_id: Optional[int] = None
    # Indexing settings (tools.document_loaders) and what the form can choose from
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    pdf_loader: Optional[str] = None
    pdf_loader_options: Dict[str, Any] = {}
    indexing_options: Optional[IndexingOptionsSchema] = None
    # Form data
    output_parsers: List[Dict[str, Any]]
    embedding_services: List[EmbeddingServiceOptionSchema]
    vector_db_options: List[Dict[str, Any]] = []
    # AI services usable for media: [{service_id, name, supports_video}]
    ai_services: List[Dict[str, Any]] = []
    # Metadata definition fields for playground
    metadata_fields: Optional[List[Dict[str, Any]]] = None
    is_frozen: bool = False
    lifecycle_warnings: List[ServiceLifecycleWarningSchema] = []

    model_config = ConfigDict(from_attributes=True)


class CreateSiloSchema(IndexingSettingsInput):
    """Schema for creating a new silo (vector_db_type is settable on creation only)"""
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    output_parser_id: Optional[int] = None
    embedding_service_id: Optional[int] = None
    vector_db_type: Optional[str] = None
    # Optional: needed only to index video/audio into the silo
    transcription_service_id: Optional[int] = None
    video_ai_service_id: Optional[int] = None


class UpdateSiloSchema(IndexingSettingsInput):
    """Schema for updating an existing silo (vector_db_type and embedding_service_id are immutable after creation)"""
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    output_parser_id: Optional[int] = None
    # Optional media services; omitted fields keep their value, null clears them
    transcription_service_id: Optional[int] = None
    video_ai_service_id: Optional[int] = None


# Kept for backward compatibility with the public API router
CreateUpdateSiloSchema = CreateSiloSchema


class SiloCountRequestSchema(BaseModel):
    """Request body for the count-documents endpoint."""
    filter_metadata: Optional[Dict[str, Any]] = None
    min_content_length: Optional[int] = None
    max_content_length: Optional[int] = None


class SiloSearchSchema(BaseModel):
    """Schema for searching within a silo.

    `limit` — max results. Defaults to DEFAULT_SEARCH_LIMIT (100), capped at MAX_SEARCH_LIMIT (200).
    `search_type` — one of "similarity" (default), "similarity_score_threshold", "mmr".
    `score_threshold` — float 0-1, only meaningful when search_type="similarity_score_threshold".
    `fetch_k` — candidate pool size for MMR, only meaningful when search_type="mmr".
    `lambda_mult` — diversity factor 0-1 for MMR (1=max relevance, 0=max diversity). Default 0.5.
    `search_method` — one of SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS ("dense", "bm25").
        None/"dense" keeps the default similarity-search code path unchanged (same `_score`
        in results); any other value routes the search through the RetrievalPipeline instead.
    Reranking is not part of this schema (the public API never exposes it); the silo playground uses
    `PlaygroundSiloSearchSchema`.
    """
    query: str
    limit: Optional[int] = None
    filter_metadata: Optional[Dict[str, Any]] = None
    search_type: str = "similarity"
    score_threshold: Optional[float] = None
    fetch_k: Optional[int] = None
    lambda_mult: Optional[float] = None
    min_content_length: Optional[int] = None   # inclusive lower bound on chunk character count
    max_content_length: Optional[int] = None   # inclusive upper bound on chunk character count
    search_method: Optional[str] = None

    @field_validator("search_type")
    @classmethod
    def validate_search_type(cls, v: str) -> str:
        allowed = {"similarity", "similarity_score_threshold", "mmr"}
        if v not in allowed:
            raise ValueError(f"search_type must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("search_method")
    @classmethod
    def validate_search_method(cls, v: Optional[str]) -> Optional[str]:
        from tools.retrieval.search_methods.search_method_factory import SearchMethodFactory
        if v is not None and v not in SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS:
            raise ValueError(
                f"search_method must be one of {sorted(SearchMethodFactory.IMPLEMENTED_SEARCH_METHODS)}"
            )
        return v

    @model_validator(mode="after")
    def validate_param_consistency(self) -> "SiloSearchSchema":
        if self.score_threshold is not None and self.search_type != "similarity_score_threshold":
            raise ValueError(
                "score_threshold is only valid when search_type='similarity_score_threshold'"
            )
        if (self.fetch_k is not None or self.lambda_mult is not None) and self.search_type != "mmr":
            raise ValueError(
                "fetch_k and lambda_mult are only valid when search_type='mmr'"
            )
        if self.min_content_length is not None and self.min_content_length < 0:
            raise ValueError("min_content_length must be >= 0")
        if self.max_content_length is not None and self.max_content_length < 0:
            raise ValueError("max_content_length must be >= 0")
        if (
            self.min_content_length is not None
            and self.max_content_length is not None
            and self.min_content_length > self.max_content_length
        ):
            raise ValueError("min_content_length must be <= max_content_length")
        return self

class PlaygroundSiloSearchSchema(SiloSearchSchema):
    """Silo playground search (internal API only): adds the optional rerank second stage.

    `rerank` — rerank the first-stage candidates with a rerank service. The query and the candidate chunks
        are sent to that service's endpoint (data egress), so it is opt-in per request.
    `rerank_service_id` — explicit service (owned by the app or a platform service); None inherits the app
        default, then the platform default.
    `rerank_top_n` — documents kept after reranking (1-50); None uses the default (30).
    With rerank on, `limit` only sizes the first stage (candidates = max(limit, top N), capped by the server);
    the response holds up to `rerank_top_n` documents even when it is larger than `limit`.
    """
    rerank: bool = Field(
        default=False,
        description="Rerank the candidates as a second stage. Sends the query and candidate chunks to the "
                    "rerank service endpoint (data egress).",
    )
    rerank_service_id: Optional[int] = Field(
        default=None, description="Explicit rerank service; None inherits the app default, then the platform default."
    )
    rerank_top_n: Optional[int] = Field(
        default=None, description="Documents kept after reranking (1-50); None uses the default (30). The response is cut to "
                    "this value, not to 'limit', which only sizes the first stage when reranking."
    )

    @field_validator("rerank_top_n")
    @classmethod
    def validate_rerank_top_n(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1 <= v <= 50):
            raise ValueError("rerank_top_n must be between 1 and 50")
        return v


class RerankStatusSchema(BaseModel):
    """Outcome of the rerank stage of one search."""
    requested: bool = True
    applied: bool = False
    service_id: Optional[int] = None
    service_name: Optional[str] = None
    source: Optional[str] = None
    skipped_reason: Optional[str] = None
    latency_ms: Optional[int] = None


class SiloSearchResultSchema(BaseModel):
    """One search hit; `rerank_score` is set only when the rerank stage was applied."""
    page_content: str
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None
    rerank_score: Optional[float] = None


class SiloSearchResponseSchema(BaseModel):
    """Internal (playground) search response; `rerank` is present only when reranking was requested."""
    query: str
    results: List[SiloSearchResultSchema]
    total_results: int
    filter_metadata: Optional[Dict[str, Any]] = None
    rerank: Optional[RerankStatusSchema] = None

