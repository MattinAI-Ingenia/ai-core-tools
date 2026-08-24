from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from schemas.embedding_service_schemas import EmbeddingServiceOptionSchema


class SiloListItemSchema(BaseModel):
    """Schema for silo list items"""
    silo_id: int
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    created_at: Optional[datetime] = None
    docs_count: int
    vector_db_type: Optional[str] = None
    lightrag_vector_db_type: Optional[str] = None
    is_frozen: bool = False

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
    lightrag_vector_db_type: Optional[str] = None
    # Current values for editing
    metadata_definition_id: Optional[int] = None
    embedding_service_id: Optional[int] = None
    indexing_service_id: Optional[int] = None  # legacy alias for extract_service_id
    # LightRAG 2026.05 role-specific LLM configuration
    extract_service_id: Optional[int] = None
    keywords_service_id: Optional[int] = None
    vlm_service_id: Optional[int] = None
    lightrag_chunk_strategy: Optional[str] = None
    lightrag_chunk_token_size: Optional[int] = None
    lightrag_chunk_overlap_token_size: Optional[int] = None
    lightrag_language: Optional[str] = None
    lightrag_entity_extract_max_gleaning: Optional[int] = None
    lightrag_max_source_ids_per_entity: Optional[int] = None
    lightrag_max_source_ids_per_relation: Optional[int] = None
    lightrag_entity_types: Optional[str] = None
    lightrag_entity_types_mode: Optional[Literal['infer', 'manual']] = None
    # True once the silo has a successful index: lightrag_language, chunking
    # and lightrag_entity_types stop being editable, and the entity-type
    # inference gate must stop showing (there is nothing left to infer for).
    lightrag_config_locked: bool = False
    # Form data
    output_parsers: List[Dict[str, Any]]
    embedding_services: List[EmbeddingServiceOptionSchema]
    ai_services: List[Dict[str, Any]] = []
    vector_db_options: List[Dict[str, Any]] = []
    # Metadata definition fields for playground
    metadata_fields: Optional[List[Dict[str, Any]]] = None
    is_frozen: bool = False

    model_config = ConfigDict(from_attributes=True)


class CreateSiloSchema(BaseModel):
    """Schema for creating a new silo (vector_db_type is settable on creation only)"""
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    output_parser_id: Optional[int] = None
    embedding_service_id: Optional[int] = None
    vector_db_type: Optional[str] = None
    lightrag_vector_db_type: Optional[str] = None
    indexing_service_id: Optional[int] = None  # legacy alias for extract_service_id
    # LightRAG 2026.05 role-specific LLM configuration
    extract_service_id: Optional[int] = None
    keywords_service_id: Optional[int] = None
    vlm_service_id: Optional[int] = None
    lightrag_chunk_strategy: Optional[str] = None
    lightrag_chunk_token_size: Optional[int] = None
    lightrag_chunk_overlap_token_size: Optional[int] = None
    lightrag_language: Optional[str] = None
    lightrag_entity_extract_max_gleaning: Optional[int] = None
    lightrag_max_source_ids_per_entity: Optional[int] = None
    lightrag_max_source_ids_per_relation: Optional[int] = None
    lightrag_entity_types: Optional[str] = None
    lightrag_entity_types_mode: Optional[Literal['infer', 'manual']] = None


class UpdateSiloSchema(BaseModel):
    """Schema for updating an existing silo. vector_db_type, embedding_service_id,
    extract/vlm/indexing_service_id, chunking, lightrag_language, and
    lightrag_entity_types are immutable after creation. keywords_service_id is
    query-time only and stays editable."""
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    output_parser_id: Optional[int] = None
    keywords_service_id: Optional[int] = None
    # Editable only until the silo's first successful index (enforced in
    # SiloService.create_or_update_silo via is_lightrag_config_locked) — this
    # is the window that lets a human infer/confirm the entity types from the
    # uploaded documents before anything is extracted.
    lightrag_entity_types: Optional[str] = None
    lightrag_entity_types_mode: Optional[Literal['infer', 'manual']] = None


class _ContentLengthFilterSchema(BaseModel):
    """Shared metadata and content-length filters for silo operations."""
    filter_metadata: Optional[Dict[str, Any]] = None
    min_content_length: Optional[int] = None
    max_content_length: Optional[int] = None

    @field_validator("min_content_length", "max_content_length")
    @classmethod
    def validate_content_length(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 0:
            raise ValueError("content length filters must be greater than or equal to 0")
        return value

    @model_validator(mode="after")
    def validate_content_length_range(self) -> "_ContentLengthFilterSchema":
        if (
            self.min_content_length is not None
            and self.max_content_length is not None
            and self.min_content_length > self.max_content_length
        ):
            raise ValueError("min_content_length cannot be greater than max_content_length")
        return self


class SiloSearchSchema(_ContentLengthFilterSchema):
    """Schema for searching in a silo."""
    query: str
    limit: Optional[int] = None
    search_type: Optional[Literal["similarity", "mmr", "similarity_score_threshold"]] = "similarity"
    fetch_k: Optional[int] = None
    lambda_mult: Optional[float] = None
    score_threshold: Optional[float] = None
    lightrag_query_mode: Optional[Literal["local", "global", "hybrid", "mix", "naive", "bypass"]] = None

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and not (1 <= value <= 200):
            raise ValueError("limit must be between 1 and 200")
        return value

    @field_validator("fetch_k")
    @classmethod
    def validate_fetch_k(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and value < 1:
            raise ValueError("fetch_k must be at least 1")
        return value

    @field_validator("lambda_mult")
    @classmethod
    def validate_lambda_mult(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("lambda_mult must be between 0.0 and 1.0")
        return value

    @field_validator("score_threshold")
    @classmethod
    def validate_score_threshold(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not (0.0 <= value <= 1.0):
            raise ValueError("score_threshold must be between 0.0 and 1.0")
        return value

    @model_validator(mode="after")
    def validate_search_options(self) -> "SiloSearchSchema":
        if self.search_type == "similarity_score_threshold" and self.score_threshold is None:
            raise ValueError(
                "score_threshold is required when search_type is 'similarity_score_threshold'"
            )
        if self.search_type != "similarity_score_threshold" and self.score_threshold is not None:
            raise ValueError(
                "score_threshold can only be used when search_type is 'similarity_score_threshold'"
            )
        if self.search_type != "mmr" and self.fetch_k is not None:
            raise ValueError("fetch_k can only be used when search_type is 'mmr'")
        if self.search_type != "mmr" and self.lambda_mult is not None:
            raise ValueError("lambda_mult can only be used when search_type is 'mmr'")
        return self


# Kept for backward compatibility with the public API router
CreateUpdateSiloSchema = CreateSiloSchema


class SiloCountRequestSchema(_ContentLengthFilterSchema):
    """Request body for the count-documents endpoint."""
    retrieval_config: Optional[Dict[str, Any]] = None


class CostEstimationRequestSchema(BaseModel):
    """Request body for cost estimation — list of documents to be indexed."""
    documents: List[Dict[str, Any]]  # Each dict has 'content' (str) and optional 'metadata'


class CostEstimationResponseSchema(BaseModel):
    """Cost estimation result returned to the frontend."""
    total_chunks: int
    chunk_token_size: int
    estimated_llm_calls: int
    estimated_embedding_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_embedding_tokens: Optional[int] = None
    estimated_cost_min: Optional[float] = None
    estimated_cost_max: Optional[float] = None
    currency: str = "USD"  # ISO 4217 code (USD, EUR, etc.)
    model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None
    # Time estimates (seconds)
    estimated_indexing_time_min: Optional[float] = None  # optimistic (faster calls)
    estimated_indexing_time_max: Optional[float] = None  # pessimistic (slower calls)
    estimated_indexing_time_avg: Optional[float] = None  # average estimate
    warnings: List[str] = []


class SiloSearchSchema(BaseModel):
    """Schema for searching within a silo.

    `limit` — max results. Defaults to DEFAULT_SEARCH_LIMIT (100), capped at MAX_SEARCH_LIMIT (200).
    `search_type` — one of "similarity" (default), "similarity_score_threshold", "mmr".
    `score_threshold` — float 0-1, only meaningful when search_type="similarity_score_threshold".
    `fetch_k` — candidate pool size for MMR, only meaningful when search_type="mmr".
    `lambda_mult` — diversity factor 0-1 for MMR (1=max relevance, 0=max diversity). Default 0.5.
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
    lightrag_query_mode: Optional[Literal["local", "global", "hybrid", "mix", "naive", "bypass"]] = None

    @field_validator("search_type")
    @classmethod
    def validate_search_type(cls, v: str) -> str:
        allowed = {"similarity", "similarity_score_threshold", "mmr"}
        if v not in allowed:
            raise ValueError(f"search_type must be one of {sorted(allowed)}, got '{v}'")
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


class InferEntityTypesRequest(BaseModel):
    """Body for POST /silos/{id}/lightrag/infer-entity-types."""

    # Not persisted anywhere: inference runs once and a smarter model is worth
    # paying for on a choice that cannot be undone after indexing. Unset falls
    # back to the silo's own extraction service.
    ai_service_id: Optional[int] = None
    # Set when inferring from a CSV import review that hasn't been confirmed
    # yet: on a brand-new silo there are no Resource rows to read from, so the
    # inference falls back to this job's staged (downloaded but not yet
    # ingested) PDFs.
    import_job_id: Optional[int] = None
