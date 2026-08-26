from schemas.media_schemas import MediaResponse
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class MetadataFieldSchema(BaseModel):
    """Schema for metadata field information"""
    name: str
    type: str
    description: str


class RepositoryListItemSchema(BaseModel):
    """Schema for repository list items"""
    repository_id: int
    name: str
    type: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    resource_count: int
    vector_db_type: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class RepositoryDetailSchema(BaseModel):
    """Schema for detailed repository information"""
    repository_id: int
    name: str
    type: Optional[str] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    resources: List[Dict[str, Any]]
    folders: List[Dict[str, Any]] = []
    embedding_services: List[Dict[str, Any]]
    embedding_service_id: Optional[int] = None
    silo_id: Optional[int] = None
    vector_db_type: Optional[str] = None
    lightrag_vector_db_type: Optional[str] = None
    vector_db_options: List[Dict[str, Any]] = []
    metadata_fields: Optional[List[MetadataFieldSchema]] = []
    media: List[MediaResponse] = []
    ai_services: List[Dict[str, Any]] = []
    transcription_service_id: Optional[int] = None
    video_ai_service_id: Optional[int] = None
    indexing_service_id: Optional[int] = None  # legacy alias for extract_service_id
    # LightRAG 2026.05 role-specific LLM configuration
    extract_service_id: Optional[int] = None
    keywords_service_id: Optional[int] = None
    vlm_service_id: Optional[int] = None
    total_indexing_duration_seconds: Optional[float] = None
    # LightRAG extraction config, read back from the silo. Returned so the UI can
    # SHOW what a repository was indexed with: it was previously write-only
    # (present on CreateUpdateRepositorySchema but not here), so the edit form
    # fell back to its hardcoded defaults and displayed 1200/100/fixed_token for
    # a silo actually running 2000/300/paragraph_semantic — plausible and wrong.
    lightrag_chunk_strategy: Optional[str] = None
    lightrag_chunk_token_size: Optional[int] = None
    lightrag_chunk_overlap_token_size: Optional[int] = None
    lightrag_language: Optional[str] = None
    lightrag_entity_extract_max_gleaning: Optional[int] = None
    lightrag_max_source_ids_per_entity: Optional[int] = None
    lightrag_max_source_ids_per_relation: Optional[int] = None
    lightrag_entity_types: Optional[str] = None
    lightrag_entity_types_mode: Optional[str] = None
    # True once something has been indexed: the fields above shaped how entities
    # were extracted, so they must stop changing (see is_lightrag_config_locked).
    lightrag_config_locked: bool = False

    model_config = ConfigDict(from_attributes=True)


class CreateRepositorySchema(BaseModel):
    """Schema for creating a new repository (vector_db_type is settable on creation only)"""
    name: str
    type: Optional[str] = None
    status: Optional[str] = None
    embedding_service_id: Optional[int] = None
    vector_db_type: Optional[str] = None
    lightrag_vector_db_type: Optional[str] = None
    transcription_service_id: Optional[int] = None
    video_ai_service_id: Optional[int] = None
    indexing_service_id: Optional[int] = None  # legacy alias for extract_service_id
    # LightRAG 2026.05 role-specific LLM configuration
    extract_service_id: Optional[int] = None
    keywords_service_id: Optional[int] = None
    vlm_service_id: Optional[int] = None
    # LightRAG chunking config (forwarded to the auto-created silo)
    lightrag_chunk_strategy: Optional[str] = None
    lightrag_chunk_token_size: Optional[int] = None
    lightrag_chunk_overlap_token_size: Optional[int] = None
    lightrag_language: Optional[str] = None
    lightrag_entity_extract_max_gleaning: Optional[int] = None
    lightrag_max_source_ids_per_entity: Optional[int] = None
    lightrag_max_source_ids_per_relation: Optional[int] = None
    lightrag_entity_types: Optional[str] = None
    lightrag_entity_types_mode: Optional[Literal['infer', 'manual']] = None


class UpdateRepositorySchema(BaseModel):
    """Schema for updating an existing repository (vector_db_type is immutable after creation)"""
    name: str
    type: Optional[str] = None
    status: Optional[str] = None
    embedding_service_id: Optional[int] = None
    transcription_service_id: Optional[int] = None
    video_ai_service_id: Optional[int] = None
    indexing_service_id: Optional[int] = None  # legacy alias for extract_service_id
    # Absent before, so Pydantic dropped them from every update: the form sent
    # them and the backend never saw them. What a silo will accept is decided by
    # SiloService.is_lightrag_config_locked, not here.
    extract_service_id: Optional[int] = None
    keywords_service_id: Optional[int] = None
    vlm_service_id: Optional[int] = None
    lightrag_vector_db_type: Optional[str] = None
    lightrag_chunk_strategy: Optional[str] = None
    lightrag_chunk_token_size: Optional[int] = None
    lightrag_chunk_overlap_token_size: Optional[int] = None
    lightrag_language: Optional[str] = None
    lightrag_entity_extract_max_gleaning: Optional[int] = None
    lightrag_max_source_ids_per_entity: Optional[int] = None
    lightrag_max_source_ids_per_relation: Optional[int] = None
    lightrag_entity_types: Optional[str] = None
    lightrag_entity_types_mode: Optional[Literal['infer', 'manual']] = None


# Backward-compatible alias used by internal callers that haven't been updated yet
CreateUpdateRepositorySchema = CreateRepositorySchema


class RepositorySearchSchema(BaseModel):
    """Schema for searching within a repository"""
    query: str
    limit: Optional[int] = 10
    filter_metadata: Optional[Dict[str, Any]] = None
