from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.rerank_service import RerankProvider
from schemas.service_lifecycle_schemas import ServiceLifecycleFieldsMixin
from tools.rerank.endpoints import normalize_endpoint

_RERANK_NOTE = (
    "Reranking is an optional second stage applied after the configured search method; "
    "the query and the text of the candidate chunks are sent to this endpoint."
)

LifecycleStatusLiteral = Literal["active", "deprecated", "retired"]


class RerankServiceListItemSchema(BaseModel):
    """Schema for rerank service list items."""
    service_id: int
    name: str
    provider: Optional[str] = None
    model_name: str = Field(..., description=f"Reranker model sent in each request. {_RERANK_NOTE}")
    base_url: str = Field("", description=f"Base URL of the Cohere-compatible reranker. {_RERANK_NOTE}")
    created_at: Optional[datetime] = None
    # Keys are optional (self-hosted rerankers); true only for the CHANGE_ME import placeholder
    needs_api_key: bool = False
    is_system: bool = False
    is_platform_default: bool = False
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None

    end_of_service_at: Optional[datetime] = None
    deprecation_notice: Optional[str] = None
    lifecycle_status: LifecycleStatusLiteral = "active"

    model_config = ConfigDict(from_attributes=True)


class RerankServiceDetailSchema(RerankServiceListItemSchema):
    """Schema for detailed rerank service information (the API key is always masked)."""
    api_key: str = ""
    available_providers: List[Dict[str, Any]] = []


class CreateUpdateRerankServiceSchema(ServiceLifecycleFieldsMixin):
    """Schema for creating, updating or testing a rerank service."""
    name: str = Field(..., min_length=1, max_length=100)
    provider: str = Field(..., description="One of the RerankProvider values")
    model_name: str = Field(..., min_length=1, max_length=255, description=f"Reranker model. {_RERANK_NOTE}")
    api_key: Optional[str] = Field("", description="Optional bearer key; a masked value keeps the stored key")
    base_url: str = Field(..., description=f"Base URL of the reranker (normalised). {_RERANK_NOTE}")

    @field_validator("name", "model_name", "api_key", "base_url", mode="before")
    @classmethod
    def _strip_credentials(cls, v):
        # Trim whitespace/newlines that sneak in when pasting keys or URLs.
        return v.strip() if isinstance(v, str) else v

    @field_validator("name", "model_name", mode="after")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        if v not in {p.value for p in RerankProvider}:
            raise ValueError("Unknown rerank provider")
        return v

    @field_validator("base_url")
    @classmethod
    def _normalise_base_url(cls, v: str) -> str:
        return normalize_endpoint(v)


class RerankServiceOptionSchema(BaseModel):
    """Lightweight schema for rerank service selectors (retired ones are flagged, not hidden)."""
    service_id: int
    name: str
    provider: Optional[str] = None
    model_name: str = ""
    is_system: bool = False
    is_platform_default: bool = False
    end_of_service_at: Optional[datetime] = None
    deprecation_notice: Optional[str] = None
    lifecycle_status: LifecycleStatusLiteral = "active"

    model_config = ConfigDict(from_attributes=True)


class RerankServiceOptionsSchema(BaseModel):
    """Selector data: options plus the app, platform and effective (non-retired) defaults."""
    options: List[RerankServiceOptionSchema] = []
    app_default: Optional[RerankServiceOptionSchema] = None
    platform_default: Optional[RerankServiceOptionSchema] = None
    effective_default: Optional[RerankServiceOptionSchema] = None


class SetPlatformDefaultSchema(BaseModel):
    """Body of the platform-default switch; null clears the default."""
    service_id: Optional[int] = Field(None, description="System rerank service to mark as platform default, or null")
