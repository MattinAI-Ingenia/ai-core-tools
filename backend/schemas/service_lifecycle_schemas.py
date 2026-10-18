"""Shared end-of-service schema fragments for AI, embedding and sandbox services."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from utils.service_lifecycle import MAX_DEPRECATION_NOTICE_LENGTH, normalize_deprecation_notice, to_utc


class ServiceLifecycleFieldsMixin(BaseModel):
    """Editable end-of-service fields (create/update payloads and export files).

    ``lifecycle_status`` is read-only and deliberately not part of this mixin. Services only apply
    a field that is in ``model_fields_set``, so omitting it keeps the stored value.
    """

    end_of_service_at: Optional[datetime] = None
    deprecation_notice: Optional[str] = Field(None, max_length=MAX_DEPRECATION_NOTICE_LENGTH)

    @field_validator("end_of_service_at", mode="after")
    @classmethod
    def _end_of_service_to_utc(cls, v):
        return to_utc(v)

    @field_validator("deprecation_notice", mode="before")
    @classmethod
    def _clean_deprecation_notice(cls, v):
        return normalize_deprecation_notice(v) if isinstance(v, str) else v


class ServiceLifecycleWarningSchema(BaseModel):
    """A deprecated or retired service a resource depends on (directly or through a silo / middleware)."""

    service_id: int
    service_name: str
    kind: Literal["ai", "embedding"]
    role: Literal[
        "llm", "ocr_vision", "transcription", "video", "media_embedding", "silo_embedding", "middleware", "embedding"
    ]
    # Silo name (role silo_embedding) or middleware name (role middleware).
    via: Optional[str] = None
    lifecycle_status: Literal["deprecated", "retired"]
    end_of_service_at: datetime
    deprecation_notice: Optional[str] = None
