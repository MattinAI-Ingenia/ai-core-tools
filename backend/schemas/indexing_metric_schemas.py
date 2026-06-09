"""Pydantic schemas for indexing metric API responses."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class IndexingMetricSchema(BaseModel):
    """Single indexing-run record for a resource."""

    metric_id: int
    silo_id: int
    resource_id: Optional[int] = None
    content_ref: Optional[str] = None
    status: str  # 'success' | 'failed' | 'partial'

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_source: Optional[str] = None  # 'provider' | 'estimated'

    embedding_tokens: Optional[int] = None

    llm_calls: int = 0
    duration_seconds: Optional[float] = None

    cost: Optional[float] = None
    currency: Optional[str] = None
    model_name: Optional[str] = None
    embedding_model_name: Optional[str] = None

    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SiloIndexingTotalsSchema(BaseModel):
    """Aggregated token/cost totals for an entire silo."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: Optional[float] = None
    currency: Optional[str] = None
    total_llm_calls: int = 0
    indexed_resources: int = 0


class SiloIndexingMetricsResponseSchema(BaseModel):
    """List of per-resource latest metrics together with silo-level totals."""

    metrics: List[IndexingMetricSchema] = Field(default_factory=list)
    totals: SiloIndexingTotalsSchema = Field(default_factory=SiloIndexingTotalsSchema)
    count: int = 0
