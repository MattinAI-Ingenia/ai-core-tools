"""IndexingMetric — per-document LLM token/time/cost record for LightRAG indexing."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import relationship

from db.database import Base


class IndexingMetric(Base):
    """Stores actual LLM usage recorded during a single document indexing run.

    One row is created per indexing run (success, partial, or failed).
    Re-indexing a document inserts a *new* row; history is preserved.
    The UI always displays the row with the highest ``created_at`` for a
    given ``resource_id``.
    """

    __tablename__ = "indexing_metric"

    metric_id = Column(Integer, primary_key=True, autoincrement=True)

    # Tenant / scope
    app_id = Column(Integer, ForeignKey("App.app_id", ondelete="CASCADE"), nullable=False)
    silo_id = Column(Integer, ForeignKey("Silo.silo_id", ondelete="CASCADE"), nullable=False)
    resource_id = Column(Integer, ForeignKey("Resource.resource_id", ondelete="SET NULL"), nullable=True)
    # Free-form ref for non-Resource content (media, domain URL, etc.)
    content_ref = Column(String(1000), nullable=True)

    # Outcome
    status = Column(String(20), nullable=False)  # success | failed | partial

    # LLM token counts (sum across all roles: extract/keyword/query/vlm)
    prompt_tokens = Column(Integer, nullable=False, default=0)
    completion_tokens = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)

    # Embedding tokens (best-effort)
    embedding_tokens = Column(Integer, nullable=True)

    # How token counts were obtained
    tokens_source = Column(String(12), nullable=False, default="provider")  # provider | estimated

    # Number of LLM invocations during the run
    llm_calls = Column(Integer, nullable=True)

    # Wall-clock duration for this document
    duration_seconds = Column(Float, nullable=False, default=0.0)

    # Monetary cost (NULL when model not in PricingCatalog)
    cost = Column(Float, nullable=True)
    currency = Column(String(3), nullable=True)  # ISO 4217

    # Model identifiers
    model_name = Column(String(255), nullable=True)
    embedding_model_name = Column(String(255), nullable=True)

    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Relationships (read-only references; no back-populates needed here)
    app = relationship("App", foreign_keys=[app_id])
    silo = relationship("Silo", foreign_keys=[silo_id])
    resource = relationship("Resource", foreign_keys=[resource_id])

    __table_args__ = (
        Index("idx_indexing_metric_silo_id", "silo_id"),
        Index("idx_indexing_metric_resource_id", "resource_id"),
        Index("idx_indexing_metric_app_id", "app_id"),
        Index("idx_indexing_metric_resource_created", "resource_id", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<IndexingMetric(metric_id={self.metric_id}, resource_id={self.resource_id}, "
            f"status={self.status}, total_tokens={self.total_tokens}, "
            f"duration={self.duration_seconds:.1f}s, cost={self.cost})>"
        )
