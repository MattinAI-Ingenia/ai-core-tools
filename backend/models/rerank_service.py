import enum
from sqlalchemy import Column, String, Integer, Boolean, ForeignKey, CheckConstraint, Index, text
from sqlalchemy.orm import relationship
from models.base_service import BaseService
from models.service_lifecycle import ServiceLifecycleMixin
from models.service_health import ServiceHealthMixin, track_service_health


class RerankProvider(enum.Enum):
    CohereCompatible = "CohereCompatible"


class RerankService(ServiceLifecycleMixin, ServiceHealthMixin, BaseService):
    __tablename__ = 'rerank_service'

    provider = Column(String(45), nullable=False)
    model_name = Column(String(255), nullable=False)
    app_id = Column(Integer, ForeignKey('App.app_id'), nullable=True, index=True)  # NULL = system/platform service
    # At most one platform default (partial unique index); only a system service (app_id IS NULL) may be it.
    is_platform_default = Column(Boolean, nullable=False, default=False, server_default='false')
    app = relationship('App', back_populates='rerank_services', foreign_keys=[app_id])

    __table_args__ = (
        CheckConstraint('NOT is_platform_default OR app_id IS NULL',
                        name='ck_rerank_service_platform_default_is_system'),
        Index('uq_rerank_service_platform_default', 'is_platform_default', unique=True,
              postgresql_where=text('is_platform_default')),
    )


track_service_health(RerankService)
