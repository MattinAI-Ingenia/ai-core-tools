from typing import List, Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from models.rerank_service import RerankService


class RerankServiceRepository:
    """Data access for rerank services (app-owned rows have app_id set, system rows have app_id NULL)."""

    @staticmethod
    def get_by_app_id(db: Session, app_id: int) -> List[RerankService]:
        """Get all rerank services owned by an app."""
        stmt = select(RerankService).where(RerankService.app_id == app_id).order_by(RerankService.service_id)
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def get_by_id_and_app_id(db: Session, service_id: int, app_id: int) -> Optional[RerankService]:
        """Get a rerank service by ID, only if it belongs to the app."""
        stmt = select(RerankService).where(RerankService.service_id == service_id, RerankService.app_id == app_id)
        return db.execute(stmt).scalars().first()

    @staticmethod
    def get_by_id(db: Session, service_id: int) -> Optional[RerankService]:
        """Get a rerank service by ID regardless of scope."""
        stmt = select(RerankService).where(RerankService.service_id == service_id)
        return db.execute(stmt).scalars().first()

    @staticmethod
    def get_visible_by_id(db: Session, service_id: int, app_id: int) -> Optional[RerankService]:
        """Get a service by ID if it is owned by the app or is a system service."""
        stmt = select(RerankService).where(
            RerankService.service_id == service_id,
            or_(RerankService.app_id == app_id, RerankService.app_id.is_(None)),
        )
        return db.execute(stmt).scalars().first()

    @staticmethod
    def get_system_services(db: Session) -> List[RerankService]:
        """Return all system (platform) services, i.e. app_id IS NULL."""
        stmt = select(RerankService).where(RerankService.app_id.is_(None)).order_by(RerankService.service_id)
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def get_platform_default(db: Session) -> Optional[RerankService]:
        """Return the platform default service, if one is flagged."""
        stmt = select(RerankService).where(RerankService.is_platform_default.is_(True))
        return db.execute(stmt).scalars().first()

    @staticmethod
    def lock_platform_defaults(db: Session) -> List[RerankService]:
        """Lock (SELECT ... FOR UPDATE) the currently flagged rows so a switch is serialised."""
        stmt = select(RerankService).where(RerankService.is_platform_default.is_(True)).with_for_update().execution_options(populate_existing=True)
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def create(db: Session, service: RerankService) -> RerankService:
        """Persist a new rerank service."""
        db.add(service)
        db.commit()
        db.refresh(service)
        return service

    @staticmethod
    def update(db: Session, service: RerankService) -> RerankService:
        """Persist changes to a rerank service."""
        db.add(service)
        db.commit()
        db.refresh(service)
        return service

    @staticmethod
    def delete(db: Session, service: RerankService) -> None:
        """Delete a rerank service."""
        db.delete(service)
        db.commit()
