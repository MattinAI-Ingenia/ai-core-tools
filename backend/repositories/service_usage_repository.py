"""Column-only lookups used to compute end-of-service dependency warnings without N+1 queries.

Every method issues exactly one statement, whatever the number of rows involved, and selects plain
columns (no ORM entities), so the results stay cheap and detached from the session.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from models.agent import Agent
from models.ai_service import AIService
from models.embedding_service import EmbeddingService
from models.middleware import AgentMiddleware, Middleware, MiddlewareType
from models.ocr_agent import OCRAgent
from models.silo import Silo


@dataclass(frozen=True)
class ServiceSnapshot:
    """The lifecycle-relevant columns of an AI / embedding / sandbox service."""

    service_id: int
    name: str
    app_id: Optional[int]
    end_of_service_at: Optional[datetime]
    deprecation_notice: Optional[str]


@dataclass(frozen=True)
class SiloRefs:
    """The services a silo points at."""

    silo_id: int
    name: str
    embedding_service_id: Optional[int]
    transcription_service_id: Optional[int]
    video_ai_service_id: Optional[int]


@dataclass(frozen=True)
class AgentRefs:
    """The service ids an agent points at (``vision_service_id`` only exists on OCR agents)."""

    agent_id: int
    service_id: Optional[int]
    vision_service_id: Optional[int]
    transcription_service_id: Optional[int]
    video_ai_service_id: Optional[int]
    media_embedding_service_id: Optional[int]
    silo_id: Optional[int]


@dataclass(frozen=True)
class MiddlewareRow:
    """Raw middleware row; the config is parsed by the service layer."""

    middleware_id: int
    name: str
    middleware_type: MiddlewareType
    config: Optional[dict]


class ServiceUsageRepository:
    """Read-only, tenant-scoped lookups for lifecycle warnings."""

    @staticmethod
    def _snapshots(db: Session, model, app_id: int) -> Dict[int, ServiceSnapshot]:
        stmt = select(
            model.service_id, model.name, model.app_id, model.end_of_service_at, model.deprecation_notice
        ).where(or_(model.app_id == app_id, model.app_id.is_(None)))
        return {row.service_id: ServiceSnapshot(*row) for row in db.execute(stmt)}

    @staticmethod
    def ai_service_snapshots(db: Session, app_id: int) -> Dict[int, ServiceSnapshot]:
        """AI services visible to the app (its own + system), keyed by id. One query."""
        return ServiceUsageRepository._snapshots(db, AIService, app_id)

    @staticmethod
    def embedding_service_snapshots(db: Session, app_id: int) -> Dict[int, ServiceSnapshot]:
        """Embedding services visible to the app (its own + system), keyed by id. One query."""
        return ServiceUsageRepository._snapshots(db, EmbeddingService, app_id)

    @staticmethod
    def silo_refs(db: Session, app_id: int) -> Dict[int, SiloRefs]:
        """The app's silos with the services they use, keyed by silo id. One query."""
        stmt = select(
            Silo.silo_id, Silo.name, Silo.embedding_service_id, Silo.transcription_service_id,
            Silo.video_ai_service_id,
        ).where(Silo.app_id == app_id)
        return {row.silo_id: SiloRefs(*row) for row in db.execute(stmt)}

    @staticmethod
    def lifecycle_middlewares(db: Session, app_id: int) -> List[MiddlewareRow]:
        """The app's middlewares that can reference an AI service (PII, SUMMARIZATION). One query."""
        stmt = select(
            Middleware.middleware_id, Middleware.name, Middleware.middleware_type, Middleware.config
        ).where(
            Middleware.app_id == app_id,
            Middleware.middleware_type.in_([MiddlewareType.PII, MiddlewareType.SUMMARIZATION]),
        )
        return [MiddlewareRow(*row) for row in db.execute(stmt)]

    @staticmethod
    def agent_middleware_ids(
        db: Session, app_id: int, agent_ids: Optional[Sequence[int]] = None
    ) -> Dict[int, List[int]]:
        """``{agent_id: [middleware_id, ...]}`` for the app's agents. One query.

        ``agent_ids=None`` means every agent of the app (no unbounded IN list); an empty sequence
        runs no query.
        """
        result: Dict[int, List[int]] = {}
        if agent_ids is not None and not agent_ids:
            return result
        stmt = (
            select(AgentMiddleware.agent_id, AgentMiddleware.middleware_id)
            .join(Agent, Agent.agent_id == AgentMiddleware.agent_id)
            .where(Agent.app_id == app_id)
        )
        if agent_ids is not None:
            stmt = stmt.where(AgentMiddleware.agent_id.in_(list(agent_ids)))
        for agent_id, middleware_id in db.execute(stmt):
            result.setdefault(agent_id, []).append(middleware_id)
        return result

    @staticmethod
    def agent_refs(db: Session, app_id: int, agent_ids: Optional[Iterable[int]] = None) -> Dict[int, AgentRefs]:
        """Service ids of the app's agents (OCR vision service included). One query.

        ``agent_ids=None`` means every agent of the app; an empty iterable runs no query.
        """
        ids = None if agent_ids is None else list(agent_ids)
        if ids is not None and not ids:
            return {}
        # The OCRAgent *table* is joined directly: joining the mapped subclass would drag the Agent table in twice.
        ocr = OCRAgent.__table__
        stmt = (
            select(
                Agent.agent_id, Agent.service_id, ocr.c.vision_service_id, Agent.transcription_service_id,
                Agent.video_ai_service_id, Agent.media_embedding_service_id, Agent.silo_id,
            )
            .select_from(Agent)
            .outerjoin(ocr, ocr.c.agent_id == Agent.agent_id)
            .where(Agent.app_id == app_id)
        )
        if ids is not None:
            stmt = stmt.where(Agent.agent_id.in_(ids))
        return {row.agent_id: AgentRefs(*row) for row in db.execute(stmt)}
