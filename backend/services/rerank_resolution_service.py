"""Which rerank service an agent uses at runtime (AD-9).

Order: the agent's explicit service, then the app default, then the platform default. The first candidate that
is *set* decides: if it is retired, deleted or not visible to the app, reranking is SKIPPED, never substituted
by the next level (a silent fallback would send the app's documents to a provider nobody chose for it).
"""

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from models.app import App
from models.rerank_service import RerankService
from utils.service_lifecycle import is_retired

SOURCE_AGENT = "agent"
SOURCE_APP_DEFAULT = "app_default"
SOURCE_PLATFORM_DEFAULT = "platform_default"

SKIP_NONE_CONFIGURED = "no_service_configured"
SKIP_RETIRED = "service_retired"
SKIP_UNAVAILABLE = "service_unavailable"


@dataclass(frozen=True)
class RerankResolution:
    """Outcome of the resolution: a service to use, or the reason reranking is skipped."""

    service: Optional[RerankService]
    source: Optional[str]
    skip_reason: Optional[str] = None


def _visible(service: RerankService, app_id: int) -> bool:
    return service.app_id is None or service.app_id == app_id


def _check(service: Optional[RerankService], source: str, app_id: int) -> RerankResolution:
    if service is None or not _visible(service, app_id):
        return RerankResolution(None, source, SKIP_UNAVAILABLE)
    if is_retired(service):
        return RerankResolution(None, source, SKIP_RETIRED)
    return RerankResolution(service, source)


def choose_rerank_service(
    explicit_id: Optional[int],
    explicit: Optional[RerankService],
    app_default: Optional[RerankService],
    platform_default: Optional[RerankService],
    app_id: int,
    app_default_id: Optional[int] = None,
) -> RerankResolution:
    """Pure AD-9 decision from already loaded rows (no DB access).

    Args:
        explicit_id: the agent's ``rag_rerank_service_id`` (None = inherit).
        explicit: the row for ``explicit_id``, or None when it does not exist.
        app_default: the app's default row, or None.
        platform_default: the row flagged as platform default, or None.
        app_id: tenant of the agent; a non-system service of another app is treated as unavailable.
        app_default_id: the app's default id, so a set-but-missing default is a skip, not a fallthrough.

    Returns:
        A resolution with the service, or ``service=None`` and a ``skip_reason``. No substitution ever happens
        once a level is set.
    """
    if explicit_id is not None:
        return _check(explicit, SOURCE_AGENT, app_id)
    if app_default is not None or app_default_id is not None:
        return _check(app_default, SOURCE_APP_DEFAULT, app_id)
    if platform_default is not None:
        return _check(platform_default, SOURCE_PLATFORM_DEFAULT, app_id)
    return RerankResolution(None, None, SKIP_NONE_CONFIGURED)


class RerankResolutionService:
    """Loads the candidates and applies :func:`choose_rerank_service`."""

    @staticmethod
    def resolve(db: Session, app_id: int, explicit_service_id: Optional[int]) -> RerankResolution:
        """Resolve the service for an agent of ``app_id`` with ONE query for all three candidates."""
        # Driven from App with an outer join: ONE statement always returns the app's default id (even when no
        # service row matches) plus every matching candidate.
        conditions = [RerankService.service_id == App.default_rerank_service_id,
                      RerankService.is_platform_default.is_(True)]
        if explicit_service_id is not None:
            conditions.append(RerankService.service_id == explicit_service_id)
        stmt = (
            select(App.default_rerank_service_id, RerankService)
            .select_from(App)
            .outerjoin(RerankService, or_(*conditions))
            .where(App.app_id == app_id)
        )
        rows = db.execute(stmt).all()

        app_default_id: Optional[int] = rows[0][0] if rows else None
        rows = [(row[1], row[0]) for row in rows if row[1] is not None]
        by_id = {row[0].service_id: row[0] for row in rows}
        platform_default = next((row[0] for row in rows if row[0].is_platform_default), None)
        return choose_rerank_service(
            explicit_service_id,
            by_id.get(explicit_service_id) if explicit_service_id is not None else None,
            by_id.get(app_default_id) if app_default_id is not None else None,
            platform_default,
            app_id,
            app_default_id,
        )

    @staticmethod
    def effective_default(db: Session, app_id: int) -> RerankResolution:
        """What an agent that inherits would use (app default, then platform default); used by the options
        endpoint and by save-time validation."""
        return RerankResolutionService.resolve(db, app_id, None)
