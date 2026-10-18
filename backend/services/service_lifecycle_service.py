from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from repositories.service_usage_repository import (
    AgentRefs,
    ServiceSnapshot,
    ServiceUsageRepository,
    SiloRefs,
)
from schemas.middleware_schemas import parse_middleware_config, referenced_ai_service_ids
from schemas.service_lifecycle_schemas import ServiceLifecycleWarningSchema
from services.system_settings_service import SystemSettingsService
from utils.logger import get_logger
from utils.service_lifecycle import (
    SERVICE_KIND_AI,
    SERVICE_KIND_EMBEDDING,
    LifecycleStatus,
    compute_lifecycle_status,
    to_utc,
)

logger = get_logger(__name__)

WARNING_DAYS_SETTING_KEY = "service_end_of_service_warning_days"
DEFAULT_WARNING_DAYS = 30


@dataclass
class LifecycleContext:
    """Everything the warning builders need, loaded once per request.

    ``services`` is keyed by service kind (``ai`` / ``embedding``; a sandbox entry can be added without
    touching the builders) and then by service id. The builders only read these dicts, never ORM
    relationships, so building warnings for N resources costs no extra queries.
    """

    warning_days: int
    now: datetime
    services: Dict[str, Dict[int, ServiceSnapshot]] = field(default_factory=dict)
    silos: Dict[int, SiloRefs] = field(default_factory=dict)
    # middleware_id -> (name, AI service ids it references)
    middlewares: Dict[int, Tuple[str, List[int]]] = field(default_factory=dict)
    agent_middlewares: Dict[int, List[int]] = field(default_factory=dict)
    agent_refs: Dict[int, AgentRefs] = field(default_factory=dict)


class ServiceLifecycleService:
    """DB-aware parts of the service end-of-service lifecycle (the pure rules live in
    ``utils.service_lifecycle``)."""

    @staticmethod
    def warning_days(db: Session) -> int:
        """Days before end of service during which a service is shown as deprecated.

        Read once per request by callers. Falls back to 30 if the setting cannot be read,
        and clamps negative values to 0.
        """
        try:
            value = SystemSettingsService(db).get_setting(WARNING_DAYS_SETTING_KEY)
            days = DEFAULT_WARNING_DAYS if value is None else int(value)
        except Exception as exc:
            if isinstance(exc, SQLAlchemyError):
                # Leave the caller's session usable for the rest of the request.
                db.rollback()
            logger.warning("Could not read %s, using %s: %s", WARNING_DAYS_SETTING_KEY, DEFAULT_WARNING_DAYS, exc)
            return DEFAULT_WARNING_DAYS
        if days < 0:
            logger.warning("%s is negative (%s), using 0", WARNING_DAYS_SETTING_KEY, days)
            return 0
        return days

    @staticmethod
    def log_change(
        service: Any, kind: str, old: Optional[datetime], new: Optional[datetime], actor_id: Optional[int]
    ) -> None:
        """INFO audit line for an end-of-service date change."""
        logger.info(
            "service_lifecycle.updated kind=%s service_id=%s app_id=%s actor_id=%s old=%s new=%s",
            kind,
            service.service_id,
            service.app_id,
            actor_id,
            old.isoformat() if old else None,
            new.isoformat() if new else None,
        )

    @staticmethod
    def apply_lifecycle_fields(service: Any, kind: str, data: Any, actor_id: Optional[int] = None) -> None:
        """Copy the lifecycle fields present in ``data`` (a pydantic model) onto ``service``.

        A field absent from ``data.model_fields_set`` is left untouched; an explicit null clears it.
        Logs at INFO when the date changes on an already persisted service (creates are logged by
        the caller once the row has an id, via ``log_change``).
        """
        fields_set = data.model_fields_set
        old = to_utc(service.end_of_service_at)
        if "end_of_service_at" in fields_set:
            service.end_of_service_at = to_utc(data.end_of_service_at)
        if "deprecation_notice" in fields_set:
            service.deprecation_notice = data.deprecation_notice
        new = to_utc(service.end_of_service_at)
        if old != new and service.service_id is not None:
            ServiceLifecycleService.log_change(service, kind, old, new, actor_id)

    # ==================== Dependency warnings (AD-8) ====================

    @staticmethod
    def build_context(
        db: Session, app_id: int, agent_ids: Optional[Iterable[int]] = None, all_agents: bool = False
    ) -> LifecycleContext:
        """Load the lookup context for warnings in one fixed set of queries.

        Always: the warning-days setting, AI and embedding services, the app's silos. Agent data
        (service references, middleware links, middlewares) is loaded only when ``all_agents`` is
        true (every agent of the app, for list endpoints) or ``agent_ids`` is non-empty. The count
        does not depend on how many agents/silos are involved.
        """
        ids: Optional[List[int]] = None
        if not all_agents:
            ids = sorted({i for i in (agent_ids or []) if i})
        needs_agents = all_agents or bool(ids)
        warning_days = ServiceLifecycleService.warning_days(db)
        middlewares: Dict[int, Tuple[str, List[int]]] = {}
        for row in (ServiceUsageRepository.lifecycle_middlewares(db, app_id) if needs_agents else []):
            try:
                config = parse_middleware_config(row.middleware_type, row.config)
            except ValidationError:
                continue  # an invalid stored config references nothing we can resolve
            middlewares[row.middleware_id] = (row.name, referenced_ai_service_ids(config))
        return LifecycleContext(
            warning_days=warning_days,
            now=datetime.now(timezone.utc),
            services={
                SERVICE_KIND_AI: ServiceUsageRepository.ai_service_snapshots(db, app_id),
                SERVICE_KIND_EMBEDDING: ServiceUsageRepository.embedding_service_snapshots(db, app_id),
            },
            silos=ServiceUsageRepository.silo_refs(db, app_id),
            middlewares=middlewares,
            agent_middlewares=(
                ServiceUsageRepository.agent_middleware_ids(db, app_id, ids) if needs_agents else {}
            ),
            agent_refs=ServiceUsageRepository.agent_refs(db, app_id, ids) if needs_agents else {},
        )

    @staticmethod
    def option_fields(service: Any, warning_days: int) -> Dict[str, Any]:
        """Lifecycle data carried by a service option in a selection dropdown."""
        eos = getattr(service, "end_of_service_at", None)
        return {
            "end_of_service_at": to_utc(eos),
            "deprecation_notice": getattr(service, "deprecation_notice", None),
            "lifecycle_status": compute_lifecycle_status(eos, warning_days).value,
        }

    @staticmethod
    def _entry(
        ctx: LifecycleContext, kind: str, service_id: Optional[int], role: str, via: Optional[str] = None
    ) -> Optional[ServiceLifecycleWarningSchema]:
        if service_id is None:
            return None
        snapshot = ctx.services.get(kind, {}).get(service_id)
        if snapshot is None or snapshot.end_of_service_at is None:
            return None
        status = compute_lifecycle_status(snapshot.end_of_service_at, ctx.warning_days, ctx.now)
        if status is LifecycleStatus.ACTIVE:
            return None
        return ServiceLifecycleWarningSchema(
            service_id=snapshot.service_id,
            service_name=snapshot.name,
            kind=kind,
            role=role,
            via=via,
            lifecycle_status=status.value,
            end_of_service_at=to_utc(snapshot.end_of_service_at),
            deprecation_notice=snapshot.deprecation_notice,
        )

    @staticmethod
    def _finish(entries: Iterable[Optional[ServiceLifecycleWarningSchema]]) -> List[ServiceLifecycleWarningSchema]:
        """Drop empties and duplicates, retired first, then by end-of-service date."""
        unique: Dict[Tuple[int, str, str, Optional[str]], ServiceLifecycleWarningSchema] = {}
        for entry in entries:
            if entry is not None:
                unique.setdefault((entry.service_id, entry.kind, entry.role, entry.via), entry)
        return sorted(
            unique.values(),
            key=lambda e: (0 if e.lifecycle_status == LifecycleStatus.RETIRED.value else 1, e.end_of_service_at),
        )

    @staticmethod
    def warnings_for_agent(agent_row: Any, ctx: LifecycleContext) -> List[ServiceLifecycleWarningSchema]:
        """Warnings for the services an agent depends on.

        ``agent_row`` only needs the service-id attributes (an ``AgentRefs`` or a loaded agent); the
        linked silo and middlewares are resolved through the context dicts.
        """
        if agent_row is None:
            return []
        entry = ServiceLifecycleService._entry
        ai, emb = SERVICE_KIND_AI, SERVICE_KIND_EMBEDDING
        entries = [
            entry(ctx, ai, getattr(agent_row, "service_id", None), "llm"),
            entry(ctx, ai, getattr(agent_row, "vision_service_id", None), "ocr_vision"),
            entry(ctx, ai, getattr(agent_row, "transcription_service_id", None), "transcription"),
            entry(ctx, ai, getattr(agent_row, "video_ai_service_id", None), "video"),
            entry(ctx, emb, getattr(agent_row, "media_embedding_service_id", None), "media_embedding"),
        ]
        silo = ctx.silos.get(getattr(agent_row, "silo_id", None))
        if silo is not None:
            entries.append(entry(ctx, emb, silo.embedding_service_id, "silo_embedding", silo.name))
        for middleware_id in ctx.agent_middlewares.get(getattr(agent_row, "agent_id", None), []):
            name, service_ids = ctx.middlewares.get(middleware_id, (None, []))
            entries.extend(entry(ctx, ai, service_id, "middleware", name) for service_id in service_ids)
        return ServiceLifecycleService._finish(entries)

    @staticmethod
    def warnings_for_silo(silo_id: Optional[int], ctx: LifecycleContext) -> List[ServiceLifecycleWarningSchema]:
        """Warnings for the embedding, transcription and video services of a silo."""
        silo = ctx.silos.get(silo_id)
        if silo is None:
            return []
        entry = ServiceLifecycleService._entry
        return ServiceLifecycleService._finish([
            entry(ctx, SERVICE_KIND_EMBEDDING, silo.embedding_service_id, "embedding"),
            entry(ctx, SERVICE_KIND_AI, silo.transcription_service_id, "transcription"),
            entry(ctx, SERVICE_KIND_AI, silo.video_ai_service_id, "video"),
        ])

    @staticmethod
    def warnings_for_task(task: Any, ctx: LifecycleContext) -> List[ServiceLifecycleWarningSchema]:
        """A scheduled task depends on what its agent depends on."""
        return ServiceLifecycleService.warnings_for_agent(ctx.agent_refs.get(getattr(task, "agent_id", None)), ctx)
