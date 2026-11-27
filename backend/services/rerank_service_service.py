import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import config
from core.export_constants import PLACEHOLDER_API_KEY
from models.rerank_service import RerankProvider, RerankService
from repositories.app_repository import AppRepository
from repositories.rerank_service_repository import RerankServiceRepository
from schemas.service_lifecycle_schemas import ServiceUsageSchema
from schemas.rerank_service_schemas import (
    CreateUpdateRerankServiceSchema,
    RerankServiceDetailSchema,
    RerankServiceListItemSchema,
    RerankServiceOptionSchema,
    RerankServiceOptionsSchema,
)
from services.service_lifecycle_service import ServiceLifecycleService
from tools.rerank.client import test_rerank_connection
from tools.rerank.types import RerankTarget
from utils.logger import get_logger
from utils.secret_utils import is_masked_key, mask_api_key
from utils.service_lifecycle import (
    SERVICE_KIND_RERANK,
    RetiredServiceAssignmentError,
    compute_lifecycle_status,
    ensure_assignable,
    is_retired,
    to_utc,
)
from utils.ssrf_guard import endpoint_is_disallowed

logger = get_logger(__name__)

RERANK_SERVICE_NOT_FOUND_MSG = "Rerank service not found"
IN_USE_MESSAGE = (
    "This rerank service is in use (agents, an app default or the platform default). "
    "Move or clear those references first."
)


class RerankServiceError(Exception):
    """A business-rule failure with the HTTP status the router should answer with."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def detail(self) -> Any:
        """The HTTP error detail the router sends."""
        return self.message




class RerankServiceInUseError(RerankServiceError):
    """The service is referenced explicitly by an agent, is an app default or is the platform default (HTTP 409)."""

    def __init__(self, usage: ServiceUsageSchema):
        super().__init__(IN_USE_MESSAGE, 409)
        self.usage = usage

    def detail(self) -> Any:
        """The 409 body: a message plus the full usage so the UI can list what blocks the delete."""
        return {"message": self.message, "usage": self.usage.model_dump(mode="json")}


def _is_real_key(key: Optional[str]) -> bool:
    """True when ``key`` is an actual secret (not empty, not the CHANGE_ME placeholder, not masked)."""
    return bool(key) and key != PLACEHOLDER_API_KEY and not is_masked_key(key)


_DEFAULT_PORTS = {"http": 80, "https": 443}
_KEY_HOST_CHANGE_MSG = "Re-enter the API key when changing the endpoint host."


def _origin(url: Optional[str]) -> tuple:
    """(scheme, host, port) of a URL, with default ports filled in."""
    try:
        parts = urlsplit((url or "").strip())
        scheme = (parts.scheme or "").lower()
        return scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme)
    except ValueError:
        return "", "", None


def _check_key_reuse(stored: Optional[RerankService], new_endpoint: str) -> None:
    """Refuse to reuse the stored key for a different scheme/host/port (it would be sent to the new host)."""
    if stored is not None and stored.api_key and _origin(stored.endpoint) != _origin(new_endpoint):
        raise RerankServiceError(_KEY_HOST_CHANGE_MSG, 422)


def _provider_value(service: RerankService) -> Optional[str]:
    return service.provider.value if hasattr(service.provider, "value") else service.provider


class RerankServiceService:
    """Business logic for rerank services (app-scoped and system-scoped)."""

    # ---------------------------------------------------------------- mapping

    @staticmethod
    def _list_fields(service: RerankService, warning_days: int, expose_base_url: bool = True) -> Dict[str, Any]:
        return dict(
            service_id=service.service_id,
            name=service.name,
            provider=_provider_value(service),
            model_name=service.model_name or "",
            base_url=(service.endpoint or "") if expose_base_url else "",
            created_at=service.create_date,
            needs_api_key=service.api_key == PLACEHOLDER_API_KEY,
            is_system=service.app_id is None,
            is_platform_default=bool(service.is_platform_default),
            last_error=service.last_error,
            last_error_at=service.last_error_at,
            end_of_service_at=to_utc(service.end_of_service_at),
            deprecation_notice=service.deprecation_notice,
            lifecycle_status=compute_lifecycle_status(service.end_of_service_at, warning_days).value,
        )

    @staticmethod
    def to_list_item(
        service: RerankService, *, warning_days: int, expose_base_url: bool = True
    ) -> RerankServiceListItemSchema:
        """Convert an ORM row to a list item (is_system derives from app_id IS NULL)."""
        return RerankServiceListItemSchema(
            **RerankServiceService._list_fields(service, warning_days, expose_base_url)
        )

    @staticmethod
    def to_detail(
        service: RerankService, warning_days: int, providers: Optional[List[Dict[str, Any]]] = None
    ) -> RerankServiceDetailSchema:
        """Convert an ORM row to the detail schema (masked key, lifecycle status)."""
        return RerankServiceDetailSchema(
            **RerankServiceService._list_fields(service, warning_days),
            api_key=mask_api_key(service.api_key),
            available_providers=providers or [],
        )

    @staticmethod
    def _providers() -> List[Dict[str, Any]]:
        return [{"value": p.value, "name": p.value} for p in RerankProvider]

    # ------------------------------------------------------------- validation

    @staticmethod
    def validate_endpoint(endpoint: str, effective_api_key: Optional[str], *, app_scoped: bool) -> None:
        """Reject unsafe endpoints (blocking: resolves DNS, call from a worker thread).

        - App-scoped endpoints on private/internal addresses are rejected unless
          ``RERANK_ALLOW_PRIVATE_ENDPOINTS`` is set (AD-12); system services are exempt.
        - A real API key is never sent over plain http:// to a non-private host.
        """
        private = endpoint_is_disallowed(endpoint, allow_unresolvable=True)
        if app_scoped and private and not config.RERANK_ALLOW_PRIVATE_ENDPOINTS:
            raise RerankServiceError(
                "The endpoint points to a private or internal address, which is not allowed for workspace "
                "rerank services.",
                422,
            )
        if endpoint.lower().startswith("http://") and _is_real_key(effective_api_key) and not private:
            raise RerankServiceError(
                "An API key cannot be sent over plain http:// to a public host. Use https://.", 422
            )

    # ------------------------------------------------------------------ reads

    @staticmethod
    def get_rerank_services_list(db: Session, app_id: int) -> List[RerankServiceListItemSchema]:
        """Services of the app plus the platform-level system services."""
        warning_days = ServiceLifecycleService.warning_days(db)
        rows = RerankServiceRepository.get_by_app_id(db, app_id) + RerankServiceRepository.get_system_services(db)
        # The endpoint of a platform service is internal infrastructure: only the OMNIADMIN detail shows it.
        return [
            RerankServiceService.to_list_item(s, warning_days=warning_days, expose_base_url=s.app_id is not None)
            for s in rows
        ]

    @staticmethod
    def get_system_services_detail(db: Session) -> List[RerankServiceDetailSchema]:
        """All system services with masked keys (platform admin)."""
        warning_days = ServiceLifecycleService.warning_days(db)
        return [
            RerankServiceService.to_detail(s, warning_days)
            for s in RerankServiceRepository.get_system_services(db)
        ]

    @staticmethod
    def get_system_service_detail(db: Session, service_id: int) -> Optional[RerankServiceDetailSchema]:
        """Detail of a system service (masked key, real base_url); None when missing or not a system service."""
        service = RerankServiceRepository.get_by_id(db, service_id)
        if not service or service.app_id is not None:
            return None
        return RerankServiceService.to_detail(service, ServiceLifecycleService.warning_days(db))

    @staticmethod
    def get_rerank_service_detail(db: Session, app_id: int, service_id: int) -> Optional[RerankServiceDetailSchema]:
        """Detail of an app-owned service; ``service_id == 0`` returns an empty template."""
        providers = RerankServiceService._providers()
        if service_id == 0:
            return RerankServiceDetailSchema(
                service_id=0, name="", provider=None, model_name="", base_url="", api_key="",
                available_providers=providers,
            )
        service = RerankServiceRepository.get_by_id_and_app_id(db, service_id, app_id)
        if not service:
            return None
        return RerankServiceService.to_detail(service, ServiceLifecycleService.warning_days(db), providers)

    @staticmethod
    def options(db: Session, app_id: int) -> RerankServiceOptionsSchema:
        """Selector data (AD-19): visible services, app default, platform default and the effective default."""
        warning_days = ServiceLifecycleService.warning_days(db)

        def to_option(s: RerankService) -> RerankServiceOptionSchema:
            f = RerankServiceService._list_fields(s, warning_days)
            return RerankServiceOptionSchema(**{k: f[k] for k in RerankServiceOptionSchema.model_fields})

        rows = RerankServiceRepository.get_by_app_id(db, app_id) + RerankServiceRepository.get_system_services(db)
        options = [to_option(s) for s in rows]
        by_id = {o.service_id: o for o in options}

        app = AppRepository(db).get_by_id(app_id)
        app_default = by_id.get(app.default_rerank_service_id) if app and app.default_rerank_service_id else None
        platform_row = next((s for s in rows if s.is_platform_default), None)
        platform_default = by_id.get(platform_row.service_id) if platform_row else None

        # AC-8: the app default wins over the platform default; a retired pick means reranking is skipped
        # (no fallback to the platform default).
        chosen = app_default if app_default is not None else platform_default
        effective = chosen if chosen is not None and chosen.lifecycle_status != "retired" else None
        return RerankServiceOptionsSchema(
            options=options, app_default=app_default, platform_default=platform_default, effective_default=effective
        )

    # ----------------------------------------------------------------- writes

    @staticmethod
    def _apply(
        service: RerankService, data: CreateUpdateRerankServiceSchema, actor_id: Optional[int], is_new: bool = False
    ) -> None:
        service.name = data.name
        service.provider = data.provider
        service.model_name = data.model_name
        service.endpoint = data.base_url
        key = data.api_key or ""
        if is_masked_key(key):
            if is_new:  # nothing stored to keep
                service.api_key = None
        else:  # a masked value on update means "keep the stored key"
            service.api_key = key
        ServiceLifecycleService.apply_lifecycle_fields(service, SERVICE_KIND_RERANK, data, actor_id)

    @staticmethod
    def _save(
        db: Session, service: RerankService, is_new: bool, actor_id: Optional[int]
    ) -> RerankService:
        if is_new:
            service = RerankServiceRepository.create(db, service)
            if service.end_of_service_at is not None:
                ServiceLifecycleService.log_change(
                    service, SERVICE_KIND_RERANK, None, service.end_of_service_at, actor_id
                )
            return service
        return RerankServiceRepository.update(db, service)

    @staticmethod
    def create_or_update_rerank_service(
        db: Session,
        app_id: int,
        service_id: int,
        data: CreateUpdateRerankServiceSchema,
        actor_id: Optional[int] = None,
    ) -> Optional[RerankService]:
        """Create (service_id == 0) or update an app-owned service; None when not found in the app.

        Raises:
            RerankServiceError: 422 for an unsafe endpoint.
        """
        is_new = service_id == 0
        if is_new:
            service = RerankService()
            service.app_id = app_id
            service.is_platform_default = False
            service.create_date = datetime.now()
            stored_key = None
        else:
            service = RerankServiceRepository.get_by_id_and_app_id(db, service_id, app_id)
            if not service:
                return None
            stored_key = service.api_key
        effective_key = stored_key if is_masked_key(data.api_key or "") else data.api_key
        if not is_new and is_masked_key(data.api_key or ""):
            _check_key_reuse(service, data.base_url)
        RerankServiceService.validate_endpoint(data.base_url, effective_key, app_scoped=True)
        RerankServiceService._apply(service, data, actor_id, is_new)
        return RerankServiceService._save(db, service, is_new, actor_id)

    @staticmethod
    def create_system_service(
        db: Session, data: CreateUpdateRerankServiceSchema, actor_id: Optional[int] = None
    ) -> RerankService:
        """Create a platform-level service (OMNIADMIN)."""
        RerankServiceService.validate_endpoint(data.base_url, data.api_key, app_scoped=False)
        service = RerankService()
        service.app_id = None
        service.is_platform_default = False
        service.create_date = datetime.now()
        RerankServiceService._apply(service, data, actor_id, True)
        return RerankServiceService._save(db, service, True, actor_id)

    @staticmethod
    def update_system_service(
        db: Session, service_id: int, data: CreateUpdateRerankServiceSchema, actor_id: Optional[int] = None
    ) -> Optional[RerankService]:
        """Update a platform-level service; None when it does not exist or is not a system service."""
        service = RerankServiceRepository.get_by_id(db, service_id)
        if not service or service.app_id is not None:
            return None
        if is_masked_key(data.api_key or ""):
            _check_key_reuse(service, data.base_url)
        effective_key = service.api_key if is_masked_key(data.api_key or "") else data.api_key
        RerankServiceService.validate_endpoint(data.base_url, effective_key, app_scoped=False)
        RerankServiceService._apply(service, data, actor_id)
        return RerankServiceService._save(db, service, False, actor_id)

    @staticmethod
    def _delete_row(db: Session, service: RerankService) -> None:
        """Delete the row. The FK is the safety net: on a violation roll back, recompute usage and answer with the
        same 409 body as the pre-check."""
        service_id, app_id = service.service_id, service.app_id
        try:
            RerankServiceRepository.delete(db, service)
        except IntegrityError:
            db.rollback()
            usage = ServiceLifecycleService.get_usage(db, SERVICE_KIND_RERANK, service_id, app_id)
            if usage is None:
                raise RerankServiceError("Rerank service is in use", 409) from None
            raise RerankServiceInUseError(usage) from None

    @staticmethod
    def _ensure_not_in_use(db: Session, service: RerankService) -> None:
        """Raise RerankServiceInUseError on an explicit agent reference, an app default or the platform flag.

        Agents that only inherit the service do not block: the default reference that makes them inherit does.
        A system service is checked across every app, an app-owned one within its app.
        """
        usage = ServiceLifecycleService.get_usage(db, SERVICE_KIND_RERANK, service.service_id, service.app_id)
        if usage is None:
            return
        explicit = any("rerank" in agent.roles for agent in usage.agents)
        if explicit or usage.apps_using_as_default or usage.is_platform_default:
            raise RerankServiceInUseError(usage)  # the request session closes and releases the row lock

    @staticmethod
    def delete_rerank_service(db: Session, app_id: int, service_id: int) -> bool:
        """Delete an app-owned service; False when not found in the app.

        Raises:
            RerankServiceInUseError: an agent references it explicitly or it is the app default.
        """
        # Row lock first (held until the delete commits): a concurrent default/agent assignment reads the row FOR
        # SHARE and therefore waits for this transaction instead of slipping a reference in after the usage check.
        service = RerankServiceRepository.get_for_update(db, service_id)
        if not service or service.app_id != app_id:
            return False
        RerankServiceService._ensure_not_in_use(db, service)
        RerankServiceService._delete_row(db, service)
        return True

    @staticmethod
    def delete_for_app_cascade(db: Session, service: RerankService) -> None:
        """Delete an app-owned service while the app is being deleted. No usage guard: the app's agents and its
        default reference are already gone. Flushes but never commits: the caller (AppService.delete_app) owns the
        transaction, so a later failure rolls the whole deletion back. An IntegrityError propagates (it is not
        turned into a 409)."""
        db.delete(service)
        db.flush()

    @staticmethod
    def delete_system_service(db: Session, service_id: int) -> bool:
        """Delete a platform-level service; False when not found or not a system service."""
        service = RerankServiceRepository.get_for_update(db, service_id)  # see delete_rerank_service
        if not service or service.app_id is not None:
            return False
        RerankServiceService._ensure_not_in_use(db, service)
        RerankServiceService._delete_row(db, service)
        return True

    @staticmethod
    def set_platform_default(db: Session, service_id: Optional[int], actor_id: Optional[int] = None) -> None:
        """Mark one system service as the platform default (None clears it), in one transaction (AD-14).

        Raises:
            RerankServiceError: 404 unknown/non-system service, 422 retired service, 409 lost a concurrent race.
        """
        target: Optional[RerankService] = None
        if service_id is not None:
            target = RerankServiceRepository.get_by_id(db, service_id)
            if not target or target.app_id is not None:
                raise RerankServiceError(RERANK_SERVICE_NOT_FOUND_MSG, 404)

        current = RerankServiceRepository.lock_platform_defaults(db)
        if target is not None:
            db.refresh(target)  # never trust attributes read before the lock
            try:
                ensure_assignable(
                    target, current[0].service_id if current else None,
                    "Platform default rerank service", SERVICE_KIND_RERANK,
                )
            except RetiredServiceAssignmentError as exc:
                raise RerankServiceError(str(exc), 422) from None
        try:
            for row in current:
                if target is None or row.service_id != target.service_id:
                    row.is_platform_default = False
            db.flush()
            if target is not None:
                target.is_platform_default = True
            db.commit()
        except IntegrityError:
            db.rollback()
            raise RerankServiceError("The platform default was changed concurrently; retry.", 409) from None
        logger.info("rerank.platform_default service_id=%s actor_id=%s", service_id, actor_id)

    # -------------------------------------------------------- test connection

    @staticmethod
    async def test_connection_with_config(
        db: Session,
        config_data: CreateUpdateRerankServiceSchema,
        *,
        app_scoped: bool,
        app_id: Optional[int] = None,
        service_id: Optional[int] = None,
    ) -> Dict[str, str]:
        """Test an unsaved configuration; nothing is persisted and the key is never echoed.

        In edit mode (``service_id`` given) an empty, CHANGE_ME or masked key is replaced by the stored
        key of the same scope.

        Raises:
            RerankServiceError: 404 when ``service_id`` is not in scope, 422 for an unsafe endpoint.
        """
        api_key = config_data.api_key or ""
        if service_id:
            stored = (
                RerankServiceRepository.get_by_id_and_app_id(db, service_id, app_id)
                if app_scoped
                else RerankServiceRepository.get_by_id(db, service_id)
            )
            if not stored or (not app_scoped and stored.app_id is not None):
                raise RerankServiceError(RERANK_SERVICE_NOT_FOUND_MSG, 404)
            if not _is_real_key(api_key) and stored.api_key:
                _check_key_reuse(stored, config_data.base_url)
                api_key = stored.api_key

        await asyncio.to_thread(
            RerankServiceService.validate_endpoint, config_data.base_url, api_key, app_scoped=app_scoped
        )
        target = RerankTarget(
            service_id=service_id or None,
            app_id=app_id if app_scoped else None,
            name=config_data.name,
            provider=config_data.provider,
            endpoint=config_data.base_url,
            model=config_data.model_name,
            api_key=api_key if _is_real_key(api_key) else None,
            system=not app_scoped,
        )
        return await test_rerank_connection(target)
