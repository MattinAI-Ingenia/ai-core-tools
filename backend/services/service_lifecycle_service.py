from datetime import datetime
from typing import Any, Optional

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from services.system_settings_service import SystemSettingsService
from utils.logger import get_logger
from utils.service_lifecycle import to_utc

logger = get_logger(__name__)

WARNING_DAYS_SETTING_KEY = "service_end_of_service_warning_days"
DEFAULT_WARNING_DAYS = 30


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
