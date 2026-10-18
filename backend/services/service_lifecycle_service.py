from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from services.system_settings_service import SystemSettingsService
from utils.logger import get_logger

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
