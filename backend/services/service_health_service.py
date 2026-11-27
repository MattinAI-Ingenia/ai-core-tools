from datetime import datetime, timedelta
from typing import Any, Optional, Type, Union

from sqlalchemy import or_

from db.database import SessionLocal
from models.ai_service import AIService
from models.embedding_service import EmbeddingService
from models.rerank_service import RerankService
from utils.logger import get_logger
from utils.provider_errors import ProviderError

logger = get_logger(__name__)

ServiceModel = Type[Union[AIService, EmbeddingService, RerankService]]

FAILURE_MESSAGE_THROTTLE_SECONDS = 60


class ServiceHealthService:
    """Records provider failures on the service that caused them, so a revoked key or an
    exhausted quota is visible on the service (and everything using it) instead of only in logs.

    Writes go through their own short session: callers are usually mid-way through a failing
    operation whose transaction may be rolled back, and the failure must survive that.
    Bulk updates are used on purpose, so ``track_service_health``'s reconfigure hook (which
    clears the error when settings change) does not fire for these writes.
    """

    @staticmethod
    def record_failure(model: ServiceModel, service: Optional[Any], error: ProviderError) -> None:
        """Store ``error`` on ``service`` (a loaded AIService / EmbeddingService) when its fix is
        on the service (key, quota)."""
        service_id = getattr(service, "service_id", None)
        if not service_id or not error.concerns_service:
            return
        session = SessionLocal()
        try:
            session.query(model).filter(model.service_id == service_id).update(
                {model.last_error: error.message, model.last_error_at: datetime.now()},
                synchronize_session=False,
            )
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Could not record failure on %s %s: %s", model.__name__, service_id, exc)
        finally:
            session.close()

    @staticmethod
    def record_failure_message(
        model: ServiceModel,
        service_id: Optional[int],
        message: str,
        throttle_seconds: int = FAILURE_MESSAGE_THROTTLE_SECONDS,
    ) -> None:
        """Store a short, key-free failure message on a service whatever the failure kind (rerank: 5xx, timeouts,
        invalid bodies as well as auth errors).

        One bulk UPDATE in its own session. The same message already written less than ``throttle_seconds`` ago is
        skipped in SQL, so an outage costs about one write per minute instead of one per request.
        """
        if not service_id or not message:
            return
        now = datetime.now()
        cutoff = now - timedelta(seconds=throttle_seconds)
        session = SessionLocal()
        try:
            session.query(model).filter(
                model.service_id == service_id,
                or_(
                    model.last_error.is_distinct_from(message),
                    model.last_error_at.is_(None),
                    model.last_error_at < cutoff,
                ),
            ).update({model.last_error: message, model.last_error_at: now}, synchronize_session=False)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Could not record failure on %s %s: %s", model.__name__, service_id, exc)
        finally:
            session.close()

    @staticmethod
    def record_success(model: ServiceModel, service: Optional[Any]) -> None:
        """Clear a recorded failure after ``service`` worked again.

        Checks the caller's loaded instance first, so the usual healthy case costs no query.
        """
        if service is None or not getattr(service, "last_error", None):
            return
        service_id = service.service_id
        session = SessionLocal()
        try:
            session.query(model).filter(
                model.service_id == service_id, model.last_error.isnot(None)
            ).update({model.last_error: None, model.last_error_at: None}, synchronize_session=False)
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("Could not clear failure on %s %s: %s", model.__name__, service_id, exc)
        finally:
            session.close()
