"""Startup tasks executed when the FastAPI app starts."""

import asyncio
from db.database import SessionLocal
from services.pricing_service import PricingService
from utils.logger import get_logger

logger = get_logger(__name__)


async def reset_stuck_resources() -> None:
    """Reset resources left in 'indexing' or 'pending' state by a previous crash.

    The background thread that owns those rows died with the previous process,
    so nothing will ever advance them again. Marking them 'error' is what tells
    the user they need to re-upload / re-index.

    With more than one uvicorn worker, "previous process" is not "every
    process" — a resource can be 'indexing' because a sibling worker's thread
    is still alive and working on it right now. Silo indexing already has a
    cross-process liveness signal for exactly this (the advisory lock in
    ``silo_indexing_lock``, also used by ``get_ingestion_liveness``), so a
    silo whose lock is still held is skipped: those rows are not stuck, they
    belong to a run this worker just doesn't own.
    """
    try:
        from models.resource import Resource
        from models.repository import Repository
        from services import silo_indexing_lock

        db = SessionLocal()
        stuck = (
            db.query(Resource)
            .join(Repository, Resource.repository_id == Repository.repository_id)
            .filter(Resource.status.in_(["indexing", "pending"]))
            .all()
        )
        reset_count = 0
        skipped_count = 0
        locked_silos: dict[int, bool] = {}
        for r in stuck:
            silo_id = r.repository.silo_id if r.repository else None
            if silo_id is not None:
                if silo_id not in locked_silos:
                    locked_silos[silo_id] = silo_indexing_lock.is_locked(db.connection(), silo_id)
                if locked_silos[silo_id]:
                    skipped_count += 1
                    continue
            r.status = "error"
            reset_count += 1
        if reset_count or skipped_count:
            db.commit()
        if reset_count:
            logger.warning(
                "Reset %d stuck resource(s) from indexing/pending → error on startup"
                " (%d skipped: their silo's indexing lock is still held by a live run)",
                reset_count, skipped_count,
            )
        else:
            logger.info("No stuck resources found on startup.")
    except Exception as e:
        logger.warning("Failed to reset stuck resources on startup: %s", e)
    finally:
        db.close()


async def initialize_pricing_catalog() -> None:
    """Initialize pricing catalog on startup.

    Fetches pricing from all providers and caches in the database.
    Runs asynchronously in the background to avoid blocking app startup.
    """
    try:
        db = SessionLocal()
        logger.info("Initializing pricing catalog...")
        result = PricingService.update_pricing_catalog(db)
        logger.info(f"Pricing catalog initialized: {result['details']}")
    except Exception as e:
        logger.warning(f"Failed to initialize pricing catalog on startup: {e}. Fallback to hardcoded prices.")
    finally:
        db.close()
