"""Startup tasks executed when the FastAPI app starts."""

import asyncio
from db.database import SessionLocal
from services.pricing_service import PricingService
from utils.logger import get_logger

logger = get_logger(__name__)


async def reset_stuck_resources() -> None:
    """Reset resources left in 'indexing' or 'pending' state by a previous crash.

    IngestionProgressManager is in-memory and starts empty on every boot.
    Any resource still marked 'indexing' or 'pending' after a restart will
    never be updated by a background thread, so we mark them 'error' so
    the user can see they need to re-upload / re-index.
    """
    try:
        from models.resource import Resource
        db = SessionLocal()
        stuck = (
            db.query(Resource)
            .filter(Resource.status.in_(["indexing", "pending"]))
            .all()
        )
        if stuck:
            for r in stuck:
                r.status = "error"
            db.commit()
            logger.warning(
                "Reset %d stuck resource(s) from indexing/pending → error on startup",
                len(stuck),
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
