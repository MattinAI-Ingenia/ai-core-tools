"""Startup tasks executed when the FastAPI app starts."""

import asyncio
from db.database import SessionLocal
from services.pricing_service import PricingService
from utils.logger import get_logger

logger = get_logger(__name__)


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
