"""
In-memory thread-safe cache for distinct metadata field values per silo.

Consumed by the dynamic retrieval tool builder (steps 005/006) to build
field-value enumerations without hitting the vector store on every agent
invocation.

Cache key: (silo_id, field)
TTL: configurable via METADATA_VALUES_CACHE_TTL_SECONDS env var (default 600 s).

Multi-process staleness contract (NFR-3):
    This cache is process-local.  When the application is deployed with multiple
    Uvicorn workers (e.g. UVICORN_WORKERS=2 in docker/.env) or multiple pod
    replicas, each process maintains its own independent cache state.
    ``invalidate()`` only evicts the entry in the calling process; the other
    workers will continue to serve stale values until their TTL window expires
    (up to 600 s by default).  This is an accepted trade-off: the cost of a
    cross-process cache (Redis pub/sub) is not justified for metadata hint
    lists that are non-critical and naturally refresh within the TTL.  Clients
    that need immediate consistency after a write should not rely on this cache.

Cycle-of-import note:
    This module imports SiloService lazily (inside get_distinct_values) to break
    the potential circular dependency:
        silo_service → metadata_values_cache_service → silo_service
    The deferred import is documented below in the method body.
"""

import os
import threading
import time
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 600
_ENV_VAR_NAME = "METADATA_VALUES_CACHE_TTL_SECONDS"

# Fixed sampling limit passed to SiloService.get_metadata_field_values.
# Caps Qdrant's in-memory scroll at 50 values; PGVector DISTINCT is unaffected
# because SiloService.get_metadata_field_values clamps limit to 1–500.
_SAMPLING_LIMIT = 50


def _parse_ttl_from_env() -> int:
    """Read and validate the TTL from the environment variable.

    Returns:
        Parsed integer TTL in seconds. Falls back to _DEFAULT_TTL_SECONDS with
        a WARNING log when the env var is set but not parseable as a positive int.
    """
    raw = os.getenv(_ENV_VAR_NAME)
    if raw is None:
        return _DEFAULT_TTL_SECONDS
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"TTL must be positive, got {value}")
        return value
    except ValueError:
        logger.warning(
            "Invalid value for %s: %r — must be a positive integer. "
            "Using default TTL of %d seconds.",
            _ENV_VAR_NAME,
            raw,
            _DEFAULT_TTL_SECONDS,
        )
        return _DEFAULT_TTL_SECONDS


class _CacheEntry:
    """Single cached item with expiry timestamp."""

    __slots__ = ("values", "expires_at")

    def __init__(self, values: list[str], ttl_seconds: int) -> None:
        self.values: list[str] = values
        self.expires_at: float = time.monotonic() + ttl_seconds


class MetadataValuesCacheService:
    """Singleton in-memory cache for distinct metadata field values per silo.

    Thread-safe via a single threading.Lock. All public methods are class
    methods so callers never need to manage an instance.

    Cache-miss policy:
        On a miss the lock is released before calling SiloService (which may
        block on I/O), then re-acquired to write. Two concurrent misses for the
        same key will both query the vector store; the second write simply
        overwrites the first. This "allow-duplicate-fetch" approach avoids
        holding the lock during I/O and prevents any deadlock scenario.

    Error policy:
        If the underlying vector store raises during sampling, a WARNING is
        logged and an empty list is returned. The result is intentionally NOT
        cached so that a transient error does not poison the cache for the full
        TTL window.
    """

    _cache: dict[tuple[int, str], _CacheEntry] = {}
    _lock: threading.Lock = threading.Lock()
    _ttl_seconds: int = _parse_ttl_from_env()

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    @classmethod
    def get_distinct_values(
        cls,
        silo_id: int,
        field: str,
        db,
    ) -> list[str]:
        """Return distinct values for a metadata field in a silo's vector collection.

        Serves from cache when the entry exists and has not expired; queries
        SiloService.get_metadata_field_values on a miss and stores the result.

        Args:
            silo_id: Primary key of the target silo.
            field: Metadata field name (e.g. ``"doc_type"``, ``"language"``).
            db: SQLAlchemy Session — forwarded to SiloService on a cache miss.

        Returns:
            Sorted list of distinct string values (up to _SAMPLING_LIMIT).
            Returns an empty list if the silo has no values for the field or if
            the vector store raises during sampling.
        """
        cache_key = (silo_id, field)

        # --- Fast path: check cache under lock ---
        with cls._lock:
            entry = cls._cache.get(cache_key)
            if entry is not None and time.monotonic() < entry.expires_at:
                logger.debug(
                    "metadata_values_cache HIT silo=%d field=%r", silo_id, field
                )
                return list(entry.values)

        # --- Slow path: query vector store without holding the lock ---
        logger.info(
            "metadata_values_cache MISS silo=%d field=%r — querying vector store",
            silo_id,
            field,
        )
        query_start = time.monotonic()

        values = cls._fetch_from_vector_store(silo_id, field, db)

        elapsed_ms = (time.monotonic() - query_start) * 1000

        # Only cache a successful (non-error) result. _fetch_from_vector_store
        # returns None on error to distinguish "empty collection" from "error".
        if values is not None:
            logger.info(
                "metadata_values_cache: fetched %d value(s) for silo=%d field=%r in %.1f ms",
                len(values),
                silo_id,
                field,
                elapsed_ms,
            )
            with cls._lock:
                cls._cache[cache_key] = _CacheEntry(values, cls._ttl_seconds)
            return list(values)

        # Error case: return empty list, do not cache.
        return []

    @classmethod
    def invalidate(cls, silo_id: int) -> None:
        """Remove all cached entries for a silo.

        Must be called from every write path in SiloService that changes the
        vector collection (index, reindex, delete). The call is wrapped in a
        try/except by callers so that a cache failure never propagates to the
        write path.

        Args:
            silo_id: Primary key of the silo whose cache entries should be evicted.
        """
        with cls._lock:
            keys_to_delete = [k for k in cls._cache if k[0] == silo_id]
            for key in keys_to_delete:
                del cls._cache[key]

        if keys_to_delete:
            logger.debug(
                "metadata_values_cache: invalidated %d entry/entries for silo=%d",
                len(keys_to_delete),
                silo_id,
            )

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    @classmethod
    def _fetch_from_vector_store(
        cls,
        silo_id: int,
        field: str,
        db,
    ) -> Optional[list[str]]:
        """Delegate to SiloService.get_metadata_field_values.

        Returns the list of values on success, or None on any exception.

        Note on deferred import:
            SiloService is imported inside this method to avoid a circular
            import cycle:
                silo_service.py → (writes) → metadata_values_cache_service.py
                                             → (reads) → silo_service.py
            Python's module import machinery would normally handle this via
            partial modules, but the combination of module-level usage and the
            singleton pattern makes the cycle fragile. The deferred import is
            safe here because this method is only called at runtime (never at
            module load time).
        """
        try:
            # Deferred import — see docstring above.
            from services.silo_service import SiloService  # noqa: PLC0415

            return SiloService.get_metadata_field_values(
                silo_id=silo_id,
                field=field,
                prefix=None,
                limit=_SAMPLING_LIMIT,
                db=db,
            )
        except Exception as exc:
            logger.warning(
                "metadata_values_cache: error fetching values for silo=%d field=%r: %s — "
                "returning empty list (result will not be cached)",
                silo_id,
                field,
                exc,
            )
            return None

    @classmethod
    def _reset_for_testing(cls) -> None:
        """Clear all cache state. For use in unit tests only.

        Resets TTL to the compile-time default constant directly — no env-var
        re-parse — so test isolation is unaffected by whatever the environment
        happens to have set.
        """
        with cls._lock:
            cls._cache.clear()
            cls._ttl_seconds = _DEFAULT_TTL_SECONDS
