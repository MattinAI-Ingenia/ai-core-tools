"""Track progress of document ingestion into silos.

Thread-safe implementation: indexing runs in background threads, the SSE
endpoint reads state from the async context.  The shared _trackers dict is
protected by a threading.Lock so both worlds can read/write safely.
"""

from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Optional, Dict
from utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_trackers: Dict[str, "IngestionProgress"] = {}


@dataclass
class IngestionProgress:
    """Current ingestion progress for one indexing session."""
    session_id: str
    silo_id: int
    total_chunks: int
    processed_chunks: int = 0
    failed_chunks: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    current_chunk_name: str = ""
    estimated_total_time_seconds: Optional[float] = None

    @property
    def progress_percent(self) -> float:
        if self.total_chunks == 0:
            return 100.0
        return min(100.0, (self.processed_chunks / self.total_chunks) * 100)

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.utcnow() - self.start_time).total_seconds()

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        if self.processed_chunks == 0:
            return None
        velocity = self.processed_chunks / max(self.elapsed_seconds, 0.001)
        remaining = self.total_chunks - self.processed_chunks
        return remaining / velocity if velocity > 0 else None

    def to_dict(self) -> dict:
        remaining = self.estimated_remaining_seconds
        return {
            'session_id': self.session_id,
            'silo_id': self.silo_id,
            'total_chunks': self.total_chunks,
            'processed_chunks': self.processed_chunks,
            'failed_chunks': self.failed_chunks,
            'progress_percent': self.progress_percent,
            'current_chunk_name': self.current_chunk_name,
            'elapsed_seconds': round(self.elapsed_seconds, 1),
            'estimated_remaining_seconds': round(remaining, 1) if remaining else None,
            'estimated_total_time_seconds': self.estimated_total_time_seconds,
        }


class IngestionProgressManager:
    """Thread-safe manager for multiple concurrent ingestion sessions.

    All public methods are synchronous — call them from background threads or
    from async code (they release the GIL while waiting for the lock).
    """

    @staticmethod
    def create_session(
        session_id: str,
        silo_id: int,
        total_chunks: int,
        estimated_total_time: Optional[float] = None,
    ) -> IngestionProgress:
        """Register a new ingestion session."""
        with _lock:
            progress = IngestionProgress(
                session_id=session_id,
                silo_id=silo_id,
                total_chunks=total_chunks,
                estimated_total_time_seconds=estimated_total_time,
            )
            _trackers[session_id] = progress
            logger.info(
                "Ingestion session created: %s (silo %s, %s chunks)",
                session_id, silo_id, total_chunks,
            )
            return progress

    @staticmethod
    def update_progress(
        session_id: str,
        processed: int,
        failed: int = 0,
        chunk_name: str = "",
    ) -> Optional[IngestionProgress]:
        """Update progress counters for an active session."""
        with _lock:
            progress = _trackers.get(session_id)
            if progress is None:
                return None
            progress.processed_chunks = processed
            progress.failed_chunks = failed
            progress.current_chunk_name = chunk_name
            return progress

    @staticmethod
    def complete_session(session_id: str) -> Optional[IngestionProgress]:
        """Mark a session as complete (keeps it in memory for final SSE read)."""
        with _lock:
            progress = _trackers.get(session_id)
            if progress is None:
                return None
            progress.processed_chunks = progress.total_chunks
            logger.info(
                "Ingestion complete: %s (%s chunks, %s failed, %.1fs)",
                session_id,
                progress.total_chunks,
                progress.failed_chunks,
                progress.elapsed_seconds,
            )
            return progress

    @staticmethod
    def get_progress(session_id: str) -> Optional[IngestionProgress]:
        """Return current progress snapshot (None if session unknown)."""
        with _lock:
            return _trackers.get(session_id)

    @staticmethod
    def has_active_session_for_silo(silo_id: int) -> bool:
        """Return True if the silo has an ingestion that is still in progress."""
        with _lock:
            return any(
                p.silo_id == silo_id and p.processed_chunks < p.total_chunks
                for p in _trackers.values()
            )

    @staticmethod
    def list_sessions_by_silo(silo_id: int):
        """Return a list of progress snapshots for active sessions on a silo."""
        with _lock:
            return [p.to_dict() for p in _trackers.values() if p.silo_id == silo_id]

    @staticmethod
    def cleanup_session(session_id: str) -> None:
        """Remove session from memory after SSE stream closes."""
        with _lock:
            _trackers.pop(session_id, None)
