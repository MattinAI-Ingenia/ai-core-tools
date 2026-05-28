"""Background sweep that enforces TTL on uploaded files.

Two concerns live here:

- **Ephemeral orphans**: files under ``data/tmp/ephemeral/{session_key}/``
  that the per-request ``finally`` failed to clean up (request crashed,
  worker was killed mid-stream, etc.). These should never be older than
  the duration of a single chat turn; ``TMP_EPHEMERAL_ORPHAN_HOURS``
  (default 1h) is the safety net.

- **Persistent TTL**: files under ``data/tmp/persistent/{session_key}/``
  associated with a memory-enabled conversation. We expire them after
  ``TMP_PERSISTENT_TTL_DAYS`` of inactivity (mtime-based), matching the
  OpenAI Threads convention of "7 days from last activity".

The worker follows the existing ``services.crawl.worker`` pattern:
asyncio task started from ``main.lifespan``, ``asyncio.CancelledError``
for graceful shutdown, no extra dependencies. The actual filesystem
walk is blocking I/O so it runs in a thread (``asyncio.to_thread``) to
keep the event loop free.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)


def _config() -> dict:
    """Read cleanup config lazily so tests can monkey-patch env vars."""
    from utils.config import get_app_config

    return get_app_config()


def _collect_expired_files(root: str, cutoff: float) -> List[str]:
    """Return paths under ``root`` whose mtime is older than ``cutoff``."""
    victims: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            try:
                if os.path.getmtime(full) < cutoff:
                    victims.append(full)
            except OSError:
                # File disappeared mid-walk; harmless.
                continue
    return victims


def _remove_files(paths: List[str], label: str) -> int:
    """Best-effort removal; returns count of files actually removed."""
    removed = 0
    for path in paths:
        try:
            os.remove(path)
            removed += 1
        except OSError as exc:
            logger.warning(
                "file_cleanup_worker[%s]: could not remove %s: %s",
                label, path, exc,
            )
    return removed


def _prune_empty_dirs(root: str) -> None:
    """Remove now-empty subdirectories of ``root`` bottom-up."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == root:
            continue
        if dirnames or filenames:
            continue
        try:
            os.rmdir(dirpath)
        except OSError:
            # Race with a new upload re-creating the directory; skip.
            pass


def _purge_tree_older_than(root: str, ttl_seconds: int, *, label: str) -> int:
    """Walk ``root`` and remove files whose mtime is older than the TTL.

    After per-file removal, empty directories under ``root`` are removed
    too so the tree does not accumulate empty session_key folders.

    Returns the number of files removed.
    """
    if not os.path.isdir(root):
        return 0

    cutoff = time.time() - ttl_seconds
    try:
        victims = _collect_expired_files(root, cutoff)
        removed = _remove_files(victims, label)
        _prune_empty_dirs(root)
        return removed
    except Exception as exc:
        logger.error(
            "file_cleanup_worker[%s]: unexpected error during sweep: %s",
            label, exc, exc_info=True,
        )
        return 0


def _run_one_sweep_sync() -> None:
    """Single pass over both ephemeral and persistent trees (blocking)."""
    cfg = _config()
    if not cfg.get('TMP_CLEANUP_ENABLED', True):
        logger.debug("file_cleanup_worker: cleanup disabled via TMP_CLEANUP_ENABLED=false")
        return

    tmp_base = cfg['TMP_BASE_FOLDER']
    ephemeral_root = os.path.join(tmp_base, "ephemeral")
    persistent_root = os.path.join(tmp_base, "persistent")
    uploads_root = os.path.join(tmp_base, "uploads")

    ephemeral_ttl_seconds = int(cfg['TMP_EPHEMERAL_ORPHAN_HOURS']) * 3600
    persistent_ttl_seconds = int(cfg['TMP_PERSISTENT_TTL_DAYS']) * 86400

    ephemeral_removed = _purge_tree_older_than(
        ephemeral_root, ephemeral_ttl_seconds, label="ephemeral",
    )
    uploads_removed = _purge_tree_older_than(
        uploads_root, ephemeral_ttl_seconds, label="uploads",
    )
    persistent_removed = _purge_tree_older_than(
        persistent_root, persistent_ttl_seconds, label="persistent",
    )

    if ephemeral_removed or uploads_removed or persistent_removed:
        logger.info(
            "file_cleanup_worker: removed %d ephemeral, %d upload, %d persistent file(s)",
            ephemeral_removed, uploads_removed, persistent_removed,
        )


async def _worker_loop() -> None:
    """Periodic loop. Sleeps between sweeps, cancelable from lifespan.

    The actual sweep is blocking I/O, so it runs in a thread via
    ``asyncio.to_thread`` to avoid stalling the event loop on large trees.
    """
    cfg = _config()
    interval_seconds = max(1, int(cfg['TMP_CLEANUP_INTERVAL_MINUTES']) * 60)

    logger.info(
        "file_cleanup_worker: starting (interval=%ds, persistent_ttl=%dd, ephemeral_orphan=%dh)",
        interval_seconds,
        int(cfg['TMP_PERSISTENT_TTL_DAYS']),
        int(cfg['TMP_EPHEMERAL_ORPHAN_HOURS']),
    )

    while True:
        try:
            await asyncio.to_thread(_run_one_sweep_sync)
        except asyncio.CancelledError:
            logger.info("file_cleanup_worker: shutting down")
            raise
        except Exception as exc:
            logger.error(
                "file_cleanup_worker: sweep failed: %s", exc, exc_info=True,
            )
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("file_cleanup_worker: shutting down")
            raise


def start_file_cleanup_worker() -> asyncio.Task:
    """Start the cleanup task. Called from FastAPI lifespan startup."""
    return asyncio.create_task(_worker_loop(), name="file-cleanup-worker")


async def stop_file_cleanup_worker(task: asyncio.Task) -> None:
    """Cancel the cleanup task. Called from FastAPI lifespan shutdown."""
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
