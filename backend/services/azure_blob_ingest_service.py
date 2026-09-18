"""One-shot Azure Blob Storage ingestion into an existing Repository.

Validation-tool feature (see .claude/specs/azure-blob-ingestion/spec.md): lists
the blobs of an Azure container and feeds them through the exact same
``ResourceService.create_multiple_resources`` pipeline a manual upload uses —
inheriting extension validation, ``App.max_file_size_mb``, the per-silo
advisory indexing lock, background indexing and SSE progress for free.
``SiloService.index_single_content`` is never called (AD-1).

Idempotency (FR-3/AD-2) lives entirely in ``Resource.extra_metadata`` — no new
table, keyed on ``(account_url, container, blob_name)``. A blob already
ingested with the same ETag is skipped; one whose ETag changed has its new
content indexed first and only then has its previous ``Resource`` deleted
(index-then-swap — mirrors ``SiloService.reindex_resource``'s "index first,
delete second" so a lock race or mid-batch crash can never leave a blob with
zero indexed Resources).

Batching (AD-6): the diff and the *first* batch run synchronously in the
calling thread — bounded by ``BLOB_INGEST_BATCH_SIZE`` — so
``trigger_ingestion`` can return a real ``session_id`` (AC-1). Any remaining
batches continue in a background thread with its own DB session, waiting for
the silo's advisory lock to free between batches. The whole run (diff + every
batch) is serialized per repository by a dedicated advisory lock — see
``_run_lock_id``.
"""
import hashlib
import os
import random
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from repositories.app_repository import AppRepository
from repositories.repository_repository import RepositoryRepository
from services import silo_indexing_lock
from services.blob import azure_blob_client
from services.blob.azure_blob_client import BlobSourceConfig, RemoteBlob
from services.resource_service import ResourceService
from services.staged_file import StagedFileAdapter
from utils.blob_url_guard import assert_allowed_account_url
from utils.config import Config, get_app_config
from utils.error_handlers import ValidationError
from utils.logger import get_logger

logger = get_logger(__name__)


class ConflictError(Exception):
    """Raised when this repository/silo already has an active ingestion run.

    Module-local rather than imported from ``crawl_job_service`` — this domain
    has its own conflict source (a stale in-progress run) and its own message,
    mirroring the per-domain ``ConflictError`` already used by
    ``import_job_service`` and ``crawl_job_service`` instead of implicitly
    re-exporting theirs to the router.
    """


# How long a background batch waits for the silo's advisory lock to free
# before giving up on the remaining blobs of this run. Generous on purpose:
# a single LightRAG batch can legitimately run for the better part of an hour
# (see ResourceService._index_resources_background's docstring) — this is a
# ceiling against a genuinely stuck lock, not a normal-case timeout.
_LOCK_POLL_INTERVAL_SECONDS = 5
_LOCK_POLL_MAX_SECONDS = 3600

# Local filename truncation, per plan.md: keep the extensionless part to this
# many characters and disambiguate with a short hash of the full blob name.
_MAX_BASENAME_LENGTH = 160


def _run_lock_id(repository_id: int) -> int:
    """Advisory-lock id serializing one repository's whole ingestion run.

    Deliberately a *different* id than the real per-silo lock
    ``create_multiple_resources`` already acquires for each batch's indexing
    (``silo_indexing_lock.acquire(silo_id)``, a positive id): holding that
    same lock ourselves for the run's duration would make every batch's own
    acquisition attempt fail with a false 409, since PostgreSQL session-level
    advisory locks are not reentrant across different connections. Instead
    this reserves a negative id per repository — the same sentinel-offset
    trick ``ResourceService._LIGHTRAG_POOL_LOCK_ID`` (``-1``) already uses to
    carve out a second, independent lock in the same namespace — so it can be
    held for the whole run (diff + every batch, including the background
    ones) without ever contending with the per-silo lock.
    """
    return -(1_000_000 + repository_id)


@dataclass(frozen=True)
class PendingBlob:
    """One remote blob queued to be downloaded and (re)ingested this run."""

    kind: str  # 'new' | 'changed'
    blob: RemoteBlob
    resource_id: Optional[int]  # set only for kind == 'changed'


@dataclass(frozen=True)
class BlobDiffResult:
    """Outcome of classifying a remote listing against existing ``Resource`` rows."""

    to_ingest: List[PendingBlob]
    unchanged_count: int
    unsupported_count: int


@dataclass
class BatchOutcome:
    created_count: int
    failed_entries: List[dict] = field(default_factory=list)
    session_id: Optional[str] = None


def _existing_blob_index(resources: Iterable, account_url: str, container: str) -> dict:
    """Map ``blob_name -> (resource_id, etag)`` from this repository's Resources.

    Only Resources whose ``extra_metadata`` marks them as ``source_type ==
    'azure_blob'`` for this same ``(account_url, container)`` are considered
    (FR-3/AD-2). ``account_url`` is part of the key so a second storage
    account that happens to have a same-named container can never be mistaken
    for a prior run of *this* account's blobs — which would otherwise
    supersede-and-delete the first account's Resources (corpus poisoning).
    """
    index: dict = {}
    for resource in resources:
        meta = getattr(resource, 'extra_metadata', None) or {}
        if (
            meta.get('source_type') == 'azure_blob'
            and meta.get('account_url') == account_url
            and meta.get('container') == container
        ):
            blob_name = meta.get('blob_name')
            if blob_name:
                index[blob_name] = (resource.resource_id, meta.get('etag'))
    return index


def classify_blobs(
    resources: Iterable, account_url: str, container: str, remote_blobs: Iterable[RemoteBlob],
    supported_extensions: set,
) -> BlobDiffResult:
    """Classify each remote blob as unchanged / changed / new / unsupported.

    Pure and DB-free: ``resources`` is any iterable of objects exposing
    ``.resource_id`` and ``.extra_metadata`` (a ``Resource`` ORM instance or a
    lightweight test fixture), so this is unit-testable without a database,
    Azure, or threads.

    Rules (FR-3, edge cases):
        - a blob with no matching ``Resource`` is 'new';
        - a blob whose matching ``Resource`` carries the same ETag is
          'unchanged' and never touched again;
        - a blob with no remote ETag, or one that differs from the stored
          one, is always 'changed' — a missing ETag is never treated as a
          match, so the worst case is a redundant re-ingest, not a missed
          update;
        - a blob whose extension is not in ``supported_extensions`` is
          'unsupported' (defense in depth: the caller's ``file_extension_filters``
          is already validated as a subset of the pipeline's supported types,
          so this should normally never fire in production).

    Returns:
        A ``BlobDiffResult`` with the ordered list of blobs to ingest
        (preserving the listing order of ``remote_blobs``) plus the unchanged/
        unsupported counts.
    """
    index = _existing_blob_index(resources, account_url, container)
    to_ingest: List[PendingBlob] = []
    unchanged_count = 0
    unsupported_count = 0
    for blob in remote_blobs:
        extension = os.path.splitext(blob.name)[1].lower()
        if supported_extensions and extension not in supported_extensions:
            unsupported_count += 1
            continue
        existing = index.get(blob.name)
        if existing is None:
            to_ingest.append(PendingBlob(kind='new', blob=blob, resource_id=None))
            continue
        resource_id, existing_etag = existing
        if blob.etag is not None and blob.etag == existing_etag:
            unchanged_count += 1
        else:
            to_ingest.append(PendingBlob(kind='changed', blob=blob, resource_id=resource_id))
    return BlobDiffResult(to_ingest=to_ingest, unchanged_count=unchanged_count, unsupported_count=unsupported_count)


def _sample_pendings(
    pendings: List[PendingBlob], sample_size: Optional[int], rng=random,
) -> List[PendingBlob]:
    """Apply the validation-phase cap: at most ``sample_size`` pending blobs, at random.

    The cap applies to the *pending* set (new + changed) — unchanged and
    unsupported blobs are never candidates — so every run (initial load or
    "Actualizar") ingests up to ``sample_size`` random not-yet-ingested blobs,
    and repeated runs eventually cover the rest. ``sample_size=None`` means no
    cap: everything pending is ingested (the post-validation behavior).

    Pure and injectable via ``rng`` (defaults to the ``random`` module) so
    tests can seed determinism without monkeypatching module state.
    """
    if sample_size is None or len(pendings) <= sample_size:
        return list(pendings)
    return rng.sample(pendings, sample_size)


def _normalize_extension_filters(raw: Optional[List[str]]) -> set:
    """Validate ``file_extension_filters`` as a subset of the pipeline's supported types.

    Defaults to every extension ``ResourceService`` supports when omitted.

    Raises:
        ValidationError: If ``raw`` contains an extension the pipeline cannot process.
    """
    supported = {ext.lower() for ext in ResourceService.SUPPORTED_EXTENSIONS}
    if not raw:
        return supported
    normalized = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in raw}
    unsupported = normalized - supported
    if unsupported:
        raise ValidationError(
            f"Unsupported file_extension_filters: {sorted(unsupported)}. "
            f"Supported: {sorted(supported)}"
        )
    return normalized


def _max_file_size_bytes(db: Session, app_id: int) -> Optional[int]:
    """The App's configured max upload size in bytes, or None when unlimited (0/unset)."""
    app = AppRepository(db).get_by_id(app_id)
    max_mb = (app.max_file_size_mb or 0) if app else 0
    return max_mb * 1024 * 1024 if max_mb > 0 else None


def _sanitize_blob_filename(blob_name: str) -> str:
    """Turn a blob name (which may contain ``/``) into a safe local filename.

    Rejects absolute paths and any ``..`` path segment outright. Collapses
    ``/`` into ``__`` so two blobs with the same basename under different
    prefixes never collide (edge case in spec.md), then truncates an
    overlong name to ``_MAX_BASENAME_LENGTH`` characters plus a short hash of
    the *original* blob name so truncated names stay distinguishable.

    Raises:
        ValueError: If ``blob_name`` is empty, absolute, contains ``..``, or
            contains a NUL byte.
    """
    if (
        not blob_name
        or '\x00' in blob_name
        or blob_name.startswith('/')
        or any(seg == '..' for seg in blob_name.split('/'))
    ):
        raise ValueError(f"Rejected unsafe blob name: {blob_name!r}")

    flattened = blob_name.strip('/').replace('/', '__')
    if not flattened:
        raise ValueError(f"Rejected unsafe blob name: {blob_name!r}")

    base, extension = os.path.splitext(flattened)
    # Unconditional, not just when truncating for length: two different blob
    # names can flatten to the identical string (e.g. "a/b.pdf" and
    # "a__b.pdf" both become "a__b.pdf") without ever exceeding
    # _MAX_BASENAME_LENGTH. A length-gated hash would leave that collision in
    # place, letting two concurrent downloads in the same batch race to write
    # the same staging path (torn content) and, across runs, two different
    # Resources alias the same file. Deterministic per blob_name so the same
    # blob still maps to the same local slot on every run — required for the
    # 'changed'-blob supersede step in _process_batch to land the new
    # content on the exact path the superseded Resource used.
    digest = hashlib.sha1(blob_name.encode('utf-8')).hexdigest()[:8]
    if len(base) > _MAX_BASENAME_LENGTH:
        base = base[:_MAX_BASENAME_LENGTH]
    return f"{base}_{digest}{extension}"


def _resolve_dest_path(staging_dir: str, filename: str) -> str:
    """Join ``filename`` onto ``staging_dir`` and assert the result cannot escape it.

    Defense in depth on top of ``_sanitize_blob_filename``: an Azure blob name
    is attacker-influenced (it comes from whatever storage account the caller
    points this at), so a bare ``os.path.join`` is never trusted on its own —
    the resolved path is required to stay inside ``staging_dir`` before it is
    ever opened for writing.

    Raises:
        ValueError: If the resolved path would land outside ``staging_dir``.
    """
    dest_path = os.path.join(staging_dir, filename)
    real_dest = os.path.realpath(dest_path)
    real_root = os.path.realpath(staging_dir)
    if not (real_dest == real_root or real_dest.startswith(real_root + os.sep)):
        raise ValueError(f"Resolved path escapes staging directory: {dest_path!r}")
    return dest_path


def _download_one(
    cfg: BlobSourceConfig, pending: PendingBlob, staging_dir: str, max_bytes: Optional[int],
) -> Tuple[PendingBlob, Optional[str], Optional[str], Optional[str]]:
    """Download one blob to staging. Returns (pending, dest_path, filename, failure_reason).

    Exactly one of (``dest_path``/``filename``) or ``failure_reason`` is set.
    Never raises — every failure is classified and returned so one bad blob
    cannot sink the rest of the batch.
    """
    blob = pending.blob
    if max_bytes is not None and blob.size_bytes is not None and blob.size_bytes > max_bytes:
        return pending, None, None, 'FILE_TOO_LARGE'

    try:
        filename = _sanitize_blob_filename(blob.name)
        dest_path = _resolve_dest_path(staging_dir, filename)
    except ValueError as exc:
        logger.warning("Rejecting unsafe blob name in container %s: %s", cfg.container, exc)
        return pending, None, None, 'INVALID_BLOB_NAME'

    try:
        azure_blob_client.download_to_path(cfg, blob.name, dest_path, max_bytes=max_bytes)
    except azure_blob_client.BlobConnectionError as exc:
        return pending, None, None, exc.code

    return pending, dest_path, filename, None


def _download_batch(
    cfg: BlobSourceConfig, pendings: List[PendingBlob], staging_dir: str, max_bytes: Optional[int], concurrency: int,
) -> List[Tuple[PendingBlob, Optional[str], Optional[str], Optional[str]]]:
    """Download every blob of one batch, bounded by ``concurrency`` concurrent downloads."""
    results: List[Optional[Tuple]] = [None] * len(pendings)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_to_index = {
            pool.submit(_download_one, cfg, pending, staging_dir, max_bytes): i
            for i, pending in enumerate(pendings)
        }
        for future, index in future_to_index.items():
            try:
                results[index] = future.result()
            except Exception:  # noqa: BLE001 - one blob's bug must not sink the batch
                logger.exception("Unexpected error downloading blob %r", pendings[index].blob.name)
                results[index] = (pendings[index], None, None, 'DOWNLOAD_FAILED')
    return results


def _process_batch(
    pendings: List[PendingBlob], db: Session, repository_id: int, cfg: BlobSourceConfig,
    max_bytes: Optional[int], concurrency: int,
) -> BatchOutcome:
    """Download, ingest, then supersede-and-delete one batch of blobs.

    Index-then-swap (AC-3): the new content is created (and its background
    indexing kicked off) *before* a 'changed' blob's previous ``Resource`` is
    deleted, mirroring ``SiloService.reindex_resource``. If the silo's
    advisory lock is lost between the caller's read-only pre-check and
    ``create_multiple_resources``'s own acquire, or the process dies
    mid-batch, the old content is still there to be re-diffed next run —
    never both gone at once (reliability-auditor H-3 / code-reviewer HIGH).

    Always removes the batch's staging directory before returning (success or
    failure) — NFR-2's "staging cleared after each blob" plus a backstop for
    ``file_cleanup_worker`` if the process dies mid-batch.
    """
    tmp_base = get_app_config()['TMP_BASE_FOLDER']
    staging_dir = os.path.join(tmp_base, 'blob_ingest_staging', str(uuid.uuid4()))
    os.makedirs(staging_dir, exist_ok=True)

    failed_entries: List[dict] = []
    try:
        download_results = _download_batch(cfg, pendings, staging_dir, max_bytes, concurrency)

        files: List[StagedFileAdapter] = []
        extra_metadata: dict = {}
        pending_by_filename: dict = {}
        for pending, dest_path, filename, failure_reason in download_results:
            if failure_reason:
                failed_entries.append({'blob_name': pending.blob.name, 'reason': failure_reason})
                continue

            index = len(files)
            files.append(StagedFileAdapter(dest_path, filename))
            extra_metadata[index] = {
                'source_type': 'azure_blob',
                'account_url': cfg.account_url,
                'container': cfg.container,
                'blob_name': pending.blob.name,
                'blob_url': pending.blob.url,
                'etag': pending.blob.etag,
            }
            pending_by_filename[filename] = pending

        session_id = None
        created_count = 0
        if files:
            created, pipeline_failed, session_id = ResourceService.create_multiple_resources(
                files=files, repository_id=repository_id, db=db, extra_metadata=extra_metadata,
            )
            created_count = len(created)
            failed_entries.extend(pipeline_failed)

            # Only now that a 'changed' blob's new Resource is durably created
            # (and indexing on it started) do we delete the stale Resource it
            # supersedes. ``created`` carries only the entries that actually
            # succeeded, matched back to their PendingBlob by the (unique per
            # batch) staged filename — a blob whose new content failed inside
            # create_multiple_resources is left with its old Resource intact.
            for resource in created:
                pending = pending_by_filename.get(resource.uri)
                if pending is None or pending.kind != 'changed' or pending.resource_id is None:
                    continue
                if pending.resource_id == resource.resource_id:
                    # ResourceService._process_single_file reused the same
                    # (uri, repository_id, folder_id) row instead of inserting
                    # a new one — it found it in status='error' and reset it
                    # to 'pending' in place. `created` and `pending` here are
                    # the same Resource, already updated, not a row to
                    # supersede: deleting it would delete the file/vectors
                    # that are being indexed right now.
                    continue
                # `_sanitize_blob_filename` is deterministic per blob_name, so
                # a 'changed' blob's new Resource lands on the exact same
                # `uri`/local path the superseded Resource used —
                # StagedFileAdapter.save() already overwrote that file with
                # the new content. Deleting the *file* here would delete the
                # new Resource's own backing file; only the stale DB row and
                # its vectors (keyed by resource_id, unaffected by the path
                # reuse) must go (reliability-auditor / security-auditor HIGH).
                if not ResourceService.delete_resource(pending.resource_id, db, delete_file=False):
                    # Not fatal — the new content is already indexed — but
                    # must not be silently swallowed (code-reviewer HIGH).
                    failed_entries.append({
                        'blob_name': pending.blob.name,
                        'reason': 'SUPERSEDE_CLEANUP_FAILED',
                    })

        if failed_entries:
            logger.warning("Azure blob ingestion: %d failed entr(ies): %r", len(failed_entries), failed_entries)

        return BatchOutcome(created_count=created_count, failed_entries=failed_entries, session_id=session_id)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


def _wait_for_lock_free(db: Session, silo_id: int) -> bool:
    """Poll ``silo_id``'s advisory lock until it frees, or the bound elapses.

    Read-only and never forced (AD-6): a previous batch's
    ``create_multiple_resources`` call holds this lock for as long as its
    background indexing thread runs.

    Returns:
        True once the lock is free; False if ``_LOCK_POLL_MAX_SECONDS`` elapsed first.
    """
    if not silo_id:
        return True
    deadline = time.monotonic() + _LOCK_POLL_MAX_SECONDS
    while True:
        locked = silo_indexing_lock.is_locked(db, silo_id)
        # is_locked's SELECT autobegins a transaction on this Session; end it
        # every iteration so a poll that can legitimately run for up to an
        # hour never pins a pool connection idle-in-transaction (mirrors
        # stream_ingestion_progress._read in routers/internal/repositories.py).
        db.rollback()
        if not locked:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_LOCK_POLL_INTERVAL_SECONDS)


def _log_run_summary(
    app_id: int, repository_id: int, silo_id: int, counters: dict, session_id: Optional[str],
    elapsed_seconds: float, listed: int,
) -> None:
    """One structured log line per run (NFR-5). Never interpolates a secret."""
    logger.info(
        "Azure blob ingestion run finished: app_id=%s repository_id=%s silo_id=%s listed=%s "
        "pending=%s sampled=%s queued=%s skipped_unchanged=%s skipped_unsupported=%s failed=%s cancelled=%s "
        "duration_s=%.1f session_id=%s",
        app_id, repository_id, silo_id, listed,
        counters.get('pending', 0), counters.get('sampled', 0),
        counters['queued'], counters['skipped_unchanged'], counters['skipped_unsupported'], counters['failed'],
        counters.get('cancelled', 0), elapsed_seconds, session_id,
    )


def _run_remaining_batches(
    batches: List[List[PendingBlob]], app_id: int, repository_id: int, silo_id: int, cfg: BlobSourceConfig,
    max_bytes: Optional[int], concurrency: int, counters: dict, start_time: float, listed: int,
    run_lock_conn, run_lock_id: int,
) -> None:
    """Background continuation of a run: batches 2..N, each on its own DB session.

    Mirrors ``ResourceService._index_resources_background``'s "own SessionLocal,
    never the caller's" rule. Waits for the silo's advisory lock to free
    between batches instead of forcing it (AD-6); if the wait times out, every
    still-unprocessed blob is counted as failed and the run stops early rather
    than hanging the thread forever. Also checks for an operator-requested stop
    at the top of every batch, so a runaway multi-batch run can be halted.
    Releases the per-repository run lock (``run_lock_conn``/``run_lock_id``,
    acquired by ``trigger_ingestion`` before the diff) once this is the last
    thing touching the run, win or lose.
    """
    from db.database import SessionLocal

    last_session_id = None
    try:
        db = SessionLocal()
        try:
            for i, batch in enumerate(batches):
                if ResourceService.ingestion_stop_mode(repository_id):
                    remaining = sum(len(b) for b in batches[i:])
                    logger.info(
                        "Azure blob ingestion: stop requested; cancelling %s remaining blob(s) for repository %s",
                        remaining, repository_id,
                    )
                    counters['cancelled'] = counters.get('cancelled', 0) + remaining
                    break
                if not _wait_for_lock_free(db, silo_id):
                    remaining = sum(len(b) for b in batches[i:])
                    logger.warning(
                        "Azure blob ingestion: timed out waiting for silo %s to free; "
                        "aborting %s remaining blob(s) for repository %s",
                        silo_id, remaining, repository_id,
                    )
                    counters['failed'] += remaining
                    break
                try:
                    outcome = _process_batch(batch, db, repository_id, cfg, max_bytes, concurrency)
                except Exception:  # noqa: BLE001 - one batch's crash must not orphan the rest of the log
                    # A failed batch can leave the session's transaction dirty; if
                    # that carries into the next iteration's queries (including the
                    # next _wait_for_lock_free poll), it poisons every batch after
                    # it too (reliability-auditor H-2).
                    db.rollback()
                    logger.exception(
                        "Azure blob ingestion: batch %s/%s failed for repository %s",
                        i + 1, len(batches), repository_id,
                    )
                    counters['failed'] += len(batch)
                    continue
                counters['queued'] += outcome.created_count
                counters['failed'] += len(outcome.failed_entries)
                if outcome.session_id:
                    last_session_id = outcome.session_id
        finally:
            # Guaranteed even if the loop above raised something not caught
            # by its own per-batch try/except.
            db.close()
    except Exception:  # noqa: BLE001 - SessionLocal()/db.close()/the loop itself must never leak the run lock
        logger.exception("Azure blob ingestion: background run crashed for repository %s", repository_id)
    finally:
        # Guaranteed last regardless of what failed above — a leaked run lock
        # (and its pooled connection) blocks this repository from ever being
        # ingested again until a process restart (reliability-auditor HIGH).
        silo_indexing_lock.release(run_lock_conn, run_lock_id)
        elapsed = time.monotonic() - start_time
        _log_run_summary(app_id, repository_id, silo_id, counters, last_session_id, elapsed, listed)


class AzureBlobIngestService:
    """Entry point for the ``POST .../ingest-azure-blobs`` endpoint."""

    @staticmethod
    def trigger_ingestion(app_id: int, repository_id: int, data: dict, db: Session) -> dict:
        """Validate, diff, and start a one-shot Azure Blob ingestion run.

        Runs the idempotency diff and the first ingestion batch synchronously
        (bounded by ``BLOB_INGEST_BATCH_SIZE``) so the caller gets a real
        ``session_id`` for the existing SSE progress endpoint (AC-1). Any
        remaining batches continue in a background thread.

        The whole run — diff plus every batch, including the background ones
        — is serialized per repository by a dedicated advisory lock (distinct
        from the per-silo one ``create_multiple_resources`` already manages;
        see ``_run_lock_id``), so two concurrent triggers on the same
        repository can never both pass the diff and duplicate Resources.

        Args:
            app_id: The owning App, for tenant-isolation and ``max_file_size_mb``.
            repository_id: Destination repository; must belong to ``app_id`` and have a silo.
            data: The validated request body (``IngestAzureBlobsRequestSchema.model_dump()``).
            db: The caller's DB session — used for validation, the diff, and the first batch only.

        Returns:
            A dict matching ``IngestAzureBlobsResponseSchema``.

        Raises:
            HTTPException: 404 if the repository does not belong to ``app_id``.
            ValidationError: On an invalid body (no silo, bad auth_mode/sas_token
                combination, or an unsupported extension filter) — 422 at the router.
            BlobUrlRejected: If ``account_url`` fails the anti-SSRF check (AC-4) — 422 at the router.
            ConflictError: If the destination silo is already indexing (AC-6), or another Azure
                blob ingestion run is already active for this repository — 409 at the router.
        """
        repo = RepositoryRepository.get_by_id(db, repository_id)
        if not repo or repo.app_id != app_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")
        if not repo.silo_id:
            raise ValidationError("Repository has no silo configured")

        auth_mode = data.get('auth_mode', 'ANONYMOUS')
        sas_token = data.get('sas_token')
        if auth_mode == 'SAS_TOKEN' and not sas_token:
            raise ValidationError("auth_mode=SAS_TOKEN requires a non-empty sas_token")
        if auth_mode == 'ANONYMOUS' and sas_token:
            raise ValidationError("auth_mode=ANONYMOUS must not include a sas_token")

        prefix = data.get('prefix')
        blob_name = (data.get('blob_name') or '').strip() or None
        if blob_name and prefix:
            raise ValidationError("prefix and blob_name are mutually exclusive")

        extensions = _normalize_extension_filters(data.get('file_extension_filters'))

        # AC-4: reject before any persistence or outbound call. azure_blob_client
        # re-validates again immediately before every SDK call (defense in depth, AD-4).
        account_url = assert_allowed_account_url(data['account_url'])
        cfg = BlobSourceConfig(
            account_url=account_url,
            container=data['container'],
            # blob_name doubles as the listing prefix — the cheapest way to
            # fetch just that blob (and near-miss names, filtered to an exact
            # match below).
            prefix=blob_name if blob_name else prefix,
            auth_mode=auth_mode,
            sas_token=sas_token,
        )

        silo_id = repo.silo_id
        if silo_indexing_lock.is_locked(db, silo_id):
            raise ConflictError(
                f"Silo {silo_id} already has an active ingestion in progress. "
                "Please wait for it to complete before triggering an Azure blob ingestion."
            )

        # H-4: claim the whole run for this repository before the diff even
        # runs, so a second concurrent trigger cannot read the same
        # pre-ingestion state and duplicate Resources. Held through every
        # batch, including the background ones — released in this function's
        # `finally` for a single-batch (or early-return) run, or handed off
        # to _run_remaining_batches to release when a multi-batch run ends.
        run_lock_id = _run_lock_id(repository_id)
        run_lock_conn = silo_indexing_lock.acquire(run_lock_id)
        if run_lock_conn is None:
            raise ConflictError(
                f"Repository {repository_id} already has an active Azure blob ingestion run in progress. "
                "Please wait for it to complete before triggering another one."
            )

        handed_off = False
        try:
            start_time = time.monotonic()

            page_size = Config.get_int_env_var('BLOB_LIST_PAGE_SIZE', int(Config.DEFAULTS['BLOB_LIST_PAGE_SIZE']))
            max_items = Config.get_int_env_var('BLOB_LIST_MAX_ITEMS', int(Config.DEFAULTS['BLOB_LIST_MAX_ITEMS']))
            remote_blobs = list(
                azure_blob_client.list_blobs(cfg, extensions=extensions, page_size=page_size, max_items=max_items)
            )

            if blob_name:
                # The prefix listing may include near-miss names
                # ("CDOC004211.pdfx" for "CDOC004211.pdf"); a single-file run
                # is exact-match only, and "not found" is a caller error.
                remote_blobs = [blob for blob in remote_blobs if blob.name == blob_name]
                if not remote_blobs:
                    raise ValidationError(
                        f"Blob {blob_name!r} not found in container {cfg.container!r} "
                        "(or its extension is not supported)"
                    )

            # The listing succeeded, so this source is proven reachable —
            # remember it (minus secrets) for the repository's "Actualizar"
            # button. auth_mode is kept so an SAS_TOKEN source re-asks for the
            # token in the UI instead of silently downgrading to ANONYMOUS.
            # The original prefix is persisted, not a single-file run's
            # blob_name, so Update keeps scanning the whole (prefix-scoped)
            # container. Commit before any further reads: the commit expires
            # this session's loaded instances, and the diff below wants a
            # fresh, cheaply-loaded view of the repository's Resources anyway.
            repo.azure_blob_source = {
                'account_url': cfg.account_url,
                'container': cfg.container,
                'prefix': prefix,
                'auth_mode': auth_mode,
            }
            db.commit()
            existing_resources = ResourceService.get_resources_by_repo_id(repository_id, db)

            diff = classify_blobs(existing_resources, cfg.account_url, cfg.container, remote_blobs, extensions)
            sampled_pendings = _sample_pendings(diff.to_ingest, data.get('sample_size'))
            counters = {
                'queued': 0,
                'skipped_unchanged': diff.unchanged_count,
                'skipped_unsupported': diff.unsupported_count,
                'failed': 0,
                'cancelled': 0,
                'pending': len(diff.to_ingest),
                'sampled': len(sampled_pendings),
            }

            if not sampled_pendings:
                # AC-2: nothing changed — no download, no create_multiple_resources
                # call, no silo lock ever acquired.
                elapsed = time.monotonic() - start_time
                _log_run_summary(app_id, repository_id, silo_id, counters, None, elapsed, len(remote_blobs))
                # 'cancelled'/'pending'/'sampled' are internal to the run
                # summary log line only — IngestAzureBlobsResponseSchema
                # forbids unknown fields.
                return {
                    'queued': counters['queued'],
                    'skipped_unchanged': counters['skipped_unchanged'],
                    'skipped_unsupported': counters['skipped_unsupported'],
                    'failed': counters['failed'],
                    'session_id': None,
                    'total_blobs': len(remote_blobs),
                    'pending_blobs': counters['pending'],
                }

            max_bytes = _max_file_size_bytes(db, app_id)
            batch_size = Config.get_int_env_var(
                'BLOB_INGEST_BATCH_SIZE', int(Config.DEFAULTS['BLOB_INGEST_BATCH_SIZE'])
            )
            concurrency = Config.get_int_env_var(
                'BLOB_INGEST_DOWNLOAD_CONCURRENCY', int(Config.DEFAULTS['BLOB_INGEST_DOWNLOAD_CONCURRENCY'])
            )
            batches = [sampled_pendings[i:i + batch_size] for i in range(0, len(sampled_pendings), batch_size)]

            first_outcome = _process_batch(batches[0], db, repository_id, cfg, max_bytes, concurrency)
            counters['queued'] += first_outcome.created_count
            counters['failed'] += len(first_outcome.failed_entries)
            session_id = first_outcome.session_id
            # 'cancelled'/'pending'/'sampled' are internal to the run summary
            # log line only — IngestAzureBlobsResponseSchema forbids unknown
            # fields.
            response = {
                'queued': counters['queued'],
                'skipped_unchanged': counters['skipped_unchanged'],
                'skipped_unsupported': counters['skipped_unsupported'],
                'failed': counters['failed'],
                'session_id': session_id,
                'total_blobs': len(remote_blobs),
                'pending_blobs': counters['pending'],
            }

            if len(batches) > 1:
                # handed_off is set only *after* .start() returns successfully:
                # if starting the thread itself raises (e.g. the process is out
                # of OS threads), the run must fall through to this function's
                # own `finally` and release the run lock here — otherwise it
                # leaks for the life of the process (reliability-auditor HIGH).
                thread = threading.Thread(
                    target=_run_remaining_batches,
                    args=(
                        batches[1:], app_id, repository_id, silo_id, cfg, max_bytes, concurrency,
                        counters, start_time, len(remote_blobs), run_lock_conn, run_lock_id,
                    ),
                    daemon=True,
                    name=f"blob-ingest-{repository_id}",
                )
                thread.start()
                handed_off = True
            else:
                elapsed = time.monotonic() - start_time
                _log_run_summary(app_id, repository_id, silo_id, counters, session_id, elapsed, len(remote_blobs))

            return response
        finally:
            if not handed_off:
                silo_indexing_lock.release(run_lock_conn, run_lock_id)
