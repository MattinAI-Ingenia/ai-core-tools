"""
Cross-session user memory — App-level service resolution (FR-7, D-16, D-21, D-24).

Decides which models do the memory work for an App and builds their clients.
Reads only the `App`, never the `Agent`: no agent column participates in
service selection.

Three things here are non-obvious:

- **No system-level fallback.** An unset service id means memory is inactive
  for that App, full stop — this module never reads `SystemSettingsService`.
  A fallback would route conversation excerpts and fact text through the
  *operator's* provider credentials by default, breaking NFR-5. An App may
  still explicitly choose a system-level (`app_id IS NULL`) service; only the
  implicit substitute is gone.
- **Dimensionality is probed, not read.** `EmbeddingService` has no width
  column (confirmed empirically — `decisions.md` D-24), so the width is
  measured once per service with `embed_query` and cached. A mismatch against
  the deployment-wide width returns the embeddings side as `None`; a probe
  that *fails* attempts the operation anyway rather than disabling memory on
  incomplete metadata, since the store's own dimension error is the backstop.
- **Nothing here raises.** Every failure degrades that side to `None` plus a
  warning; a memory failure must never break a chat turn (NFR-4).

No provider API key is ever read from the environment in this module (AC-24).
"""
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from sqlalchemy.orm import Session

from repositories.ai_service_repository import AIServiceRepository
from repositories.embedding_service_repository import EmbeddingServiceRepository
from tools.aiServiceTools import create_llm_from_service
from tools.embeddingTools import get_embeddings_model
from utils.logger import get_logger

from . import settings
from .settings import FAIL_ERROR, SKIP_DIMS_MISMATCH, SKIP_NO_AI_SERVICE, SKIP_NO_EMBEDDING_SERVICE, log_memory_event

logger = get_logger(__name__)

# Probed embedding width per `embedding_service_id`, populated the first time
# `build_memory_clients` sees that id and never re-probed afterward (D-24).
# Only successful probes are cached: a probe failure is not memoized, so a
# transient provider outage does not permanently disable the dimensionality
# guard for that service for the lifetime of the process.
_DIMS_CACHE: Dict[int, int] = {}

# Applied only to the `/internal` App-configuration boundary's probe
# (`probe_embedding_dims_for_config`), never to the runtime probe inside
# `build_memory_clients` — a slow/unreachable Custom/Ollama endpoint must not
# be able to hold an admin's `PUT /internal/apps/{id}` request (and the
# threadpool worker running it) open indefinitely (security-audit HIGH,
# step_009 follow-up).
_CONFIG_PROBE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class ResolvedMemoryServices:
    """A snapshot of which services (if any) do memory work for one App (AD-5).

    Attributes:
        extraction_service_id: The `AIService.service_id` to use for fact
            extraction, or `None` if unresolved.
        extraction_source: `"app_config"` if the App's own configuration was
            accepted, `"unresolved"` otherwise. There is no `"system_default"`
            tag any more (D-21) — a system-level row the App explicitly
            pointed at is still `"app_config"`.
        extraction_is_system: Whether the accepted extraction service is a
            system-level row (`app_id IS NULL`) — quota-relevant (D-10).
        embedding_service_id: The `EmbeddingService.service_id` to use for
            memory vectors, or `None` if unresolved.
        embedding_source: Same convention as `extraction_source`.
        embedding_is_system: Same convention as `extraction_is_system`.
    """

    extraction_service_id: Optional[int]
    extraction_source: str
    extraction_is_system: bool
    embedding_service_id: Optional[int]
    embedding_source: str
    embedding_is_system: bool


def _resolve_one(
    db: Session,
    app_id: Optional[int],
    configured_id: Optional[int],
    loader: Callable[[Session, int], Any],
    skip_reason: str,
) -> Tuple[Optional[int], str, bool]:
    """Resolve a single App-configured service id to `(id, source, is_system)`.

    Shared logic for both the extraction and the embedding side (FR-7,
    D-16, D-21): unset configuration, a nonexistent id, a load failure and a
    cross-App reference are all treated identically — as `"unresolved"` plus
    one WARNING — because a caller must never be able to distinguish "App
    configured nothing" from "App configured something invalid" well enough
    to be tempted to special-case one of them into silently using it.

    Args:
        db: Active DB session (read-only use here).
        app_id: The App this resolution is being performed for.
        configured_id: `app.memory_extraction_service_id` or
            `app.memory_embedding_service_id` — `None` means "not configured".
        loader: `AIServiceRepository.get_by_id` or
            `EmbeddingServiceRepository.get_by_id`.
        skip_reason: The `SKIP_*` constant to log when this side resolves to
            unresolved for any reason other than an unexpected exception.

    Returns:
        `(service_id, source, is_system)` — `service_id` is `None` and
        `source` is `"unresolved"` whenever the configured service could not
        be safely accepted.
    """
    if configured_id is None:
        log_memory_event(
            "resolve", app_id=app_id, agent_id=None, active=False, reason=skip_reason, source="unresolved", level="warning"
        )
        return None, "unresolved", False

    try:
        svc = loader(db, configured_id)
    except Exception:
        logger.warning(
            "global_memory resolution failed loading service_id=%s for app_id=%s", configured_id, app_id
        )
        log_memory_event("resolve", app_id=app_id, agent_id=None, active=False, reason=FAIL_ERROR, source="unresolved", level="warning")
        return None, "unresolved", False

    if svc is None:
        logger.warning(
            "global_memory configured service_id=%s not found for app_id=%s", configured_id, app_id
        )
        log_memory_event(
            "resolve", app_id=app_id, agent_id=None, active=False, reason=skip_reason, source="unresolved", level="warning"
        )
        return None, "unresolved", False

    svc_app_id = getattr(svc, "app_id", None)
    if svc_app_id is not None and svc_app_id != app_id:
        # Cross-App reference (the App-scoped vs. system-level trap of D-4):
        # never silently used, regardless of how it ended up on this row.
        logger.warning(
            "global_memory configured service_id=%s belongs to app_id=%s, not %s — treating as unresolved",
            configured_id,
            svc_app_id,
            app_id,
        )
        log_memory_event(
            "resolve", app_id=app_id, agent_id=None, active=False, reason=skip_reason, source="unresolved", level="warning"
        )
        return None, "unresolved", False

    return svc.service_id, "app_config", svc_app_id is None


def resolve_memory_services(db: Session, app: Any) -> ResolvedMemoryServices:
    """Resolve which services (if any) do memory work for `app` (FR-7, AD-5).

    Deterministic, no LLM calls, DB reads only. Takes the **App**, never the
    Agent (D-16) — no agent column participates in service selection
    (AC-40), and this function never falls back to `agent.service_id` for
    extraction nor to a silo's embedding service for memory embeddings.

    Args:
        db: Active DB session.
        app: The `App` ORM row (or any object exposing `app_id`,
            `memory_extraction_service_id` and `memory_embedding_service_id`
            — kept duck-typed for testability).

    Returns:
        A `ResolvedMemoryServices` snapshot. Either side may be independently
        unresolved; callers must treat that side as inactive.
    """
    app_id = getattr(app, "app_id", None)

    extraction_service_id, extraction_source, extraction_is_system = _resolve_one(
        db=db,
        app_id=app_id,
        configured_id=getattr(app, "memory_extraction_service_id", None),
        loader=AIServiceRepository.get_by_id,
        skip_reason=SKIP_NO_AI_SERVICE,
    )
    embedding_service_id, embedding_source, embedding_is_system = _resolve_one(
        db=db,
        app_id=app_id,
        configured_id=getattr(app, "memory_embedding_service_id", None),
        loader=EmbeddingServiceRepository.get_by_id,
        skip_reason=SKIP_NO_EMBEDDING_SERVICE,
    )

    return ResolvedMemoryServices(
        extraction_service_id=extraction_service_id,
        extraction_source=extraction_source,
        extraction_is_system=extraction_is_system,
        embedding_service_id=embedding_service_id,
        embedding_source=embedding_source,
        embedding_is_system=embedding_is_system,
    )


def _probe_embedding_dims(
    embeddings: Any, embedding_service_id: int, timeout: Optional[float] = None
) -> Optional[int]:
    """Probe an embeddings client's output width once, caching it per service id (D-24).

    `EmbeddingService` has no dimensionality column (see the module
    docstring), so the only way to learn the width is to call the model.
    Cached in `_DIMS_CACHE` so a turn never re-probes a service it has
    already measured; a failed probe is not cached (see `_DIMS_CACHE`'s
    comment) so a transient provider outage self-heals on the next turn.

    Args:
        embeddings: A constructed embeddings client (`get_embeddings_model`
            output).
        embedding_service_id: Cache key.
        timeout: When given, the underlying `embed_query` call is bounded to
            this many seconds, run on a throwaway single-worker executor so a
            hung HTTP call can't block the caller past the deadline (the
            worker thread itself is not forcibly killed — the executor is
            shut down without waiting — since plain `concurrent.futures` has
            no cancellation for a call already in flight). `None` (the
            default) preserves the original unbounded behaviour used by the
            runtime probe inside `build_memory_clients`, which must not
            change (security-audit HIGH, step_009 follow-up).

    Returns:
        The embedding width, or `None` if the probe itself failed or timed
        out (in which case the caller must attempt the operation anyway
        rather than disabling memory on incomplete metadata — the store's
        own dimension error is the backstop, per AD-12/D-24).
    """
    if embedding_service_id in _DIMS_CACHE:
        return _DIMS_CACHE[embedding_service_id]

    try:
        if timeout is not None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(embeddings.embed_query, "dimension probe")
                dims = len(future.result(timeout=timeout))
            finally:
                # wait=False: never block on a still-running probe past the
                # deadline we just enforced above.
                executor.shutdown(wait=False)
        else:
            dims = len(embeddings.embed_query("dimension probe"))
    except Exception:
        logger.warning(
            "global_memory could not probe embedding dimensionality for embedding_service_id=%s",
            embedding_service_id,
        )
        return None

    _DIMS_CACHE[embedding_service_id] = dims
    return dims


def probe_embedding_dims_for_config(embedding_service: Any) -> Optional[int]:
    """Best-effort probe of one `EmbeddingService`'s output width (D-13/AC-46).

    Used by the `/internal` App-configuration boundary (FR-3(b)) to reject a
    `memory_embedding_service_id` whose width does not match
    `GLOBAL_MEMORY_EMBEDDING_DIMS` at configuration time, so the mismatch is
    visible to the administrator instead of silently skipping the turn later.
    Shares `_DIMS_CACHE` with the runtime probe (D-24), so a service measured
    here is not re-probed on the next turn and vice versa.

    Never raises: any failure to build the client or to probe it (bad
    credentials, unreachable provider, etc.) returns `None` so the caller can
    accept the configuration and let the runtime guard
    (`build_memory_clients`) be the backstop, per AD-12 — configuration must
    not be blocked on a transient probe failure or on genuinely
    undeterminable metadata.

    Bounded to `_CONFIG_PROBE_TIMEOUT_SECONDS` (5s): this is called from the
    synchronous `AppService._validate_memory_service_ids`, itself run off the
    event loop via `run_in_threadpool` by `PUT /internal/apps/{id}` — without
    an explicit deadline here a slow/unreachable Custom/Ollama endpoint could
    still tie up that threadpool worker for minutes per request
    (security-audit HIGH, step_009 follow-up).

    Args:
        embedding_service: The `EmbeddingService` ORM row to probe.

    Returns:
        The embedding width, or `None` if it could not be determined (which
        includes timing out).
    """
    try:
        embeddings = get_embeddings_model(embedding_service)
        return _probe_embedding_dims(
            embeddings, embedding_service.service_id, timeout=_CONFIG_PROBE_TIMEOUT_SECONDS
        )
    except Exception:
        logger.warning(
            "global_memory could not probe embedding dimensionality for embedding_service_id=%s "
            "during App configuration validation",
            getattr(embedding_service, "service_id", None),
        )
        return None


def build_memory_clients(
    resolved: ResolvedMemoryServices, db: Session
) -> Tuple[Optional[Any], Optional[Any], Optional[int]]:
    """Build the LangChain clients for a resolved App-level memory configuration.

    Loads the accepted `AIService`/`EmbeddingService` rows and constructs
    `create_llm_from_service(ai_service, temperature=0)` and
    `get_embeddings_model(embedding_service)`. Each side is independently
    `None` when unresolved, and any exception building a client makes that
    side `None` plus a WARNING — this function never raises.

    The embedding side additionally applies the D-13/AD-12 dimensionality
    guard: the embedding width is probed once (`_probe_embedding_dims`,
    D-24) and compared against `settings.embedding_dims()`. On a known
    mismatch the embeddings client is discarded (returned as `None`) with a
    WARNING naming both widths — the existing FR-7 contract then skips the
    read path and the dedup stage, so no new failure mode is introduced. When
    the width cannot be determined at all, the embeddings client is returned
    as-is (an *attempt*, not a block) — the store's own dimension error is
    the backstop (its exact type is recorded in `decisions.md`).

    Args:
        resolved: The `ResolvedMemoryServices` snapshot from
            `resolve_memory_services`.
        db: Active DB session used to load the accepted service rows.

    Returns:
        `(llm, embeddings, dims)` — `llm` and `embeddings` are `None` when
        unresolved or when construction failed; `dims` is the probed
        embedding width when known, else `None`.
    """
    llm: Optional[Any] = None
    if resolved.extraction_service_id is not None:
        try:
            ai_service = AIServiceRepository.get_by_id(db, resolved.extraction_service_id)
            if ai_service is not None:
                llm = create_llm_from_service(ai_service, temperature=0)
        except Exception:
            logger.warning(
                "global_memory failed to build extraction LLM for service_id=%s",
                resolved.extraction_service_id,
            )
            log_memory_event(
                "resolve", app_id=None, agent_id=None, active=False, reason=FAIL_ERROR, source="unresolved", level="warning"
            )
            llm = None

    embeddings: Optional[Any] = None
    dims: Optional[int] = None
    if resolved.embedding_service_id is not None:
        try:
            embedding_service = EmbeddingServiceRepository.get_by_id(db, resolved.embedding_service_id)
            if embedding_service is not None:
                embeddings = get_embeddings_model(embedding_service)
        except Exception:
            logger.warning(
                "global_memory failed to build embeddings client for service_id=%s",
                resolved.embedding_service_id,
            )
            log_memory_event(
                "resolve", app_id=None, agent_id=None, active=False, reason=FAIL_ERROR, source="unresolved", level="warning"
            )
            embeddings = None

        if embeddings is not None:
            dims = _probe_embedding_dims(embeddings, resolved.embedding_service_id)
            expected_dims = settings.embedding_dims()
            if dims is not None and dims != expected_dims:
                logger.warning(
                    "global_memory embedding dimensionality mismatch for embedding_service_id=%s: "
                    "got %s dims, deployment expects %s",
                    resolved.embedding_service_id,
                    dims,
                    expected_dims,
                )
                log_memory_event(
                    "resolve",
                    app_id=None,
                    agent_id=None,
                    active=False,
                    source="unresolved",
                    reason=SKIP_DIMS_MISMATCH,
                    level="warning",
                )
                embeddings = None

    return llm, embeddings, dims
