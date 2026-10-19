"""
Cross-session user memory — App-level service resolution (FR-7, D-16, D-21).

Decides which models do the memory work for an App and builds their clients.
Reads only the `App`, never the `Agent`: no agent column participates in
service selection.

Two things here are non-obvious:

- **No system-level fallback.** An unset service id means memory is inactive
  for that App, full stop — this module never reads `SystemSettingsService`.
  A fallback would route conversation excerpts and fact text through the
  *operator's* provider credentials by default, breaking NFR-5. An App may
  still explicitly choose a system-level (`app_id IS NULL`) service; only the
  implicit substitute is gone.
- **Nothing here raises.** Every failure degrades that side to `None` plus a
  warning; a memory failure must never break a chat turn (NFR-4).

Any embedding width is accepted: the store's vector column is
dimension-agnostic and every vector search is filtered to the service that
embedded the facts (see `tools/memory/client.py`), so there is no
deployment-wide width to check against.

No provider API key is ever read from the environment in this module (AC-24).
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from sqlalchemy.orm import Session

from repositories.ai_service_repository import AIServiceRepository
from repositories.embedding_service_repository import EmbeddingServiceRepository
from tools.aiServiceTools import create_llm_from_service
from tools.embeddingTools import get_embeddings_model
from utils.logger import get_logger

from . import settings
from .settings import FAIL_ERROR, SKIP_NO_AI_SERVICE, SKIP_NO_EMBEDDING_SERVICE, log_memory_event

logger = get_logger(__name__)


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
        language: The code facts are written in (`App.memory_language`,
            English when unset) — see `settings.MEMORY_LANGUAGES`.
    """

    extraction_service_id: Optional[int]
    extraction_source: str
    extraction_is_system: bool
    embedding_service_id: Optional[int]
    embedding_source: str
    embedding_is_system: bool
    language: str = "en"


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
        language=settings.memory_language(app),
    )


def build_memory_clients(
    resolved: ResolvedMemoryServices, db: Session
) -> Tuple[Optional[Any], Optional[Any]]:
    """Build the LangChain clients for a resolved App-level memory configuration.

    Loads the accepted `AIService`/`EmbeddingService` rows and constructs
    `create_llm_from_service(ai_service, temperature=0)` and
    `get_embeddings_model(embedding_service)`. Each side is independently
    `None` when unresolved, and any exception building a client makes that
    side `None` plus a WARNING — this function never raises.

    Args:
        resolved: The `ResolvedMemoryServices` snapshot from
            `resolve_memory_services`.
        db: Active DB session used to load the accepted service rows.

    Returns:
        `(llm, embeddings)` — each is `None` when unresolved or when
        construction failed.
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

    return llm, embeddings
