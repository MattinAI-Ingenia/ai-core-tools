"""
Cross-session user memory — scope identity and eligibility (spec
`user-memory-mem0`, FR-4, D-2, D-22, AD-2).

Every memory namespace is the tuple `(NAMESPACE_ROOT, str(app_id),
str(user_id))` (D-2/D-12). **No other code in this codebase may build a
memory namespace from `user_id` alone**, and no read may omit the `app_id`
component. `resolve_identity` in this module is the only place that
namespace tuple may be constructed; the store seam (`client.py`, a later
step) refuses to accept anything that is not a `MemoryIdentity`.

`GLOBAL_MEMORY_NAMESPACE_ROOT` is reserved for this feature specifically
because any other LangGraph store instance in this system (e.g. a future
one) shares the same four physical tables (`store`, `store_vectors`,
`store_migrations`, `vector_migrations`) — only the namespace prefix keeps
them apart.

Ineligible identities are **excluded**, never bucketed: an `apikey_*`
identity is shared by every end user of an integration, so treating it as a
single "user" would pool unrelated people's facts into one scope and serve
them back to all of them.
"""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from sqlalchemy.orm import Session

from models.agent import MarketplaceVisibility
from repositories.user_repository import UserRepository
from services.system_settings_service import SystemSettingsService
from utils.logger import get_logger

from . import settings
from .settings import FAIL_ERROR, log_memory_event

logger = get_logger(__name__)

# A single `apikey_*` identity is shared by every end user of an integration
# (`backend/routers/public/v1/auth.py:154-160` builds
# `f"apikey_{sha256(...)[:16]}"`). Bucketing them under one memory scope is a
# hard prohibition, not an optimisation.
APIKEY_USER_ID_PREFIX = "apikey_"

# Known synthetic/placeholder identities to reject outright (defence in
# depth — see the module docstring on why this must never be the *decision*).
INELIGIBLE_USER_IDS = frozenset({"anonymous", "default", ""})

# The `'default'` sentinel a caller-less context falls back to for app_id
# (`agent_execution_service.py:552`).
INELIGIBLE_APP_ID_SENTINEL = "default"

# System setting key that designates the platform-wide chatbot agent
# (`backend/routers/internal/platform_chatbot.py`). Unset/`-1` means "no
# platform chatbot configured" — the same sentinel that router already uses.
_PLATFORM_CHATBOT_SETTING_KEY = "platform_chatbot_agent_id"
_PLATFORM_CHATBOT_UNSET_SENTINEL = -1


@dataclass(frozen=True)
class MemoryIdentity:
    """The one object every memory read/write is keyed by (AD-2).

    Attributes:
        app_id: The tenant boundary — the agent's App (not necessarily the
            caller's own App; see the module docstring on D-22).
        user_id: The real `User` primary key of the caller.
        agent_id: The agent the fact/turn originated from. Recorded as
            per-fact metadata (OQ-9), **not** part of the namespace, so a
            future switch to per-agent isolation is a read-policy change
            with no data migration.
        namespace: `(NAMESPACE_ROOT, str(app_id), str(user_id))` — asserted
            against `app_id`/`user_id` in `__post_init__` so a namespace can
            never silently drift from the fields it is supposed to encode.
    """

    app_id: int
    user_id: int
    agent_id: int
    namespace: Tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (settings.namespace_root(), str(self.app_id), str(self.user_id))
        if self.namespace != expected:
            raise AssertionError(
                f"MemoryIdentity.namespace {self.namespace!r} does not match the expected "
                f"(root, app_id, user_id) tuple {expected!r} — a memory namespace must only "
                "ever be built by resolve_identity()."
            )


def is_eligible_user_id(user_id: Any) -> bool:
    """The single FR-4 predicate for whether a `user_id` shape is eligible.

    Kept as defence in depth (deny-list), never as the decision itself: see
    `resolve_identity` for the positive-proof checks that actually gate
    activation. True only for a positive integer identity — a real `User`
    primary key, either as an `int` or an all-digits `str`.

    Rejects, in order: `None`; the known synthetic placeholders in
    `INELIGIBLE_USER_IDS` (`'anonymous'`, `'default'`, `''`); any string
    starting with `apikey_` (one identity shared by every end user of an
    integration); booleans (a `bool` is a `int` subclass in Python and must
    never be mistaken for a PK); and anything that is not an `int` or an
    all-digits `str`.

    Args:
        user_id: The raw `user_id` value from a `user_context` dict.

    Returns:
        True only if `user_id` is shaped like a real, positive `User` PK.
    """
    if user_id is None or isinstance(user_id, bool):
        return False
    if isinstance(user_id, str):
        if user_id in INELIGIBLE_USER_IDS or user_id.startswith(APIKEY_USER_ID_PREFIX):
            return False
        if not (user_id.isascii() and user_id.isdigit()):
            return False
        return int(user_id) > 0
    if isinstance(user_id, int):
        return user_id > 0
    return False


def is_eligible_app_id(app_id: Any) -> bool:
    """Shape-only eligibility check for `app_id` (defence in depth, FR-4).

    False for `None` and for the literal `'default'` fallback
    (`agent_execution_service.py:552`, used when a context carries no App at
    all). True for a positive integer, whether `int` or an all-digits `str`.

    Args:
        app_id: The raw `app_id` value (typically `agent.app_id`).

    Returns:
        True only if `app_id` is shaped like a real, positive App PK.
    """
    if app_id is None or isinstance(app_id, bool):
        return False
    if isinstance(app_id, str):
        if app_id == INELIGIBLE_APP_ID_SENTINEL or not (app_id.isascii() and app_id.isdigit()):
            return False
        return int(app_id) > 0
    if isinstance(app_id, int):
        return app_id > 0
    return False


def _is_platform_chatbot(db: Session, agent_id: Optional[int]) -> bool:
    """Return whether `agent_id` is the deployment's designated platform chatbot.

    Mirrors the unset-sentinel handling in
    `backend/routers/internal/platform_chatbot.py::_get_agent_id`: both
    `None` and `-1` mean "no platform chatbot configured".

    Fails **closed**: if the setting cannot be resolved (malformed value via
    `AICT_SYSTEM_PLATFORM_CHATBOT_AGENT_ID`, or the key ever leaving
    `system_defaults.yaml`), this returns `True` — "cannot determine" is
    treated as "this is the platform chatbot", i.e. memory stays off, never
    the other way around (NFR-4).
    """
    if agent_id is None:
        return False
    try:
        configured_id = SystemSettingsService(db).get_setting(_PLATFORM_CHATBOT_SETTING_KEY)
    except Exception:
        log_memory_event(
            "identity", app_id=None, agent_id=agent_id, active=False, reason=FAIL_ERROR, level="error"
        )
        return True
    if configured_id is None or configured_id == _PLATFORM_CHATBOT_UNSET_SENTINEL:
        return False
    return configured_id == agent_id


def resolve_identity(db: Session, user_context: Optional[dict], agent: Any) -> Optional[MemoryIdentity]:
    """Resolve the caller's `MemoryIdentity` for a turn, or `None` if ineligible (FR-4).

    This is the **only** place in the codebase that may construct a memory
    namespace (AD-2, AC-12), and — per D-22 — the fix for a CRITICAL
    tenant-isolation defect found in design: `user_context['app_id']` is the
    **agent's** App, not necessarily the caller's
    (`backend/routers/internal/marketplace.py:626`/`:700`,
    `backend/routers/internal/platform_chatbot.py:83` all build it as
    `{"user_id": <caller>, "app_id": agent.app_id}`), while `PUBLIC`
    marketplace agents are chattable by *any* authenticated user of the
    deployment (`backend/services/marketplace_service.py:64`). Without a
    membership check, an outsider's personal facts would be stored inside
    the publishing App's tenant.

    Eligibility is therefore a **positive proof**, not a deny-list: the shape
    checks below are a cheap first guard, never sufficient on their own,
    because any future synthetic-but-numeric identity would slip through
    them. See the five numbered checks in the body.

    Returns `None` when ineligible — memory is simply inactive for the turn:
    no read, no write, no error.
    """
    if not user_context:
        return None

    if user_context.get("oauth") is not True:
        return None

    raw_user_id = user_context.get("user_id")
    if not is_eligible_user_id(raw_user_id):
        return None
    user_id = int(raw_user_id)

    app_id = getattr(agent, "app_id", None)
    if not is_eligible_app_id(app_id):
        return None
    app_id = int(app_id)

    # Positive proof #2: user_id resolves to a real User row.
    if UserRepository(db).get_by_id(user_id) is None:
        return None

    # Imported lazily, only once the cheap checks above have already passed:
    # `routers.controls.role_authorization` sits behind `routers.internal`'s
    # package __init__ (`apps.py` imports it back), so importing it before
    # `routers.internal` has ever been loaded (e.g. a bare unit test that
    # imports only this module and exercises the ineligible-shape cases
    # above) raises a circular-import error. By the time any real turn
    # reaches this function the app has already fully loaded
    # `routers.internal` (`main.py:54`), so this import always resolves
    # against an already-initialized module in production.
    from routers.controls.role_authorization import AppRole, has_min_role, resolve_user_app_role

    # Positive proof #3 (D-22/R-22): the caller holds >= VIEWER in the
    # agent's own App — NOT merely "authenticated on the deployment".
    role = resolve_user_app_role(db, app_id, user_id)
    if role is None or not has_min_role(role, AppRole.VIEWER):
        return None

    # Positive proof #4 (D-22): the agent's audience must be App-confined.
    if getattr(agent, "marketplace_visibility", None) == MarketplaceVisibility.PUBLIC:
        return None
    agent_id = getattr(agent, "agent_id", None)
    if _is_platform_chatbot(db, agent_id):
        return None

    # Positive proof #5: `agent_id` must itself be shaped like a real PK — it
    # is recorded as per-fact provenance (OQ-9) and a `None`/malformed value
    # would write facts nothing can later attribute to an agent.
    if not is_eligible_app_id(agent_id):
        return None
    agent_id = int(agent_id)

    namespace = (settings.namespace_root(), str(app_id), str(user_id))
    return MemoryIdentity(app_id=app_id, user_id=user_id, agent_id=agent_id, namespace=namespace)
