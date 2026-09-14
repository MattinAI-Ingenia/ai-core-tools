"""
Cross-session user memory — kill switch, tunables, turn-index arithmetic and
the single structured logging helper (spec `user-memory-mem0`, FR-10, AD-3,
AD-13).

Every `GLOBAL_MEMORY_*` constant lives in `config.py` (module-level
`os.getenv`, per this repo's `SANDBOX_*` convention). Call sites — including
`identity.py` in this same package — must go through the thin accessors below
rather than importing `config` directly, so every tunable has exactly one
read path to reason about.

This module is pure: no DB session, no network I/O, no LangGraph/store
import.
"""
from typing import Any, Optional

import config
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level flag so the disabled state is logged once per process, not
# once per turn (FR-10's "log once, not per turn").
_disabled_logged = False

# Closed set of machine-readable skip / failure reasons (FR-10). Kept as
# plain string constants (not an Enum) so log lines and `reason=` comparisons
# in tests stay simple string equality.
SKIP_DISABLED = "SKIP_DISABLED"
SKIP_FLAG_OFF = "SKIP_FLAG_OFF"
SKIP_INELIGIBLE_IDENTITY = "SKIP_INELIGIBLE_IDENTITY"
SKIP_NO_AI_SERVICE = "SKIP_NO_AI_SERVICE"
SKIP_NO_EMBEDDING_SERVICE = "SKIP_NO_EMBEDDING_SERVICE"
SKIP_DIMS_MISMATCH = "SKIP_DIMS_MISMATCH"
SKIP_QUOTA = "SKIP_QUOTA"
SKIP_NO_FACTS = "SKIP_NO_FACTS"
SKIP_DUPLICATE = "SKIP_DUPLICATE"
SKIP_ZERO_MATCHES = "SKIP_ZERO_MATCHES"
SKIP_WRITE_NOT_DUE = "SKIP_WRITE_NOT_DUE"
SKIP_READ_NOT_DUE = "SKIP_READ_NOT_DUE"
# step_008: the fire-and-forget write path's own back-pressure bound
# (R-8/NFR-7) — scheduling was skipped because `max_inflight_writes()`
# concurrent write tasks were already running. Kept distinct from
# `FAIL_ERROR` so an operator can tell "the feature is overloaded" apart
# from "something raised" in the logs.
SKIP_BACKPRESSURE = "SKIP_BACKPRESSURE"
FAIL_TIMEOUT = "FAIL_TIMEOUT"
FAIL_ERROR = "FAIL_ERROR"

# Closed set every `reason=` must belong to (AC-34, NFR-5). A caller that
# builds `reason` from an exception message (the natural thing to write on a
# `FAIL_*` path) must never let that string reach the log line verbatim — an
# embedding/LLM provider 400 commonly echoes the payload it choked on, which
# could contain fact contents. `log_memory_event` coerces anything outside
# this set to `FAIL_ERROR` rather than trusting the caller.
_ALLOWED_REASONS = frozenset({
    SKIP_DISABLED,
    SKIP_FLAG_OFF,
    SKIP_INELIGIBLE_IDENTITY,
    SKIP_NO_AI_SERVICE,
    SKIP_NO_EMBEDDING_SERVICE,
    SKIP_DIMS_MISMATCH,
    SKIP_QUOTA,
    SKIP_NO_FACTS,
    SKIP_DUPLICATE,
    SKIP_ZERO_MATCHES,
    SKIP_WRITE_NOT_DUE,
    SKIP_READ_NOT_DUE,
    SKIP_BACKPRESSURE,
    FAIL_TIMEOUT,
    FAIL_ERROR,
})

# Closed set every `source=` must belong to (same rationale as
# `_ALLOWED_REASONS`). Anything outside this set is coerced to `"unresolved"`
# — the existing "could not determine" tag — rather than logged verbatim.
_ALLOWED_SOURCES = frozenset({"app_config", "unresolved", "tool"})
_UNKNOWN_SOURCE_FALLBACK = "unresolved"


def is_global_memory_enabled() -> bool:
    """Return whether the deployment-wide memory kill switch is on (FR-10).

    Reads `config.GLOBAL_MEMORY_ENABLED`. When disabled, logs the disabled
    state at INFO exactly once per process (a module-level flag guards
    repeats) so a per-turn hook that calls this every turn does not spam the
    log — FR-10 requires "log once, not per turn".

    Returns:
        True if memory is enabled for this deployment, False otherwise.
    """
    global _disabled_logged
    if config.GLOBAL_MEMORY_ENABLED:
        return True
    if not _disabled_logged:
        logger.info("Global user memory is disabled by AICT_GLOBAL_MEMORY_ENABLED")
        _disabled_logged = True
    return False


def top_k() -> int:
    """Max facts injected into the read-path context block per turn."""
    return config.GLOBAL_MEMORY_TOP_K


def read_timeout_s() -> float:
    """Timeout budget, in seconds, for a single retrieval (read) operation."""
    return config.GLOBAL_MEMORY_READ_TIMEOUT_S


def write_timeout_s() -> float:
    """Timeout budget, in seconds, for the fire-and-forget write path."""
    return config.GLOBAL_MEMORY_WRITE_TIMEOUT_S


def context_max_chars() -> int:
    """Hard character cap on the rendered "remembered context" block (AD-7)."""
    return config.GLOBAL_MEMORY_CONTEXT_MAX_CHARS


def extraction_input_max_chars() -> int:
    """Hard character cap on the extraction input window (FR-5, R-6)."""
    return config.GLOBAL_MEMORY_EXTRACTION_INPUT_MAX_CHARS


def max_inflight_writes() -> int:
    """Back-pressure bound on concurrent fire-and-forget write tasks (D-6)."""
    return config.GLOBAL_MEMORY_MAX_INFLIGHT_WRITES


def namespace_root() -> str:
    """The reserved namespace root segment shared by every memory scope (D-2).

    This is a code constant in `config.py`, never an env var (R-25): it
    participates in every namespace, so making it configurable would strand
    existing facts beyond both enumeration (AD-10) and erasure (FR-8).
    """
    return config.GLOBAL_MEMORY_NAMESPACE_ROOT


def max_facts_per_extraction() -> int:
    """Max number of facts a single extraction call may produce (D-23)."""
    return config.GLOBAL_MEMORY_MAX_FACTS_PER_EXTRACTION


def embedding_dims() -> int:
    """Deployment-wide memory vector width (D-13)."""
    return config.GLOBAL_MEMORY_EMBEDDING_DIMS


def dedup_threshold() -> float:
    """Similarity ceiling above which a candidate fact is dropped (D-14)."""
    return config.GLOBAL_MEMORY_DEDUP_THRESHOLD


def write_every_n_turns() -> int:
    """The Nth-turn batching threshold for the write trigger (D-15)."""
    return config.GLOBAL_MEMORY_WRITE_EVERY_N_TURNS


def read_first_n_turns() -> int:
    """The turn-index ceiling for reads on `has_memory=True` agents (D-17)."""
    return config.GLOBAL_MEMORY_READ_FIRST_N_TURNS


def fact_max_chars() -> int:
    """Max length of a single fact string (D-23, also used by the FR-11 tool)."""
    return config.GLOBAL_MEMORY_FACT_MAX_CHARS


def tool_enabled() -> bool:
    """Whether the FR-11 `remember` tool is enabled (off by default, D-19/R-18)."""
    return config.GLOBAL_MEMORY_TOOL_ENABLED


def tool_max_writes_per_turn() -> int:
    """Max `remember`-tool calls honored within a single turn (FR-11, R-20)."""
    return config.GLOBAL_MEMORY_TOOL_MAX_WRITES_PER_TURN


def tool_max_writes_per_day() -> int:
    """Max `remember`-tool writes honored per user per day (FR-11, R-20)."""
    return config.GLOBAL_MEMORY_TOOL_MAX_WRITES_PER_DAY


def turn_index(conversation: Any) -> int:
    """Return the 1-based turn index for a conversation (AD-13).

    Both the write trigger (D-15) and the read condition (D-17) need "which
    turn is this?", and they must agree — this is the single helper both use.

    The source is `Conversation.message_count` (`backend/models/conversation.py`),
    which is already loaded on the execution context and already maintained:
    `ConversationService.increment_message_count` is called with
    `increment_by=2` from the shared `_finalize_turn`
    (`backend/services/agent_execution_service.py:913-918`), once per
    completed turn (one user message + one assistant reply). So after N
    completed turns, `message_count == 2 * N`, and `message_count // 2 + 1`
    is the index of the turn currently in progress.

    The read hook (`_prepare_turn`) runs *before* this increment, so it sees
    `message_count` as left by the *previous* turn — exactly what "which turn
    is this" needs. The write hook (`_finalize_turn`) runs around the same
    increment; step_007/step_008 must read the counter at a well-defined
    point relative to it (see AD-13) — this helper only supplies the
    arithmetic, not *when* it is called.

    A conversation with no row yet (a path that never persists one) or a
    conversation whose `message_count` is falsy resolves to turn 1, i.e. the
    turn reads/extracts rather than being silently skipped — erring toward
    doing the work once beats never doing it.

    Args:
        conversation: The `Conversation` ORM row (or `None`, or any object
            exposing a `message_count` attribute — kept duck-typed so this
            stays pure and unit-testable without the ORM).

    Returns:
        The 1-based turn index.
    """
    return (getattr(conversation, "message_count", 0) or 0) // 2 + 1


def should_write(turn_index_value: int) -> bool:
    """Return whether the batched extraction trigger fires for this turn (D-15).

    Fires on turn 1 and every `GLOBAL_MEMORY_WRITE_EVERY_N_TURNS`th turn
    after. The `turn_index_value == 1` clause is load-bearing (R-16), not an
    optimisation: a threshold-only trigger would never fire on a conversation
    that ends before the first threshold, and opening a conversation with a
    stable preference is the commonest way a durable fact arrives. Do not
    "simplify" it away.

    Args:
        turn_index_value: The value returned by `turn_index()`.

    Returns:
        True if extraction should run for this turn.
    """
    return turn_index_value == 1 or turn_index_value % write_every_n_turns() == 0


def should_read(turn_index_value: int, has_memory: bool) -> bool:
    """Return whether pre-turn retrieval should run for this turn (D-17).

    The two branches exist because they answer different questions:

    - `has_memory` falsy: there is no in-thread checkpointer state, so
      nothing carries a previously injected memory block forward. Retrieval
      must run on *every* turn or a later turn would lose the earlier
      context entirely.
    - `has_memory` truthy: the injected block becomes part of the thread
      state (AD-7), so once it has been injected, re-reading on a later turn
      would only re-fetch what the conversation already contains. Retrieval
      is therefore limited to the first `GLOBAL_MEMORY_READ_FIRST_N_TURNS`
      turns.

    Args:
        turn_index_value: The value returned by `turn_index()`.
        has_memory: `Agent.has_memory` — whether this agent carries in-thread
            checkpointer state forward.

    Returns:
        True if retrieval should run for this turn.
    """
    if not has_memory:
        return True
    return turn_index_value <= read_first_n_turns()


def log_memory_event(
    event: str,
    *,
    app_id: Optional[int],
    agent_id: Optional[int],
    active: bool,
    count: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
    source: Optional[str] = None,
    reason: Optional[str] = None,
    level: str = "info",
) -> None:
    """Emit the single structured log line for a memory read/write event (FR-10).

    This is the ONLY logging helper the feature may use for memory events.
    It deliberately accepts no free-text content parameter — no fact text, no
    user message body — so it is structurally impossible to leak PII through
    it (AC-34, NFR-5). `reason` and `source` are each constrained to a closed
    allow-list (`_ALLOWED_REASONS` / `_ALLOWED_SOURCES`); any value outside it
    is coerced (to `FAIL_ERROR` / `"unresolved"` respectively) rather than
    logged verbatim, so a caller cannot smuggle arbitrary text — e.g. a raw
    exception message — into this line even by mistake.

    Args:
        event: Short event name, e.g. `"read"`, `"write"`, `"erase"`.
        app_id: The App the event is scoped to (may be `None` if resolution
            never got that far).
        agent_id: The agent the turn belongs to.
        active: Whether memory was active for this turn/event.
        count: Fact/match count, when applicable (e.g. matches returned,
            facts written, records purged).
        elapsed_ms: Elapsed wall-clock time in milliseconds, when applicable.
        source: Resolution source tag (e.g. `"app_config"`, `"unresolved"`),
            when applicable.
        reason: One of the `SKIP_*` / `FAIL_*` constants above, when the
            event represents a skip or a failure.
        level: One of `"debug"`, `"info"`, `"warning"`, `"error"`.
    """
    log_fn = {
        "debug": logger.debug,
        "info": logger.info,
        "warning": logger.warning,
        "error": logger.error,
    }.get(level, logger.info)

    if reason is not None and reason not in _ALLOWED_REASONS:
        reason = FAIL_ERROR
    if source is not None and source not in _ALLOWED_SOURCES:
        source = _UNKNOWN_SOURCE_FALLBACK

    log_fn(
        "global_memory event=%s app_id=%s agent_id=%s active=%s count=%s "
        "elapsed_ms=%s source=%s reason=%s",
        event,
        app_id,
        agent_id,
        active,
        count,
        elapsed_ms,
        source,
        reason,
    )
