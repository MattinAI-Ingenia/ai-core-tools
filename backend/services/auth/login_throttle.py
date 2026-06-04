"""Per-IP and per-account login throttle for LOCAL auth mode.

Provides a thread-safe, in-process fixed-window rate limiter keyed on
arbitrary string keys (callers pass ``"ip:<addr>"`` or ``"account:<email>"``).
Mirrors the ``RateLimitService`` window logic but is intentionally a separate
module — the concern (brute-force on the login surface) is different from
per-app agent-execution limits.

Module-level singleton ``login_throttle`` is the single shared instance.
The ``LoginThrottle`` Protocol makes it trivially swappable to a Redis-backed
implementation without touching the router.

FastAPI dependency ``enforce_login_throttle`` enforces the per-IP limit and is
wired onto the throttled endpoints via ``Depends()``.  The per-account limit
cannot be a dependency (the body is not yet parsed), so it is applied inline
inside the ``login`` handler.

Configuration (environment variables, read once at module import):
    LOCAL_LOGIN_THROTTLE_PER_IP_PER_MIN      — max login attempts per IP per
        minute window (default: 20).  0 disables the IP throttle.
    LOCAL_LOGIN_THROTTLE_PER_ACCOUNT_PER_MIN — max login attempts per account
        (email) per minute window (default: 10).  0 disables the account
        throttle.  Callers read this constant directly.
    TRUST_PROXY_HEADERS                      — when True (default), the first
        value in ``X-Forwarded-For`` is used as the client IP; otherwise
        ``request.client.host`` is used.  Set to False when the backend is
        exposed directly to the internet (no trusted proxy in front).
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request, status

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read once at module import to match credential_service.py
# ---------------------------------------------------------------------------

_PER_IP_PER_MIN: int = Config.get_int_env_var(
    "LOCAL_LOGIN_THROTTLE_PER_IP_PER_MIN", default=20
)
_PER_ACCOUNT_PER_MIN: int = Config.get_int_env_var(
    "LOCAL_LOGIN_THROTTLE_PER_ACCOUNT_PER_MIN", default=10
)
_TRUST_PROXY: bool = Config.get_bool_env_var("TRUST_PROXY_HEADERS", default=True)

# Generic 429 detail — intentionally vague to prevent leaking whether the
# throttle is IP- or account-based (mirrors the AccountLockedError path).
# Exported as a public constant so the router can import it and avoid duplicating
# the string.
ERR_TOO_MANY_REQUESTS = "Too many requests. Please try again later."

# ---------------------------------------------------------------------------
# ThrottleState
# ---------------------------------------------------------------------------


@dataclass
class ThrottleState:
    """Result returned by ``LoginThrottle.hit``.

    Attributes:
        allowed: True when the request is within the allowed rate.
        remaining: How many more calls are permitted in the current window.
            -1 means unlimited (throttle is disabled for this key).
        retry_after_seconds: Seconds until the current window resets.
            Meaningful only when ``allowed`` is False.
    """

    allowed: bool
    remaining: int
    retry_after_seconds: int


# ---------------------------------------------------------------------------
# LoginThrottle Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LoginThrottle(Protocol):
    """Protocol for swappable login throttle implementations.

    Both the in-process and any future Redis-backed implementation must satisfy
    this interface.
    """

    def hit(self, key: str) -> ThrottleState:
        """Consume one slot for *key* and return the resulting state.

        Args:
            key: Arbitrary throttle key, e.g. ``"ip:1.2.3.4"`` or
                ``"account:user@example.com"``.

        Returns:
            ``ThrottleState`` with ``allowed=True`` when the request is within
            limits, ``allowed=False`` when the window is exhausted.
        """
        ...

    def reset(self, key: str) -> None:
        """Clear the counter for *key*.

        Used after a successful login to release any per-account backpressure
        accumulated by earlier failed attempts in the same minute window.

        Args:
            key: The throttle key to clear.
        """
        ...


# ---------------------------------------------------------------------------
# InProcessLoginThrottle
# ---------------------------------------------------------------------------


class InProcessLoginThrottle:
    """Thread-safe in-memory fixed-window login throttle.

    Each key has an independent counter that resets at the start of each
    calendar minute.  Stale entries are pruned lazily when the tracked-key
    count exceeds ``_cleanup_threshold`` to bound memory consumption.

    Thread safety is provided by a single ``threading.RLock``.  The lock is
    held only for the duration of the counter mutation — no I/O inside.

    Args:
        per_minute: Maximum hits allowed per key per minute window.
            0 means unlimited.
        cleanup_threshold: Prune stale entries when more than this many keys
            are tracked.  Defaults to 500.
    """

    def __init__(self, per_minute: int = 20, cleanup_threshold: int = 500) -> None:
        self._per_minute = per_minute
        self._cleanup_threshold = cleanup_threshold
        # key -> {"window_start": int (epoch-minute), "count": int}
        self._counters: dict[str, dict[str, int]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public interface (satisfies LoginThrottle Protocol)
    # ------------------------------------------------------------------

    def hit(self, key: str) -> ThrottleState:
        """Consume one slot for *key* and return the resulting ``ThrottleState``.

        When ``per_minute`` is 0, the throttle is disabled — returns
        ``ThrottleState(allowed=True, remaining=-1, retry_after_seconds=0)``.

        Args:
            key: Throttle key (e.g. ``"ip:1.2.3.4"``).

        Returns:
            ``ThrottleState`` describing whether the request is allowed.
        """
        if self._per_minute <= 0:
            return ThrottleState(allowed=True, remaining=-1, retry_after_seconds=0)

        current_time = time.time()
        current_minute = int(current_time // 60)
        window_reset_epoch = (current_minute + 1) * 60
        retry_after = max(0, int(window_reset_epoch - current_time))

        with self._lock:
            counter = self._counters.get(key)
            if counter is None:
                self._counters[key] = {"window_start": current_minute, "count": 0}
                counter = self._counters[key]
            elif counter["window_start"] < current_minute:
                # New window — reset counter.
                counter["window_start"] = current_minute
                counter["count"] = 0

            # Run cleanup unconditionally so stale entries are pruned even when
            # every tracked key is over its limit (prevents unbounded memory growth
            # under saturation where the allowed branch is never reached).
            self._cleanup_if_needed()

            if counter["count"] >= self._per_minute:
                return ThrottleState(
                    allowed=False,
                    remaining=0,
                    retry_after_seconds=retry_after,
                )

            counter["count"] += 1
            remaining = self._per_minute - counter["count"]
            return ThrottleState(
                allowed=True,
                remaining=remaining,
                retry_after_seconds=retry_after,
            )

    def reset(self, key: str) -> None:
        """Clear the counter for *key*.

        Args:
            key: Throttle key to clear.
        """
        with self._lock:
            self._counters.pop(key, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cleanup_if_needed(self) -> None:
        """Prune stale entries when the tracked-key count exceeds the threshold.

        Called inside the lock — no additional locking needed.  Removes entries
        whose window started before the current minute (they will reset anyway
        on the next ``hit`` call).
        """
        if len(self._counters) <= self._cleanup_threshold:
            return

        current_minute = int(time.time() // 60)
        stale = [
            k for k, c in self._counters.items()
            if c["window_start"] < current_minute - 1
        ]
        for k in stale:
            del self._counters[k]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

login_throttle: LoginThrottle = InProcessLoginThrottle(
    per_minute=_PER_IP_PER_MIN,
    cleanup_threshold=500,
)

# A second instance keyed on accounts shares the same window logic but uses
# the per-account limit.  Typed as ``LoginThrottle`` (the Protocol) so that a
# future Redis-backed swap replaces both singletons through one interface.
_account_throttle: LoginThrottle = InProcessLoginThrottle(
    per_minute=_PER_ACCOUNT_PER_MIN,
    cleanup_threshold=500,
)


# ---------------------------------------------------------------------------
# IP extraction helper (DRY — used by dependency and can be used in handlers)
# ---------------------------------------------------------------------------


def extract_client_ip(request: Request) -> Optional[str]:
    """Extract the real client IP from the request.

    Trusts ``X-Forwarded-For`` (first hop) when ``TRUST_PROXY_HEADERS`` is
    True (the default), because all deployments sit behind Caddy or a k8s
    ingress.  When the flag is False the raw socket peer is used.

    Args:
        request: The incoming FastAPI ``Request`` object.

    Returns:
        Client IP string, or ``None`` if it cannot be determined.
    """
    if _TRUST_PROXY:
        forwarded_for: Optional[str] = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


# ---------------------------------------------------------------------------
# Per-account throttle helpers (called from the login route body)
# ---------------------------------------------------------------------------


def hit_account_throttle(email: str) -> ThrottleState:
    """Consume one per-account throttle slot and return the state.

    The caller is responsible for raising ``HTTPException(429)`` if
    ``state.allowed`` is False, passing ``headers={"Retry-After": ...}``
    directly on the exception so Starlette's default handler carries the header
    onto the response (setting headers on the injected ``Response`` dependency
    object does NOT work for ``HTTPException`` responses).

    Kept as a plain function (not a dependency) because the account key is
    derived from the parsed request body, which is not available in a FastAPI
    dependency.

    Args:
        email: The submitted login email address (should already be normalised
            to lowercase by the ``LoginRequest`` schema validator).

    Returns:
        ``ThrottleState`` for the account key.
    """
    return _account_throttle.hit(f"account:{email}")


def reset_account_throttle(email: str) -> None:
    """Clear the per-account throttle counter on successful login.

    Prevents a user from being throttled by their own earlier typos within
    the same minute window after a successful authentication.

    Args:
        email: The authenticated user's email address.
    """
    _account_throttle.reset(f"account:{email}")


# ---------------------------------------------------------------------------
# FastAPI dependency: per-IP throttle
# ---------------------------------------------------------------------------


async def enforce_login_throttle(request: Request) -> None:
    """FastAPI dependency that enforces the per-IP login throttle.

    Extracts the client IP, consumes one slot from the IP-keyed throttle, and
    raises ``HTTP 429`` with a generic body and a ``Retry-After`` header if the
    window is exhausted.

    The ``Retry-After`` header is attached directly to ``HTTPException`` so that
    Starlette's default exception handler carries it onto the response.  Setting
    it on an injected ``Response`` dependency object does NOT work for exceptions
    because Starlette replaces the response object entirely when rendering a 4xx.

    The error detail is intentionally generic — it does not reveal whether the
    throttle is IP-based or account-based (anti-enumeration).

    Args:
        request: Incoming FastAPI request (IP extraction).

    Raises:
        HTTPException: 429 when the per-IP window is exhausted.
    """
    ip = extract_client_ip(request)
    if ip is None:
        # Cannot determine IP — allow and log; prefer availability over blocking.
        logger.warning("auth:throttle_no_ip path=%s", request.url.path)
        return

    state = login_throttle.hit(f"ip:{ip}")
    if not state.allowed:
        logger.warning(
            "auth:throttle_ip_exceeded ip=%s path=%s retry_after=%s",
            ip,
            request.url.path,
            state.retry_after_seconds,
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=ERR_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(state.retry_after_seconds)},
        )
