"""Rerank call entry points. Raise RerankError on failure; the caller decides the fallback."""
import asyncio
import concurrent.futures
import json
import logging
import sys
import threading
import time
from contextlib import contextmanager

import httpx

import config
from tools.rerank import http_clients
from tools.rerank.errors import (
    InvalidRerankResponseError, RerankAuthError, RerankBusyError, RerankConnectError, RerankError, RerankLocalError,
    RerankHTTPError, RerankTimeoutError,
)
from tools.rerank.providers import get_adapter
from tools.rerank.types import RerankResult, RerankTarget
from utils import ssrf_guard

logger = logging.getLogger(__name__)

_TEST_QUERY = "What is the capital of France?"
_TEST_DOCS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


# Per-target in-flight cap: one hung endpoint must not take the whole shared connection pool (50).
MAX_INFLIGHT_PER_TARGET = 8
_inflight: dict = {}
_inflight_lock = threading.Lock()
# A shared (system) service also has a per-calling-app sub-limit of half the cap so one tenant cannot starve others.
_SUB_LIMIT = max(1, MAX_INFLIGHT_PER_TARGET // 2)


@contextmanager
def _inflight_slot(target: RerankTarget, caller_app_id=None):
    key = target.service_id if target.service_id is not None else target.endpoint
    sub = (key, caller_app_id) if target.system and caller_app_id is not None else None
    with _inflight_lock:
        if _inflight.get(key, 0) >= MAX_INFLIGHT_PER_TARGET or (sub is not None and _inflight.get(sub, 0) >= _SUB_LIMIT):
            raise RerankBusyError("Too many concurrent rerank calls for this service")
        for k in (key, sub):
            if k is not None:
                _inflight[k] = _inflight.get(k, 0) + 1
    try:
        yield
    finally:
        with _inflight_lock:
            for k in (key, sub):
                if k is not None:
                    _inflight[k] -= 1
                    if _inflight[k] <= 0:
                        _inflight.pop(k, None)


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _total() -> float:
    return max(config.RERANK_TIMEOUT_SECONDS, config.RERANK_CONNECT_TIMEOUT_SECONDS)


def _too_large() -> InvalidRerankResponseError:
    return InvalidRerankResponseError("Response body too large")


def _map_exc(exc: Exception) -> RerankError:
    """Map transport-level exceptions to short, key-free RerankErrors (never echoes URLs or bodies)."""
    if isinstance(exc, (ssrf_guard.FetchUnresolvableError,)):
        return RerankConnectError("Could not resolve the reranker host")
    if isinstance(exc, ssrf_guard.SsrfBlockedError):
        return RerankConnectError("Endpoint resolves to a disallowed address")
    if isinstance(exc, httpx.PoolTimeout):
        return RerankBusyError("Reranker busy (local connection pool exhausted)")
    if isinstance(exc, httpx.ConnectTimeout):
        return RerankTimeoutError(f"Connect timed out after {config.RERANK_CONNECT_TIMEOUT_SECONDS:g}s")
    if isinstance(exc, httpx.TimeoutException):
        return RerankTimeoutError(f"Reranker timed out after {_total():g}s")
    if isinstance(exc, httpx.DecodingError):
        return InvalidRerankResponseError("Response body could not be decoded")
    if isinstance(exc, httpx.InvalidURL):
        return RerankError("Invalid reranker endpoint URL", kind="config")
    if isinstance(exc, httpx.TransportError):
        return RerankConnectError("Could not connect to the reranker")
    return RerankLocalError("Reranker request failed")


def _headers(req) -> dict:
    # Identity encoding: the size cap is applied to the bytes as received, with no decompression amplification.
    return {**req.headers, "Accept-Encoding": "identity"}


def _check_status(response: httpx.Response) -> None:
    status = response.status_code
    if status in (401, 403):
        raise RerankAuthError(f"Authentication failed (HTTP {status}): check the API key.")
    if not 200 <= status < 300:
        raise RerankHTTPError(f"Reranker returned HTTP {status}", status_code=status)
    encoding = response.headers.get("content-encoding", "identity").strip().lower()
    if encoding not in ("", "identity"):
        raise InvalidRerankResponseError("Unsupported Content-Encoding in response")
    declared = response.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > config.RERANK_MAX_RESPONSE_BYTES:
                raise _too_large()
        except ValueError:
            pass  # malformed header: the streaming cap still applies


def _parse(adapter, target, status, raw: bytes, n_docs, top_n, start) -> list[RerankResult]:
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise InvalidRerankResponseError("Response body is not valid JSON") from None
    results = adapter.parse_response(status, body, n_docs, top_n)
    logger.info(
        "rerank.call service_id=%s app_id=%s provider=%s candidates=%d top_n=%d latency_ms=%d outcome=ok",
        target.service_id, target.app_id, target.provider, n_docs, top_n, _ms(start),
    )
    return results


def _needs_guard(target: RerankTarget) -> bool:
    """Fail-closed: only an explicit system-scoped service (no app) skips the SSRF guard."""
    if config.RERANK_ALLOW_PRIVATE_ENDPOINTS:
        return False
    return not (target.system and target.app_id is None)


def _prepare(target, query, documents, top_n):
    adapter = get_adapter(target.provider)
    try:
        req = adapter.build_request(target, query, documents, top_n)
    except ValueError as exc:  # invalid endpoint
        raise RerankError(str(exc), kind="config") from None
    logger.debug("rerank.request service_id=%s provider=%s candidates=%d top_n=%d",
                 target.service_id, target.provider, len(documents), top_n)
    return adapter, req


_warned_sites: set = set()


def _warn_running_loop_once() -> None:
    """Log the running-loop misuse once per call site (rate limit)."""
    frame = sys._getframe(2)
    site = (frame.f_code.co_filename, frame.f_lineno)
    if site in _warned_sites:
        return
    _warned_sites.add(site)
    logger.warning("rerank_sync called from a thread with a running event loop: it blocks that loop "
                   "(use rerank_async or asyncio.to_thread); further calls from %s:%s are not logged",
                   site[0], site[1])


def rerank_sync(target: RerankTarget, query: str, documents: list[str], top_n: int,
                caller_app_id=None) -> list[RerankResult]:
    """Blocking entry point: runs rerank_async on the single dedicated background loop (one implementation)."""
    loop = http_clients.get_background_loop()
    try:
        if asyncio.get_running_loop() is loop:
            raise RerankLocalError("rerank_sync must not be called from the rerank background loop")
    except RuntimeError:
        pass  # no running loop in this thread: the normal case
    else:
        _warn_running_loop_once()
    fut = asyncio.run_coroutine_threadsafe(rerank_async(target, query, documents, top_n, caller_app_id), loop)
    try:
        return fut.result(timeout=_total() + 0.5)
    except concurrent.futures.TimeoutError:
        fut.cancel()
        raise RerankTimeoutError(f"Reranker timed out after {_total():g}s") from None
    except concurrent.futures.CancelledError:
        raise RerankLocalError("Rerank call was cancelled") from None


async def rerank_async(target: RerankTarget, query: str, documents: list[str], top_n: int,
                       caller_app_id=None) -> list[RerankResult]:
    adapter, req = _prepare(target, query, documents, top_n)
    start = time.monotonic()
    cap = config.RERANK_MAX_RESPONSE_BYTES
    try:
        with _inflight_slot(target, caller_app_id):
            async with asyncio.timeout(_total()):
                client = http_clients.get_async_client(guarded=_needs_guard(target))
                async with client.stream(
                        "POST", req.url, headers=_headers(req), json=req.json,
                        timeout=http_clients.build_timeout()) as resp:
                    _check_status(resp)
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) > cap:
                            raise _too_large()
                    status = resp.status_code
    except RerankError:
        raise
    except TimeoutError:
        raise RerankTimeoutError(f"Reranker timed out after {_total():g}s") from None
    except Exception as exc:
        raise _map_exc(exc) from None
    return _parse(adapter, target, status, bytes(buf), len(documents), top_n, start)


async def test_rerank_connection(target: RerankTarget) -> dict:
    """Minimal rerank request; returns {"status", "message"}. Never includes the key."""
    try:
        results = await rerank_async(target, _TEST_QUERY, _TEST_DOCS, len(_TEST_DOCS))
    except RerankAuthError as exc:
        return {"status": "error", "message": exc.message}
    except RerankHTTPError as exc:
        return {"status": "error", "message": f"The reranker returned HTTP {exc.status_code}."}
    except (RerankTimeoutError, RerankBusyError) as exc:
        return {"status": "error", "message": f"{exc.message}."}
    except RerankConnectError as exc:
        return {"status": "error", "message": f"{exc.message}. Check the URL and network access."}
    except InvalidRerankResponseError as exc:
        return {"status": "error", "message": f"Invalid response from the reranker: {exc.message}."}
    except RerankError as exc:
        return {"status": "error", "message": exc.message}
    except Exception:
        logger.warning("rerank.test_connection unexpected failure service_id=%s", target.service_id)
        return {"status": "error", "message": "Unexpected error while testing the reranker."}
    return {"status": "success", "message": f"Connection successful ({len(results)} results)."}


test_rerank_connection.__test__ = False  # not a pytest test when imported into test modules
