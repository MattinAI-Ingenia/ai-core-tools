"""Shared httpx clients (AD-7): one async client per running event loop and SSRF policy; sync callers use one background loop."""
import asyncio
import logging
import threading
from typing import Optional

import httpx

import config

logger = logging.getLogger(__name__)

_LIMITS = httpx.Limits(max_connections=50, max_keepalive_connections=20)
_lock = threading.Lock()
_bg_lock = threading.Lock()
_bg_loop: Optional[asyncio.AbstractEventLoop] = None
_bg_thread: Optional[threading.Thread] = None
# Plain dict: keep-alive transports keep a dead loop alive anyway, so a weak map would never evict.
# Entries of closed loops are evicted explicitly in get_async_client().
_async_clients: dict[tuple[asyncio.AbstractEventLoop, bool], httpx.AsyncClient] = {}
_test_async: Optional[httpx.AsyncClient] = None


def build_timeout() -> httpx.Timeout:
    connect = config.RERANK_CONNECT_TIMEOUT_SECONDS
    total = max(config.RERANK_TIMEOUT_SECONDS, connect)
    return httpx.Timeout(total, connect=connect, pool=connect)


def _pool_kwargs() -> dict:
    # httpx ignores `limits` when a custom transport is given, so pass them to the transport explicitly.
    return dict(max_connections=_LIMITS.max_connections,
                max_keepalive_connections=_LIMITS.max_keepalive_connections,
                keepalive_expiry=_LIMITS.keepalive_expiry)


def get_background_loop() -> asyncio.AbstractEventLoop:
    """The single dedicated loop (daemon thread, lazy) that serves every synchronous rerank call."""
    global _bg_loop, _bg_thread
    with _bg_lock:
        if _bg_loop is None or _bg_loop.is_closed() or not _bg_thread.is_alive():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="rerank-sync-loop", daemon=True)
            thread.start()
            _bg_loop, _bg_thread = loop, thread
        return _bg_loop


def get_async_client(guarded: bool = False) -> httpx.AsyncClient:
    """Client for the running loop. `guarded` connects through the SSRF-pinned transport."""
    if _test_async is not None:
        return _test_async
    loop = asyncio.get_running_loop()
    with _lock:
        for key in [k for k in _async_clients if k[0].is_closed()]:
            _async_clients.pop(key, None)  # drop the reference; the dead loop's sockets are gone
            logger.debug("rerank: evicted async client of a closed event loop")
        client = _async_clients.get((loop, guarded))
        if client is None or client.is_closed:
            kwargs = dict(follow_redirects=False, limits=_LIMITS, trust_env=False)
            if guarded:
                from utils.ssrf_guard import pinned_async_transport
                kwargs["transport"] = pinned_async_transport(
                    connect_timeout_s=config.RERANK_CONNECT_TIMEOUT_SECONDS, **_pool_kwargs())
            client = httpx.AsyncClient(**kwargs)
            _async_clients[(loop, guarded)] = client
        return client


async def close_rerank_clients() -> None:
    """Close every shared client. Clients bound to other loops are closed on their own loop when alive."""
    with _lock:
        clients = list(_async_clients.items())
        _async_clients.clear()
    running = asyncio.get_running_loop()
    for (loop, _guarded), client in clients:
        if loop is running:
            await client.aclose()
        elif loop.is_running() and not loop.is_closed():
            try:
                fut = asyncio.run_coroutine_threadsafe(client.aclose(), loop)
                await asyncio.wait_for(asyncio.wrap_future(fut), timeout=2)
            except Exception as exc:  # best effort at shutdown
                logger.debug("rerank: closing a client on another loop failed: %s", type(exc).__name__)
        else:
            logger.debug("rerank: skipped closing a client of a stopped event loop")
    _stop_background_loop()


def _stop_background_loop() -> None:
    global _bg_loop, _bg_thread
    with _bg_lock:
        loop, thread, _bg_loop, _bg_thread = _bg_loop, _bg_thread, None, None
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        if not thread.is_alive():
            loop.close()


def _set_clients_for_tests(async_: Optional[httpx.AsyncClient] = None) -> None:
    global _test_async
    _test_async = async_
