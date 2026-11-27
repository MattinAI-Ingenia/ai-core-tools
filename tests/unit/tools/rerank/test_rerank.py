import asyncio
import gc
import http.server
import logging
import threading
import time
from unittest.mock import MagicMock
from types import SimpleNamespace

import httpx
import pytest

import config
from tools.rerank import http_clients
from tools.rerank.client import rerank_async, rerank_sync, test_rerank_connection
from tools.rerank.providers import PreparedRequest
from tools.rerank.endpoints import normalize_endpoint, rerank_url
from tools.rerank.errors import (
    InvalidRerankResponseError, RerankAuthError, RerankConnectError, RerankHTTPError,
    RerankTimeoutError, UnknownRerankProviderError,
)
from tools.rerank.providers import build_request, parse_response
from tools.rerank.types import RerankTarget

KEY = "sk-super-secret-key-123"
QUERY = "secret query text"
DOCS = ["alpha document body", "beta document body", "gamma document body"]


def make_target(api_key=KEY, provider="CohereCompatible", endpoint="http://rerank.test:7997"):
    return RerankTarget.from_service(SimpleNamespace(
        service_id=7, app_id=3, name="svc", provider=provider, endpoint=endpoint,
        model_name="m", api_key=api_key, last_error=None))


def install(handler):
    http_clients._set_clients_for_tests(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", True)  # tests use fake hosts / loopback
    yield
    http_clients._set_clients_for_tests(None)


def ok_handler(captured=None):
    def h(request):
        if captured is not None:
            captured.append(request)
        return httpx.Response(200, json={"results": [
            {"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}]})
    return h


def test_from_service_and_repr_hides_key():
    t = make_target()
    assert t.model == "m" and t.api_key == KEY
    assert KEY not in repr(t)


def test_dispatch_and_unknown_provider():
    assert build_request(make_target(), "q", ["a"], 1).url
    with pytest.raises(UnknownRerankProviderError):
        build_request(make_target(provider="Voyage"), "q", ["a"], 1)
    with pytest.raises(UnknownRerankProviderError):
        rerank_sync(make_target(provider="Nope"), "q", ["a"], 1)


def test_top_n_is_capped_to_the_number_of_documents():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    install(handler)
    rerank_sync(make_target(), QUERY, DOCS[:2], 50)
    import json as _j
    assert _j.loads(seen[0].content)["top_n"] == 2


def test_background_loop_unavailable_is_a_local_error(monkeypatch):
    from tools.rerank.errors import RerankLocalError
    monkeypatch.setattr(http_clients, "get_background_loop", MagicMock(side_effect=RuntimeError("closed")))
    with pytest.raises(RerankLocalError) as e:
        rerank_sync(make_target(), QUERY, DOCS, 2)
    assert e.value.service_failure is False and e.value.message == "Rerank background loop unavailable"


def test_request_body_exact_and_auth():
    seen = []
    install(ok_handler(seen))
    rerank_sync(make_target(), QUERY, DOCS, 2)
    req = seen[0]
    import json as _j
    body = _j.loads(req.content)
    assert set(body) == {"model", "query", "documents", "top_n"}
    assert body["documents"] == DOCS and all(isinstance(x, str) for x in body["documents"])
    assert body["top_n"] == 2 and str(req.url) == "http://rerank.test:7997/rerank"
    assert req.headers["authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("key", [None, "", "  ", "CHANGE_ME"])
def test_no_authorization_header(key):
    seen = []
    install(ok_handler(seen))
    rerank_sync(make_target(api_key=key), "q", DOCS, 2)
    assert "authorization" not in seen[0].headers


@pytest.mark.parametrize("raw,expected", [
    ("http://h:7997", "http://h:7997"),
    ("http://h:7997/", "http://h:7997"),
    ("  http://h:7997/rerank ", "http://h:7997"),
    ("http://h:7997/RERANK/", "http://h:7997"),
    ("https://api.cohere.com/v2", "https://api.cohere.com/v2"),
    ("https://api.jina.ai/v1/rerank", "https://api.jina.ai/v1"),
    ("http://vllm:8000/v1/", "http://vllm:8000/v1"),
])
def test_normalize_endpoint(raw, expected):
    assert normalize_endpoint(raw) == expected
    assert rerank_url(raw) == expected + "/rerank"


@pytest.mark.parametrize("raw", ["", "   ", "h:7997", "ftp://h/x", "http://", "/rerank", "http:///x"])
def test_normalize_endpoint_invalid(raw):
    with pytest.raises(ValueError):
        normalize_endpoint(raw)


@pytest.mark.parametrize("body", [
    [], {}, {"results": "x"}, {"results": [{"index": 5, "relevance_score": 1}]},
    {"results": [{"index": -1, "relevance_score": 1}]},
    {"results": [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 0.5}]},
    {"results": [{"index": 0}]}, {"results": [{"relevance_score": 1}]},
    {"results": [{"index": 0, "relevance_score": "high"}]},
    {"results": [{"index": 0, "relevance_score": 1}] * 4},
    {"results": [1]},
])
def test_invalid_responses(body):
    with pytest.raises(InvalidRerankResponseError):
        parse_response(200, body, 3, 3)


def test_sorting_stable_and_truncation():
    body = {"results": [
        {"index": 1, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5}]}
    res = parse_response(200, body, 3, 3)
    assert [r.index for r in res] == [0, 1, 2]
    assert [r.index for r in parse_response(200, body, 3, 2)] == [0, 1]
    assert len(parse_response(200, {"results": body["results"][:1]}, 3, 3)) == 1


def test_status_mapping_sync():
    install(lambda r: httpx.Response(401, text="bad key " + KEY))
    with pytest.raises(RerankAuthError) as e:
        rerank_sync(make_target(), "q", DOCS, 2)
    assert KEY not in e.value.message and e.value.kind == "auth"
    assert e.value.message == "Authentication failed (HTTP 401): check the API key."
    install(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(RerankHTTPError) as e:
        rerank_sync(make_target(), "q", DOCS, 2)
    assert e.value.status_code == 500 and "boom" not in e.value.message
    install(lambda r: httpx.Response(200, text="not json"))
    with pytest.raises(InvalidRerankResponseError):
        rerank_sync(make_target(), "q", DOCS, 2)


def test_transport_errors_mapped():
    def timeout(r):
        raise httpx.ReadTimeout("slow", request=r)

    def connect(r):
        raise httpx.ConnectError("refused", request=r)
    install(timeout)
    with pytest.raises(RerankTimeoutError):
        rerank_sync(make_target(), "q", DOCS, 2)
    install(connect)
    with pytest.raises(RerankConnectError):
        rerank_sync(make_target(), "q", DOCS, 2)


@pytest.mark.asyncio
async def test_async_ok_and_errors():
    install(ok_handler())
    res = await rerank_async(make_target(), "q", DOCS, 2)
    assert [r.index for r in res] == [2, 0]

    def connect(r):
        raise httpx.ConnectError("refused", request=r)
    install(connect)
    with pytest.raises(RerankConnectError):
        await rerank_async(make_target(), "q", DOCS, 2)
    install(lambda r: httpx.Response(403))
    with pytest.raises(RerankAuthError):
        await rerank_async(make_target(), "q", DOCS, 2)


@pytest.mark.asyncio
async def test_connection_messages_distinct_and_keyless():
    install(lambda r: httpx.Response(401, text=KEY))
    auth = await test_rerank_connection(make_target())
    install(lambda r: httpx.Response(503))
    http = await test_rerank_connection(make_target())

    def timeout(r):
        raise httpx.ConnectTimeout("t", request=r)
    install(timeout)
    to = await test_rerank_connection(make_target())
    install(lambda r: httpx.Response(200, json={"results": [{"index": 9, "relevance_score": 1}]}))
    bad = await test_rerank_connection(make_target())
    install(lambda r: httpx.Response(200, json={"results": [
        {"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}]}))
    good = await test_rerank_connection(make_target())
    assert good["status"] == "success"
    msgs = [auth, http, to, bad]
    assert all(m["status"] == "error" for m in msgs)
    assert len({m["message"] for m in msgs}) == 4
    assert "503" in http["message"] and "2s" in to["message"] and "out of range" in bad["message"]
    assert all(KEY not in m["message"] for m in msgs + [good])


@pytest.mark.asyncio
async def test_logs_never_contain_query_docs_or_key(caplog):
    caplog.set_level(logging.DEBUG)
    install(ok_handler())
    rerank_sync(make_target(), QUERY, DOCS, 2)
    await rerank_async(make_target(), QUERY, DOCS, 2)
    install(lambda r: httpx.Response(401, text=KEY))
    await test_rerank_connection(make_target())
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "rerank.call service_id=7 app_id=3 provider=CohereCompatible candidates=3 top_n=2" in text
    assert "outcome=ok" in text
    for secret in (KEY, QUERY, *DOCS):
        assert secret not in text


@pytest.mark.asyncio
async def test_hard_timeout_against_slow_server(monkeypatch):
    class Slow(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            time.sleep(2)
            try:
                self.send_response(200)
                self.end_headers()
            except OSError:
                pass

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "RERANK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(config, "RERANK_CONNECT_TIMEOUT_SECONDS", 0.3)
    http_clients._set_clients_for_tests(None)
    target = make_target(endpoint=f"http://127.0.0.1:{srv.server_address[1]}")
    start = time.monotonic()
    try:
        with pytest.raises(RerankTimeoutError):
            await rerank_async(target, "q", DOCS, 2)
        assert time.monotonic() - start < 0.8
    finally:
        await http_clients.close_rerank_clients()
        srv.shutdown()
        srv.server_close()


# ---- review-board additions ----

@pytest.mark.parametrize("raw", [
    "http://h/x?a=1", "http://h/x#frag", "http://u:p@h", "http://u@h", "http://h:99999", "http://h:abc", "http://h?",
])
def test_normalize_endpoint_rejects_query_fragment_credentials_port(raw):
    with pytest.raises(ValueError):
        normalize_endpoint(raw)


def test_normalize_keeps_path_and_lowercases_scheme():
    assert normalize_endpoint("HTTPS://Host:8443/a/b/rerank/") == "https://Host:8443/a/b"


def test_top_n_and_repr_and_overflow():
    from tools.rerank.errors import RerankError
    with pytest.raises(RerankError) as e:
        build_request(make_target(), "q", DOCS, 0)
    assert e.value.kind == "config"
    req = build_request(make_target(), QUERY, DOCS, 1)
    assert KEY not in repr(req) and QUERY not in repr(req)
    with pytest.raises(InvalidRerankResponseError):
        parse_response(200, {"results": [{"index": 0, "relevance_score": 10**400}]}, 1, 1)


def test_clients_ignore_env():
    http_clients._set_clients_for_tests(None)

    async def go():
        return http_clients.get_async_client()._trust_env
    assert asyncio.run(go()) is False
    http_clients._async_clients.clear()


def test_async_clients_evict_dead_loops():
    http_clients._set_clients_for_tests(None)
    for _ in range(3):
        async def go():
            http_clients.get_async_client()
        asyncio.run(go())
    gc.collect()
    # at most one stale entry (the last loop) can remain; it is evicted by the next call
    assert len(http_clients._async_clients) <= 1
    async def live():
        http_clients.get_async_client()
        return list(http_clients._async_clients)
    keys = asyncio.run(live())
    assert len(keys) == 1
    http_clients._async_clients.clear()


@pytest.mark.parametrize("exc,kind,fragment", [
    (httpx.PoolTimeout("p"), "busy", "busy"),
    (httpx.ConnectTimeout("c"), "timeout", "Connect timed out after 2s"),
    (httpx.ReadTimeout("r"), "timeout", "timed out after 5s"),
    (httpx.DecodingError("d"), "invalid_response", "decoded"),
    (httpx.InvalidURL("u"), "config", "URL"),
    (httpx.TooManyRedirects("t"), "config", "failed"),
])
def test_exception_mapping(exc, kind, fragment):
    def h(r):
        raise exc
    install(h)
    from tools.rerank.errors import RerankError
    with pytest.raises(RerankError) as e:
        rerank_sync(make_target(), "q", DOCS, 2)
    assert e.value.kind == kind and fragment in e.value.message
    # busy (local pool) and the unmapped fallback ("Reranker request failed", local) never count against the service
    assert e.value.service_failure is (kind != "busy" and fragment != "failed")


@pytest.fixture
def server():
    made = []

    def start(handler_cls):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        made.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"
    yield start
    for srv in made:
        srv.shutdown()
        srv.server_close()


class Trickle(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            self.send_response(200)
            self.end_headers()
            for _ in range(40):
                self.wfile.write(b" ")
                self.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass

    def log_message(self, *a):
        pass


class Big(http.server.BaseHTTPRequestHandler):
    declare = True

    def do_POST(self):
        try:
            self.send_response(200)
            if self.declare:
                self.send_header("Content-Length", "5000")
            self.end_headers()
            self.wfile.write(b" " * 5000)
        except OSError:
            pass

    def log_message(self, *a):
        pass


class BigNoLength(Big):
    declare = False


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(config, "RERANK_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(config, "RERANK_CONNECT_TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(config, "RERANK_MAX_RESPONSE_BYTES", 1000)
    http_clients._set_clients_for_tests(None)
    yield
    http_clients._set_clients_for_tests(None)


def test_sync_trickle_hits_total_deadline(server, small):
    t = make_target(endpoint=server(Trickle))
    start = time.monotonic()
    with pytest.raises(RerankTimeoutError):
        rerank_sync(t, "q", DOCS, 2)
    assert time.monotonic() - start < 1.0


@pytest.mark.asyncio
async def test_async_trickle_hits_total_deadline(server, small):
    t = make_target(endpoint=server(Trickle))
    start = time.monotonic()
    try:
        with pytest.raises(RerankTimeoutError):
            await rerank_async(t, "q", DOCS, 2)
        assert time.monotonic() - start < 0.8
    finally:
        await http_clients.close_rerank_clients()


@pytest.mark.parametrize("handler", [Big, BigNoLength])
@pytest.mark.asyncio
async def test_oversized_body_rejected(server, small, handler):
    t = make_target(endpoint=server(handler))
    with pytest.raises(InvalidRerankResponseError, match="too large"):
        rerank_sync(t, "q", DOCS, 2)
    try:
        with pytest.raises(InvalidRerankResponseError, match="too large"):
            await rerank_async(t, "q", DOCS, 2)
    finally:
        await http_clients.close_rerank_clients()


@pytest.mark.asyncio
async def test_ssrf_guard_blocks_loopback_for_app_targets(server, small, monkeypatch):
    monkeypatch.setattr(config, "RERANK_ALLOW_PRIVATE_ENDPOINTS", False)
    url = server(Big)
    with pytest.raises(RerankConnectError, match="disallowed"):
        rerank_sync(make_target(endpoint=url), "q", DOCS, 2)
    try:
        with pytest.raises(RerankConnectError, match="disallowed"):
            await rerank_async(make_target(endpoint=url), "q", DOCS, 2)
        # system targets (app_id None) keep the plain client and may reach loopback
        ns = SimpleNamespace(
            service_id=1, app_id=None, name="s", provider="CohereCompatible", endpoint=url, model_name="m")
        sys_t = RerankTarget.from_service(ns, system=True)
        with pytest.raises(InvalidRerankResponseError):
            await rerank_async(sys_t, "q", DOCS, 2)
        with pytest.raises(InvalidRerankResponseError):
            rerank_sync(sys_t, "q", DOCS, 2)
        # fail-closed: app_id None without an explicit system flag is guarded
        unflagged = RerankTarget.from_service(ns)
        assert unflagged.system is False
        with pytest.raises(RerankConnectError, match="disallowed"):
            rerank_sync(unflagged, "q", DOCS, 2)
        with pytest.raises(RerankConnectError, match="disallowed"):
            await rerank_async(unflagged, "q", DOCS, 2)
    finally:
        await http_clients.close_rerank_clients()


@pytest.mark.parametrize("raw,expected,warns", [
    (None, 2.0, False), ("", 2.0, False), ("1.5", 1.5, False), ("abc", 2.0, True), ("0", 2.0, True),
    ("-1", 2.0, True), ("nan", 2.0, True), ("inf", 2.0, True), ("11", 2.0, True), ("10", 10.0, False),
])
def test_env_float_or_default(monkeypatch, caplog, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("X_T", raising=False)
    else:
        monkeypatch.setenv("X_T", raw)
    with caplog.at_level(logging.WARNING):
        assert config._env_float_or_default("X_T", 2.0, maximum=10.0) == expected
    assert bool([r for r in caplog.records if "X_T" in r.getMessage()]) is warns


def test_system_flag_requires_real_system_row_and_no_app():
    from models.rerank_service import RerankService
    row = RerankService(name="s", provider="CohereCompatible", model_name="m", endpoint="http://x", app_id=None)
    assert RerankTarget.from_service(row).system is True
    row.app_id = 5
    assert RerankTarget.from_service(row).system is False
    assert RerankTarget.from_service(row, system=True).system is False  # an app-owned row can never be system


def test_guarded_client_applies_pool_limits():
    http_clients._set_clients_for_tests(None)

    async def go():
        pool = http_clients.get_async_client(guarded=True)._transport._pool
        return pool._max_connections, pool._max_keepalive_connections
    assert asyncio.run(go()) == (50, 20)
    http_clients._async_clients.clear()


def test_identity_encoding_requested_and_foreign_encoding_rejected():
    seen = []

    def h(r):
        seen.append(r.headers.get("accept-encoding"))
        return httpx.Response(200, headers={"content-encoding": "br"}, content=b"{}")
    install(h)
    with pytest.raises(InvalidRerankResponseError, match="Content-Encoding"):
        rerank_sync(make_target(), "q", DOCS, 2)
    assert seen == ["identity"]


class Drip(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            self.wfile.write(b"HTTP/1.0 200 OK\r\n")
            self.wfile.flush()
            for _ in range(20):
                time.sleep(0.4)
                self.wfile.write(b"X-Drip: 1\r\n")
                self.wfile.flush()
        except OSError:
            pass

    def log_message(self, *a):
        pass


@pytest.fixture
def one_second(monkeypatch):
    monkeypatch.setattr(config, "RERANK_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(config, "RERANK_CONNECT_TIMEOUT_SECONDS", 0.3)
    http_clients._set_clients_for_tests(None)
    yield
    http_clients._set_clients_for_tests(None)


def test_acceptance_header_drip_sync(server, one_second):
    t = make_target(endpoint=server(Drip))
    start = time.monotonic()
    with pytest.raises(RerankTimeoutError):
        rerank_sync(t, "q", DOCS, 2)
    assert time.monotonic() - start < 1.3


@pytest.mark.asyncio
async def test_acceptance_header_drip_async(server, one_second):
    t = make_target(endpoint=server(Drip))
    start = time.monotonic()
    try:
        with pytest.raises(RerankTimeoutError):
            await rerank_async(t, "q", DOCS, 2)
        assert time.monotonic() - start < 1.3
    finally:
        await http_clients.close_rerank_clients()


def test_gzip_bomb_rejected_without_buffering(server, monkeypatch):
    import gzip
    import tracemalloc
    bomb = gzip.compress(b"\0" * (300 * 1024 * 1024), compresslevel=9)
    assert len(bomb) < 400 * 1024

    class Bomb(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(bomb)))
                self.end_headers()
                self.wfile.write(bomb)
            except OSError:
                pass

        def log_message(self, *a):
            pass
    monkeypatch.setattr(config, "RERANK_MAX_RESPONSE_BYTES", 100_000)
    http_clients._set_clients_for_tests(None)
    t = make_target(endpoint=server(Bomb))
    tracemalloc.start()
    try:
        with pytest.raises(InvalidRerankResponseError, match="Content-Encoding"):
            rerank_sync(t, "q", DOCS, 2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20 * 1024 * 1024  # a 300 MB inflation would dwarf this


def test_background_loop_single_and_sync_from_loop_rejected():
    loop = http_clients.get_background_loop()
    assert http_clients.get_background_loop() is loop
    install(ok_handler())

    async def inner():
        return rerank_sync(make_target(), "q", DOCS, 2)
    from tools.rerank.errors import RerankError
    with pytest.raises(RerankError):
        asyncio.run_coroutine_threadsafe(inner(), loop).result(5)


@pytest.mark.parametrize("raw,expected,warns", [
    (None, 1048576, False), ("2048", 2048, False), ("abc", 1048576, True), ("0", 1048576, True), ("-5", 1048576, True),
])
def test_env_int_or_default(monkeypatch, caplog, raw, expected, warns):
    if raw is None:
        monkeypatch.delenv("X_I", raising=False)
    else:
        monkeypatch.setenv("X_I", raw)
    with caplog.at_level(logging.WARNING):
        assert config._env_int_or_default("X_I", 1048576) == expected
    assert bool([r for r in caplog.records if "X_I" in r.getMessage()]) is warns
