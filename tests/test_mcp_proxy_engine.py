"""Characterization and reliability tests for ``dataquery.mcp_proxy``.

The proxy runs against an in-process aiohttp server that plays both the remote
MCP endpoint and its OAuth token endpoint, so no real network is used. Every
URL (MCP and token) points at that fake server. ``run_mcp_proxy`` is driven by
replacing ``_read_stdin_lines`` with a fed ``asyncio.StreamReader`` and
``_emit_stdout`` with a list, so whole sessions run in-process.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
import json
import os
import socket
import sys
import threading
import time
import types
import typing
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import aiohttp
import pytest
import pytest_asyncio
import structlog
from aiohttp import web
from aiohttp.test_utils import TestServer

import dataquery
from dataquery import mcp_proxy
from dataquery.mcp_proxy import StreamableHttpProxy, run_mcp_proxy
from dataquery.transport.auth import OAuthManager
from dataquery.types.exceptions import AuthenticationError, ConfigurationError
from dataquery.types.models import ClientConfig

pytestmark = pytest.mark.asyncio

SESSION_ID = "sess-1"
PROTOCOL_VERSION = "2025-06-18"
_REAL_SLEEP = asyncio.sleep
_client_ids = itertools.count()

PostHook = Callable[[web.Request, Any], Awaitable[web.StreamResponse | None]]
GetHook = Callable[[web.Request], Awaitable[web.StreamResponse]]


# --------------------------------------------------------------------------- #
# Fake MCP + OAuth server
# --------------------------------------------------------------------------- #
class FakeMcp:
    """Remote MCP endpoint plus token endpoint, with opt-in hooks per test."""

    def __init__(self) -> None:
        self.token_calls = 0
        self.token_status: int | None = None
        self.revoked: set[str] = set()
        self.posts: list[dict[str, Any]] = []
        self.gets: list[dict[str, str]] = []
        self.deletes: list[dict[str, str]] = []
        self.post_hook: PostHook | None = None
        self.get_hook: GetHook | None = None
        self.delete_gate: asyncio.Event | None = None
        self.release = asyncio.Event()
        self.server: TestServer | None = None

    @property
    def base_url(self) -> str:
        assert self.server is not None
        return str(self.server.make_url("")).rstrip("/")

    @property
    def mcp_url(self) -> str:
        return f"{self.base_url}/mcp"

    @property
    def token_url(self) -> str:
        return f"{self.base_url}/oauth/token"

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/oauth/token", self._token)
        app.router.add_post("/mcp", self._post)
        app.router.add_get("/mcp", self._get)
        app.router.add_delete("/mcp", self._delete)
        return app

    async def _token(self, request: web.Request) -> web.Response:
        self.token_calls += 1
        if self.token_status is not None:
            return web.Response(status=self.token_status)
        return web.json_response({"access_token": f"tok{self.token_calls}", "token_type": "Bearer", "expires_in": 3600})

    async def _post(self, request: web.Request) -> web.StreamResponse:
        message = await request.json()
        self.posts.append({"headers": dict(request.headers), "body": message})
        if request.headers.get("Authorization", "") in self.revoked:
            return web.Response(status=401)
        if self.post_hook is not None:
            resp = await self.post_hook(request, message)
            if resp is not None:
                return resp
        if not isinstance(message, dict):
            return web.json_response({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "batch"}})
        msg_id = message.get("id")
        method = message.get("method")
        if msg_id is None:
            return web.Response(status=202)
        if method == "initialize":
            return web.json_response(
                {"jsonrpc": "2.0", "id": msg_id, "result": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}}},
                headers={"Mcp-Session-Id": SESSION_ID},
            )
        if method == "sse":
            note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
            reply = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": []}}
            return await sse(request, [f"data: {json.dumps(note)}\n\n", f"data: {json.dumps(reply)}\n\n"])
        if method == "slow":
            await self.release.wait()
        return web.json_response({"jsonrpc": "2.0", "id": msg_id, "result": {"echo": method}})

    async def _get(self, request: web.Request) -> web.StreamResponse:
        self.gets.append(dict(request.headers))
        if self.get_hook is not None:
            return await self.get_hook(request)
        return web.Response(status=405)

    async def _delete(self, request: web.Request) -> web.Response:
        self.deletes.append(dict(request.headers))
        if self.delete_gate is not None:
            await self.delete_gate.wait()
        return web.Response(status=204)


async def sse(
    request: web.Request,
    chunks: list[str | bytes],
    *,
    drop: bool = False,
    pause: float = 0.0,
) -> web.StreamResponse:
    """Stream ``chunks`` as an SSE body; ``drop`` cuts the connection mid-body."""
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for chunk in chunks:
        await resp.write(chunk.encode() if isinstance(chunk, str) else chunk)
        if pause:
            await _REAL_SLEEP(pause)
    if drop:
        assert request.transport is not None
        request.transport.close()
        return resp
    await resp.write_eof()
    return resp


def event(payload: Any) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def reply(msg_id: Any, **result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def request(msg_id: Any, method: str = "ping", **params: Any) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params:
        msg["params"] = params
    return msg


def initialize(msg_id: int = 1) -> dict[str, Any]:
    return request(msg_id, "initialize", protocolVersion=PROTOCOL_VERSION, capabilities={})


INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def closed_port_url(path: str = "/mcp") -> str:
    """A localhost URL nothing listens on (connection refused, no real network)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}{path}"


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before the deadline")
        await _REAL_SLEEP(0.01)


def errors_for(emitted: list[Any], msg_id: Any) -> list[dict[str, Any]]:
    return [m for m in emitted if isinstance(m, dict) and m.get("id") == msg_id and "error" in m]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _isolate_env_and_logging(tmp_path: Path) -> Iterator[None]:
    """Isolate DATAQUERY_* env vars and restore the structlog config afterwards."""
    saved_logging = structlog.get_config()
    with patch.dict(os.environ):
        for key in [k for k in os.environ if k.startswith("DATAQUERY_")]:
            del os.environ[key]
        os.environ["DATAQUERY_CONFIG_DIR"] = str(tmp_path / "home")
        yield
    structlog.configure(**saved_logging)


@pytest_asyncio.fixture
async def fake() -> typing.AsyncIterator[FakeMcp]:
    server = FakeMcp()
    server.server = TestServer(server.app())
    await server.server.start_server()
    try:
        yield server
    finally:
        server.release.set()
        if server.delete_gate is not None:
            server.delete_gate.set()
        await server.server.close()


@pytest_asyncio.fixture
async def http() -> typing.AsyncIterator[aiohttp.ClientSession]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30)) as session:
        yield session


@pytest.fixture
def make_proxy(
    fake: FakeMcp, http: aiohttp.ClientSession, tmp_path: Path
) -> Callable[..., tuple[StreamableHttpProxy, list[Any]]]:
    def _make(*, bearer: str | None = None, url: str | None = None) -> tuple[StreamableHttpProxy, list[Any]]:
        n = next(_client_ids)
        fields: dict[str, Any] = {
            "base_url": fake.base_url,
            "oauth_token_url": fake.token_url,
            "download_dir": str(tmp_path),
            "token_storage_enabled": True,
            "token_storage_dir": str(tmp_path / f"tokens{n}"),
        }
        if bearer is not None:
            fields["bearer_token"] = bearer
        else:
            fields["client_id"] = f"cid{n}"
            fields["client_secret"] = "secret"
        emitted: list[Any] = []
        proxy = StreamableHttpProxy(
            url or fake.mcp_url, OAuthManager(ClientConfig.model_validate(fields)), http, emitted.append
        )
        return proxy, emitted

    return _make


@pytest.fixture
def proxy_env(fake: FakeMcp, tmp_path: Path) -> dict[str, str]:
    env = {
        "DATAQUERY_MCP_URL": fake.mcp_url,
        "DATAQUERY_OAUTH_TOKEN_URL": fake.token_url,
        "DATAQUERY_CLIENT_ID": f"cid{next(_client_ids)}",
        "DATAQUERY_CLIENT_SECRET": "secret",
        "DATAQUERY_TOKEN_STORAGE_ENABLED": "true",
        "DATAQUERY_TOKEN_STORAGE_DIR": str(tmp_path / "proxy-tokens"),
    }
    os.environ.update(env)
    return env


class Stdin:
    """Stands in for the proxy's stdin; lines are fed while it runs."""

    def __init__(self, limit: int | None = None) -> None:
        self.reader = asyncio.StreamReader() if limit is None else asyncio.StreamReader(limit=limit)

    def send(self, message: Any) -> None:
        self.raw((json.dumps(message) + "\n").encode())

    def raw(self, data: bytes) -> None:
        self.reader.feed_data(data)

    def eof(self) -> None:
        self.reader.feed_eof()


class Session:
    """A running ``run_mcp_proxy`` with its fed stdin and captured stdout."""

    def __init__(self, stdin: Stdin, emitted: list[Any], task: asyncio.Task[int]) -> None:
        self.stdin = stdin
        self.emitted = emitted
        self.task = task

    async def finish(self, timeout: float = 10.0) -> int:
        self.stdin.eof()
        return await asyncio.wait_for(self.task, timeout)

    async def answered(self, msg_id: Any) -> dict[str, Any]:
        await wait_until(lambda: any(isinstance(m, dict) and m.get("id") == msg_id for m in self.emitted))
        return next(m for m in self.emitted if isinstance(m, dict) and m.get("id") == msg_id)


@pytest_asyncio.fixture
async def start(monkeypatch: pytest.MonkeyPatch) -> typing.AsyncIterator[Callable[..., Session]]:
    tasks: list[asyncio.Task[int]] = []

    def _start(*, url: str | None = None, env_file: Path | None = None, limit: int | None = None) -> Session:
        stdin = Stdin(limit)
        emitted: list[Any] = []

        async def _fake_stdin() -> asyncio.StreamReader:
            return stdin.reader

        monkeypatch.setattr(mcp_proxy, "_read_stdin_lines", _fake_stdin)
        monkeypatch.setattr(mcp_proxy, "_emit_stdout", emitted.append)
        task = asyncio.create_task(run_mcp_proxy(url=url, env_file=env_file))
        tasks.append(task)
        return Session(stdin, emitted, task)

    yield _start
    # A failed test can leave the proxy running or its exception unretrieved.
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class SleepRecorder:
    """Stands in for ``mcp_proxy.asyncio``; records sleeps, stops after ``limit``."""

    def __init__(self, limit: int = 10) -> None:
        self.sleeps: list[float] = []
        self.limit = limit

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        if len(self.sleeps) >= self.limit:
            raise _StopLoop
        await _REAL_SLEEP(0)


class _StopLoop(Exception):
    pass


# --------------------------------------------------------------------------- #
# Public API pins
# --------------------------------------------------------------------------- #
def _norm(tp: Any) -> Any:
    """Normalise ``Dict``/``dict`` and ``typing``/``collections.abc`` spellings."""
    if tp is None:  # collections.abc.Callable keeps a bare None return
        return type(None)
    origin = typing.get_origin(tp)
    if origin is None:
        return tp
    args = typing.get_args(tp)
    if origin is typing.Union or origin is types.UnionType:
        return ("union", frozenset(_norm(a) for a in args))
    if args and isinstance(args[0], list):  # Callable[[params], ret]
        return (origin, tuple(_norm(a) for a in args[0]), _norm(args[1]))
    return (origin, tuple(_norm(a) for a in args))


def _hints(obj: Any) -> dict[str, Any]:
    return {k: _norm(v) for k, v in typing.get_type_hints(obj).items()}


async def test_public_names_are_unchanged() -> None:
    defined = {
        name
        for name, value in vars(mcp_proxy).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == mcp_proxy.__name__
    }
    assert defined == {"StreamableHttpProxy", "run_mcp_proxy"}
    assert not hasattr(mcp_proxy, "__all__")
    assert "StreamableHttpProxy" not in dataquery.__all__
    assert "run_mcp_proxy" not in dataquery.__all__
    assert mcp_proxy._SESSION_HEADER == "Mcp-Session-Id"
    assert mcp_proxy._PROTOCOL_HEADER == "MCP-Protocol-Version"
    assert mcp_proxy._GET_BACKOFF_INITIAL == 1.0
    assert mcp_proxy._GET_BACKOFF_MAX == 30.0


async def test_public_signatures_are_unchanged() -> None:
    def shape(fn: Any) -> list[tuple[str, Any, Any]]:
        return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]

    pk = inspect.Parameter.POSITIONAL_OR_KEYWORD
    empty = inspect.Parameter.empty
    assert shape(StreamableHttpProxy.__init__) == [
        ("self", pk, empty),
        ("url", pk, empty),
        ("oauth", pk, empty),
        ("session", pk, empty),
        ("emit", pk, empty),
    ]
    assert shape(StreamableHttpProxy.handle_message) == [("self", pk, empty), ("message", pk, empty)]
    assert shape(StreamableHttpProxy.listen_get_stream) == [("self", pk, empty)]
    assert shape(StreamableHttpProxy.close) == [("self", pk, empty)]
    assert shape(run_mcp_proxy) == [("url", pk, None), ("env_file", pk, None)]
    for fn in (
        StreamableHttpProxy.handle_message,
        StreamableHttpProxy.listen_get_stream,
        StreamableHttpProxy.close,
        run_mcp_proxy,
        mcp_proxy._read_stdin_lines,
    ):
        assert inspect.iscoroutinefunction(fn), fn
    # Seams used by tests and callers that drive the proxy in-process.
    assert shape(mcp_proxy._read_stdin_lines) == []
    assert shape(mcp_proxy._emit_stdout) == [("message", pk, empty)]
    assert shape(mcp_proxy._configure_stderr_logging) == []
    assert shape(mcp_proxy._build_oauth_manager) == [("url", pk, empty), ("env_file", pk, empty)]


async def test_public_type_hints_are_unchanged() -> None:
    json_obj = (dict, (str, Any))
    init = _hints(StreamableHttpProxy.__init__)
    assert set(init) == {"url", "oauth", "session", "emit"}
    assert init["url"] is str
    assert init["oauth"] is OAuthManager
    assert init["session"] is aiohttp.ClientSession
    emit_origin, emit_args, emit_ret = init["emit"]
    assert emit_origin is typing.get_origin(Callable[[], None])
    assert emit_args == (json_obj,)
    assert emit_ret is type(None)
    assert _hints(StreamableHttpProxy.handle_message) == {"message": json_obj, "return": type(None)}
    assert _hints(StreamableHttpProxy.listen_get_stream) == {"return": type(None)}
    assert _hints(StreamableHttpProxy.close) == {"return": type(None)}
    assert _hints(run_mcp_proxy) == {
        "url": ("union", frozenset({str, type(None)})),
        "env_file": ("union", frozenset({Path, type(None)})),
        "return": int,
    }
    assert typing.get_type_hints(mcp_proxy._read_stdin_lines)["return"] is asyncio.StreamReader


async def test_instance_attributes(make_proxy: Callable[..., Any], fake: FakeMcp, http: aiohttp.ClientSession) -> None:
    proxy, emitted = make_proxy()
    assert proxy.url == fake.mcp_url
    assert isinstance(proxy.oauth, OAuthManager)
    assert proxy.session is http
    assert proxy.emit == emitted.append
    assert proxy.session_id is None
    assert proxy.protocol_version is None


# --------------------------------------------------------------------------- #
# handle_message: today's behaviour
# --------------------------------------------------------------------------- #
async def test_initialize_sends_auth_and_accept_and_captures_session(
    make_proxy: Callable[..., Any], fake: FakeMcp
) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())

    first = fake.posts[0]["headers"]
    assert first["Authorization"] == "Bearer tok1"
    assert first["Accept"] == "application/json, text/event-stream"
    assert first["Content-Type"] == "application/json"
    assert "Mcp-Session-Id" not in first
    assert "MCP-Protocol-Version" not in first
    assert emitted == [reply(1, protocolVersion=PROTOCOL_VERSION, capabilities={})]
    assert (proxy.session_id, proxy.protocol_version) == (SESSION_ID, PROTOCOL_VERSION)

    await proxy.handle_message(request(2))
    later = fake.posts[1]["headers"]
    assert later["Mcp-Session-Id"] == SESSION_ID
    assert later["MCP-Protocol-Version"] == PROTOCOL_VERSION
    assert emitted[-1] == reply(2, echo="ping")
    assert fake.token_calls == 1


async def test_session_id_is_replaced_when_the_server_issues_a_new_one(
    make_proxy: Callable[..., Any], fake: FakeMcp
) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.json_response(reply(msg["id"]), headers={"Mcp-Session-Id": f"s{msg['id']}"})

    fake.post_hook = hook
    proxy, _ = make_proxy()
    await proxy.handle_message(request(1))
    await proxy.handle_message(request(2))
    assert proxy.session_id == "s2"
    assert fake.posts[1]["headers"]["Mcp-Session-Id"] == "s1"


async def test_protocol_version_only_comes_from_an_object_result(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    bodies = iter(
        [[reply(1, protocolVersion="x")], reply(2, protocolVersion=""), {"jsonrpc": "2.0", "id": 3, "result": "v"}]
    )

    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.json_response(next(bodies))

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    for i in (1, 2, 3):
        await proxy.handle_message(request(i))
    assert proxy.protocol_version is None
    assert emitted[0] == [reply(1, protocolVersion="x")]  # non-object payloads pass through unchanged
    assert len(emitted) == 3


async def test_json_body_without_content_type_is_parsed(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(body=json.dumps(reply(msg["id"], ok=True)).encode(), content_type="text/plain")

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(4))
    assert emitted == [reply(4, ok=True)]


@pytest.mark.parametrize("status", [202, 204])
async def test_accepted_without_body_emits_nothing(make_proxy: Callable[..., Any], fake: FakeMcp, status: int) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=status, headers={"Mcp-Session-Id": "ignored"})

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(1))
    assert emitted == []
    assert proxy.session_id is None  # only captured from a 2xx with a body


async def test_notification_emits_nothing(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(INITIALIZED)
    assert emitted == []
    assert fake.posts[0]["body"] == INITIALIZED


@pytest.mark.parametrize("status", [400, 403, 500, 503])
async def test_http_error_becomes_jsonrpc_error(make_proxy: Callable[..., Any], fake: FakeMcp, status: int) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=status, text="x" * 2000)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(7))
    assert emitted == [
        {"jsonrpc": "2.0", "id": 7, "error": {"code": -32001, "message": f"MCP endpoint returned {status}"}}
    ]


async def test_error_for_id_zero_is_still_answered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=500)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(0))
    assert errors_for(emitted, 0)


async def test_http_error_for_a_notification_is_not_answered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=500)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(INITIALIZED)
    assert emitted == []


async def test_404_with_a_session_reports_expiry(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())

    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=404)

    fake.post_hook = hook
    await proxy.handle_message(request(8))
    assert emitted[-1]["error"] == {"code": -32001, "message": "MCP session expired; restart the connection"}
    assert proxy.session_id == SESSION_ID


async def test_404_without_a_session_is_a_plain_http_error(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=404)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(8))
    assert emitted[-1]["error"]["message"] == "MCP endpoint returned 404"


async def test_401_refreshes_the_token_and_retries_once(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())
    fake.revoked.add("Bearer tok1")

    await proxy.handle_message(request(2))
    assert [p["headers"]["Authorization"] for p in fake.posts] == ["Bearer tok1", "Bearer tok1", "Bearer tok2"]
    assert fake.token_calls == 2
    assert emitted[-1] == reply(2, echo="ping")


async def test_second_401_is_reported(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(status=401)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(3))
    assert len(fake.posts) == 2
    assert emitted == [{"jsonrpc": "2.0", "id": 3, "error": {"code": -32001, "message": "MCP endpoint returned 401"}}]


async def test_bearer_token_is_sent_without_token_calls(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy(bearer="static-token")
    await proxy.handle_message(request(1))
    assert fake.posts[0]["headers"]["Authorization"] == "Bearer static-token"
    assert fake.token_calls == 0
    assert emitted == [reply(1, echo="ping")]


async def test_unreachable_endpoint_becomes_jsonrpc_error(make_proxy: Callable[..., Any]) -> None:
    proxy, emitted = make_proxy(url=closed_port_url())
    await proxy.handle_message(request(5))
    assert emitted[0]["id"] == 5
    assert emitted[0]["error"]["code"] == -32001
    assert emitted[0]["error"]["message"].startswith("MCP endpoint unreachable: ")


# --------------------------------------------------------------------------- #
# SSE parsing
# --------------------------------------------------------------------------- #
async def test_sse_response_emits_every_event_in_order(make_proxy: Callable[..., Any]) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(2, "sse"))
    assert [m.get("method", "reply") for m in emitted] == ["notifications/progress", "reply"]
    assert emitted[-1] == reply(2, tools=[])


async def test_sse_framing_rules(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(
            req,
            [
                ": comment line\n",
                "event: message\nid: 1\nretry: 10\n",
                'data:{"jsonrpc":"2.0",\n',
                'data:   "method":"a"}\n\n',  # multi-line data, joined with "\n"; leading spaces stripped
                "data: not json\n\n",  # skipped
                '\r\ndata: {"jsonrpc":"2.0","method":"b"}\r\n\r\n',  # CRLF line endings
                "\n\n",  # blank lines with no data dispatch nothing
                'data: {"jsonrpc":"2.0","method":"c"}\n',  # no blank line before EOF: dropped
            ],
        )

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(1))
    assert emitted == [{"jsonrpc": "2.0", "method": "a"}, {"jsonrpc": "2.0", "method": "b"}]


async def test_sse_events_split_across_chunks_and_multibyte_text(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    text = "caf\u00e9 \u20ac \U0001f600"
    body = f"data: {json.dumps(reply(1, text=text), ensure_ascii=False)}\n\n".encode()
    euro = body.index("\u20ac".encode())
    # Split inside the "data:" field name, inside a multi-byte character and inside the "\n\n".
    pieces = [body[:2], body[2 : euro + 1], body[euro + 1 : euro + 2], body[euro + 2 : -1], body[-1:]]

    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(req, list(pieces), pause=0.01)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(1))
    assert emitted == [reply(1, text=text)]


async def test_sse_protocol_version_is_captured(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(req, [event(reply(msg["id"], protocolVersion="2099-01-01"))])

    fake.post_hook = hook
    proxy, _ = make_proxy()
    await proxy.handle_message(request(1))
    assert proxy.protocol_version == "2099-01-01"


# --------------------------------------------------------------------------- #
# listen_get_stream and close
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [404, 405, 501])
async def test_get_stream_not_offered_returns(make_proxy: Callable[..., Any], fake: FakeMcp, status: int) -> None:
    async def hook(req: web.Request) -> web.StreamResponse:
        return web.Response(status=status)

    fake.get_hook = hook
    proxy, emitted = make_proxy()
    await asyncio.wait_for(proxy.listen_get_stream(), 5)
    assert len(fake.gets) == 1
    assert emitted == []


@pytest.mark.parametrize("status", [400, 401, 429, 503])
async def test_get_stream_gives_up_on_other_errors_today(
    make_proxy: Callable[..., Any], fake: FakeMcp, status: int
) -> None:
    async def hook(req: web.Request) -> web.StreamResponse:
        return web.Response(status=status)

    fake.get_hook = hook
    proxy, _ = make_proxy()
    await asyncio.wait_for(proxy.listen_get_stream(), 5)
    assert len(fake.gets) == 1


async def test_get_stream_pumps_events_and_reconnects_with_backoff(
    make_proxy: Callable[..., Any], fake: FakeMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SleepRecorder(limit=6)

    async def hook(req: web.Request) -> web.StreamResponse:
        # aiohttp may transparently retry a dropped GET, so phase on the sleeps seen.
        phase = len(recorder.sleeps)
        if phase in (0, 3):
            return await sse(req, [event({"jsonrpc": "2.0", "method": f"server/{phase}"})])
        assert req.transport is not None
        req.transport.close()  # dropped connection -> aiohttp.ClientError
        return web.Response()

    fake.get_hook = hook
    monkeypatch.setattr(mcp_proxy, "asyncio", recorder)
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())
    with pytest.raises(_StopLoop):
        await proxy.listen_get_stream()

    assert [m["method"] for m in emitted[1:]] == ["server/0", "server/3"]
    # success resets the delay; each drop doubles it.
    assert recorder.sleeps == [1.0, 2.0, 4.0, 1.0, 2.0, 4.0]
    assert fake.gets[0]["Mcp-Session-Id"] == SESSION_ID
    assert fake.gets[0]["MCP-Protocol-Version"] == PROTOCOL_VERSION
    assert fake.gets[0]["Accept"] == "application/json, text/event-stream"


async def test_get_stream_backoff_is_capped(
    make_proxy: Callable[..., Any], fake: FakeMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def hook(req: web.Request) -> web.StreamResponse:
        assert req.transport is not None
        req.transport.close()
        return web.Response()

    fake.get_hook = hook
    recorder = SleepRecorder(limit=8)
    monkeypatch.setattr(mcp_proxy, "asyncio", recorder)
    proxy, _ = make_proxy()
    with pytest.raises(_StopLoop):
        await proxy.listen_get_stream()
    assert recorder.sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]


async def test_get_stream_token_failure_raises_today(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    fake.token_status = 500
    proxy, _ = make_proxy()
    with pytest.raises(AuthenticationError):
        await proxy.listen_get_stream()


async def test_close_deletes_the_session(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, _ = make_proxy()
    await proxy.handle_message(initialize())
    await proxy.close()
    assert len(fake.deletes) == 1
    assert fake.deletes[0]["Mcp-Session-Id"] == SESSION_ID
    assert fake.deletes[0]["Authorization"] == "Bearer tok1"


async def test_close_without_a_session_sends_nothing(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, _ = make_proxy()
    await proxy.close()
    assert fake.deletes == []


async def test_close_swallows_errors(make_proxy: Callable[..., Any]) -> None:
    proxy, _ = make_proxy(url=closed_port_url())
    proxy.session_id = "s"
    await proxy.close()


# --------------------------------------------------------------------------- #
# run_mcp_proxy: today's behaviour
# --------------------------------------------------------------------------- #
async def test_full_session_round_trip(start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]) -> None:
    s = start()
    s.stdin.send(initialize())
    assert (await s.answered(1))["result"]["protocolVersion"] == PROTOCOL_VERSION
    s.stdin.raw(b"\n   \n")  # blank lines are ignored
    s.stdin.raw(b"{not json\n")  # malformed lines are skipped
    s.stdin.send(INITIALIZED)
    s.stdin.send(INITIALIZED)  # the GET stream is only opened once
    s.stdin.send(request(2, "sse"))
    await s.answered(2)
    await wait_until(lambda: len(fake.gets) == 1)

    assert await s.finish() == 0
    assert [p["body"].get("method") for p in fake.posts] == [
        "initialize",
        "notifications/initialized",
        "notifications/initialized",
        "sse",
    ]
    assert len(fake.gets) == 1
    assert len(fake.deletes) == 1
    assert fake.deletes[0]["Mcp-Session-Id"] == SESSION_ID


async def test_url_defaults_to_the_mcp_url_env_var(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]
) -> None:
    s = start()
    s.stdin.send(request(1))
    await s.answered(1)
    assert await s.finish() == 0
    assert len(fake.posts) == 1


async def test_base_url_is_never_derived_from_the_mcp_url(fake: FakeMcp, proxy_env: dict[str, str]) -> None:
    # get_env_var("BASE_URL") falls back to the model default, so the
    # derivation branch in _build_oauth_manager never runs (see findings N3).
    oauth = mcp_proxy._build_oauth_manager(fake.mcp_url, None)
    assert "DATAQUERY_BASE_URL" not in os.environ
    assert oauth.config.base_url == ClientConfig.model_fields["base_url"].default


async def test_env_file_supplies_the_configuration(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str], tmp_path: Path
) -> None:
    env_file = tmp_path / "proxy.env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in proxy_env.items()) + "\n")
    for key in proxy_env:
        del os.environ[key]
    s = start(env_file=env_file)
    s.stdin.send(request(1))
    await s.answered(1)
    assert await s.finish() == 0


async def test_requests_run_concurrently(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]
) -> None:
    s = start()
    s.stdin.send(request(1, "slow"))
    s.stdin.send(request(2))
    await s.answered(2)
    assert all(m["id"] == 2 for m in s.emitted)
    fake.release.set()
    await s.answered(1)
    assert await s.finish() == 0


async def test_missing_credentials_exit_1(
    start: Callable[..., Session], fake: FakeMcp, capsys: pytest.CaptureFixture[str]
) -> None:
    s = start(url=fake.mcp_url)
    assert await asyncio.wait_for(s.task, 10) == 1
    assert "dataquery mcp connect: No credentials configured" in capsys.readouterr().err
    assert fake.posts == []


async def test_explicitly_empty_mcp_url_exits_1(
    start: Callable[..., Session], proxy_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    os.environ["DATAQUERY_MCP_URL"] = ""
    s = start(url="")
    assert await asyncio.wait_for(s.task, 10) == 1
    assert "No MCP endpoint configured" in capsys.readouterr().err


async def test_rejected_credentials_exit_1(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    fake.token_status = 401
    s = start()
    assert await asyncio.wait_for(s.task, 10) == 1
    assert "dataquery mcp connect: " in capsys.readouterr().err


async def test_drain_is_bounded(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_proxy, "_DRAIN_TIMEOUT", 0.2, raising=False)
    s = start()
    s.stdin.send(request(1, "slow"))  # never released
    await wait_until(lambda: len(fake.posts) == 1)
    assert await s.finish(timeout=5) == 0
    assert s.emitted == []


async def test_build_oauth_manager_requires_credentials(fake: FakeMcp) -> None:
    with pytest.raises(ConfigurationError, match="No credentials configured"):
        mcp_proxy._build_oauth_manager(fake.mcp_url, None)


async def test_build_oauth_manager_keeps_an_explicit_base_url(fake: FakeMcp, proxy_env: dict[str, str]) -> None:
    os.environ["DATAQUERY_BASE_URL"] = "https://explicit.example"
    oauth = mcp_proxy._build_oauth_manager(fake.mcp_url, None)
    assert oauth.config.base_url == "https://explicit.example"
    assert oauth.is_authenticated()


async def test_emit_stdout_writes_compact_json_lines(capsys: pytest.CaptureFixture[str]) -> None:
    mcp_proxy._emit_stdout({"jsonrpc": "2.0", "id": 1, "result": {"a": [1, 2]}})
    mcp_proxy._emit_stdout({"jsonrpc": "2.0", "method": "n"})
    assert capsys.readouterr().out == (
        '{"jsonrpc":"2.0","id":1,"result":{"a":[1,2]}}\n{"jsonrpc":"2.0","method":"n"}\n'
    )


async def test_logging_goes_to_stderr_only(capsys: pytest.CaptureFixture[str]) -> None:
    mcp_proxy._configure_stderr_logging()
    # A fresh logger: other tests may have cached the module logger's old config.
    structlog.get_logger("dataquery.mcp_proxy").info("proxy-log-line", key="v")
    out = capsys.readouterr()
    assert out.out == ""
    assert "proxy-log-line" in out.err
    assert "key=v" in out.err


# --------------------------------------------------------------------------- #
# Defects: fixed (F*) and not fixed (N*, strict xfail)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("line", [b'[{"jsonrpc":"2.0","id":9,"method":"ping"}]\n', b"42\n", b'"text"\n', b"null\n"])
async def test_non_object_line_is_skipped(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str], line: bytes
) -> None:
    s = start()
    s.stdin.raw(line)
    s.stdin.send(request(2))
    await s.answered(2)
    assert await s.finish() == 0
    assert [p["body"] for p in fake.posts] == [request(2)]


async def test_non_json_2xx_body_is_answered_with_an_error(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(text="<html>gateway</html>", content_type="text/html")

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(3))
    assert [e["error"]["code"] for e in errors_for(emitted, 3)] == [-32001]
    assert len(emitted) == 1


async def test_non_json_2xx_body_for_a_notification_is_logged_only(
    make_proxy: Callable[..., Any], fake: FakeMcp
) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(text="oops", content_type="text/plain")

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(INITIALIZED)
    assert emitted == []


async def test_token_failure_mid_session_is_answered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())
    proxy.oauth.token_manager.clear_token()
    fake.token_status = 500
    await proxy.handle_message(request(2))
    errors = errors_for(emitted, 2)
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == -32001


async def test_token_failure_during_401_refresh_is_answered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())
    fake.revoked.add("Bearer tok1")
    fake.token_status = 500
    await proxy.handle_message(request(2))
    assert len(errors_for(emitted, 2)) == 1


async def test_dropped_sse_before_the_reply_is_answered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(req, [event({"jsonrpc": "2.0", "method": "progress"}), 'data: {"jsonrpc"'], drop=True)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(4))
    assert emitted[0] == {"jsonrpc": "2.0", "method": "progress"}
    assert len(errors_for(emitted, 4)) == 1


async def test_dropped_sse_after_the_reply_is_not_answered_twice(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(req, [event(reply(msg["id"], done=True)), "data: {"], drop=True)

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(4))
    assert emitted == [reply(4, done=True)]


async def test_large_sse_event_is_delivered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    big = "x" * (2 * 1024 * 1024)

    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(req, [event(reply(msg["id"], blob=big))])

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(5))
    assert emitted == [reply(5, blob=big)]


async def test_oversized_stdin_line_is_skipped(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]
) -> None:
    s = start(limit=1024)
    s.stdin.send(request(1, "ping", blob="y" * 4096))
    s.stdin.send(request(2))
    await s.answered(2)
    assert await s.finish() == 0
    assert [p["body"]["id"] for p in fake.posts] == [2]


async def test_oversized_last_line_at_eof_is_skipped(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]
) -> None:
    s = start(limit=1024)
    s.stdin.send(request(1))
    await s.answered(1)
    s.stdin.raw(json.dumps(request(2, "ping", blob="y" * 4096)).encode())  # no trailing newline
    assert await s.finish() == 0
    assert [p["body"]["id"] for p in fake.posts] == [1]


async def test_oversized_line_arriving_in_pieces_is_skipped_once(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    log = RecordingLogger()
    monkeypatch.setattr(mcp_proxy, "logger", log)
    s = start(limit=1024)
    line = (json.dumps(request(1, "ping", blob="y" * 8000)) + "\n").encode()
    for i in range(0, len(line), 1500):
        s.stdin.raw(line[i : i + 1500])
        await _REAL_SLEEP(0.01)
    s.stdin.send(request(2))
    await s.answered(2)
    assert await s.finish() == 0
    assert [p["body"]["id"] for p in fake.posts] == [2]
    assert [r for r in log.records if r[0] == "warning"] == [
        ("warning", "Skipping JSON-RPC line over the stdin limit", {})
    ]


async def test_last_line_without_a_newline_is_processed(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]
) -> None:
    s = start()
    s.stdin.raw(json.dumps(request(1)).encode())
    s.stdin.eof()
    assert await asyncio.wait_for(s.task, 10) == 0
    assert s.emitted == [reply(1, echo="ping")]


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Callable[..., None]:
        def log(event: str, **kw: Any) -> None:
            self.records.append((level, event, kw))

        return log


async def test_failed_background_task_is_logged(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(self: StreamableHttpProxy) -> None:
        raise RuntimeError("boom")

    log = RecordingLogger()
    monkeypatch.setattr(mcp_proxy, "logger", log)
    monkeypatch.setattr(StreamableHttpProxy, "listen_get_stream", boom)
    s = start()
    s.stdin.send(INITIALIZED)
    await wait_until(lambda: ("error", "MCP proxy task failed", {"error": "boom"}) in log.records)
    assert await s.finish() == 0


async def test_401_with_a_static_bearer_token_retries_once_without_a_token_fetch(
    make_proxy: Callable[..., Any], fake: FakeMcp
) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse | None:
        return web.Response(status=401) if len(fake.posts) == 1 else None

    fake.post_hook = hook
    proxy, emitted = make_proxy(bearer="static-token")
    await proxy.handle_message(request(1))
    assert [p["headers"]["Authorization"] for p in fake.posts] == ["Bearer static-token"] * 2
    assert fake.token_calls == 0
    assert emitted == [reply(1, echo="ping")]


def _write_all(fd: int, data: bytes) -> None:
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
    except OSError:
        pass
    finally:
        os.close(fd)


async def test_stdin_reader_accepts_large_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "rb", buffering=0))
    reader = await mcp_proxy._read_stdin_lines()
    line = json.dumps(request(1, "ping", blob="z" * (1024 * 1024))).encode() + b"\n"
    writer = threading.Thread(target=_write_all, args=(write_fd, line), daemon=True)
    writer.start()
    try:
        assert await asyncio.wait_for(reader.readline(), 10) == line
        assert await asyncio.wait_for(reader.readline(), 10) == b""
    finally:
        transport = getattr(reader, "_transport", None)
        if transport is not None:
            transport.close()
        writer.join(5)


async def test_eof_lets_in_flight_requests_finish(
    start: Callable[..., Session], fake: FakeMcp, proxy_env: dict[str, str]
) -> None:
    s = start()
    s.stdin.send(request(1, "slow"))
    await wait_until(lambda: len(fake.posts) == 1)
    s.stdin.eof()
    await _REAL_SLEEP(0.1)
    fake.release.set()
    assert await asyncio.wait_for(s.task, 10) == 0
    assert s.emitted == [reply(1, echo="slow")]


async def test_staggered_401s_share_one_refresh(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    proxy, emitted = make_proxy()
    await proxy.handle_message(initialize())

    async def hook(req: web.Request, msg: Any) -> web.StreamResponse | None:
        await _REAL_SLEEP(0.05 * int(msg["id"]))
        if req.headers["Authorization"] == "Bearer tok1":
            return web.Response(status=401)
        return None

    fake.post_hook = hook
    await asyncio.gather(*(proxy.handle_message(request(i)) for i in range(2, 6)))
    assert fake.token_calls == 2  # the initial token + one shared refresh
    assert sorted(m["id"] for m in emitted[1:]) == [2, 3, 4, 5]
    assert not any("error" in m for m in emitted)


async def test_close_gives_up_on_a_stalled_delete(
    make_proxy: Callable[..., Any], fake: FakeMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_proxy, "_CLOSE_TIMEOUT", 0.2, raising=False)
    fake.delete_gate = asyncio.Event()
    proxy, _ = make_proxy()
    await proxy.handle_message(initialize())
    await asyncio.wait_for(proxy.close(), 3)
    assert len(fake.deletes) == 1


async def test_unreachable_token_endpoint_at_startup_exits_1(
    start: Callable[..., Session], proxy_env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    os.environ["DATAQUERY_OAUTH_TOKEN_URL"] = closed_port_url("/oauth/token")
    s = start()
    assert await asyncio.wait_for(s.task, 10) == 1
    assert "dataquery mcp connect: Failed to get OAuth token" in capsys.readouterr().err


@pytest.mark.xfail(strict=True, reason="N1 (not fixed): a token failure ends the GET stream instead of backing off")
async def test_get_stream_retries_after_a_token_failure(
    make_proxy: Callable[..., Any], fake: FakeMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = SleepRecorder(limit=3)
    monkeypatch.setattr(mcp_proxy, "asyncio", recorder)
    fake.token_status = 500
    proxy, _ = make_proxy()
    with pytest.raises(_StopLoop):
        await proxy.listen_get_stream()
    assert recorder.sleeps == [1.0, 2.0, 4.0]


@pytest.mark.xfail(strict=True, reason="N1 (not fixed): a 5xx on the GET stream ends it for good")
async def test_get_stream_reconnects_after_a_server_error(
    make_proxy: Callable[..., Any], fake: FakeMcp, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = iter([503, 405])

    async def hook(req: web.Request) -> web.StreamResponse:
        return web.Response(status=next(statuses))

    fake.get_hook = hook
    monkeypatch.setattr(mcp_proxy, "asyncio", SleepRecorder())
    proxy, _ = make_proxy()
    await asyncio.wait_for(proxy.listen_get_stream(), 5)
    assert len(fake.gets) == 2


@pytest.mark.xfail(strict=True, reason="N2 (not fixed): an empty 2xx JSON body leaves the request unanswered")
@pytest.mark.parametrize("body", [b"", b"null"])
async def test_empty_2xx_body_is_answered(make_proxy: Callable[..., Any], fake: FakeMcp, body: bytes) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return web.Response(body=body, content_type="application/json")

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(6))
    assert len(errors_for(emitted, 6)) == 1


@pytest.mark.xfail(strict=True, reason="N2 (not fixed): an SSE stream that ends without the reply leaves it unanswered")
async def test_sse_stream_without_reply_is_answered(make_proxy: Callable[..., Any], fake: FakeMcp) -> None:
    async def hook(req: web.Request, msg: Any) -> web.StreamResponse:
        return await sse(req, [event({"jsonrpc": "2.0", "method": "progress"})])

    fake.post_hook = hook
    proxy, emitted = make_proxy()
    await proxy.handle_message(request(6))
    assert len(errors_for(emitted, 6)) == 1
