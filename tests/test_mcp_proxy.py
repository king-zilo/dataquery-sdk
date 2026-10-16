"""Coverage for the stdio <-> Streamable HTTP MCP proxy.

Runs the proxy against an in-process aiohttp server that plays the remote MCP
endpoint (plus its OAuth token endpoint), so no real network is needed.
"""

import asyncio
import json
import os
import sys
from unittest.mock import patch

import aiohttp
import pytest
import pytest_asyncio
import structlog
from aiohttp import web
from aiohttp.test_utils import TestServer

from dataquery import mcp_proxy
from dataquery.mcp_proxy import StreamableHttpProxy
from dataquery.transport.auth import OAuthManager
from dataquery.types.exceptions import AuthenticationError, ConfigurationError
from dataquery.types.models import ClientConfig

SESSION_ID = "sess-1"
PROTOCOL_VERSION = "2025-06-18"


def _make_app(state: dict) -> web.Application:
    """Fake remote MCP server + OAuth token endpoint."""

    async def token(request: web.Request) -> web.Response:
        state["token_calls"] += 1
        if state.get("token_status"):
            return web.Response(status=state["token_status"])
        return web.json_response(
            {
                "access_token": f"tok{state['token_calls']}",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    async def mcp_post(request: web.Request) -> web.StreamResponse:
        auth = request.headers.get("Authorization", "")
        state["seen_auth"].append(auth)
        if auth in state["revoked"]:
            return web.Response(status=401)

        message = await request.json()
        state["seen_session"].append(request.headers.get("Mcp-Session-Id"))

        if state.get("force_status"):
            return web.Response(status=state["force_status"])

        if state.get("post_override"):
            return await state["post_override"](request, message)

        if message.get("id") is None:
            return web.Response(status=202)  # notification

        if message.get("method") == "initialize":
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}},
                },
                headers={"Mcp-Session-Id": SESSION_ID},
            )

        if message.get("method") == "tools/list":
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}}
            reply = {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": []}}
            await resp.write(f"data: {json.dumps(note)}\n\n".encode())
            await resp.write(f"data: {json.dumps(reply)}\n\n".encode())
            await resp.write_eof()
            return resp

        return web.json_response({"jsonrpc": "2.0", "id": message["id"], "result": {"echo": message.get("method")}})

    async def mcp_get(request: web.Request) -> web.Response:
        state["get_sessions"].append(request.headers.get("Mcp-Session-Id"))
        return web.Response(status=405)

    async def mcp_delete(request: web.Request) -> web.Response:
        state["deleted"] = True
        return web.Response(status=204)

    app = web.Application()
    app.router.add_post("/oauth/token", token)
    app.router.add_post("/mcp", mcp_post)
    app.router.add_get("/mcp", mcp_get)
    app.router.add_delete("/mcp", mcp_delete)
    return app


def _new_state() -> dict:
    return {
        "token_calls": 0,
        "revoked": set(),
        "seen_auth": [],
        "seen_session": [],
        "get_sessions": [],
        "deleted": False,
    }


@pytest_asyncio.fixture
async def rig(tmp_path):
    """(proxy, state, emitted) wired to an in-process fake MCP server."""
    state = _new_state()
    server = TestServer(_make_app(state))
    await server.start_server()
    base_url = str(server.make_url("")).rstrip("/")

    config = ClientConfig(
        base_url=base_url,
        oauth_token_url=f"{base_url}/oauth/token",
        client_id="cid",
        client_secret="secret",
        download_dir=str(tmp_path),
    )
    oauth = OAuthManager(config)
    emitted: list = []
    async with aiohttp.ClientSession() as session:
        proxy = StreamableHttpProxy(f"{base_url}/mcp", oauth, session, emitted.append)
        yield proxy, state, emitted
    await server.close()


def _initialize_msg(msg_id: int = 1) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "initialize",
        "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}},
    }


@pytest.mark.asyncio
async def test_initialize_captures_session_and_protocol(rig):
    proxy, state, emitted = rig
    await proxy.handle_message(_initialize_msg())

    assert len(emitted) == 1
    assert emitted[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert proxy.session_id == SESSION_ID
    assert proxy.protocol_version == PROTOCOL_VERSION
    assert state["seen_auth"] == ["Bearer tok1"]


@pytest.mark.asyncio
async def test_sse_response_emits_all_events_with_session_header(rig):
    proxy, state, emitted = rig
    await proxy.handle_message(_initialize_msg())
    await proxy.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})

    assert [m.get("method", "response") for m in emitted[1:]] == [
        "notifications/progress",
        "response",
    ]
    assert emitted[-1]["id"] == 2
    assert state["seen_session"][-1] == SESSION_ID  # session echoed after init


@pytest.mark.asyncio
async def test_notification_emits_nothing(rig):
    proxy, _state, emitted = rig
    await proxy.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert emitted == []


@pytest.mark.asyncio
async def test_401_refreshes_token_and_retries(rig):
    proxy, state, emitted = rig
    await proxy.handle_message(_initialize_msg())

    state["revoked"].add("Bearer tok1")  # simulate expiry server-side
    await proxy.handle_message({"jsonrpc": "2.0", "id": 2, "method": "ping"})

    assert state["token_calls"] == 2  # initial token + forced refresh
    assert state["seen_auth"][-1] == "Bearer tok2"
    assert emitted[-1] == {"jsonrpc": "2.0", "id": 2, "result": {"echo": "ping"}}


@pytest.mark.asyncio
async def test_server_error_becomes_jsonrpc_error(rig):
    proxy, state, emitted = rig
    await proxy.handle_message(_initialize_msg())

    state["force_status"] = 500
    await proxy.handle_message({"jsonrpc": "2.0", "id": 7, "method": "ping"})

    err = emitted[-1]
    assert err["id"] == 7
    assert err["error"]["code"] == -32001
    assert "500" in err["error"]["message"]


@pytest.mark.asyncio
async def test_expired_session_404_surfaces_error(rig):
    proxy, state, emitted = rig
    await proxy.handle_message(_initialize_msg())

    state["force_status"] = 404
    await proxy.handle_message({"jsonrpc": "2.0", "id": 8, "method": "ping"})

    err = emitted[-1]
    assert err["id"] == 8
    assert "session expired" in err["error"]["message"].lower()


@pytest.mark.asyncio
async def test_get_stream_not_offered_gives_up(rig):
    proxy, _state, _emitted = rig
    await proxy.listen_get_stream()  # server answers 405; must return, not loop


@pytest.mark.asyncio
async def test_close_deletes_session(rig):
    proxy, state, _emitted = rig
    await proxy.handle_message(_initialize_msg())
    await proxy.close()
    assert state["deleted"] is True


@pytest.mark.asyncio
async def test_close_without_session_is_noop(rig):
    proxy, state, _emitted = rig
    await proxy.close()
    assert state["deleted"] is False


# --- handle_message: error paths and response shapes -------------------------


def _sse(*events: str) -> bytes:
    return "".join(events).encode()


async def _stream(request: web.Request, body: bytes) -> web.StreamResponse:
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(body)
    await resp.write_eof()
    return resp


@pytest.mark.asyncio
async def test_persistent_401_refreshes_once_then_errors(rig):
    proxy, state, emitted = rig
    state["revoked"].update({"Bearer tok1", "Bearer tok2", "Bearer tok3"})

    await proxy.handle_message({"jsonrpc": "2.0", "id": 3, "method": "ping"})

    assert state["token_calls"] == 2  # initial token + exactly one forced refresh
    assert state["seen_auth"] == ["Bearer tok1", "Bearer tok2"]
    assert emitted == [
        {"jsonrpc": "2.0", "id": 3, "error": {"code": -32001, "message": "MCP endpoint returned 401"}},
    ]


@pytest.mark.asyncio
async def test_unreachable_endpoint_becomes_jsonrpc_error(rig, monkeypatch):
    proxy, _state, emitted = rig

    async def refuse(*_args, **_kwargs):
        raise aiohttp.ClientConnectionError("connection refused")

    monkeypatch.setattr(proxy.session, "post", refuse)
    await proxy.handle_message({"jsonrpc": "2.0", "id": 4, "method": "ping"})

    assert emitted == [
        {
            "jsonrpc": "2.0",
            "id": 4,
            "error": {"code": -32001, "message": "MCP endpoint unreachable: connection refused"},
        }
    ]


@pytest.mark.asyncio
async def test_404_without_session_is_a_plain_endpoint_error(rig):
    proxy, state, emitted = rig
    state["force_status"] = 404
    await proxy.handle_message({"jsonrpc": "2.0", "id": 5, "method": "ping"})

    assert emitted[-1]["error"]["message"] == "MCP endpoint returned 404"


@pytest.mark.asyncio
async def test_errors_for_notifications_are_not_answered(rig):
    proxy, state, emitted = rig
    state["force_status"] = 500
    await proxy.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})

    assert emitted == []  # JSON-RPC: notifications never get a response


@pytest.mark.asyncio
async def test_json_null_body_emits_nothing(rig):
    proxy, state, emitted = rig

    async def null_body(_request, _message):
        return web.Response(text="null", content_type="application/json")

    state["post_override"] = null_body
    await proxy.handle_message({"jsonrpc": "2.0", "id": 6, "method": "ping"})

    assert emitted == []


@pytest.mark.asyncio
async def test_non_object_payload_is_forwarded_verbatim(rig):
    proxy, state, emitted = rig
    batch = [{"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "1999-01-01"}}]

    async def batch_body(_request, _message):
        return web.json_response(batch)

    state["post_override"] = batch_body
    await proxy.handle_message({"jsonrpc": "2.0", "id": 1, "method": "ping"})

    assert emitted == [batch]
    assert proxy.protocol_version is None  # only a top-level result is inspected


@pytest.mark.asyncio
async def test_sse_parser_handles_comments_multiline_crlf_and_bad_events(rig):
    proxy, state, emitted = rig

    async def tricky_stream(request, _message):
        body = _sse(
            ": keep-alive comment\n",
            "event: message\n",
            "\n",  # blank line with no buffered data: nothing to dispatch
            'data: {"jsonrpc": "2.0",\n',
            'data:"id": 1, "result": {}}\n',
            "\n",
            "data: not json\n\n",
            'data: {"jsonrpc": "2.0", "id": 2, "result": {}}\r\n\r\n',
        )
        return await _stream(request, body)

    state["post_override"] = tricky_stream
    await proxy.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})

    assert emitted == [
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 2, "result": {}},
    ]


@pytest.mark.asyncio
async def test_sse_event_without_trailing_blank_line_is_dropped(rig):
    proxy, state, emitted = rig

    async def truncated_stream(request, _message):
        return await _stream(request, _sse('data: {"jsonrpc": "2.0", "id": 1, "result": {}}\n'))

    state["post_override"] = truncated_stream
    await proxy.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call"})

    assert emitted == []  # matches the SSE spec: an unterminated event is not dispatched


@pytest.mark.xfail(
    raises=json.JSONDecodeError,
    strict=True,
    reason="Finding: a 2xx non-JSON body escapes handle_message and the request is never answered",
)
@pytest.mark.asyncio
async def test_non_json_success_body_should_become_jsonrpc_error(rig):
    proxy, state, emitted = rig

    async def html_body(_request, _message):
        return web.Response(text="<html>gateway login</html>", content_type="text/html")

    state["post_override"] = html_body
    await proxy.handle_message({"jsonrpc": "2.0", "id": 9, "method": "ping"})

    assert emitted and emitted[-1]["id"] == 9 and "error" in emitted[-1]


@pytest.mark.xfail(
    raises=AuthenticationError,
    strict=True,
    reason="Finding: a token failure mid-session escapes handle_message and the request is never answered",
)
@pytest.mark.asyncio
async def test_token_failure_mid_session_should_become_jsonrpc_error(rig):
    proxy, state, emitted = rig
    state["token_status"] = 500  # token endpoint down when the proxy needs a token

    await proxy.handle_message({"jsonrpc": "2.0", "id": 10, "method": "ping"})

    assert emitted and emitted[-1]["id"] == 10 and "error" in emitted[-1]


# --- listen_get_stream: give-up, backoff and reconnect ------------------------


class _StubOAuth:
    async def get_headers(self) -> dict:
        return {"Authorization": "Bearer stub"}


class _FakeGetResponse:
    def __init__(self, status: int, lines: tuple = ()):
        self.status = status
        self.content = self._iter(lines)

    @staticmethod
    async def _iter(lines):
        for line in lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _ScriptedGetSession:
    """Answers each GET with the next scripted outcome (status, SSE lines, or an exception)."""

    def __init__(self, script: list):
        self.script = list(script)
        self.headers: list = []

    def get(self, _url, headers):
        self.headers.append(headers)
        outcome = self.script.pop(0)

        class _Ctx:
            async def __aenter__(self):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            async def __aexit__(self, *_exc):
                return False

        return _Ctx()


@pytest.fixture
def recorded_sleeps(monkeypatch):
    delays: list = []

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(mcp_proxy.asyncio, "sleep", fake_sleep)
    return delays


def _get_proxy(script: list) -> tuple:
    emitted: list = []
    session = _ScriptedGetSession(script)
    proxy = StreamableHttpProxy("http://mcp.invalid/mcp", _StubOAuth(), session, emitted.append)  # type: ignore[arg-type]
    return proxy, session, emitted


@pytest.mark.asyncio
async def test_get_stream_rejected_status_gives_up(recorded_sleeps):
    proxy, session, _emitted = _get_proxy([_FakeGetResponse(403)])
    await proxy.listen_get_stream()

    assert recorded_sleeps == []
    assert session.script == []


@pytest.mark.asyncio
async def test_get_stream_backoff_doubles_up_to_cap(recorded_sleeps):
    drops = [aiohttp.ClientConnectionError("dropped") for _ in range(7)]
    proxy, _session, _emitted = _get_proxy([*drops, _FakeGetResponse(404)])
    await proxy.listen_get_stream()

    assert recorded_sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


@pytest.mark.asyncio
async def test_get_stream_emits_events_and_resets_backoff(recorded_sleeps):
    event = {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
    stream = _FakeGetResponse(200, (f"data: {json.dumps(event)}\n".encode(), b"\n"))
    script = [
        aiohttp.ClientConnectionError("dropped"),
        aiohttp.ClientConnectionError("dropped"),
        stream,
        aiohttp.ClientConnectionError("dropped"),
        _FakeGetResponse(501),
    ]
    proxy, session, emitted = _get_proxy(script)
    proxy.session_id = SESSION_ID
    await proxy.listen_get_stream()

    assert emitted == [event]
    assert recorded_sleeps == [1.0, 2.0, 1.0, 2.0]
    assert all(h["Mcp-Session-Id"] == SESSION_ID for h in session.headers)


# --- close -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_failure_is_swallowed(rig, monkeypatch):
    proxy, state, _emitted = rig
    proxy.session_id = SESSION_ID

    def broken_delete(*_args, **_kwargs):
        raise aiohttp.ClientConnectionError("gone")

    monkeypatch.setattr(proxy.session, "delete", broken_delete)
    await proxy.close()  # must not raise

    assert state["deleted"] is False


# --- module helpers ------------------------------------------------------------


@pytest.fixture
def restore_structlog():
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


@pytest.fixture
def proxy_env(tmp_path):
    """os.environ without any DATAQUERY_* vars and a private user config dir."""
    keep = {k: v for k, v in os.environ.items() if not k.startswith("DATAQUERY_")}
    with patch.dict(os.environ, keep, clear=True):
        os.environ["DATAQUERY_CONFIG_DIR"] = str(tmp_path / "home")
        yield os.environ


def test_configure_stderr_logging_keeps_stdout_clean(restore_structlog, capsys):
    mcp_proxy._configure_stderr_logging()
    structlog.get_logger("mcp-test").info("hello from proxy")

    out, err = capsys.readouterr()
    assert out == ""
    assert "hello from proxy" in err


def test_emit_stdout_writes_one_compact_json_line(capsys):
    mcp_proxy._emit_stdout({"jsonrpc": "2.0", "id": 1, "result": {"a": [1, 2]}})

    assert capsys.readouterr().out == '{"jsonrpc":"2.0","id":1,"result":{"a":[1,2]}}\n'


@pytest.mark.asyncio
async def test_read_stdin_lines_reads_from_stdin_pipe(monkeypatch):
    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "rb"))
    os.write(write_fd, b'{"id":1}\n{"id":2}\n')
    os.close(write_fd)

    reader = await mcp_proxy._read_stdin_lines()

    assert await reader.readline() == b'{"id":1}\n'
    assert await reader.readline() == b'{"id":2}\n'
    assert await reader.readline() == b""


def test_build_oauth_manager_derives_base_url_from_mcp_url(proxy_env):
    proxy_env.update({"DATAQUERY_BASE_URL": "", "DATAQUERY_CLIENT_ID": "cid", "DATAQUERY_CLIENT_SECRET": "s"})

    oauth = mcp_proxy._build_oauth_manager("https://mcp.example.test:8443/research/mcp", None)

    assert proxy_env["DATAQUERY_BASE_URL"] == "https://mcp.example.test:8443"
    assert oauth.config.base_url == "https://mcp.example.test:8443"
    assert oauth.is_authenticated()


def test_build_oauth_manager_keeps_explicit_base_url(proxy_env):
    proxy_env.update(
        {"DATAQUERY_BASE_URL": "https://api.example.test", "DATAQUERY_CLIENT_ID": "cid", "DATAQUERY_CLIENT_SECRET": "s"}
    )

    oauth = mcp_proxy._build_oauth_manager("https://mcp.example.test/mcp", None)

    assert oauth.config.base_url == "https://api.example.test"


def test_build_oauth_manager_loads_env_file(proxy_env, tmp_path):
    env_file = tmp_path / "proxy.env"
    env_file.write_text("DATAQUERY_BEARER_TOKEN=from-file\n")

    oauth = mcp_proxy._build_oauth_manager("https://mcp.example.test/mcp", env_file)

    assert oauth.config.has_bearer_token
    assert proxy_env["DATAQUERY_BEARER_TOKEN"] == "from-file"


def test_build_oauth_manager_without_credentials_raises(proxy_env):
    with pytest.raises(ConfigurationError, match="No credentials configured"):
        mcp_proxy._build_oauth_manager("https://mcp.example.test/mcp", None)


# --- run_mcp_proxy end to end ----------------------------------------------------


@pytest_asyncio.fixture
async def fake_server():
    """(base_url, state) for an in-process fake MCP server."""
    state = _new_state()
    server = TestServer(_make_app(state))
    await server.start_server()
    yield str(server.make_url("")).rstrip("/"), state
    await server.close()


@pytest.fixture
def stdio(monkeypatch):
    """(stdin StreamReader, list of messages written to stdout) for run_mcp_proxy."""
    reader = asyncio.StreamReader()
    emitted: list = []
    real_emit = mcp_proxy._emit_stdout

    async def fake_stdin():
        return reader

    def recording_emit(message):
        emitted.append(message)
        real_emit(message)

    monkeypatch.setattr(mcp_proxy, "_read_stdin_lines", fake_stdin)
    monkeypatch.setattr(mcp_proxy, "_emit_stdout", recording_emit)
    return reader, emitted


def _credentials_env(base_url: str) -> dict:
    return {
        "DATAQUERY_BASE_URL": base_url,
        "DATAQUERY_OAUTH_TOKEN_URL": f"{base_url}/oauth/token",
        "DATAQUERY_CLIENT_ID": "cid",
        "DATAQUERY_CLIENT_SECRET": "secret",
    }


async def _until(predicate, timeout: float = 5.0) -> None:
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


def _line(message) -> bytes:
    return (json.dumps(message) + "\n").encode()


@pytest.mark.asyncio
async def test_run_proxy_without_endpoint_returns_1(proxy_env, restore_structlog, capsys):
    proxy_env["DATAQUERY_MCP_URL"] = ""

    assert await mcp_proxy.run_mcp_proxy() == 1
    assert "No MCP endpoint configured" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_run_proxy_without_credentials_returns_1(proxy_env, restore_structlog, capsys):
    assert await mcp_proxy.run_mcp_proxy("https://mcp.example.test/mcp") == 1
    assert "No credentials configured" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_run_proxy_auth_failure_returns_1(proxy_env, restore_structlog, fake_server, capsys):
    base_url, state = fake_server
    proxy_env.update(_credentials_env(base_url))
    state["token_status"] = 401

    assert await mcp_proxy.run_mcp_proxy(f"{base_url}/mcp") == 1
    assert "dataquery mcp connect:" in capsys.readouterr().err
    assert state["seen_auth"] == []  # never reached the MCP endpoint


@pytest.mark.asyncio
async def test_run_proxy_bridges_stdio_until_eof(proxy_env, restore_structlog, fake_server, stdio, tmp_path, capsys):
    base_url, state = fake_server
    env_file = tmp_path / "mcp.env"
    env_file.write_text(
        "".join(f"{k}={v}\n" for k, v in {**_credentials_env(base_url), "DATAQUERY_MCP_URL": f"{base_url}/mcp"}.items())
    )
    reader, emitted = stdio

    run = asyncio.ensure_future(mcp_proxy.run_mcp_proxy(env_file=env_file))
    reader.feed_data(b"\n   \n")  # blank lines are ignored
    reader.feed_data(b"{not json\n")  # malformed lines are skipped
    reader.feed_data(_line(_initialize_msg()))
    await _until(lambda: len(emitted) == 1)

    initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    reader.feed_data(_line(initialized))
    await _until(lambda: len(state["get_sessions"]) == 1)
    reader.feed_data(_line(initialized))  # a second one must not open another GET stream
    reader.feed_data(_line({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    await _until(lambda: len(emitted) == 3)

    reader.feed_eof()
    assert await asyncio.wait_for(run, 5) == 0

    assert [m.get("id") for m in emitted] == [1, None, 2]
    assert emitted[0]["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert state["get_sessions"] == [SESSION_ID]
    assert state["seen_session"].count(SESSION_ID) == 3  # both notifications + tools/list
    assert state["deleted"] is True  # session torn down on EOF
    out_lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line) for line in out_lines] == emitted


@pytest.mark.asyncio
async def test_run_proxy_cancels_requests_in_flight_at_eof(proxy_env, restore_structlog, fake_server, stdio):
    """Documents current behavior (see PR Findings): EOF cancels unanswered requests."""
    base_url, state = fake_server
    proxy_env.update(_credentials_env(base_url))
    reader, emitted = stdio
    reader.feed_data(_line(_initialize_msg()))
    reader.feed_eof()

    assert await mcp_proxy.run_mcp_proxy(f"{base_url}/mcp") == 0

    assert emitted == []
    assert state["seen_auth"] == []  # the request was cancelled before it was sent


@pytest.mark.xfail(
    raises=AttributeError,
    strict=True,
    reason="Finding: a valid JSON line that is not an object (e.g. a batch) crashes the proxy loop",
)
@pytest.mark.asyncio
async def test_run_proxy_should_survive_non_object_json_line(proxy_env, restore_structlog, fake_server, stdio):
    base_url, _state = fake_server
    proxy_env.update(_credentials_env(base_url))
    reader, _emitted = stdio
    reader.feed_data(_line([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]))
    reader.feed_eof()

    assert await mcp_proxy.run_mcp_proxy(f"{base_url}/mcp") == 0
