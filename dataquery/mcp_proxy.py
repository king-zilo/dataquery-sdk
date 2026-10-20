"""Local stdio <-> Streamable HTTP proxy for a remote MCP server."""

import asyncio
import json
import os
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

import aiohttp
import structlog

from .config import EnvConfig
from .transport.auth import OAuthManager
from .types.exceptions import AuthenticationError, ConfigurationError, NetworkError

logger = structlog.get_logger(__name__)

_SESSION_HEADER: Final = "Mcp-Session-Id"
_PROTOCOL_HEADER: Final = "MCP-Protocol-Version"

_GET_BACKOFF_INITIAL: Final = 1.0
_GET_BACKOFF_MAX: Final = 30.0

# Longest stdin line accepted; longer lines are skipped.
_STDIN_LIMIT: Final = 64 * 1024 * 1024
# How long in-flight requests may run after stdin closes.
_DRAIN_TIMEOUT: Final = 30.0
# How long the session DELETE may take on shutdown.
_CLOSE_TIMEOUT: Final = 5.0

_TRANSPORT_ERROR_CODE: Final = -32001


class StreamableHttpProxy:
    """Forward JSON-RPC messages between a stdio client and a remote endpoint."""

    def __init__(
        self,
        url: str,
        oauth: OAuthManager,
        session: aiohttp.ClientSession,
        emit: Callable[[dict[str, Any]], None],
    ):
        self.url = url
        self.oauth = oauth
        self.session = session
        self.emit = emit
        self.session_id: str | None = None
        self.protocol_version: str | None = None

    async def _headers(self) -> dict[str, str]:
        headers = await self.oauth.get_headers()
        headers["Accept"] = "application/json, text/event-stream"
        if self.session_id:
            headers[_SESSION_HEADER] = self.session_id
        if self.protocol_version:
            headers[_PROTOCOL_HEADER] = self.protocol_version
        return headers

    async def handle_message(self, message: dict[str, Any]) -> None:
        """POST one client message to the endpoint and emit whatever comes back."""
        request_id = message.get("id")
        replied = False

        def emit(payload: Any) -> None:
            nonlocal replied
            self._emit_message(payload)
            replied = replied or _is_reply(payload, request_id)

        try:
            await self._post(message, emit)
        except Exception as exc:
            # Whatever went wrong, a request must still get an answer.
            logger.error("MCP request failed", error=_describe(exc), replied=replied)
            if not replied:
                if isinstance(exc, json.JSONDecodeError | aiohttp.ContentTypeError):
                    text = "MCP endpoint returned a response that is not valid JSON"
                else:
                    text = f"MCP request failed: {_describe(exc)}"
                self._emit_transport_error(message, text)

    async def _post(self, message: dict[str, Any], emit: Callable[[Any], None]) -> None:
        for attempt in (1, 2):
            headers = await self._headers()
            headers["Content-Type"] = "application/json"
            try:
                resp = await self.session.post(self.url, json=message, headers=headers)
            except aiohttp.ClientError as exc:
                logger.error("MCP endpoint unreachable", url=self.url, error=str(exc))
                self._emit_transport_error(message, f"MCP endpoint unreachable: {exc}")
                return

            async with resp:
                if resp.status == 401 and attempt == 1:
                    # Token likely expired between refresh checks; replace it
                    # (once, however many requests it failed) and retry once.
                    logger.info("401 from MCP endpoint, refreshing token")
                    await self.oauth.refresh_after_rejection(headers.get("Authorization", ""))
                    continue

                if resp.status in (202, 204):
                    return

                if resp.status == 404 and self.session_id:
                    # Session expired server-side. A transparent re-init is
                    # not possible mid-conversation; surface it so the MCP
                    # client restarts the proxy and re-initializes.
                    logger.error("MCP session expired (404), restart the connection")
                    self._emit_transport_error(message, "MCP session expired; restart the connection")
                    return

                if resp.status >= 400:
                    body = (await resp.text())[:500]
                    logger.error("MCP endpoint error", status=resp.status, body=body)
                    self._emit_transport_error(message, f"MCP endpoint returned {resp.status}")
                    return

                self._capture_session(resp)

                content_type = resp.headers.get("Content-Type", "")
                if content_type.startswith("text/event-stream"):
                    async for payload in _sse_events(resp):
                        emit(payload)
                elif (payload := await resp.json(content_type=None)) is not None:
                    emit(payload)
                return

    def _capture_session(self, resp: aiohttp.ClientResponse) -> None:
        session_id = resp.headers.get(_SESSION_HEADER)
        if session_id and session_id != self.session_id:
            self.session_id = session_id
            logger.info("MCP session established")

    def _emit_message(self, payload: Any) -> None:
        if isinstance(payload, dict):
            result = payload.get("result")
            if isinstance(result, dict) and result.get("protocolVersion"):
                self.protocol_version = result["protocolVersion"]
        self.emit(payload)

    def _emit_transport_error(self, request: dict[str, Any], text: str) -> None:
        """Answer a request the endpoint never processed with a JSON-RPC error."""
        request_id = request.get("id")
        if request_id is None:
            return
        self.emit(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": _TRANSPORT_ERROR_CODE, "message": text},
            }
        )

    async def _pump_sse(self, resp: aiohttp.ClientResponse) -> None:
        """Emit each ``data:`` event of an SSE stream as a JSON-RPC message."""
        async for payload in _sse_events(resp):
            self._emit_message(payload)

    async def listen_get_stream(self) -> None:
        """Pump the optional server-initiated GET stream, reconnecting politely."""
        backoff = _GET_BACKOFF_INITIAL
        while True:
            headers = await self._headers()
            try:
                async with self.session.get(self.url, headers=headers) as resp:
                    if resp.status in (404, 405, 501):
                        logger.debug("Server offers no GET stream", status=resp.status)
                        return
                    if resp.status >= 400:
                        logger.debug("GET stream rejected", status=resp.status)
                        return
                    backoff = _GET_BACKOFF_INITIAL
                    await self._pump_sse(resp)
            except aiohttp.ClientError as exc:
                logger.debug("GET stream dropped", error=str(exc))
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _GET_BACKOFF_MAX)

    async def close(self) -> None:
        """Best-effort session teardown (spec: DELETE with the session id)."""
        if not self.session_id:
            return
        try:
            async with asyncio.timeout(_CLOSE_TIMEOUT):
                headers = await self._headers()
                async with self.session.delete(self.url, headers=headers) as resp:
                    logger.debug("MCP session closed", status=resp.status)
        except Exception as exc:
            logger.debug("MCP session close failed", error=_describe(exc))


def _is_reply(payload: Any, request_id: Any) -> bool:
    """Whether ``payload`` answers the request with ``request_id``."""
    return (
        request_id is not None
        and isinstance(payload, dict)
        and payload.get("id") == request_id
        and ("result" in payload or "error" in payload)
    )


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


class _SseDecoder:
    """Incremental SSE parser: feed raw chunks, get back the JSON ``data:`` events.

    Lines are split from raw chunks so an event of any size gets through;
    aiohttp's line reader caps a line at its read buffer size.
    """

    __slots__ = ("_data_lines", "_partial")

    def __init__(self) -> None:
        self._data_lines: list[str] = []
        self._partial = bytearray()

    def feed(self, chunk: bytes) -> list[Any]:
        events: list[Any] = []
        start = 0
        while (end := chunk.find(b"\n", start)) >= 0:
            self._partial += chunk[start:end]
            start = end + 1
            line = self._partial.decode("utf-8", errors="replace").rstrip("\r")
            self._partial.clear()
            if line.startswith("data:"):
                self._data_lines.append(line[5:].lstrip())
            elif not line and self._data_lines:
                data = "\n".join(self._data_lines)
                self._data_lines.clear()
                try:
                    events.append(json.loads(data))
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON SSE event", data=data[:200])
        self._partial += chunk[start:]
        return events


async def _sse_events(resp: aiohttp.ClientResponse) -> AsyncIterator[Any]:
    """Yield each JSON ``data:`` event of an SSE response body."""
    decoder = _SseDecoder()
    async for chunk in resp.content.iter_any():
        for payload in decoder.feed(chunk):
            yield payload


def _configure_stderr_logging() -> None:
    """Route all structlog output to stderr; stdout is the JSON-RPC channel."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def _build_oauth_manager(url: str, env_file: Path | None) -> OAuthManager:
    """Build the OAuth manager from env config."""
    if env_file is not None:
        EnvConfig.load_env_file(env_file)
    if not EnvConfig.get_env_var("BASE_URL"):
        parts = urlsplit(url)
        os.environ[f"{EnvConfig.PREFIX}BASE_URL"] = f"{parts.scheme}://{parts.netloc}"
    config = EnvConfig.create_client_config()
    oauth = OAuthManager(config)
    if not oauth.is_authenticated():
        raise ConfigurationError(
            "No credentials configured: set DATAQUERY_CLIENT_ID and DATAQUERY_CLIENT_SECRET (or DATAQUERY_BEARER_TOKEN)"
        )
    return oauth


async def _read_stdin_lines() -> asyncio.StreamReader:
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=_STDIN_LIMIT)
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    return reader


async def _next_line(reader: asyncio.StreamReader) -> bytes:
    """Read one stdin line (b"" at EOF), skipping lines over the reader's limit."""
    oversized = False
    while True:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            if oversized:
                return b""
            return exc.partial  # last line without a newline, or b"" at EOF
        except asyncio.LimitOverrunError as exc:
            # Drop what is buffered of the line and keep reading to its end.
            if not oversized:
                logger.warning("Skipping JSON-RPC line over the stdin limit")
                oversized = True
            await reader.readexactly(exc.consumed)
            continue
        if not oversized:
            return line
        oversized = False


def _emit_stdout(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


async def run_mcp_proxy(url: str | None = None, env_file: Path | None = None) -> int:
    """Run the stdio <-> Streamable HTTP proxy until stdin closes.

    ``url`` defaults to ``DATAQUERY_MCP_URL``, and failing that to the PROD MCP
    endpoint carried by :class:`ClientConfig`.
    """
    _configure_stderr_logging()
    try:
        if env_file is not None:
            EnvConfig.load_env_file(env_file)
        EnvConfig.load_user_env_file()
        url = url or EnvConfig.get_env_var("MCP_URL")
        if not url:
            raise ConfigurationError("No MCP endpoint configured: pass a url or set DATAQUERY_MCP_URL")
        oauth = _build_oauth_manager(url, env_file)
        await oauth.authenticate()
    except (ConfigurationError, AuthenticationError, NetworkError) as exc:
        print(f"dataquery mcp connect: {exc}", file=sys.stderr)
        return 1

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        proxy = StreamableHttpProxy(url, oauth, session, _emit_stdout)
        logger.info("MCP proxy connected", url=url)

        get_task: asyncio.Task[None] | None = None
        pending: set[asyncio.Task[None]] = set()

        def _done(task: asyncio.Task[None]) -> None:
            pending.discard(task)
            error = None if task.cancelled() else task.exception()
            if error is not None:
                logger.error("MCP proxy task failed", error=_describe(error))

        def _spawn(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
            task = asyncio.create_task(coro)
            task.add_done_callback(_done)
            return task

        try:
            reader = await _read_stdin_lines()
            while line := await _next_line(reader):
                text = line.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON-RPC line", line=text[:200])
                    continue
                if not isinstance(message, dict):
                    # Batches were dropped from MCP in 2025-06-18; answer nothing.
                    logger.warning("Skipping non-object JSON-RPC line", line=text[:200])
                    continue
                # Each message runs as its own task so a long streamed
                # response doesn't block the next client message.
                pending.add(_spawn(proxy.handle_message(message)))
                if get_task is None and message.get("method") == "notifications/initialized":
                    get_task = _spawn(proxy.listen_get_stream())
            if pending:
                # stdin closed: let in-flight requests deliver their replies.
                logger.info("stdin closed, waiting for in-flight requests", count=len(pending))
                await asyncio.wait(set(pending), timeout=_DRAIN_TIMEOUT)
        finally:
            if get_task is not None:
                get_task.cancel()
            for task in list(pending):
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await proxy.close()

    return 0
