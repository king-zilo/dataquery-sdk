"""Failure and error-handling paths of ``dataquery.download.parallel``.

A real ``DataQueryClient`` talks to an in-memory fake Files API (patched in at
``_make_authenticated_request``) that injects faults per file, Range header and
attempt: connection errors, HTTP error statuses (handled by the real
``_handle_response``), and ``ClientPayloadError`` midway through a body.
``asyncio.sleep`` is replaced with a recorder so retries never really sleep.
Also covers the error paths of ``dataquery.download.utils``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Optional, TypeVar

import aiohttp
import pytest
from multidict import CIMultiDict

from dataquery.core.client import DataQueryClient
from dataquery.download import parallel
from dataquery.download import utils as dl_utils
from dataquery.types.models import ClientConfig, DownloadOptions, DownloadResult, DownloadStatus

PAYLOAD = bytes(range(20))
THRESHOLD = 8
FILE = "FG1"
FIRST = "bytes=0-9"
SECOND = "bytes=10-19"
PROBE = "bytes=0-0"
_REAL_SLEEP = asyncio.sleep

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Fake Files API
# --------------------------------------------------------------------------- #
@dataclass
class Fault:
    """What the fake server does for one (file, Range, attempt)."""

    connect: Optional[BaseException] = None
    status: Optional[int] = None
    body: str = ""
    midstream_after: Optional[int] = None
    midstream_exc: Optional[BaseException] = None
    wait_for: Optional[asyncio.Event] = None


class FakeResponse:
    def __init__(
        self,
        status: int,
        headers: dict,
        body: bytes = b"",
        error_text: str = "",
        fail_after: Optional[int] = None,
        fail_exc: Optional[BaseException] = None,
        on_eof: Optional[asyncio.Event] = None,
    ) -> None:
        self.status = status
        self.headers = CIMultiDict(headers)
        self.url = "https://api.example.com/group/file/download"
        self.content = self
        self._body = body
        self._error_text = error_text
        self._fail_after = fail_after
        self._fail_exc = fail_exc
        self._on_eof = on_eof

    async def text(self) -> str:
        return self._error_text

    async def iter_chunked(self, n: int):
        sent = 0
        for i in range(0, len(self._body), 2):
            if self._fail_after is not None and sent >= self._fail_after:
                assert self._fail_exc is not None
                raise self._fail_exc
            piece = self._body[i : i + 2]
            sent += len(piece)
            yield piece
        if self._fail_after is not None and sent >= self._fail_after:
            assert self._fail_exc is not None
            raise self._fail_exc
        if self._on_eof is not None:
            self._on_eof.set()

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeFilesApi:
    def __init__(self, payload: bytes = PAYLOAD, probe_headers: Optional[dict] = None) -> None:
        self.payload = payload
        self.probe_headers = probe_headers
        self.faults: dict[tuple[str, str, int], Fault] = {}
        self.attempts: Counter = Counter()
        self.requests: list[tuple[str, str]] = []
        self.cancelled: list[str] = []
        self.eof: dict[tuple[str, str], asyncio.Event] = {}

    def fail(self, rng: str, *attempts: int, fault: Fault, file: str = FILE) -> None:
        for attempt in attempts:
            self.faults[(file, rng, attempt)] = fault

    def _headers(self, file: str, extra: dict) -> dict:
        return {"content-disposition": f'attachment; filename="{file}.bin"', **extra}

    async def __call__(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        file = kwargs["params"]["file-group-id"]
        rng = (kwargs.get("headers") or {}).get("Range", "")
        attempt = self.attempts[(file, rng)]
        self.attempts[(file, rng)] += 1
        self.requests.append((file, rng))
        fault = self.faults.get((file, rng, attempt), Fault())

        if fault.wait_for is not None:
            try:
                await fault.wait_for.wait()
            except asyncio.CancelledError:
                self.cancelled.append(rng)
                raise
        if fault.connect is not None:
            raise fault.connect
        if fault.status is not None:
            return FakeResponse(fault.status, {}, error_text=fault.body)

        size = len(self.payload)
        if rng == PROBE:
            headers = self.probe_headers
            if headers is None:
                headers = {"Content-Range": f"bytes 0-0/{size}", "Content-Length": "1"}
            return FakeResponse(206, self._headers(file, headers), self.payload[:1])
        if not rng:
            return FakeResponse(200, self._headers(file, {"Content-Length": str(size)}), self.payload)

        start, end = (int(x) for x in rng.split("=")[1].split("-"))
        return FakeResponse(
            206,
            self._headers(file, {"Content-Range": f"bytes {start}-{end}/{size}"}),
            self.payload[start : end + 1],
            fail_after=fault.midstream_after,
            fail_exc=fault.midstream_exc,
            on_eof=self.eof.get((file, rng)),
        )

    def ranged(self, file: str = FILE) -> list[str]:
        return [r for f, r in self.requests if f == file and r not in ("", PROBE)]


class LogRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def __getattr__(self, level: str):
        def _log(event: str, *args: Any, **kw: Any) -> None:
            self.events.append((level, event, kw))

        return _log

    def find(self, level: str, event: str) -> list[dict]:
        return [kw for lvl, ev, kw in self.events if lvl == level and ev == event]


def payload_error(msg: str = "connection reset mid-body") -> aiohttp.ClientPayloadError:
    return aiohttp.ClientPayloadError(msg)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _low_threshold(monkeypatch):
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", THRESHOLD)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    calls: list[float] = []

    async def fake_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        calls.append(delay)
        await _REAL_SLEEP(0)

    monkeypatch.setattr(parallel.asyncio, "sleep", fake_sleep)
    return calls


@pytest.fixture
def log(monkeypatch) -> LogRecorder:
    rec = LogRecorder()
    monkeypatch.setattr(parallel, "logger", rec)
    return rec


def make_client(tmp_path: Path, monkeypatch, api: FakeFilesApi, **config: Any) -> DataQueryClient:
    cfg = ClientConfig(
        base_url="https://api.example.com",
        download_dir=str(tmp_path),
        enable_range_downloads=True,
        **config,
    )
    client = DataQueryClient(cfg)
    monkeypatch.setattr(client, "_make_authenticated_request", api)
    return client


def options(tmp_path: Path, max_retries: int = 0, retry_delay: float = 0.5) -> DownloadOptions:
    return DownloadOptions(
        destination_path=tmp_path,
        overwrite_existing=True,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )


async def drain() -> None:
    """Wait for orphaned range tasks a failed gather left running."""
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=5)


async def run(coro: Awaitable[T]) -> T:
    try:
        return await coro
    finally:
        await drain()


async def multipart(client: DataQueryClient, opts: DownloadOptions, **kw: Any) -> DownloadResult:
    return await run(parallel.download_file_multipart(client, FILE, None, opts, num_parts=2, **kw))


async def flattened(client: DataQueryClient, tmp_path: Path, **kw: Any) -> Optional[DownloadResult]:
    return await run(parallel.download_file_parallel(client, FILE, None, tmp_path, 2, asyncio.Semaphore(4), **kw))


def leftovers(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# Probe failures and unexpected probe responses
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flattened_probe_401_returns_none_and_logs_error(tmp_path, monkeypatch, log):
    api = FakeFilesApi()
    api.fail(PROBE, 0, fault=Fault(status=401))
    client = make_client(tmp_path, monkeypatch, api)

    assert await flattened(client, tmp_path) is None

    assert api.requests == [(FILE, PROBE)]  # no ranges, no single-stream fallback
    errors = log.find("error", "Flattened parallel download failed")
    assert len(errors) == 1
    assert errors[0]["file_group_id"] == FILE
    assert "Authentication failed" in errors[0]["error"]
    assert leftovers(tmp_path) == []


@pytest.mark.asyncio
async def test_multipart_probe_server_error_returns_failed_result(tmp_path, monkeypatch):
    api = FakeFilesApi()
    api.fail(PROBE, 0, fault=Fault(status=503, body='{"code": "DQ503", "description": "Files backend down"}'))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path))

    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None
    assert result.error_message.startswith("NetworkError: Server error: 503: [DQ503] Files backend down")
    assert result.bytes_downloaded == 0 and result.file_size == 0
    # No destination was resolved, so the result points at the placeholder path.
    assert result.local_path == tmp_path / f"{FILE}.tmp"
    assert api.requests == [(FILE, PROBE)]
    assert leftovers(tmp_path) == []


@pytest.mark.asyncio
async def test_multipart_probe_connection_error_returns_failed_result(tmp_path, monkeypatch):
    api = FakeFilesApi()
    api.fail(PROBE, 0, fault=Fault(connect=aiohttp.ClientConnectionError("connection refused")))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path))

    assert result.status == DownloadStatus.FAILED
    assert result.error_message == "ClientConnectionError: connection refused"
    assert api.requests == [(FILE, PROBE)]
    assert leftovers(tmp_path) == []


@pytest.mark.asyncio
async def test_probe_without_content_range_falls_back_to_single_stream(tmp_path, monkeypatch):
    # Server ignored the Range header (no Content-Range on the probe).
    api = FakeFilesApi(probe_headers={"Content-Length": str(len(PAYLOAD))})
    client = make_client(tmp_path, monkeypatch, api)

    result = await flattened(client, tmp_path)

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert api.requests == [(FILE, PROBE), (FILE, "")]
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_probe_unknown_total_uses_content_length_and_falls_back(tmp_path, monkeypatch):
    # "bytes 0-0/*" -> ValueError -> Content-Length (1 byte) is below the threshold.
    api = FakeFilesApi(probe_headers={"Content-Range": "bytes 0-0/*", "Content-Length": "1"})
    client = make_client(tmp_path, monkeypatch, api)

    probe = await parallel._probe_size(
        client, "https://x/download", {"file-group-id": FILE}, FILE, None, asyncio.Semaphore(1)
    )
    assert probe is None

    result = await flattened(client, tmp_path)
    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert api.ranged() == []
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD


@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:81-90 an unknown size ('bytes 0-0/*', no Content-Length) gives total_bytes=0, "
    "so no ranges are fetched and an empty file is reported COMPLETED",
)
@pytest.mark.asyncio
async def test_probe_unknown_total_without_content_length_downloads_full_file(tmp_path, monkeypatch):
    api = FakeFilesApi(probe_headers={"Content-Range": "bytes 0-0/*"})
    client = make_client(tmp_path, monkeypatch, api)

    result = await flattened(client, tmp_path)

    assert result is not None
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD


# --------------------------------------------------------------------------- #
# HTTP error statuses on range requests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("status", "error_type", "message"),
    [
        (401, "AuthenticationError", "Authentication failed"),
        (403, "AuthenticationError", "Access denied - insufficient permissions"),
        (404, "NotFoundError", "Resource not found"),
        (416, "ValidationError", "Client error: 416"),
        (429, "RateLimitError", "Rate limit exceeded: 429"),
        (502, "NetworkError", "Server error: 502"),
    ],
)
@pytest.mark.asyncio
async def test_range_http_error_maps_to_failed_result(tmp_path, monkeypatch, status, error_type, message):
    api = FakeFilesApi()
    api.fail(SECOND, 0, fault=Fault(status=status))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=0))

    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None
    assert result.error_message.startswith(f"{error_type}: {message}")
    assert api.attempts[(FILE, SECOND)] == 1
    assert result.local_path == tmp_path / f"{FILE}.bin"
    assert leftovers(tmp_path) == []  # temp .part discarded, nothing promoted


@pytest.mark.asyncio
async def test_range_v2_error_body_is_surfaced(tmp_path, monkeypatch):
    api = FakeFilesApi()
    api.fail(SECOND, 0, fault=Fault(status=500, body='{"errors": [{"code": 5001, "description": "Shard offline"}]}'))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path))

    assert result.error_message is not None
    assert result.error_message.startswith("NetworkError: Server error: 500: [5001] Shard offline")


@pytest.mark.asyncio
async def test_range_server_error_exhausts_retries(tmp_path, monkeypatch):
    api = FakeFilesApi()
    api.fail(SECOND, 0, 1, 2, fault=Fault(status=500))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=2))

    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith("NetworkError: Server error: 500")
    assert api.attempts[(FILE, SECOND)] == 3  # 1 try + max_retries
    assert leftovers(tmp_path) == []


@pytest.mark.asyncio
async def test_range_server_error_then_success_completes(tmp_path, monkeypatch):
    api = FakeFilesApi()
    api.fail(SECOND, 0, 1, fault=Fault(status=503))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=2))

    assert result.status == DownloadStatus.COMPLETED
    assert api.attempts[(FILE, SECOND)] == 3
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD


@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:200-202 every range exception is retried, including permanent 4xx errors like 404",
)
@pytest.mark.asyncio
async def test_range_not_found_is_not_retried(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, 0, 1, 2, 3, fault=Fault(status=404))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=3))

    assert result.status == DownloadStatus.FAILED
    assert api.attempts[(FILE, SECOND)] == 1


# --------------------------------------------------------------------------- #
# Connection and midstream errors, retry, backoff and progress rewind
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_range_connection_error_is_retried(tmp_path, monkeypatch):
    api = FakeFilesApi()
    api.fail(SECOND, 0, fault=Fault(connect=aiohttp.ClientConnectionError("reset")))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=1))

    assert result.status == DownloadStatus.COMPLETED
    assert api.attempts[(FILE, SECOND)] == 2
    assert api.attempts[(FILE, FIRST)] == 1
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_midstream_errors_back_off_exponentially_and_rewind_progress(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, 0, fault=Fault(midstream_after=4, midstream_exc=payload_error()))
    api.fail(SECOND, 1, fault=Fault(midstream_after=6, midstream_exc=payload_error()))
    client = make_client(tmp_path, monkeypatch, api)
    seen: list[int] = []

    result = await multipart(
        client,
        options(tmp_path, max_retries=3, retry_delay=0.5),
        progress_callback=lambda p: seen.append(p.bytes_downloaded),
    )

    assert result.status == DownloadStatus.COMPLETED
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD
    assert api.attempts[(FILE, SECOND)] == 3
    assert sleeps == [0.5, 1.0]  # retry_delay * 2 ** (attempt - 1)
    # The 4 + 6 bytes from failed attempts were rewound, not double counted.
    assert result.bytes_downloaded == len(PAYLOAD)
    assert seen and max(seen) <= len(PAYLOAD)


@pytest.mark.asyncio
async def test_midstream_error_exhausted_discards_partial_file(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, 0, 1, fault=Fault(midstream_after=4, midstream_exc=payload_error("truncated")))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=1, retry_delay=0.25))

    assert result.status == DownloadStatus.FAILED
    assert result.error_message == "ClientPayloadError: truncated"
    assert sleeps == [0.25]
    # Only the final attempt's 4 bytes stay counted (plus whatever the sibling got).
    assert 4 <= result.bytes_downloaded <= 14
    assert leftovers(tmp_path) == []


@pytest.mark.asyncio
async def test_flattened_midstream_error_exhausted_returns_none(tmp_path, monkeypatch, sleeps, log):
    api = FakeFilesApi()
    # download_file_parallel uses DownloadOptions defaults: max_retries=3, retry_delay=1.0.
    api.fail(SECOND, 0, 1, 2, 3, fault=Fault(midstream_after=2, midstream_exc=payload_error("eof")))
    client = make_client(tmp_path, monkeypatch, api)

    assert await flattened(client, tmp_path) is None

    assert api.attempts[(FILE, SECOND)] == 4
    assert sleeps == [1.0, 2.0, 4.0]
    errors = log.find("error", "Flattened parallel download failed")
    assert [e["error"] for e in errors] == ["eof"]
    assert leftovers(tmp_path) == []


@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:172 backoff only runs when the failed attempt wrote bytes; "
    "a connection error before the first byte is retried immediately",
)
@pytest.mark.asyncio
async def test_connection_error_before_first_byte_backs_off(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, 0, fault=Fault(connect=aiohttp.ClientConnectionError("reset")))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=1, retry_delay=0.5))

    assert result.status == DownloadStatus.COMPLETED
    assert sleeps == [0.5]


@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:172-175 part_bytes_written is reset before the retry, so a midstream failure "
    "followed by a pre-byte failure skips the second backoff",
)
@pytest.mark.asyncio
async def test_midstream_then_connection_error_backs_off_twice(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, 0, fault=Fault(midstream_after=4, midstream_exc=payload_error()))
    api.fail(SECOND, 1, fault=Fault(connect=aiohttp.ClientConnectionError("reset")))
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=2, retry_delay=0.5))

    assert result.status == DownloadStatus.COMPLETED
    assert sleeps == [0.5, 1.0]


@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:322/443 asyncio.gather without cancellation: when one range fails, "
    "sibling range requests keep running after the download has returned FAILED",
)
@pytest.mark.asyncio
async def test_failed_range_cancels_sibling_ranges(tmp_path, monkeypatch):
    gate = asyncio.Event()
    api = FakeFilesApi()
    api.fail(FIRST, 0, fault=Fault(wait_for=gate))
    api.fail(SECOND, 0, fault=Fault(status=500))
    client = make_client(tmp_path, monkeypatch, api)

    try:
        result = await parallel.download_file_multipart(client, FILE, None, options(tmp_path), num_parts=2)
        assert result.status == DownloadStatus.FAILED
        await _REAL_SLEEP(0)
        assert api.cancelled == [FIRST]
    finally:
        gate.set()
        await drain()


# --------------------------------------------------------------------------- #
# Salvage triggered by real failures
# --------------------------------------------------------------------------- #
def _error_after_full_range(api: FakeFilesApi) -> None:
    """SECOND writes all its bytes and then errors; it starts only once FIRST is done."""
    first_done = asyncio.Event()
    api.eof[(FILE, FIRST)] = first_done
    api.fail(SECOND, 0, fault=Fault(wait_for=first_done, midstream_after=10, midstream_exc=payload_error("late")))


@pytest.mark.asyncio
async def test_multipart_salvages_file_when_error_follows_last_byte(tmp_path, monkeypatch):
    api = FakeFilesApi()
    _error_after_full_range(api)
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path, max_retries=0))

    assert result.status == DownloadStatus.COMPLETED
    assert result.error_message is None
    assert result.bytes_downloaded == result.file_size == len(PAYLOAD)
    assert leftovers(tmp_path) == [f"{FILE}.bin"]
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_flattened_salvages_file_when_error_follows_last_byte(tmp_path, monkeypatch, sleeps, log):
    api = FakeFilesApi()
    _error_after_full_range(api)
    # Default max_retries=3: keep failing the late range so the error escapes the range worker.
    for attempt in (1, 2, 3):
        api.fail(SECOND, attempt, fault=Fault(midstream_after=10, midstream_exc=payload_error("late")))
    client = make_client(tmp_path, monkeypatch, api)

    result = await flattened(client, tmp_path)

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert (tmp_path / f"{FILE}.bin").read_bytes() == PAYLOAD
    assert log.find("error", "Flattened parallel download failed") == []


@pytest.mark.asyncio
async def test_multipart_promote_failure_logs_salvage_warning(tmp_path, monkeypatch, log):
    (tmp_path / f"{FILE}.bin").mkdir()  # the final rename onto a directory fails
    api = FakeFilesApi()
    client = make_client(tmp_path, monkeypatch, api)

    result = await multipart(client, options(tmp_path))

    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith("IsADirectoryError")
    warnings = log.find("warning", "Salvage cleanup failed")
    assert len(warnings) == 1
    assert warnings[0]["temp_path"] == str(tmp_path / f"{FILE}.bin.part")
    assert (tmp_path / f"{FILE}.bin").is_dir()


@pytest.mark.asyncio
async def test_flattened_promote_failure_returns_none(tmp_path, monkeypatch, log):
    (tmp_path / f"{FILE}.bin").mkdir()
    api = FakeFilesApi()
    client = make_client(tmp_path, monkeypatch, api, overwrite_existing=True)

    assert await flattened(client, tmp_path) is None

    assert len(log.find("warning", "Salvage cleanup failed")) == 1
    errors = log.find("error", "Flattened parallel download failed")
    assert len(errors) == 1 and "Is a directory" in errors[0]["error"]


def test_salvage_without_destination_derives_it_from_temp_path(tmp_path):
    client = DataQueryClient(ClientConfig(base_url="https://api.example.com"))
    temp = tmp_path / "report.csv.part"
    temp.write_bytes(b"abcd")

    result = parallel._salvage(client, FILE, None, temp, total_bytes=4, bytes_downloaded=4, start_time=0.0)

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert result.local_path == tmp_path / "report.csv"
    assert (tmp_path / "report.csv").read_bytes() == b"abcd"
    assert not temp.exists()


# --------------------------------------------------------------------------- #
# Error propagation
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_flattened_invalid_datetime_raises_before_any_request(tmp_path, monkeypatch):
    api = FakeFilesApi()
    client = make_client(tmp_path, monkeypatch, api)

    with pytest.raises(ValueError, match="Invalid file-datetime format"):
        await parallel.download_file_parallel(client, FILE, "2024-01-01", tmp_path, 2, asyncio.Semaphore(1))
    assert api.requests == []


@pytest.mark.asyncio
async def test_flattened_existing_file_reports_already_exists(tmp_path, monkeypatch):
    (tmp_path / f"{FILE}.bin").write_bytes(b"old")
    api = FakeFilesApi()
    client = make_client(tmp_path, monkeypatch, api)  # overwrite_existing defaults to False

    result = await flattened(client, tmp_path)

    assert result is not None and result.status == DownloadStatus.ALREADY_EXISTS
    assert result.error_message is not None and result.error_message.startswith("FileExistsError")
    assert api.requests == [(FILE, PROBE)]
    assert (tmp_path / f"{FILE}.bin").read_bytes() == b"old"


# --------------------------------------------------------------------------- #
# download_files_with_retry + _classify driven by real failures
# --------------------------------------------------------------------------- #
async def batch(client: DataQueryClient, tmp_path: Path, files: list[dict], max_retries: int = 2):
    return await run(
        parallel.download_files_with_retry(
            client,
            files,
            tmp_path,
            num_parts=2,
            global_semaphore=asyncio.Semaphore(4),
            intelligent_delay=0.0,
            base_retry_delay=2.0,
            max_retries=max_retries,
        )
    )


@pytest.mark.asyncio
async def test_batch_retry_recovers_file_whose_ranges_failed(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    # Exhaust the per-range retries (1 + 3) in the first pass only.
    api.fail(SECOND, 0, 1, 2, 3, fault=Fault(status=500), file="B")
    client = make_client(tmp_path, monkeypatch, api)

    ok, failed, retries = await batch(client, tmp_path, [{"file-group-id": "A"}, {"file-group-id": "B"}])

    assert failed == []
    assert retries == 1
    assert sorted(r.file_group_id for r in ok) == ["A", "B"]
    assert sleeps == [2.0]
    assert api.requests.count(("A", PROBE)) == 1
    assert api.requests.count(("B", PROBE)) == 2
    assert (tmp_path / "B.bin").read_bytes() == PAYLOAD
    assert leftovers(tmp_path) == ["A.bin", "B.bin"]


@pytest.mark.asyncio
async def test_batch_retry_exhausts_with_persistent_range_failure(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, *range(12), fault=Fault(status=500), file="B")
    client = make_client(tmp_path, monkeypatch, api)

    ok, failed, retries = await batch(client, tmp_path, [{"file-group-id": "A"}, {"file-group-id": "B"}])

    assert [r.file_group_id for r in ok] == ["A"]
    assert failed == [{"file-group-id": "B", "error": "download failed"}]
    assert retries == 2
    assert sleeps == [2.0, 4.0]
    assert api.requests.count(("B", PROBE)) == 3
    assert api.attempts[("B", SECOND)] == 12
    assert leftovers(tmp_path) == ["A.bin"]


@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:466-475 download_file_parallel returns None on failure, so _classify "
    "can only report 'download failed' instead of the real error",
)
@pytest.mark.asyncio
async def test_batch_failure_reason_preserves_range_error(tmp_path, monkeypatch, sleeps):
    api = FakeFilesApi()
    api.fail(SECOND, *range(4), fault=Fault(status=500), file="B")
    client = make_client(tmp_path, monkeypatch, api)

    _, failed, _ = await batch(client, tmp_path, [{"file-group-id": "B"}], max_retries=0)

    assert len(failed) == 1
    assert "NetworkError" in failed[0]["error"]


@pytest.mark.asyncio
async def test_batch_counts_already_existing_file_as_success(tmp_path, monkeypatch, sleeps):
    (tmp_path / "A.bin").write_bytes(b"old")
    api = FakeFilesApi()
    client = make_client(tmp_path, monkeypatch, api)

    ok, failed, retries = await batch(client, tmp_path, [{"file-group-id": "A"}])

    assert failed == [] and retries == 0 and sleeps == []
    assert [r.status for r in ok] == [DownloadStatus.ALREADY_EXISTS]
    assert api.ranged("A") == []
    assert (tmp_path / "A.bin").read_bytes() == b"old"


@pytest.mark.asyncio
async def test_batch_invalid_datetime_fails_every_attempt_without_requests(tmp_path, monkeypatch, sleeps, log):
    api = FakeFilesApi()
    client = make_client(tmp_path, monkeypatch, api)
    bad = {"file-group-id": "A", "file-datetime": "yesterday"}

    ok, failed, retries = await batch(client, tmp_path, [bad], max_retries=2)

    assert ok == []
    assert failed == [{**bad, "error": "download failed"}]
    assert retries == 2 and sleeps == [2.0, 4.0]
    assert api.requests == []
    errors = log.find("error", "Flattened parallel download failed")
    assert len(errors) == 3
    assert all(e["file_datetime"] == "yesterday" and "Invalid file-datetime" in e["error"] for e in errors)


# --------------------------------------------------------------------------- #
# dataquery.download.utils error paths
# --------------------------------------------------------------------------- #
def _stats() -> dict:
    return {"files_downloaded": 0, "files_skipped": 0, "download_failures": 0, "total_bytes_downloaded": 0}


def _client_returning(result: Any) -> SimpleNamespace:
    async def download_file_async(**kwargs: Any) -> Any:
        return result

    return SimpleNamespace(download_file_async=download_file_async)


@pytest.mark.asyncio
async def test_download_and_track_none_result_counts_failure(caplog):
    stats, done, failed = _stats(), set(), {"k": 1}
    caplog.set_level(logging.ERROR, logger=dl_utils.logger.name)

    await dl_utils.download_and_track(
        _client_returning(None), FILE, "20240101", "k", DownloadOptions(), stats, done, failed
    )

    assert failed == {"k": 2}
    assert stats["download_failures"] == 1 and stats["files_downloaded"] == 0
    assert done == set()
    assert "Download failed for 'FG1' for 20240101: No result" in caplog.text


@pytest.mark.asyncio
async def test_download_and_track_failed_result_logs_error_message(caplog):
    stats, done, failed = _stats(), set(), {}
    result = SimpleNamespace(status=DownloadStatus.FAILED, error_message="NetworkError: Server error: 500")
    caplog.set_level(logging.ERROR, logger=dl_utils.logger.name)

    await dl_utils.download_and_track(
        _client_returning(result), FILE, "20240101", "k", DownloadOptions(), stats, done, failed
    )

    assert failed == {"k": 1}
    assert stats["download_failures"] == 1
    assert "NetworkError: Server error: 500" in caplog.text


@pytest.mark.asyncio
async def test_download_and_track_ignores_unparseable_file_size():
    stats, done, failed = _stats(), set(), {"k": 3}
    result = SimpleNamespace(status=DownloadStatus.COMPLETED, file_size="n/a")

    await dl_utils.download_and_track(
        _client_returning(result), FILE, "20240101", "k", DownloadOptions(), stats, done, failed
    )

    assert stats["files_downloaded"] == 1
    assert stats["total_bytes_downloaded"] == 0
    assert done == {"k"} and failed == {}


def test_progress_wrapper_swallows_user_callback_error(caplog):
    stats = {"total_bytes_downloaded": 10}

    def broken(progress: Any) -> None:
        raise RuntimeError("callback exploded")

    wrapper = dl_utils.create_progress_wrapper(stats, broken)
    caplog.set_level(logging.ERROR, logger=dl_utils.logger.name)

    wrapper(SimpleNamespace(bytes_downloaded=25))  # type: ignore[arg-type]
    wrapper(SimpleNamespace(bytes_downloaded=5))  # type: ignore[arg-type]

    assert stats["total_bytes_downloaded"] == 25  # monotonic max
    assert caplog.text.count("Error in progress callback: callback exploded") == 2
