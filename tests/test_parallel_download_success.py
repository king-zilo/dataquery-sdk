"""Success-path coverage for dataquery.download.parallel.

Drives ``download_file_multipart``, ``download_file_parallel``,
``_download_range`` and ``_probe_size`` against an in-memory fake files server
that honours ``Range`` headers, using a real ``DataQueryClient`` whose
``_make_authenticated_request`` is replaced by the fake. Failure/retry paths
are intentionally out of scope here.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import pytest
from multidict import CIMultiDict

from dataquery import constants as C
from dataquery.core.client import DataQueryClient
from dataquery.download import parallel
from dataquery.types.models import (
    BandwidthThrottler,
    ClientConfig,
    DownloadOptions,
    DownloadProgress,
    DownloadStatus,
)

FILES_URL = "https://files.example.com/research/dataquery-authe/api/v2/group/file/download"
THRESHOLD = 64  # parallel path for payloads >= 64 bytes; smaller ones use single stream


def _payload(n: int) -> bytes:
    return bytes((i * 7 + 3) % 251 for i in range(n))


def _parse_range(value: str) -> tuple[int, int]:
    start, end = value.split("=", 1)[1].split("-", 1)
    return int(start), int(end)


# --------------------------------------------------------------------------- #
# Fake files server
# --------------------------------------------------------------------------- #
class _Content:
    def __init__(self, server: "FakeFilesServer", body: bytes, is_part: bool) -> None:
        self._server = server
        self._body = body
        self._is_part = is_part

    async def iter_chunked(self, n: int):
        self._server.chunk_sizes.append(n)
        if self._is_part and self._server.gate is not None:
            if self._server.in_flight >= self._server.gate_target:
                self._server.gate.set()
            await asyncio.wait_for(self._server.gate.wait(), timeout=5)
        for i in range(0, len(self._body), n):
            chunk = self._body[i : i + n]
            self._server.chunks_served.append(len(chunk))
            yield chunk


class _FakeResponse:
    def __init__(self, server: "FakeFilesServer", status: int, headers: dict, body: bytes, is_part: bool) -> None:
        self.status = status
        self.headers = CIMultiDict(headers)
        self.url = FILES_URL
        self.content = _Content(server, body, is_part)

    async def text(self) -> str:
        return ""


class _FakeCM:
    def __init__(self, server: "FakeFilesServer", resp: _FakeResponse) -> None:
        self._server = server
        self._resp = resp

    async def __aenter__(self) -> _FakeResponse:
        self._server.in_flight += 1
        self._server.peak_in_flight = max(self._server.peak_in_flight, self._server.in_flight)
        return self._resp

    async def __aexit__(self, *exc: Any) -> bool:
        self._server.in_flight -= 1
        return False


class FakeFilesServer:
    """Serves ``payload``; honours ``Range`` like a real 206-capable server."""

    def __init__(
        self,
        payload: bytes,
        *,
        filename: Optional[str] = "data.bin",
        content_type: Optional[str] = None,
        probe_headers: Optional[dict] = None,
        honour_part_ranges: bool = True,
        on_part_request: Optional[Callable[[], None]] = None,
    ) -> None:
        self.payload = payload
        self.filename = filename
        self.content_type = content_type
        self.probe_headers = probe_headers or {}
        self.honour_part_ranges = honour_part_ranges
        self.on_part_request = on_part_request
        self.requests: list[dict] = []
        self.chunk_sizes: list[int] = []
        self.chunks_served: list[int] = []
        self.in_flight = 0
        self.peak_in_flight = 0
        self.gate: Optional[asyncio.Event] = None
        self.gate_target = 0

    async def __call__(self, method: str, url: str, **kwargs: Any) -> _FakeCM:
        headers = dict(kwargs.get("headers") or {})
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": dict(kwargs.get("params") or {}),
                "headers": headers,
                "timeout": kwargs.get("timeout"),
            }
        )
        resp_headers: dict = {}
        if self.filename is not None:
            resp_headers["Content-Disposition"] = f'attachment; filename="{self.filename}"'
        if self.content_type is not None:
            resp_headers["Content-Type"] = self.content_type

        rng = headers.get("Range")
        is_probe = headers == C.PROBE_HEADERS and kwargs.get("timeout") is None
        total = len(self.payload)
        if rng and (is_probe or self.honour_part_ranges):
            start, end = _parse_range(rng)
            body = self.payload[start : end + 1]
            resp_headers["Content-Range"] = f"bytes {start}-{end}/{total}"
            resp_headers["Content-Length"] = str(len(body))
            status = 206
            if is_probe:
                for key, value in self.probe_headers.items():
                    resp_headers.pop(key, None)
                    if value is not None:
                        resp_headers[key] = value
        else:
            body = self.payload
            resp_headers["Content-Length"] = str(total)
            status = 200
        is_part = bool(rng) and not is_probe
        if is_part and self.on_part_request is not None:
            self.on_part_request()
        return _FakeCM(self, _FakeResponse(self, status, resp_headers, body, is_part))

    # convenience views ------------------------------------------------------
    @property
    def probe_requests(self) -> list[dict]:
        return [r for r in self.requests if r["headers"] == C.PROBE_HEADERS and r["timeout"] is None]

    @property
    def part_requests(self) -> list[dict]:
        return [r for r in self.requests if "Range" in r["headers"] and r["timeout"] is not None]

    @property
    def plain_requests(self) -> list[dict]:
        return [r for r in self.requests if "Range" not in r["headers"]]

    def part_ranges(self) -> list[tuple[int, int]]:
        return sorted(_parse_range(r["headers"]["Range"]) for r in self.part_requests)


def _make_client(
    tmp_path: Path, server: FakeFilesServer, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> DataQueryClient:
    cfg: dict[str, Any] = {
        "base_url": "https://api.example.com",
        "files_base_url": "https://files.example.com",
        "oauth_enabled": False,
        "download_dir": str(tmp_path / "default-downloads"),
        "enable_range_downloads": True,
        "timeout": 42.0,
    }
    cfg.update(overrides)
    client = DataQueryClient(ClientConfig.model_validate(cfg))
    monkeypatch.setattr(client, "_make_authenticated_request", server)
    return client


def _options(**fields: Any) -> DownloadOptions:
    return DownloadOptions.model_validate(fields)


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch: pytest.MonkeyPatch):
    """Low size threshold, time-independent progress dispatch, and no real sleeps."""
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", THRESHOLD)
    monkeypatch.setattr(parallel.C, "CALLBACK_TIME_THRESHOLD", 10_000.0)
    real_sleep = asyncio.sleep
    delays: list[float] = []

    async def _no_wait_sleep(delay: float, *args: Any, **kwargs: Any) -> None:
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(parallel.asyncio, "sleep", _no_wait_sleep)
    yield
    # Success paths must never back off.
    assert [d for d in delays if d > 0] == []


# =========================================================================== #
# download_file_multipart
# =========================================================================== #
@pytest.mark.asyncio
async def test_multipart_assembles_byte_exact_file_and_returns_completed_result(tmp_path, monkeypatch):
    payload = _payload(1024)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    dest_dir = tmp_path / "out"

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=dest_dir),
        num_parts=4,
    )

    dest = dest_dir / "data.bin"
    assert dest.read_bytes() == payload
    assert result.status == DownloadStatus.COMPLETED
    assert result.file_group_id == "FG1"
    assert result.local_path == dest
    assert result.file_size == 1024
    assert result.bytes_downloaded == 1024
    assert result.error_message is None
    assert result.download_time is not None and result.download_time >= 0
    # Temp file was promoted; nothing else left behind.
    assert sorted(p.name for p in dest_dir.iterdir()) == ["data.bin"]


@pytest.mark.asyncio
async def test_multipart_sends_probe_then_one_range_request_per_part(tmp_path, monkeypatch):
    payload = _payload(1024)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime="20240102",
        options=_options(destination_path=tmp_path),
        num_parts=4,
    )

    assert server.requests[0]["headers"] == {"Range": "bytes=0-0"}
    assert server.requests[0]["timeout"] is None
    assert len(server.probe_requests) == 1
    assert server.part_ranges() == [(0, 255), (256, 511), (512, 767), (768, 1023)]
    assert server.plain_requests == []
    for req in server.requests:
        assert req["method"] == "GET"
        assert req["url"] == FILES_URL
        assert req["params"] == {"file-group-id": "FG1", "file-datetime": "20240102"}
    for req in server.part_requests:
        timeout = req["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total is None
        assert timeout.sock_read == 42.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "size, num_parts",
    [(64, 1), (100, 3), (1000, 7), (97, 97), (4096, 16)],
)
async def test_multipart_part_count_and_ranges_follow_file_size(tmp_path, monkeypatch, size, num_parts):
    payload = _payload(size)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=num_parts,
    )

    ranges = server.part_ranges()
    assert len(ranges) == num_parts
    assert ranges == parallel._compute_ranges(size, num_parts)
    assert ranges[0][0] == 0 and ranges[-1][1] == size - 1
    assert sum(end - start + 1 for start, end in ranges) == size
    assert (tmp_path / "data.bin").read_bytes() == payload
    assert result.bytes_downloaded == size == result.file_size


@pytest.mark.asyncio
async def test_multipart_streams_with_option_chunk_size(tmp_path, monkeypatch):
    payload = _payload(1000)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path, chunk_size=64),
        num_parts=2,
    )

    assert server.chunk_sizes == [64, 64]
    # 2 parts of 500 bytes -> 7 full chunks + one 52-byte tail each.
    assert sorted(server.chunks_served) == sorted([64] * 14 + [52, 52])
    assert (tmp_path / "data.bin").read_bytes() == payload


@pytest.mark.asyncio
async def test_multipart_progress_callback_is_monotonic_and_reaches_total(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel.C, "CALLBACK_BYTE_THRESHOLD", 1)
    payload = _payload(1000)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    seen: list[tuple[str, int, int, float]] = []

    def on_progress(p: DownloadProgress) -> None:
        seen.append((p.file_group_id, p.bytes_downloaded, p.total_bytes, p.percentage))

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path, chunk_size=100),
        num_parts=4,
        progress_callback=on_progress,
    )

    assert result.status == DownloadStatus.COMPLETED
    # One dispatch per chunk written (threshold of 1 byte).
    assert len(seen) == len(server.chunks_served) == 12
    downloaded = [b for _, b, _, _ in seen]
    assert downloaded == sorted(downloaded)
    assert {fg for fg, _, _, _ in seen} == {"FG1"}
    assert {t for _, _, t, _ in seen} == {1000}
    assert seen[-1][1:] == (1000, 1000, 100.0)


@pytest.mark.asyncio
async def test_multipart_progress_callback_fires_once_at_completion_below_byte_threshold(tmp_path, monkeypatch):
    payload = _payload(500)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    seen: list[tuple[int, float]] = []

    await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path, chunk_size=50),
        num_parts=5,
        progress_callback=lambda p: seen.append((p.bytes_downloaded, p.percentage)),
    )

    # 500 bytes never crosses the 1MB byte threshold; only "reached total" dispatches.
    assert seen == [(500, 100.0)]


@pytest.mark.asyncio
async def test_multipart_writes_to_explicit_file_path_creating_parents(tmp_path, monkeypatch):
    payload = _payload(300)
    server = FakeFilesServer(payload, filename="server-name.bin")
    client = _make_client(tmp_path, server, monkeypatch)
    dest = tmp_path / "a" / "b" / "custom.csv"

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=dest),
        num_parts=3,
    )

    assert result.local_path == dest
    assert dest.read_bytes() == payload
    assert sorted(p.name for p in dest.parent.iterdir()) == ["custom.csv"]


@pytest.mark.asyncio
async def test_multipart_defaults_to_config_download_dir(tmp_path, monkeypatch):
    payload = _payload(200)
    server = FakeFilesServer(payload, filename="report.zip")
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(),
        num_parts=2,
    )

    expected = tmp_path / "default-downloads" / "report.zip"
    assert result.local_path == expected
    assert expected.read_bytes() == payload


@pytest.mark.asyncio
async def test_multipart_derives_filename_from_id_datetime_and_content_type(tmp_path, monkeypatch):
    payload = _payload(200)
    server = FakeFilesServer(payload, filename=None, content_type="text/csv; charset=utf-8")
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime="20240102",
        options=_options(destination_path=tmp_path),
        num_parts=2,
    )

    assert result.local_path == tmp_path / "FG1_20240102.csv"
    assert (tmp_path / "FG1_20240102.csv").read_bytes() == payload


@pytest.mark.asyncio
async def test_multipart_strips_directory_components_from_server_filename(tmp_path, monkeypatch):
    payload = _payload(200)
    server = FakeFilesServer(payload, filename="../../escape.bin")
    client = _make_client(tmp_path, server, monkeypatch)
    dest_dir = tmp_path / "jail" / "inner"

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=dest_dir),
        num_parts=2,
    )

    assert result.local_path == dest_dir / "escape.bin"
    assert (dest_dir / "escape.bin").read_bytes() == payload
    assert not (tmp_path / "escape.bin").exists()


@pytest.mark.asyncio
async def test_multipart_overwrite_existing_replaces_larger_old_file(tmp_path, monkeypatch):
    payload = _payload(256)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"old" * 1000)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path, overwrite_existing=True),
        num_parts=4,
    )

    assert result.status == DownloadStatus.COMPLETED
    assert dest.read_bytes() == payload
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.bin"]


@pytest.mark.asyncio
async def test_multipart_keeps_existing_file_without_overwrite(tmp_path, monkeypatch):
    server = FakeFilesServer(_payload(256))
    client = _make_client(tmp_path, server, monkeypatch)
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"keep me")

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path, overwrite_existing=False),
        num_parts=4,
    )

    assert result.status == DownloadStatus.ALREADY_EXISTS
    assert result.local_path == dest
    assert result.file_size == 0 and result.bytes_downloaded == 0
    assert result.error_message is not None and "File already exists" in result.error_message
    assert dest.read_bytes() == b"keep me"
    assert len(server.requests) == 1  # probe only; no parts fetched
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.bin"]


@pytest.mark.asyncio
async def test_multipart_streams_into_preallocated_part_file_then_promotes(tmp_path, monkeypatch):
    payload = _payload(400)
    dest = tmp_path / "data.bin"
    temp = tmp_path / ("data.bin" + C.TEMP_SUFFIX)
    observations: list[tuple[bool, bool, int]] = []

    def observe() -> None:
        observations.append((dest.exists(), temp.exists(), temp.stat().st_size if temp.exists() else -1))

    server = FakeFilesServer(payload, on_part_request=observe)
    client = _make_client(tmp_path, server, monkeypatch)

    await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=4,
    )

    # Every part was fetched while only the full-size temp file existed.
    assert observations == [(False, True, 400)] * 4
    assert dest.read_bytes() == payload
    assert not temp.exists()


@pytest.mark.asyncio
async def test_multipart_runs_all_parts_concurrently(tmp_path, monkeypatch):
    payload = _payload(800)
    server = FakeFilesServer(payload)
    server.gate = asyncio.Event()
    server.gate_target = 4  # no part streams until all 4 are open at once
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=4,
    )

    assert server.peak_in_flight == 4
    assert result.status == DownloadStatus.COMPLETED
    assert (tmp_path / "data.bin").read_bytes() == payload


@pytest.mark.asyncio
async def test_multipart_falls_back_to_single_stream_without_content_range(tmp_path, monkeypatch):
    payload = _payload(500)
    server = FakeFilesServer(payload, probe_headers={"Content-Range": None})
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=4,
    )

    assert result.status == DownloadStatus.COMPLETED
    assert (tmp_path / "data.bin").read_bytes() == payload
    assert len(server.probe_requests) == 1
    assert server.part_requests == []
    assert len(server.plain_requests) == 1


@pytest.mark.asyncio
async def test_multipart_falls_back_to_single_stream_below_threshold(tmp_path, monkeypatch):
    payload = _payload(THRESHOLD - 1)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=4,
    )

    assert result.status == DownloadStatus.COMPLETED
    assert result.file_size == THRESHOLD - 1
    assert (tmp_path / "data.bin").read_bytes() == payload
    assert server.part_requests == []
    assert len(server.plain_requests) == 1


@pytest.mark.asyncio
async def test_multipart_uses_content_length_when_total_is_not_numeric(tmp_path, monkeypatch):
    payload = _payload(600)
    server = FakeFilesServer(payload, probe_headers={"Content-Range": "bytes 0-0/*", "Content-Length": "600"})
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=3,
    )

    assert server.part_ranges() == [(0, 199), (200, 399), (400, 599)]
    assert result.file_size == 600
    assert (tmp_path / "data.bin").read_bytes() == payload


@pytest.mark.asyncio
async def test_client_download_file_async_routes_to_multipart_when_ranges_enabled(tmp_path, monkeypatch):
    payload = _payload(900)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await client.download_file_async(
        "FG1",
        options=_options(destination_path=tmp_path),
        num_parts=3,
    )

    assert result.status == DownloadStatus.COMPLETED
    assert server.part_ranges() == [(0, 299), (300, 599), (600, 899)]
    assert (tmp_path / "data.bin").read_bytes() == payload


# =========================================================================== #
# _probe_size
# =========================================================================== #
@pytest.mark.asyncio
async def test_probe_size_returns_total_and_filename_under_global_semaphore(tmp_path, monkeypatch):
    sem = asyncio.Semaphore(1)
    locked_during_request: list[bool] = []
    server = FakeFilesServer(_payload(5000), filename="probe.csv")
    original_call = server.__call__

    async def recording(method: str, url: str, **kwargs: Any) -> _FakeCM:
        locked_during_request.append(sem.locked())
        return await original_call(method, url, **kwargs)

    client = _make_client(tmp_path, server, monkeypatch)
    monkeypatch.setattr(client, "_make_authenticated_request", recording)
    params = {"file-group-id": "FG1"}

    probe = await parallel._probe_size(client, FILES_URL, params, "FG1", None, sem)

    assert probe == parallel._ProbeResult(total_bytes=5000, filename="probe.csv")
    assert server.requests == [
        {"method": "GET", "url": FILES_URL, "params": params, "headers": C.PROBE_HEADERS, "timeout": None}
    ]
    assert locked_during_request == [True]
    assert not sem.locked()


@pytest.mark.asyncio
async def test_probe_size_falls_back_to_content_length(tmp_path, monkeypatch):
    server = FakeFilesServer(_payload(10), probe_headers={"Content-Range": "bytes 0-0/*", "Content-Length": "7777"})
    client = _make_client(tmp_path, server, monkeypatch)

    probe = await parallel._probe_size(client, FILES_URL, {}, "FG1", "20240102", asyncio.Semaphore(1))

    assert probe is not None
    assert probe.total_bytes == 7777
    assert probe.filename == "data.bin"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_headers, size",
    [
        ({"Content-Range": None}, 5000),  # server ignored the probe range
        ({"Content-Range": "bytes 0-0"}, 5000),  # no "/total" part
        ({}, THRESHOLD - 1),  # small file: single stream is cheaper
    ],
    ids=["no-content-range", "no-total", "below-threshold"],
)
async def test_probe_size_returns_none_when_parallel_not_applicable(tmp_path, monkeypatch, probe_headers, size):
    server = FakeFilesServer(_payload(size), probe_headers=probe_headers)
    client = _make_client(tmp_path, server, monkeypatch)
    sem = asyncio.Semaphore(1)

    assert await parallel._probe_size(client, FILES_URL, {}, "FG1", None, sem) is None
    assert not sem.locked()


# =========================================================================== #
# _download_range
# =========================================================================== #
class _RecordingThrottler(BandwidthThrottler):
    def __init__(self) -> None:
        super().__init__(max_bytes_per_second=None)
        self.calls: list[int] = []

    async def throttle(self, bytes_to_transfer: int) -> None:
        self.calls.append(bytes_to_transfer)


def _range_ctx(
    client: DataQueryClient, temp: Path, total: int, *, chunk_size: int, throttler: Optional[BandwidthThrottler] = None
) -> parallel._RangeContext:
    reporter = parallel._ProgressReporter(
        progress=DownloadProgress(file_group_id="FG1", total_bytes=total, start_time=datetime.now()),
        total_bytes=total,
        progress_callback=None,
        show_progress=False,
        file_group_id="FG1",
    )
    return parallel._RangeContext(
        client=client,
        url=FILES_URL,
        params={"file-group-id": "FG1"},
        temp_path=temp,
        chunk_size=chunk_size,
        semaphore=asyncio.Semaphore(1),
        timeout=aiohttp.ClientTimeout(total=None, sock_read=9.0),
        throttler=throttler,
        reporter=reporter,
        max_retries=2,
        retry_delay=1.0,
    )


@pytest.mark.asyncio
async def test_download_range_writes_slice_at_offset_in_preallocated_file(tmp_path, monkeypatch):
    payload = _payload(100)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    temp = tmp_path / "blob.part"
    parallel._preallocate_file(temp, 100)
    ctx = _range_ctx(client, temp, 100, chunk_size=7)

    await parallel._download_range(ctx, 30, 59)

    data = temp.read_bytes()
    assert len(data) == 100
    assert data[30:60] == payload[30:60]
    assert data[:30] == b"\x00" * 30 and data[60:] == b"\x00" * 40
    assert ctx.reporter.bytes_downloaded == 30
    assert server.chunk_sizes == [7]
    assert server.chunks_served == [7, 7, 7, 7, 2]
    assert len(server.requests) == 1  # single attempt, no retries
    req = server.requests[0]
    assert req["headers"] == {"Range": "bytes=30-59"}
    assert req["params"] == {"file-group-id": "FG1"}
    assert req["timeout"] is ctx.timeout


@pytest.mark.asyncio
async def test_download_range_passes_every_chunk_through_throttler(tmp_path, monkeypatch):
    payload = _payload(100)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    temp = tmp_path / "blob.part"
    parallel._preallocate_file(temp, 100)
    throttler = _RecordingThrottler()
    ctx = _range_ctx(client, temp, 100, chunk_size=16, throttler=throttler)

    await parallel._download_range(ctx, 0, 49)

    assert throttler.calls == [16, 16, 16, 2]
    assert temp.read_bytes()[:50] == payload[:50]


@pytest.mark.asyncio
async def test_download_ranges_assemble_full_file_in_any_completion_order(tmp_path, monkeypatch):
    payload = _payload(90)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    temp = tmp_path / "blob.part"
    parallel._preallocate_file(temp, 90)
    ctx = _range_ctx(client, temp, 90, chunk_size=8)

    for start, end in reversed(parallel._compute_ranges(90, 3)):
        await parallel._download_range(ctx, start, end)

    assert temp.read_bytes() == payload
    assert ctx.reporter.bytes_downloaded == 90


# =========================================================================== #
# download_file_parallel
# =========================================================================== #
@pytest.mark.asyncio
async def test_parallel_assembles_byte_exact_file_and_returns_completed_result(tmp_path, monkeypatch):
    payload = _payload(2048)
    server = FakeFilesServer(payload, filename="big.parquet")
    client = _make_client(tmp_path, server, monkeypatch)
    dest_dir = tmp_path / "out"
    sem = asyncio.Semaphore(3)

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime="20240102T0930",
        destination_path=dest_dir,
        num_parts=4,
        global_semaphore=sem,
    )

    dest = dest_dir / "big.parquet"
    assert result is not None
    assert result.status == DownloadStatus.COMPLETED
    assert result.file_group_id == "FG1"
    assert result.local_path == dest
    assert result.file_size == 2048 == result.bytes_downloaded
    assert result.error_message is None
    assert dest.read_bytes() == payload
    assert sorted(p.name for p in dest_dir.iterdir()) == ["big.parquet"]
    # Probe first, then one range request per part, all with the same params.
    assert server.requests[0]["headers"] == C.PROBE_HEADERS
    assert server.part_ranges() == [(0, 511), (512, 1023), (1024, 1535), (1536, 2047)]
    assert server.plain_requests == []
    for req in server.requests:
        assert req["url"] == FILES_URL
        assert req["params"] == {"file-group-id": "FG1", "file-datetime": "20240102T0930"}
    for req in server.part_requests:
        assert req["timeout"].sock_read == 42.0
        assert req["timeout"].total is None
    assert server.chunk_sizes == [C.DEFAULT_CHUNK_SIZE] * 4
    assert not sem.locked()


@pytest.mark.asyncio
async def test_parallel_global_semaphore_bounds_probe_and_parts(tmp_path, monkeypatch):
    payload = _payload(1000)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=5,
        global_semaphore=asyncio.Semaphore(1),
    )

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert server.peak_in_flight == 1
    assert len(server.part_requests) == 5
    assert (tmp_path / "data.bin").read_bytes() == payload


@pytest.mark.asyncio
async def test_parallel_shared_semaphore_across_concurrent_files(tmp_path, monkeypatch):
    payload_a, payload_b = _payload(600), _payload(900)[::-1]
    server_a = FakeFilesServer(payload_a, filename="a.bin")
    server_b = FakeFilesServer(payload_b, filename="b.bin")
    in_flight = {"now": 0, "peak": 0}

    def tracked(server: FakeFilesServer):
        async def call(method: str, url: str, **kwargs: Any):
            cm = await server(method, url, **kwargs)

            class _Tracked:
                async def __aenter__(self):
                    in_flight["now"] += 1
                    in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
                    await asyncio.sleep(0)
                    return await cm.__aenter__()

                async def __aexit__(self, *exc: Any):
                    in_flight["now"] -= 1
                    return await cm.__aexit__(*exc)

            return _Tracked()

        return call

    client_a = _make_client(tmp_path, server_a, monkeypatch)
    client_b = _make_client(tmp_path, server_b, monkeypatch)
    monkeypatch.setattr(client_a, "_make_authenticated_request", tracked(server_a))
    monkeypatch.setattr(client_b, "_make_authenticated_request", tracked(server_b))
    sem = asyncio.Semaphore(2)

    res_a, res_b = await asyncio.gather(
        parallel.download_file_parallel(client_a, "FA", None, tmp_path, 3, sem),
        parallel.download_file_parallel(client_b, "FB", None, tmp_path, 3, sem),
    )

    assert res_a is not None and res_a.status == DownloadStatus.COMPLETED
    assert res_b is not None and res_b.status == DownloadStatus.COMPLETED
    assert (tmp_path / "a.bin").read_bytes() == payload_a
    assert (tmp_path / "b.bin").read_bytes() == payload_b
    assert in_flight["peak"] <= 2
    assert not sem.locked()


@pytest.mark.asyncio
async def test_parallel_overwrites_existing_when_config_allows(tmp_path, monkeypatch):
    payload = _payload(300)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch, overwrite_existing=True)
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"stale" * 200)

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=3,
        global_semaphore=asyncio.Semaphore(3),
    )

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert dest.read_bytes() == payload
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.bin"]


@pytest.mark.asyncio
async def test_parallel_reports_already_exists_and_keeps_file_when_config_forbids_overwrite(tmp_path, monkeypatch):
    server = FakeFilesServer(_payload(300))
    client = _make_client(tmp_path, server, monkeypatch, overwrite_existing=False)
    dest = tmp_path / "data.bin"
    dest.write_bytes(b"original")

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=3,
        global_semaphore=asyncio.Semaphore(3),
    )

    assert result is not None
    assert result.status == DownloadStatus.ALREADY_EXISTS
    assert result.local_path == dest
    assert result.bytes_downloaded == 0
    assert dest.read_bytes() == b"original"
    assert len(server.requests) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.bin"]


@pytest.mark.asyncio
async def test_parallel_writes_to_explicit_file_destination(tmp_path, monkeypatch):
    payload = _payload(256)
    server = FakeFilesServer(payload, filename="ignored.bin")
    client = _make_client(tmp_path, server, monkeypatch)
    dest = tmp_path / "nested" / "named.csv"

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=dest,
        num_parts=2,
        global_semaphore=asyncio.Semaphore(2),
    )

    assert result is not None and result.local_path == dest
    assert dest.read_bytes() == payload
    assert sorted(p.name for p in dest.parent.iterdir()) == ["named.csv"]


@pytest.mark.asyncio
async def test_parallel_progress_callback_reaches_total(tmp_path, monkeypatch):
    payload = _payload(400)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)
    seen: list[tuple[int, int, float]] = []

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=4,
        global_semaphore=asyncio.Semaphore(4),
        progress_callback=lambda p: seen.append((p.bytes_downloaded, p.total_bytes, p.percentage)),
    )

    assert result is not None and result.bytes_downloaded == 400
    assert seen == [(400, 400, 100.0)]


@pytest.mark.asyncio
async def test_parallel_logs_progress_when_no_callback(tmp_path, monkeypatch):
    class _RecordingLogger:
        def __init__(self) -> None:
            self.debug_calls: list[tuple[str, dict]] = []

        def debug(self, event: str, **kw: Any) -> None:
            self.debug_calls.append((event, kw))

        def info(self, *a: Any, **k: Any) -> None:
            pass

        warning = error = info

    rec = _RecordingLogger()
    monkeypatch.setattr(parallel, "logger", rec)
    server = FakeFilesServer(_payload(400))
    client = _make_client(tmp_path, server, monkeypatch)

    await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=2,
        global_semaphore=asyncio.Semaphore(2),
    )

    progress_logs = [kw for event, kw in rec.debug_calls if event == "Download progress (parallel)"]
    assert progress_logs == [{"file": "FG1", "percentage": "100.0%", "downloaded": "400.00 B"}]


@pytest.mark.asyncio
async def test_parallel_falls_back_to_single_stream_for_small_file(tmp_path, monkeypatch):
    payload = _payload(THRESHOLD - 1)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=4,
        global_semaphore=asyncio.Semaphore(2),
    )

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert (tmp_path / "data.bin").read_bytes() == payload
    assert len(server.probe_requests) == 1
    assert server.part_requests == []
    assert len(server.plain_requests) == 1


@pytest.mark.asyncio
async def test_parallel_non_positive_num_parts_downloads_single_stream_without_probe(tmp_path, monkeypatch):
    payload = _payload(500)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        destination_path=tmp_path,
        num_parts=0,
        global_semaphore=asyncio.Semaphore(2),
    )

    assert result is not None and result.status == DownloadStatus.COMPLETED
    assert (tmp_path / "data.bin").read_bytes() == payload
    assert [r["headers"] for r in server.requests] == [{}]


# =========================================================================== #
# Findings (suspected production bugs) — strict xfails, prod code unchanged
# =========================================================================== #
@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "_download_range never checks for a 206/Content-Range reply: if a part request is answered "
        "200 with the full body (Range ignored), every part writes the whole file at its own offset, "
        "the output grows past total_bytes and the download is still reported COMPLETED"
    ),
)
async def test_multipart_does_not_report_corrupt_file_as_completed_when_part_range_ignored(tmp_path, monkeypatch):
    payload = _payload(400)
    server = FakeFilesServer(payload, honour_part_ranges=False)
    client = _make_client(tmp_path, server, monkeypatch)

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=None,
        options=_options(destination_path=tmp_path),
        num_parts=4,
    )

    dest = tmp_path / "data.bin"
    assert result.status != DownloadStatus.COMPLETED or dest.read_bytes() == payload
