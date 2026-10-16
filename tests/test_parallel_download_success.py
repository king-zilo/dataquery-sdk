"""Success-path coverage for dataquery.download.parallel.

Drives ``download_file_multipart`` (via ``DataQueryClient.download_file_async``)
and ``download_file_parallel`` end to end through a real ``DataQueryClient``
whose transport is an in-memory server that honours ``Range: bytes=a-b``.
Every request in this module succeeds; failure/retry paths live elsewhere.
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
import pytest
from multidict import CIMultiDict

from dataquery.core.client import DataQueryClient
from dataquery.download import parallel
from dataquery.types.models import ClientConfig, DownloadOptions, DownloadResult, DownloadStatus

THRESHOLD = 64
FG = "FG_PARALLEL"
FILES_URL = "https://files.example.com/research/dataquery-authe/api/v2/group/file/download"


def _payload(size: int, seed: int = 1234) -> bytes:
    return random.Random(seed).randbytes(size)


def _expected_ranges(size: int, num_parts: int) -> List[Tuple[int, int]]:
    part = size // num_parts
    ranges = [(i * part, i * part + part - 1) for i in range(num_parts - 1)]
    ranges.append(((num_parts - 1) * part, size - 1))
    return ranges


def _parse_range(header: str) -> Tuple[int, int]:
    assert header.startswith("bytes=")
    start, end = header[len("bytes=") :].split("-")
    return int(start), int(end)


class InFlight:
    """Counts concurrently open responses (shareable across servers)."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self.current -= 1


class FakeResponse:
    def __init__(
        self,
        server: "FakeFilesServer",
        status: int,
        headers: Any,
        body: bytes,
        start: Optional[int],
    ) -> None:
        self._server = server
        self.status = status
        self.headers = headers
        self.url = FILES_URL
        self._body = body
        self._start = start
        self.content = self

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        self._server.iter_chunk_sizes.append(n)
        await self._server.wait_turn(self._start)
        step = min(n, self._server.chunk_cap) if self._server.chunk_cap else n
        for i in range(0, len(self._body), step):
            await asyncio.sleep(0)
            self._server.chunks_served[self._start] += 1
            yield self._body[i : i + step]

    async def text(self) -> str:  # pragma: no cover - only used on HTTP errors
        return ""

    async def __aenter__(self) -> "FakeResponse":
        self._server.in_flight.enter()
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        self._server.in_flight.exit()
        self._server.mark_done(self._start)
        return False


class FakeFilesServer:
    """In-memory files endpoint replacing ``client._make_authenticated_request``."""

    def __init__(
        self,
        payload: bytes,
        *,
        filename: Optional[str] = "data.bin",
        content_type: str = "application/octet-stream",
        supports_ranges: bool = True,
        unknown_total: bool = False,
        lowercase_plain_headers: bool = False,
        chunk_cap: Optional[int] = None,
        completion_order: Optional[List[int]] = None,
        in_flight: Optional[InFlight] = None,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        self.payload = payload
        self.filename = filename
        self.content_type = content_type
        self.supports_ranges = supports_ranges
        self.unknown_total = unknown_total
        self.lowercase_plain_headers = lowercase_plain_headers
        self.chunk_cap = chunk_cap
        self.completion_order = completion_order
        self.in_flight = in_flight or InFlight()
        self.semaphore = semaphore
        self.calls: List[Dict[str, Any]] = []
        self.iter_chunk_sizes: List[int] = []
        self.chunks_served: Dict[Optional[int], int] = defaultdict(int)
        self.done_order: List[Optional[int]] = []
        self._done: Dict[Optional[int], asyncio.Event] = defaultdict(asyncio.Event)

    async def wait_turn(self, start: Optional[int]) -> None:
        if self.completion_order is None or start not in self.completion_order:
            return
        idx = self.completion_order.index(start)
        if idx > 0:
            await self._done[self.completion_order[idx - 1]].wait()

    def mark_done(self, start: Optional[int]) -> None:
        self.done_order.append(start)
        self._done[start].set()

    def _headers(self, values: Dict[str, str]) -> Any:
        if self.filename is not None:
            values["Content-Disposition"] = f'attachment; filename="{self.filename}"'
        values["Content-Type"] = self.content_type
        if self.lowercase_plain_headers:
            return {k.lower(): v for k, v in values.items()}
        return CIMultiDict(values)

    @property
    def range_headers(self) -> List[str]:
        return [c["headers"]["Range"] for c in self.calls if "Range" in c["headers"]][1:]

    async def __call__(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        headers = dict(kwargs.get("headers") or {})
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": dict(kwargs.get("params") or {}),
                "headers": headers,
                "timeout": kwargs.get("timeout"),
                "semaphore_locked": self.semaphore.locked() if self.semaphore else None,
            }
        )
        size = len(self.payload)
        rng = headers.get("Range")
        if rng is None or not self.supports_ranges:
            return FakeResponse(self, 200, self._headers({"Content-Length": str(size)}), self.payload, None)
        start, end = _parse_range(rng)
        body = self.payload[start : end + 1]
        total = "*" if self.unknown_total else str(size)
        values = {"Content-Range": f"bytes {start}-{end}/{total}", "Content-Length": str(len(body))}
        return FakeResponse(self, 206, self._headers(values), body, None if rng == "bytes=0-0" else start)


def _make_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: FakeFilesServer, **cfg: Any
) -> DataQueryClient:
    config = ClientConfig(
        base_url="https://api.example.com",
        context_path="/research/dataquery-authe/api/v2",
        files_base_url="https://files.example.com",
        files_context_path="/research/dataquery-authe/api/v2",
        oauth_enabled=False,
        download_dir=str(tmp_path / "default-downloads"),
        enable_range_downloads=True,
        **cfg,
    )
    client = DataQueryClient(config)
    monkeypatch.setattr(client, "_make_authenticated_request", server)
    return client


@pytest.fixture(autouse=True)
def _low_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", THRESHOLD)


def _assert_completed(result: Optional[DownloadResult], path: Path, payload: bytes) -> None:
    assert result is not None
    assert result.status == DownloadStatus.COMPLETED
    assert result.error_message is None
    assert result.file_group_id == FG
    assert result.local_path == path
    assert result.file_size == len(payload)
    assert result.bytes_downloaded == len(payload)
    assert path.read_bytes() == payload
    assert list(path.parent.rglob("*.part")) == []


def _assert_ranges(server: FakeFilesServer, size: int, num_parts: int) -> None:
    requested = sorted(_parse_range(h) for h in server.range_headers)
    assert requested == _expected_ranges(size, num_parts)
    assert requested[0][0] == 0
    assert requested[-1][1] == size - 1
    for (_s1, e1), (s2, _e2) in zip(requested, requested[1:]):
        assert s2 == e1 + 1


async def _multipart(
    client: DataQueryClient,
    dest: Path,
    num_parts: int,
    file_datetime: Optional[str] = None,
    progress_callback: Any = None,
    **opts: Any,
) -> DownloadResult:
    return await client.download_file_async(
        file_group_id=FG,
        file_datetime=file_datetime,
        options=DownloadOptions(destination_path=dest, **opts),
        num_parts=num_parts,
        progress_callback=progress_callback,
    )


async def _parallel(
    client: DataQueryClient,
    dest: Path,
    num_parts: int,
    semaphore: Optional[asyncio.Semaphore] = None,
    file_datetime: Optional[str] = None,
    progress_callback: Any = None,
) -> Optional[DownloadResult]:
    return await parallel.download_file_parallel(
        client=client,
        file_group_id=FG,
        file_datetime=file_datetime,
        destination_path=dest,
        num_parts=num_parts,
        global_semaphore=semaphore or asyncio.Semaphore(16),
        progress_callback=progress_callback,
    )


SIZES_AND_PARTS = [(1000, 4), (1001, 3), (997, 7), (4096, 2), (257, 5), (THRESHOLD, 8), (1000, 10)]


# --------------------------------------------------------------------------- #
# download_file_multipart
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("size,num_parts", SIZES_AND_PARTS)
async def test_multipart_assembles_exact_bytes(tmp_path, monkeypatch, size, num_parts):
    payload = _payload(size)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)
    dest = tmp_path / "out"

    result = await _multipart(client, dest, num_parts)

    _assert_completed(result, dest / "data.bin", payload)
    _assert_ranges(server, size, num_parts)
    assert len(server.calls) == 1 + num_parts
    assert server.calls[0]["headers"] == {"Range": "bytes=0-0"}
    assert all(c["method"] == "GET" and c["url"] == FILES_URL for c in server.calls)


@pytest.mark.asyncio
async def test_multipart_last_part_takes_remainder(tmp_path, monkeypatch):
    payload = _payload(103)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)

    await _multipart(client, tmp_path, 4)

    assert server.range_headers == ["bytes=0-24", "bytes=25-49", "bytes=50-74", "bytes=75-102"]


@pytest.mark.asyncio
async def test_multipart_range_requests_carry_params_and_sock_read_timeout(tmp_path, monkeypatch):
    payload = _payload(500)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server, timeout=42.0)

    await _multipart(client, tmp_path, 3)

    assert [c["params"] for c in server.calls] == [{"file-group-id": FG}] * 4
    assert server.calls[0]["timeout"] is None
    for call in server.calls[1:]:
        assert call["timeout"] == aiohttp.ClientTimeout(total=None, sock_read=42.0)


@pytest.mark.asyncio
async def test_multipart_small_chunk_size_spans_many_chunks(tmp_path, monkeypatch):
    payload = _payload(1000)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _multipart(client, tmp_path, 3, chunk_size=7)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert server.iter_chunk_sizes == [7, 7, 7]
    # 333, 333 and 334 bytes in 7-byte chunks -> 48 + 48 + 48 chunks.
    assert dict(server.chunks_served) == {0: 48, 333: 48, 666: 48}


@pytest.mark.asyncio
async def test_multipart_out_of_order_completion_lands_at_right_offsets(tmp_path, monkeypatch):
    payload = _payload(800)
    starts_reversed = [s for s, _ in reversed(_expected_ranges(800, 4))]
    server = FakeFilesServer(payload, completion_order=starts_reversed, chunk_cap=50)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _multipart(client, tmp_path, 4)

    assert server.done_order == [None] + starts_reversed
    _assert_completed(result, tmp_path / "data.bin", payload)


@pytest.mark.asyncio
async def test_multipart_accepts_lowercase_content_range(tmp_path, monkeypatch):
    payload = _payload(640)
    server = FakeFilesServer(payload, filename="lower.csv", lowercase_plain_headers=True)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _multipart(client, tmp_path, 4)

    _assert_completed(result, tmp_path / "lower.csv", payload)
    _assert_ranges(server, 640, 4)


@pytest.mark.asyncio
async def test_multipart_forwards_file_datetime_and_default_filename(tmp_path, monkeypatch):
    payload = _payload(300)
    server = FakeFilesServer(payload, filename=None, content_type="text/csv; charset=utf-8")
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _multipart(client, tmp_path, 3, file_datetime="20240115")

    _assert_completed(result, tmp_path / f"{FG}_20240115.csv", payload)
    assert all(c["params"] == {"file-group-id": FG, "file-datetime": "20240115"} for c in server.calls)


@pytest.mark.asyncio
async def test_multipart_explicit_file_destination_wins_over_disposition(tmp_path, monkeypatch):
    payload = _payload(200)
    server = FakeFilesServer(payload, filename="server-name.bin")
    client = _make_client(tmp_path, monkeypatch, server)
    target = tmp_path / "nested" / "dir" / "chosen.dat"

    result = await _multipart(client, target, 2)

    _assert_completed(result, target, payload)
    assert not (tmp_path / "nested" / "dir" / "server-name.bin").exists()


@pytest.mark.asyncio
async def test_multipart_progress_callback_ends_at_100_percent(tmp_path, monkeypatch):
    payload = _payload(900)
    server = FakeFilesServer(payload, chunk_cap=60)
    client = _make_client(tmp_path, monkeypatch, server)
    seen: List[Tuple[str, int, int, float]] = []

    def on_progress(p):
        seen.append((p.file_group_id, p.bytes_downloaded, p.total_bytes, p.percentage))

    result = await _multipart(client, tmp_path, 3, progress_callback=on_progress)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert seen, "progress callback never fired"
    assert seen[-1] == (FG, 900, 900, 100.0)
    downloaded = [s[1] for s in seen]
    assert downloaded == sorted(downloaded)
    assert all(s[2] == 900 for s in seen)


@pytest.mark.asyncio
async def test_multipart_overwrite_existing_replaces_file(tmp_path, monkeypatch):
    payload = _payload(400)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)
    existing = tmp_path / "data.bin"
    existing.write_bytes(b"OLD" * 1000)

    result = await _multipart(client, tmp_path, 4, overwrite_existing=True)

    _assert_completed(result, existing, payload)


@pytest.mark.asyncio
async def test_multipart_logs_progress_when_no_callback(tmp_path, monkeypatch):
    payload = _payload(256)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)

    logged: List[Tuple[str, Dict[str, Any]]] = []
    monkeypatch.setattr(parallel, "logger", SimpleNamespace(debug=lambda event, **kw: logged.append((event, kw))))

    result = await _multipart(client, tmp_path, 2, show_progress=True)

    _assert_completed(result, tmp_path / "data.bin", payload)
    events = [kw for event, kw in logged if event == "Download progress (parallel)"]
    assert events and events[-1]["percentage"] == "100.0%"
    assert events[-1]["file"] == FG


@pytest.mark.asyncio
async def test_multipart_size_equal_to_threshold_uses_ranges(tmp_path, monkeypatch):
    payload = _payload(THRESHOLD)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _multipart(client, tmp_path, 2)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert server.range_headers == ["bytes=0-31", "bytes=32-63"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_kwargs,size",
    [
        pytest.param({"supports_ranges": False}, 500, id="no-content-range"),
        pytest.param({}, THRESHOLD - 1, id="below-threshold"),
        pytest.param({"unknown_total": True}, 500, id="unknown-total-uses-content-length"),
    ],
)
async def test_multipart_probe_fallback_to_single_stream(tmp_path, monkeypatch, server_kwargs, size):
    payload = _payload(size)
    server = FakeFilesServer(payload, **server_kwargs)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _multipart(client, tmp_path, 4)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert [c["headers"] for c in server.calls] == [{"Range": "bytes=0-0"}, {}]
    assert server.calls[1]["params"] == {"file-group-id": FG}


# --------------------------------------------------------------------------- #
# download_file_parallel / _probe_size
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("size,num_parts", SIZES_AND_PARTS)
async def test_parallel_assembles_exact_bytes(tmp_path, monkeypatch, size, num_parts):
    payload = _payload(size, seed=size)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)
    dest = tmp_path / "out"

    result = await _parallel(client, dest, num_parts)

    _assert_completed(result, dest / "data.bin", payload)
    _assert_ranges(server, size, num_parts)
    assert len(server.calls) == 1 + num_parts
    assert server.calls[0]["headers"] == {"Range": "bytes=0-0"}
    assert all(c["url"] == FILES_URL and c["params"] == {"file-group-id": FG} for c in server.calls)


@pytest.mark.asyncio
async def test_parallel_multi_chunk_ranges_assemble(tmp_path, monkeypatch):
    payload = _payload(1000)
    server = FakeFilesServer(payload, chunk_cap=13)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, 3)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert server.iter_chunk_sizes == [parallel.C.DEFAULT_CHUNK_SIZE] * 3
    # 333/333/334 bytes served 13 at a time -> 26 chunks each.
    assert dict(server.chunks_served) == {0: 26, 333: 26, 666: 26}


@pytest.mark.asyncio
async def test_parallel_forwards_file_datetime_and_disposition_name(tmp_path, monkeypatch):
    payload = _payload(500)
    server = FakeFilesServer(payload, filename="report_20240115.zip")
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, 4, file_datetime="20240115T0930")

    _assert_completed(result, tmp_path / "report_20240115.zip", payload)
    assert all(c["params"] == {"file-group-id": FG, "file-datetime": "20240115T0930"} for c in server.calls)


@pytest.mark.asyncio
async def test_parallel_default_filename_without_disposition(tmp_path, monkeypatch):
    payload = _payload(300)
    server = FakeFilesServer(payload, filename=None, content_type="application/json")
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, 2, file_datetime="20240115T093000")

    _assert_completed(result, tmp_path / f"{FG}_20240115T093000.json", payload)


@pytest.mark.asyncio
async def test_parallel_range_requests_use_sock_read_timeout(tmp_path, monkeypatch):
    server = FakeFilesServer(_payload(400))
    client = _make_client(tmp_path, monkeypatch, server, timeout=17.0)

    await _parallel(client, tmp_path, 4)

    assert server.calls[0]["timeout"] is None
    assert {c["timeout"] for c in server.calls[1:]} == {aiohttp.ClientTimeout(total=None, sock_read=17.0)}


@pytest.mark.asyncio
async def test_parallel_respects_global_semaphore(tmp_path, monkeypatch):
    payload = _payload(1200)
    server = FakeFilesServer(payload, chunk_cap=20)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, 6, semaphore=asyncio.Semaphore(2))

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert len(server.range_headers) == 6
    assert server.in_flight.peak == 2


@pytest.mark.asyncio
async def test_parallel_probe_and_ranges_hold_the_semaphore(tmp_path, monkeypatch):
    payload = _payload(600)
    sem = asyncio.Semaphore(1)
    server = FakeFilesServer(payload, chunk_cap=25, semaphore=sem)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, 4, semaphore=sem)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert len(server.calls) == 5
    assert all(c["semaphore_locked"] for c in server.calls)
    assert server.in_flight.peak == 1
    assert not sem.locked()


@pytest.mark.asyncio
async def test_parallel_semaphore_is_shared_across_concurrent_files(tmp_path, monkeypatch):
    shared = InFlight()
    sem = asyncio.Semaphore(3)
    payloads = [_payload(700, seed=1), _payload(900, seed=2)]
    servers = [FakeFilesServer(p, filename=f"f{i}.bin", chunk_cap=30, in_flight=shared) for i, p in enumerate(payloads)]
    clients = [_make_client(tmp_path, monkeypatch, s) for s in servers]

    results = await asyncio.gather(
        _parallel(clients[0], tmp_path, 5, semaphore=sem),
        _parallel(clients[1], tmp_path, 5, semaphore=sem),
    )

    for i, (res, p) in enumerate(zip(results, payloads)):
        assert res is not None and res.status == DownloadStatus.COMPLETED
        assert (tmp_path / f"f{i}.bin").read_bytes() == p
    assert shared.peak == 3
    assert shared.current == 0


@pytest.mark.asyncio
async def test_parallel_progress_callback_ends_at_total(tmp_path, monkeypatch):
    payload = _payload(750)
    server = FakeFilesServer(payload, chunk_cap=40)
    client = _make_client(tmp_path, monkeypatch, server)
    seen: List[Tuple[int, int, float]] = []

    def on_progress(p):
        seen.append((p.bytes_downloaded, p.total_bytes, p.percentage))

    result = await _parallel(client, tmp_path, 5, progress_callback=on_progress)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert seen and seen[-1] == (750, 750, 100.0)
    assert [s[0] for s in seen] == sorted(s[0] for s in seen)


@pytest.mark.asyncio
async def test_parallel_overwrite_existing_from_client_config(tmp_path, monkeypatch):
    payload = _payload(350)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server, overwrite_existing=True)
    existing = tmp_path / "data.bin"
    existing.write_bytes(b"stale-bytes" * 500)

    result = await _parallel(client, tmp_path, 3)

    _assert_completed(result, existing, payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server_kwargs,size",
    [
        pytest.param({"supports_ranges": False}, 500, id="no-content-range"),
        pytest.param({}, THRESHOLD - 1, id="below-threshold"),
        pytest.param({"unknown_total": True}, 500, id="unknown-total-uses-content-length"),
    ],
)
async def test_parallel_probe_fallback_to_single_stream(tmp_path, monkeypatch, server_kwargs, size):
    payload = _payload(size)
    server = FakeFilesServer(payload, **server_kwargs)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, 4)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert [c["headers"] for c in server.calls] == [{"Range": "bytes=0-0"}, {}]


@pytest.mark.asyncio
@pytest.mark.parametrize("num_parts", [0, -3])
async def test_parallel_non_positive_parts_downloads_single_stream(tmp_path, monkeypatch, num_parts):
    payload = _payload(500)
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)

    result = await _parallel(client, tmp_path, num_parts)

    _assert_completed(result, tmp_path / "data.bin", payload)
    assert [c["headers"] for c in server.calls] == [{}]


@pytest.mark.asyncio
async def test_probe_size_reads_total_and_filename(tmp_path, monkeypatch):
    server = FakeFilesServer(_payload(1234), filename="probe.bin")
    client = _make_client(tmp_path, monkeypatch, server)
    sem = asyncio.Semaphore(1)

    probe = await parallel._probe_size(client, FILES_URL, {"file-group-id": FG}, FG, None, sem)

    assert probe == parallel._ProbeResult(total_bytes=1234, filename="probe.bin")
    assert server.calls[0]["headers"] == {"Range": "bytes=0-0"}
    assert not sem.locked()


# --------------------------------------------------------------------------- #
# Findings (not fixed): pinned expected-correct behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "parallel.py:44-55 _compute_ranges returns [] when num_parts > total_bytes, so no range "
        "is requested and the zero-filled preallocated file is reported COMPLETED"
    ),
)
@pytest.mark.parametrize("entry", ["multipart", "parallel"])
async def test_more_parts_than_bytes_still_downloads_payload(tmp_path, monkeypatch, entry):
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", 1)
    payload = b"\x01\x02\x03\x04\x05"
    server = FakeFilesServer(payload)
    client = _make_client(tmp_path, monkeypatch, server)

    if entry == "multipart":
        result: Optional[DownloadResult] = await _multipart(client, tmp_path, 8)
    else:
        result = await _parallel(client, tmp_path, 8)

    _assert_completed(result, tmp_path / "data.bin", payload)
