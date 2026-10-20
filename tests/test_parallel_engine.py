"""Characterization and reliability tests for ``dataquery.download.parallel``.

Every public entry point (``download_file_multipart``, ``download_file_parallel``,
``download_files_with_retry``) runs end to end against an in-memory files server
that honours ``Range`` headers and injects faults per request and per attempt.
A real ``DataQueryClient`` is used, with ``_make_authenticated_request`` patched,
so HTTP errors go through the real ``_handle_response``.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
import typing
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from multidict import CIMultiDict

import dataquery.download as download_pkg
from dataquery import constants as C
from dataquery.core.client import DataQueryClient
from dataquery.download import parallel
from dataquery.types.models import (
    BandwidthThrottler,
    ClientConfig,
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
)

FILES_URL = "https://files.example.com/research/dataquery-authe/api/v2/group/file/download"
THRESHOLD = 64
SIZE = 400


def _payload(n: int = SIZE) -> bytes:
    return bytes((i * 7 + 3) % 251 for i in range(n))


def _parse_range(value: str) -> tuple[int, int]:
    start, end = value.split("=", 1)[1].split("-", 1)
    return int(start), int(end)


# --------------------------------------------------------------------------- #
# Fake files server
# --------------------------------------------------------------------------- #
@dataclass
class Reply:
    """Per-request override; fields left as None keep the default behaviour."""

    status: int | None = None
    body: bytes | None = None
    headers: dict[str, str | None] | None = None
    raise_before: BaseException | None = None
    fail_after: int | None = None
    truncate_to: int | None = None
    wait_for: asyncio.Event | None = None
    set_on_eof: asyncio.Event | None = None


class _Content:
    def __init__(self, body: bytes, reply: Reply) -> None:
        self._body = body
        self._reply = reply

    async def iter_chunked(self, n: int):
        reply = self._reply
        if reply.wait_for is not None:
            await reply.wait_for.wait()
        body = self._body if reply.truncate_to is None else self._body[: reply.truncate_to]
        sent = 0
        for i in range(0, len(body), n):
            chunk = body[i : i + n]
            if reply.fail_after is not None and sent + len(chunk) > reply.fail_after:
                chunk = chunk[: reply.fail_after - sent]
                if chunk:
                    yield chunk
                raise aiohttp.ClientPayloadError("connection reset mid-body")
            sent += len(chunk)
            yield chunk
        if reply.set_on_eof is not None:
            reply.set_on_eof.set()
        if reply.fail_after is not None and sent >= reply.fail_after:
            raise aiohttp.ClientPayloadError("connection reset after body")


class _FakeResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes, reply: Reply) -> None:
        self.status = status
        self.headers = CIMultiDict(headers)
        self.url = FILES_URL
        self.content = _Content(body, reply)

    async def text(self) -> str:
        return ""


class _FakeCM:
    def __init__(self, server: FakeServer, resp: _FakeResponse) -> None:
        self._server = server
        self._resp = resp

    async def __aenter__(self) -> _FakeResponse:
        self._server.in_flight += 1
        self._server.peak_in_flight = max(self._server.peak_in_flight, self._server.in_flight)
        return self._resp

    async def __aexit__(self, *exc: Any) -> bool:
        self._server.in_flight -= 1
        return False


class FakeServer:
    """Serves the same payload for every file group, honouring ``Range``."""

    def __init__(self, payload: bytes | None = None, *, ignore_part_ranges: bool = False) -> None:
        self.payload = _payload() if payload is None else payload
        self.ignore_part_ranges = ignore_part_ranges
        self.requests: list[dict[str, Any]] = []
        self.script: dict[tuple[str, str, int | None], list[Reply | None]] = {}
        self.in_flight = 0
        self.peak_in_flight = 0
        self._attempts: Counter[tuple[str, str, int | None]] = Counter()

    def on(self, kind: str, *replies: Reply | None, start: int | None = None, fgid: str = "FG") -> None:
        self.script[(fgid, kind, start)] = list(replies)

    async def __call__(self, method: str, url: str, **kwargs: Any) -> _FakeCM:
        headers = dict(kwargs.get("headers") or {})
        params = dict(kwargs.get("params") or {})
        timeout = kwargs.get("timeout")
        fgid = params.get("file-group-id", "")
        rng = headers.get("Range")
        if headers == C.PROBE_HEADERS and timeout is None:
            kind, start = "probe", None
        elif rng:
            kind, start = "part", _parse_range(rng)[0]
        else:
            kind, start = "plain", None
        key = (fgid, kind, start)
        attempt = self._attempts[key]
        self._attempts[key] += 1
        self.requests.append(
            {"kind": kind, "fgid": fgid, "range": rng, "params": params, "timeout": timeout, "attempt": attempt}
        )
        script = self.script.get(key, [])
        reply = (script[attempt] if attempt < len(script) else None) or Reply()
        if reply.raise_before is not None:
            raise reply.raise_before

        total = len(self.payload)
        resp_headers: dict[str, str] = {"Content-Disposition": f'attachment; filename="{fgid}.bin"'}
        if kind == "plain" or (kind == "part" and self.ignore_part_ranges):
            status, body = 200, self.payload
            resp_headers["Content-Length"] = str(total)
        else:
            assert rng is not None
            s, e = _parse_range(rng)
            status, body = 206, self.payload[s : e + 1]
            resp_headers["Content-Range"] = f"bytes {s}-{e}/{total}"
            resp_headers["Content-Length"] = str(len(body))
        if reply.status is not None:
            status = reply.status
        if reply.body is not None:
            body = reply.body
        for k, v in (reply.headers or {}).items():
            resp_headers.pop(k, None)
            if v is not None:
                resp_headers[k] = v
        return _FakeCM(self, _FakeResponse(status, resp_headers, body, reply))

    def of(self, kind: str, fgid: str = "FG") -> list[dict[str, Any]]:
        return [r for r in self.requests if r["kind"] == kind and r["fgid"] == fgid]

    def part_attempts(self, start: int, fgid: str = "FG") -> int:
        return self._attempts[(fgid, "part", start)]


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _small_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", THRESHOLD)
    monkeypatch.setattr(parallel.C, "CALLBACK_TIME_THRESHOLD", 1e9)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every positive ``asyncio.sleep`` delay instead of sleeping."""
    real_sleep = asyncio.sleep
    recorded: list[float] = []

    async def fake_sleep(delay: float, result: Any = None) -> Any:
        if delay > 0:
            recorded.append(delay)
        await real_sleep(0)
        return result

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return recorded


def make_client(server: FakeServer, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **cfg: Any) -> DataQueryClient:
    config: dict[str, Any] = {
        "base_url": "https://api.example.com",
        "files_base_url": "https://files.example.com",
        "oauth_enabled": False,
        "download_dir": str(tmp_path / "default"),
        "enable_range_downloads": True,
        "timeout": 42.0,
    }
    config.update(cfg)
    client = DataQueryClient(ClientConfig.model_validate(config))
    monkeypatch.setattr(client, "_make_authenticated_request", server)
    return client


def opts(dest: Path, **fields: Any) -> DownloadOptions:
    base: dict[str, Any] = {"destination_path": dest, "chunk_size": 32, "retry_delay": 0.5}
    base.update(fields)
    return DownloadOptions.model_validate(base)


async def multipart(client: DataQueryClient, dest: Path, num_parts: int = 4, **fields: Any) -> DownloadResult:
    return await parallel.download_file_multipart(
        client=client,
        file_group_id="FG",
        file_datetime=None,
        options=opts(dest, **fields),
        num_parts=num_parts,
    )


async def flat(
    client: DataQueryClient, dest: Path, num_parts: int = 4, sem: asyncio.Semaphore | None = None, **kw: Any
) -> DownloadResult | None:
    return await parallel.download_file_parallel(
        client=client,
        file_group_id="FG",
        file_datetime=kw.pop("file_datetime", None),
        destination_path=dest,
        num_parts=num_parts,
        global_semaphore=sem or asyncio.Semaphore(8),
        **kw,
    )


async def run_entry(entry: str, client: DataQueryClient, dest: Path) -> DownloadResult | None:
    if entry == "multipart":
        return await multipart(client, dest)
    return await flat(client, dest)


async def drain_orphans() -> None:
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current]
    if pending:
        await asyncio.wait(pending, timeout=5)


def part_files(root: Path) -> list[Path]:
    return list(root.rglob(f"*{C.TEMP_SUFFIX}"))


# --------------------------------------------------------------------------- #
# Public API contract
# --------------------------------------------------------------------------- #
PUBLIC_SIGNATURES = {
    "download_file_parallel": [
        ("client", inspect.Parameter.empty),
        ("file_group_id", inspect.Parameter.empty),
        ("file_datetime", inspect.Parameter.empty),
        ("destination_path", inspect.Parameter.empty),
        ("num_parts", inspect.Parameter.empty),
        ("global_semaphore", inspect.Parameter.empty),
        ("progress_callback", None),
    ],
    "download_file_multipart": [
        ("client", inspect.Parameter.empty),
        ("file_group_id", inspect.Parameter.empty),
        ("file_datetime", inspect.Parameter.empty),
        ("options", inspect.Parameter.empty),
        ("num_parts", inspect.Parameter.empty),
        ("progress_callback", None),
    ],
    "download_files_with_retry": [
        ("client", inspect.Parameter.empty),
        ("files", inspect.Parameter.empty),
        ("destination_dir", inspect.Parameter.empty),
        ("num_parts", inspect.Parameter.empty),
        ("global_semaphore", inspect.Parameter.empty),
        ("intelligent_delay", inspect.Parameter.empty),
        ("base_retry_delay", inspect.Parameter.empty),
        ("max_retries", inspect.Parameter.empty),
        ("progress_callback", None),
        ("on_file_complete", None),
    ],
}


@pytest.mark.parametrize("name", sorted(PUBLIC_SIGNATURES))
def test_public_signature_is_unchanged(name: str) -> None:
    fn = getattr(parallel, name)
    assert inspect.iscoroutinefunction(fn)
    params = inspect.signature(fn).parameters.values()
    assert [(p.name, p.default) for p in params] == PUBLIC_SIGNATURES[name]
    assert {p.kind for p in params} == {inspect.Parameter.POSITIONAL_OR_KEYWORD}


def test_public_type_hints_are_unchanged() -> None:
    def hints(fn: Any) -> dict[str, Any]:
        return typing.get_type_hints(fn, localns={"DataQueryClient": DataQueryClient})

    flat_hints = hints(parallel.download_file_parallel)
    assert flat_hints["return"] == DownloadResult | None
    assert flat_hints["file_datetime"] == str | None
    assert flat_hints["destination_path"] is Path
    assert flat_hints["global_semaphore"] is asyncio.Semaphore
    assert flat_hints["client"] is DataQueryClient

    multi_hints = hints(parallel.download_file_multipart)
    assert multi_hints["return"] is DownloadResult
    assert multi_hints["options"] is DownloadOptions

    batch_hints = hints(parallel.download_files_with_retry)
    assert batch_hints["return"] == tuple[list[DownloadResult], list[dict], int]
    assert batch_hints["files"] == list[dict]


def test_package_exports_are_unchanged() -> None:
    assert download_pkg.__all__ == [
        "create_progress_wrapper",
        "download_and_track",
        "download_file_parallel",
        "file_exists_locally",
    ]
    assert download_pkg.download_file_parallel is parallel.download_file_parallel


# --------------------------------------------------------------------------- #
# _compute_ranges
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("total", "parts"),
    [(t, p) for t in (1, 2, 7, 64, 400, 1001) for p in (1, 2, 3, 4, 7) if p <= t],
)
def test_compute_ranges_is_contiguous_and_complete(total: int, parts: int) -> None:
    ranges = parallel._compute_ranges(total, parts)
    assert len(ranges) == parts
    assert ranges[0][0] == 0 and ranges[-1][1] == total - 1
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
        assert next_start == prev_end + 1
    sizes = [e - s + 1 for s, e in ranges]
    assert sizes[:-1] == [total // parts] * (parts - 1)


@pytest.mark.xfail(strict=True, reason="F2: returns [] when num_parts > total_bytes")
def test_compute_ranges_caps_parts_at_byte_count() -> None:
    assert parallel._compute_ranges(3, 5) == [(0, 0), (1, 1), (2, 2)]


@pytest.mark.xfail(strict=True, raises=ZeroDivisionError, reason="F2: num_parts=0 divides by zero")
def test_compute_ranges_treats_zero_parts_as_one() -> None:
    assert parallel._compute_ranges(10, 0) == [(0, 9)]


def test_compute_ranges_empty_file() -> None:
    assert parallel._compute_ranges(0, 4) == []


# --------------------------------------------------------------------------- #
# download_file_multipart — happy path and fallbacks
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multipart_downloads_all_ranges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    seen: list[float] = []

    result = await parallel.download_file_multipart(
        client=client,
        file_group_id="FG",
        file_datetime="20240102",
        options=opts(tmp_path / "out"),
        num_parts=4,
        progress_callback=lambda p: seen.append(p.percentage),
    )

    dest = tmp_path / "out" / "FG.bin"
    assert result.status is DownloadStatus.COMPLETED
    assert result.local_path == dest
    assert result.file_size == SIZE and result.bytes_downloaded == SIZE
    assert result.error_message is None
    assert dest.read_bytes() == server.payload
    assert part_files(tmp_path) == []
    assert seen == [100.0]

    probe, *parts = server.requests
    assert probe["kind"] == "probe" and probe["timeout"] is None
    assert probe["params"] == {"file-group-id": "FG", "file-datetime": "20240102"}
    assert sorted(r["range"] for r in parts) == [
        "bytes=0-99",
        "bytes=100-199",
        "bytes=200-299",
        "bytes=300-399",
    ]
    assert {r["timeout"] for r in parts} == {aiohttp.ClientTimeout(total=None, sock_read=42.0)}
    assert server.peak_in_flight <= 4


@pytest.mark.asyncio
async def test_multipart_destination_with_suffix_is_used_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    target = tmp_path / "nested" / "custom.dat"
    result = await multipart(client, target)
    assert result.status is DownloadStatus.COMPLETED
    assert target.read_bytes() == server.payload


@pytest.mark.asyncio
async def test_multipart_logs_progress_when_no_callback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class _Log:
        def debug(self, event: str, **_: Any) -> None:
            events.append(event)

        def __getattr__(self, _name: str) -> Any:
            return lambda *a, **k: None

    monkeypatch.setattr(parallel, "logger", _Log())
    client = make_client(FakeServer(), monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out", show_progress=True)
    assert result.status is DownloadStatus.COMPLETED
    assert events == ["Download progress (parallel)"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_headers",
    [
        {"Content-Range": None},
        {"Content-Range": "bytes 0-0"},
        {"Content-Range": "bytes 0-0/*", "Content-Length": "1"},
        {"Content-Range": f"bytes 0-0/{THRESHOLD - 1}"},
    ],
    ids=["no-content-range", "no-total", "unknown-total-with-length", "below-threshold"],
)
async def test_multipart_falls_back_to_single_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe_headers: dict[str, str | None]
) -> None:
    server = FakeServer()
    server.on("probe", Reply(headers=probe_headers))
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload
    assert [r["kind"] for r in server.requests] == ["probe", "plain"]


@pytest.mark.asyncio
@pytest.mark.parametrize("num_parts", [0, 1])
async def test_client_uses_single_stream_for_one_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, num_parts: int
) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    result = await client.download_file_async("FG", options=opts(tmp_path / "out"), num_parts=num_parts)
    assert result.status is DownloadStatus.COMPLETED
    assert [r["kind"] for r in server.requests] == ["plain"]


@pytest.mark.asyncio
async def test_client_routes_multi_part_downloads_to_multipart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    result = await client.download_file_async("FG", options=opts(tmp_path / "out"), num_parts=3)
    assert result.status is DownloadStatus.COMPLETED
    assert len(server.of("part")) == 3


@pytest.mark.asyncio
async def test_multipart_existing_file_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    dest = tmp_path / "out" / "FG.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"old")
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.ALREADY_EXISTS
    assert result.error_message is not None and result.error_message.startswith("FileExistsError")
    assert dest.read_bytes() == b"old"
    assert server.of("part") == []


@pytest.mark.asyncio
async def test_multipart_overwrites_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    dest = tmp_path / "out" / "FG.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"old")
    result = await multipart(client, tmp_path / "out", overwrite_existing=True)
    assert result.status is DownloadStatus.COMPLETED
    assert dest.read_bytes() == server.payload


@pytest.mark.asyncio
async def test_multipart_probe_http_error_is_failed_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("probe", Reply(status=404))
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith("NotFoundError")
    assert result.bytes_downloaded == 0
    assert server.of("part") == []


# --------------------------------------------------------------------------- #
# Range retries
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_midstream_failure_is_retried_with_backoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    server = FakeServer()
    server.on("part", Reply(fail_after=40), Reply(fail_after=10), start=100)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.COMPLETED
    assert result.bytes_downloaded == SIZE
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload
    assert server.part_attempts(100) == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_range_gives_up_after_max_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("part", *[Reply(fail_after=5)] * 3, start=200)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out", max_retries=2)
    await drain_orphans()
    assert result.status is DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith("ClientPayloadError")
    assert server.part_attempts(200) == 3
    assert not (tmp_path / "out" / "FG.bin").exists()
    assert part_files(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_transient_http_errors_are_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    server = FakeServer()
    server.on("part", Reply(status=status), start=0)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.COMPLETED
    assert server.part_attempts(0) == 2


@pytest.mark.asyncio
async def test_salvage_promotes_when_every_byte_arrived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    first_done = asyncio.Event()
    server.on("part", Reply(set_on_eof=first_done), start=0)
    server.on("part", Reply(wait_for=first_done, fail_after=200), start=200)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out", num_parts=2, max_retries=0)
    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload


@pytest.mark.asyncio
async def test_connect_error_before_first_byte_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("part", Reply(raise_before=aiohttp.ClientConnectionError("refused")), start=300)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.COMPLETED
    assert server.part_attempts(300) == 2


# --------------------------------------------------------------------------- #
# download_file_parallel (flattened batch path)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_parallel_downloads_under_global_semaphore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    seen: list[int] = []
    result = await flat(
        client,
        tmp_path / "out",
        sem=asyncio.Semaphore(2),
        file_datetime="20240102",
        progress_callback=lambda p: seen.append(p.bytes_downloaded),
    )
    assert result is not None and result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload
    assert server.peak_in_flight <= 2
    assert len(server.of("part")) == 4
    assert server.of("probe")[0]["params"] == {"file-group-id": "FG", "file-datetime": "20240102"}
    assert seen == [SIZE]


@pytest.mark.asyncio
async def test_parallel_falls_back_when_ranges_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("probe", Reply(headers={"Content-Range": None}))
    client = make_client(server, monkeypatch, tmp_path)
    result = await flat(client, tmp_path / "out")
    assert result is not None and result.status is DownloadStatus.COMPLETED
    assert [r["kind"] for r in server.requests] == ["probe", "plain"]


@pytest.mark.asyncio
async def test_parallel_respects_config_overwrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    dest = tmp_path / "out" / "FG.bin"
    dest.parent.mkdir()
    dest.write_bytes(b"old")

    keep = await flat(make_client(server, monkeypatch, tmp_path), tmp_path / "out")
    assert keep is not None and keep.status is DownloadStatus.ALREADY_EXISTS
    assert dest.read_bytes() == b"old"

    replace = await flat(make_client(server, monkeypatch, tmp_path, overwrite_existing=True), tmp_path / "out")
    assert replace is not None and replace.status is DownloadStatus.COMPLETED
    assert dest.read_bytes() == server.payload


@pytest.mark.asyncio
async def test_parallel_failure_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("part", *[Reply(fail_after=1)] * 4, start=100)
    client = make_client(server, monkeypatch, tmp_path)
    result = await flat(client, tmp_path / "out")
    await drain_orphans()
    assert result is None
    assert not (tmp_path / "out" / "FG.bin").exists()
    assert part_files(tmp_path) == []


@pytest.mark.asyncio
async def test_parallel_invalid_datetime_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(FakeServer(), monkeypatch, tmp_path)
    with pytest.raises(Exception):
        await flat(client, tmp_path / "out", file_datetime="not-a-date")


# --------------------------------------------------------------------------- #
# download_files_with_retry
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_batch_retries_failed_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    server = FakeServer()
    server.on("probe", Reply(status=503), fgid="B")
    client = make_client(server, monkeypatch, tmp_path)
    completed: list[str] = []

    async def on_done(result: DownloadResult) -> None:
        completed.append(result.file_group_id)

    ok, failed, retries = await parallel.download_files_with_retry(
        client=client,
        files=[{"file-group-id": "A"}, {"file_group_id": "B", "file_datetime": "20240102"}],
        destination_dir=tmp_path / "out",
        num_parts=2,
        global_semaphore=asyncio.Semaphore(4),
        intelligent_delay=0.25,
        base_retry_delay=2.0,
        max_retries=2,
        on_file_complete=on_done,
    )
    assert sorted(r.file_group_id for r in ok) == ["A", "B"]
    assert failed == [] and retries == 1
    assert sorted(completed) == ["A", "B"]
    assert (tmp_path / "out" / "B.bin").read_bytes() == server.payload
    assert sleeps == [0.25, 2.0]


@pytest.mark.asyncio
async def test_batch_reports_generic_reason_when_parallel_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = FakeServer()
    server.on("probe", *[Reply(status=503)] * 2)
    client = make_client(server, monkeypatch, tmp_path)
    ok, failed, retries = await parallel.download_files_with_retry(
        client=client,
        files=[{"file-group-id": "FG"}],
        destination_dir=tmp_path / "out",
        num_parts=2,
        global_semaphore=asyncio.Semaphore(4),
        intelligent_delay=0.0,
        base_retry_delay=1.0,
        max_retries=1,
    )
    assert ok == [] and retries == 1
    assert failed == [{"file-group-id": "FG", "error": "download failed"}]


# --------------------------------------------------------------------------- #
# Reliability defects (F1-F10, see PR findings table)
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(strict=True, reason="F1: unknown size gives a 0-byte COMPLETED file")
@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["multipart", "parallel"])
async def test_unknown_total_size_falls_back_to_single_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    server = FakeServer()
    server.on("probe", Reply(headers={"Content-Range": "bytes 0-0/*", "Content-Length": None}))
    client = make_client(server, monkeypatch, tmp_path)
    result = await run_entry(entry, client, tmp_path / "out")
    assert result is not None and result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload


@pytest.mark.xfail(strict=True, reason="F3: a 200 reply (Range ignored) is written at every offset")
@pytest.mark.asyncio
async def test_part_answered_without_range_fails_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(ignore_part_ranges=True)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    await drain_orphans()
    assert result.status is DownloadStatus.FAILED
    assert not (tmp_path / "out" / "FG.bin").exists()
    assert part_files(tmp_path) == []
    assert all(server.part_attempts(s) <= 1 for s in (0, 100, 200, 300))


@pytest.mark.xfail(strict=True, reason="F3: a 206 for a different range is accepted")
@pytest.mark.asyncio
async def test_part_with_mismatched_content_range_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("part", *[Reply(headers={"Content-Range": f"bytes 0-99/{SIZE}"})] * 4, start=100)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    await drain_orphans()
    assert result.status is DownloadStatus.FAILED
    assert not (tmp_path / "out" / "FG.bin").exists()


@pytest.mark.xfail(strict=True, reason="F4: a short part (clean EOF) leaves a zero-filled hole")
@pytest.mark.asyncio
async def test_short_part_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("part", Reply(truncate_to=40), start=100)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload
    assert result.bytes_downloaded == SIZE
    assert server.part_attempts(100) == 2


@pytest.mark.xfail(strict=True, reason="F4: extra bytes overwrite the next range")
@pytest.mark.asyncio
async def test_oversized_part_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer()
    server.on("part", *[Reply(body=b"\xff" * 200)] * 4, start=0)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    await drain_orphans()
    assert result.status is DownloadStatus.FAILED
    assert not (tmp_path / "out" / "FG.bin").exists()


@pytest.mark.xfail(strict=True, reason="F5: a failed range leaves its siblings running")
@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["multipart", "parallel"])
async def test_failed_range_cancels_siblings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str) -> None:
    server = FakeServer()
    never = asyncio.Event()
    server.on("part", *[Reply(raise_before=aiohttp.ClientConnectionError("down"))] * 4, start=0)
    for start in (100, 200, 300):
        server.on("part", Reply(wait_for=never), start=start)
    client = make_client(server, monkeypatch, tmp_path)
    try:
        if entry == "multipart":
            result = await multipart(client, tmp_path / "out", max_retries=3)
            assert result.status is DownloadStatus.FAILED
        else:
            assert await flat(client, tmp_path / "out") is None
        assert server.in_flight == 0
    finally:
        never.set()
        await drain_orphans()


@pytest.mark.xfail(strict=True, reason="F6: no backoff when a retry follows a pre-byte failure")
@pytest.mark.asyncio
async def test_pre_byte_failure_backs_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    server = FakeServer()
    server.on(
        "part",
        Reply(fail_after=10),
        Reply(raise_before=aiohttp.ClientConnectionError("refused")),
        start=100,
    )
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    assert result.status is DownloadStatus.COMPLETED
    assert sleeps == [0.5, 1.0]


@pytest.mark.xfail(strict=True, reason="F7: permanent 4xx errors are retried")
@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 404, 416])
async def test_permanent_http_errors_are_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status: int
) -> None:
    server = FakeServer()
    server.on("part", *[Reply(status=status)] * 4, start=100)
    client = make_client(server, monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out")
    await drain_orphans()
    assert result.status is DownloadStatus.FAILED
    assert server.part_attempts(100) == 1
    assert sleeps == []


@pytest.mark.xfail(strict=True, reason="F8: max_bandwidth_mbps is ignored by the multipart path")
@pytest.mark.asyncio
async def test_multipart_applies_bandwidth_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[_RecordingThrottler] = []

    class _RecordingThrottler(BandwidthThrottler):
        def __init__(self, max_bytes_per_second: int | None = None) -> None:
            super().__init__(max_bytes_per_second)
            self.calls: list[int] = []
            created.append(self)

        async def throttle(self, bytes_to_transfer: int) -> None:
            self.calls.append(bytes_to_transfer)

    monkeypatch.setattr(parallel, "BandwidthThrottler", _RecordingThrottler)
    client = make_client(FakeServer(), monkeypatch, tmp_path)
    result = await multipart(client, tmp_path / "out", max_bandwidth_mbps=2.0)
    assert result.status is DownloadStatus.COMPLETED
    assert len(created) == 1
    assert created[0].max_bytes_per_second == 2 * C.MBPS_TO_BYTES_PER_SECOND
    assert sum(created[0].calls) == SIZE


@pytest.mark.xfail(strict=True, reason="F9: cancelling a download leaks the .part file")
@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["multipart", "parallel"])
async def test_cancellation_removes_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str) -> None:
    server = FakeServer()
    gate = asyncio.Event()
    for start in (0, 100, 200, 300):
        server.on("part", Reply(wait_for=gate), start=start)
    client = make_client(server, monkeypatch, tmp_path)
    task = asyncio.ensure_future(run_entry(entry, client, tmp_path / "out"))
    try:
        for _ in range(200):
            if server.in_flight == 4:
                break
            await asyncio.sleep(0)
        assert server.in_flight == 4
        assert part_files(tmp_path)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert part_files(tmp_path) == []
        assert not (tmp_path / "out" / "FG.bin").exists()
    finally:
        gate.set()
        await drain_orphans()


@pytest.mark.xfail(strict=True, reason="F10: parts open a file handle before waiting for the semaphore")
@pytest.mark.asyncio
async def test_parts_open_files_only_inside_the_semaphore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"open": 0, "peak": 0}

    class _Tracked:
        def __init__(self, fh: Any) -> None:
            self._fh = fh

        def __enter__(self) -> _Tracked:
            return self

        def __exit__(self, *exc: Any) -> None:
            state["open"] -= 1
            self._fh.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._fh, name)

    def tracking_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        fh = builtins.open(file, mode, *args, **kwargs)
        if mode != "r+b":
            return fh
        state["open"] += 1
        state["peak"] = max(state["peak"], state["open"])
        return _Tracked(fh)

    monkeypatch.setattr(parallel, "open", tracking_open, raising=False)
    server = FakeServer()
    client = make_client(server, monkeypatch, tmp_path)
    result = await flat(client, tmp_path / "out", sem=asyncio.Semaphore(1))
    assert result is not None and result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "out" / "FG.bin").read_bytes() == server.payload
    assert state["peak"] == 1
