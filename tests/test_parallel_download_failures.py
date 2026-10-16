"""Failure, retry and cleanup paths of ``dataquery.download.parallel``.

Every test drives a real ``DataQueryClient`` whose transport
(``_make_authenticated_request``) is replaced by an in-memory range server
that can inject a failure per range and per attempt: a connection error
before the response, an interrupted body stream, or an HTTP error status
(raised by the real ``_handle_response``). ``asyncio.sleep`` is patched so
retry backoff is recorded instead of slept.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import pytest
from multidict import CIMultiDict

from dataquery import constants as C
from dataquery.core.client import DataQueryClient
from dataquery.download import parallel
from dataquery.types.models import ClientConfig, DownloadOptions, DownloadResult, DownloadStatus

PAYLOAD = bytes((i * 7 + 3) % 251 for i in range(64))
NUM_PARTS = 4
RANGES = ["bytes=0-15", "bytes=16-31", "bytes=32-47", "bytes=48-63"]
PROBE = C.PROBE_HEADERS["Range"]
FILE_GROUP_ID = "FG1"
FILE_DT = "20240102"
FILENAME = "data.bin"

CONNECT = "connect"

# Retry settings per entry point: multipart takes caller options, the
# parallel path always uses DownloadOptions() defaults.
MULTIPART_RETRIES, MULTIPART_DELAY = 2, 0.5
PARALLEL_RETRIES, PARALLEL_DELAY = 3, 1.0


def midstream(after: int) -> tuple[str, int]:
    """Fault: deliver ``after`` bytes of the range, then drop the stream."""
    return ("midstream", after)


# --------------------------------------------------------------------------- #
# In-memory range server
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(
        self,
        status: int,
        headers: dict,
        body: bytes,
        cut: Optional[int] = None,
        gate: Optional[asyncio.Event] = None,
        on_cancel: Optional[Callable[[], None]] = None,
    ) -> None:
        self.status = status
        self.headers = CIMultiDict(headers)
        self.url = "https://files.example.com/group/file/download"
        self.content = self
        self._body = body
        self._cut = cut
        self._gate = gate
        self._on_cancel = on_cancel

    async def text(self) -> str:
        return self._body.decode(errors="replace")

    async def iter_chunked(self, n: int):
        if self._gate is not None:
            try:
                await self._gate.wait()
            except asyncio.CancelledError:
                if self._on_cancel is not None:
                    self._on_cancel()
                raise
        limit = len(self._body) if self._cut is None else self._cut
        for i in range(0, limit, n):
            yield self._body[i : min(i + n, limit)]
        if self._cut is not None:
            raise aiohttp.ClientPayloadError("Response payload is not completed")


class FakeCM:
    def __init__(self, resp: FakeResponse) -> None:
        self._resp = resp

    async def __aenter__(self) -> FakeResponse:
        return self._resp

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class RangeServer:
    """Serves ``payload`` by byte range with scripted per-attempt faults.

    ``faults`` maps a Range header to a list of faults, one per attempt
    (``None`` = success, attempts beyond the list succeed). ``always`` maps a
    Range header to a fault applied on every attempt.
    """

    def __init__(
        self,
        payload: bytes = PAYLOAD,
        faults: Optional[dict] = None,
        always: Optional[dict] = None,
        probe_total: Optional[str] = None,
        probe_content_length: bool = True,
    ) -> None:
        self.payload = payload
        self.faults = faults or {}
        self.always = always or {}
        self.probe_total = str(len(payload)) if probe_total is None else probe_total
        self.probe_content_length = probe_content_length
        self.requests: list[dict] = []
        self.attempts: Counter = Counter()
        self.gates: dict[str, asyncio.Event] = {}
        self.cancelled: list[str] = []

    def range_requests(self) -> list[dict]:
        return [r for r in self.requests if r["range"] != PROBE]

    async def __call__(self, method: str, url: str, **kwargs: Any) -> FakeCM:
        rng = (kwargs.get("headers") or {}).get("Range")
        self.requests.append(
            {
                "method": method,
                "url": url,
                "params": dict(kwargs.get("params") or {}),
                "range": rng,
                "timeout": kwargs.get("timeout"),
            }
        )
        attempt = self.attempts[rng]
        self.attempts[rng] += 1
        scripted = self.faults.get(rng, [])
        fault = self.always.get(rng) or (scripted[attempt] if attempt < len(scripted) else None)

        if fault == CONNECT:
            raise aiohttp.ClientConnectionError(f"connection reset before response ({rng})")
        if isinstance(fault, int):
            return FakeCM(FakeResponse(fault, {"Content-Type": "text/plain"}, b""))

        if rng is None:
            return FakeCM(FakeResponse(200, {"Content-Length": str(len(self.payload))}, self.payload))

        start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
        body = self.payload[start : end + 1]
        headers = {
            "Content-Range": f"bytes {start}-{end}/{self.probe_total if rng == PROBE else len(self.payload)}",
            "Content-Disposition": f'attachment; filename="{FILENAME}"',
        }
        if rng != PROBE or self.probe_content_length:
            headers["Content-Length"] = str(len(body))
        cut = fault[1] if isinstance(fault, tuple) else None
        return FakeCM(
            FakeResponse(
                206,
                headers,
                body,
                cut=cut,
                gate=self.gates.get(rng),
                on_cancel=lambda: self.cancelled.append(rng),
            )
        )


class RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict]] = []

    def _log(self, level: str, event: str, **kw: Any) -> None:
        self.records.append((level, event, kw))

    def debug(self, event: str, **kw: Any) -> None:
        self._log("debug", event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._log("info", event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._log("warning", event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._log("error", event, **kw)

    def of(self, level: str) -> list[tuple[str, dict]]:
        return [(e, kw) for lvl, e, kw in self.records if lvl == level]


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def low_threshold(monkeypatch):
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", 1)


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, result: Any = None) -> Any:
        delays.append(delay)
        await real_sleep(0)
        return result

    monkeypatch.setattr(parallel.asyncio, "sleep", fake_sleep)
    return delays


@pytest.fixture
def log(monkeypatch) -> RecordingLogger:
    rec = RecordingLogger()
    monkeypatch.setattr(parallel, "logger", rec)
    return rec


@pytest.fixture
def out_dir(tmp_path) -> Path:
    return tmp_path / "out"


def make_client(tmp_path: Path, server: RangeServer, **config: Any) -> DataQueryClient:
    client = DataQueryClient(
        ClientConfig.model_validate(
            {
                "base_url": "https://api.example.com",
                "client_id": "id",
                "client_secret": "secret",
                "oauth_enabled": False,
                "download_dir": str(tmp_path),
                "enable_range_downloads": True,
                **config,
            }
        )
    )
    client._make_authenticated_request = server  # type: ignore[method-assign]
    return client


async def drain_background_tasks() -> None:
    """Let range tasks orphaned by a failed gather finish before the loop closes."""
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    if pending:
        await asyncio.wait(pending, timeout=5)


async def start_multipart(client: DataQueryClient, out_dir: Path, **opts: Any) -> DownloadResult:
    opts.setdefault("max_retries", MULTIPART_RETRIES)
    opts.setdefault("retry_delay", MULTIPART_DELAY)
    options = DownloadOptions(destination_path=out_dir, chunk_size=4, **opts)
    return await client.download_file_async(FILE_GROUP_ID, file_datetime=FILE_DT, options=options, num_parts=NUM_PARTS)


async def start_parallel(client: DataQueryClient, out_dir: Path, **_: Any) -> Optional[DownloadResult]:
    return await parallel.download_file_parallel(
        client, FILE_GROUP_ID, FILE_DT, out_dir, num_parts=NUM_PARTS, global_semaphore=asyncio.Semaphore(8)
    )


async def run_multipart(client: DataQueryClient, out_dir: Path, **opts: Any) -> DownloadResult:
    try:
        return await start_multipart(client, out_dir, **opts)
    finally:
        await drain_background_tasks()


async def run_parallel(client: DataQueryClient, out_dir: Path, **_: Any) -> Optional[DownloadResult]:
    try:
        return await start_parallel(client, out_dir)
    finally:
        await drain_background_tasks()


MODES = {
    "multipart": (run_multipart, MULTIPART_RETRIES, MULTIPART_DELAY),
    "parallel": (run_parallel, PARALLEL_RETRIES, PARALLEL_DELAY),
}


def part_files(root: Path) -> list[Path]:
    return sorted(root.rglob(f"*{C.TEMP_SUFFIX}"))


def all_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def expected_params() -> dict:
    return {"file-group-id": FILE_GROUP_ID, "file-datetime": FILE_DT}


# --------------------------------------------------------------------------- #
# Interrupted download, then recovery
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multipart_midstream_interruption_recovers_with_exact_bytes(tmp_path, out_dir, sleeps):
    server = RangeServer(faults={"bytes=16-31": [midstream(6)]})
    client = make_client(tmp_path, server)
    seen: list[int] = []

    options = DownloadOptions(destination_path=out_dir, chunk_size=4, max_retries=2, retry_delay=0.5)
    result = await client.download_file_async(
        FILE_GROUP_ID,
        file_datetime=FILE_DT,
        options=options,
        num_parts=NUM_PARTS,
        progress_callback=lambda p: seen.append(p.bytes_downloaded),
    )

    dest = out_dir / FILENAME
    assert result.status == DownloadStatus.COMPLETED
    assert result.error_message is None
    assert dest.read_bytes() == PAYLOAD
    # The 6 bytes from the interrupted attempt were rewound, not double-counted.
    assert result.bytes_downloaded == len(PAYLOAD)
    assert result.file_size == len(PAYLOAD)
    assert seen and max(seen) <= len(PAYLOAD)
    assert server.attempts["bytes=16-31"] == 2
    assert all(server.attempts[r] == 1 for r in RANGES if r != "bytes=16-31")
    assert sleeps == [0.5]
    retried = [r for r in server.requests if r["range"] == "bytes=16-31"]
    assert [r["params"] for r in retried] == [expected_params()] * 2
    assert part_files(tmp_path) == []


@pytest.mark.asyncio
async def test_multipart_backoff_doubles_on_each_interrupted_attempt(tmp_path, out_dir, sleeps):
    server = RangeServer(faults={"bytes=32-47": [midstream(1), midstream(9), midstream(15)]})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir, max_retries=3, retry_delay=0.25)

    assert result.status == DownloadStatus.COMPLETED
    assert (out_dir / FILENAME).read_bytes() == PAYLOAD
    assert result.bytes_downloaded == len(PAYLOAD)
    assert server.attempts["bytes=32-47"] == 4
    # retry_delay * 2 ** (attempt - 1) for attempts 1, 2, 3
    assert sleeps == [0.25, 0.5, 1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [CONNECT, 404, 500, 503], ids=["connect", "404", "500", "503"])
async def test_multipart_failure_before_first_byte_recovers(tmp_path, out_dir, sleeps, fault):
    server = RangeServer(faults={"bytes=48-63": [fault]})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir)

    assert result.status == DownloadStatus.COMPLETED
    assert (out_dir / FILENAME).read_bytes() == PAYLOAD
    assert result.bytes_downloaded == len(PAYLOAD)
    assert server.attempts["bytes=48-63"] == 2
    assert part_files(tmp_path) == []


@pytest.mark.asyncio
async def test_parallel_midstream_interruption_recovers_with_default_backoff(tmp_path, out_dir, sleeps):
    server = RangeServer(faults={"bytes=0-15": [midstream(10)]})
    client = make_client(tmp_path, server)

    result = await run_parallel(client, out_dir)

    assert result is not None
    assert result.status == DownloadStatus.COMPLETED
    assert (out_dir / FILENAME).read_bytes() == PAYLOAD
    assert result.bytes_downloaded == len(PAYLOAD)
    assert server.attempts["bytes=0-15"] == 2
    assert sleeps == [PARALLEL_DELAY]
    for req in server.range_requests():
        assert req["params"] == expected_params()
        assert req["timeout"].total is None
        assert req["timeout"].sock_read == client.config.timeout
    assert part_files(tmp_path) == []


# --------------------------------------------------------------------------- #
# Retry exhaustion
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "error_prefix", "expected_sleeps"),
    [
        (CONNECT, "ClientConnectionError: connection reset before response", None),
        (midstream(3), "ClientPayloadError: Response payload is not completed", [0.5, 1.0]),
        (503, "NetworkError: Server error: 503", None),
    ],
    ids=["connect", "midstream", "503"],
)
async def test_multipart_range_exhausts_retries_and_fails(
    tmp_path, out_dir, sleeps, fault, error_prefix, expected_sleeps
):
    server = RangeServer(always={"bytes=32-47": fault})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir)

    assert isinstance(result, DownloadResult)
    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith(error_prefix)
    assert result.file_size == 0
    assert result.local_path == out_dir / FILENAME
    assert server.attempts["bytes=32-47"] == MULTIPART_RETRIES + 1
    if expected_sleeps is not None:
        assert sleeps == expected_sleeps
    assert not (out_dir / FILENAME).exists()
    assert part_files(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "error_text", "expected_sleeps"),
    [
        (midstream(5), "Response payload is not completed", [1.0, 2.0, 4.0]),
        (CONNECT, "connection reset before response (bytes=48-63)", None),
    ],
    ids=["midstream", "connect"],
)
async def test_parallel_range_exhausts_retries_returns_none_and_logs(
    tmp_path, out_dir, sleeps, log, fault, error_text, expected_sleeps
):
    server = RangeServer(always={"bytes=48-63": fault})
    client = make_client(tmp_path, server)

    result = await run_parallel(client, out_dir)

    assert result is None
    assert server.attempts["bytes=48-63"] == PARALLEL_RETRIES + 1
    if expected_sleeps is not None:
        assert sleeps == expected_sleeps
    errors = log.of("error")
    assert len(errors) == 1
    event, kw = errors[0]
    assert event == "Flattened parallel download failed"
    assert kw["file_group_id"] == FILE_GROUP_ID
    assert error_text in kw["error"]
    assert all_files(tmp_path) == []


# --------------------------------------------------------------------------- #
# Probe failure
# --------------------------------------------------------------------------- #
PROBE_FAULTS = [
    (404, "NotFoundError"),
    (500, "NetworkError: Server error: 500"),
    (503, "NetworkError: Server error: 503"),
    (401, "AuthenticationError: Authentication failed"),
    (CONNECT, "ClientConnectionError"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("fault", "error_prefix"), PROBE_FAULTS, ids=["404", "500", "503", "401", "connect"])
async def test_multipart_probe_failure_returns_failed_and_creates_nothing(
    tmp_path, out_dir, sleeps, fault, error_prefix
):
    server = RangeServer(always={PROBE: fault})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir)

    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith(error_prefix)
    assert result.bytes_downloaded == 0
    assert [r["range"] for r in server.requests] == [PROBE]
    assert sleeps == []
    assert all_files(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("fault", "error_prefix"), PROBE_FAULTS, ids=["404", "500", "503", "401", "connect"])
async def test_parallel_probe_failure_returns_none_and_creates_nothing(
    tmp_path, out_dir, sleeps, log, fault, error_prefix
):
    server = RangeServer(always={PROBE: fault})
    client = make_client(tmp_path, server)

    result = await run_parallel(client, out_dir)

    assert result is None
    assert [r["range"] for r in server.requests] == [PROBE]
    assert [e for e, _ in log.of("error")] == ["Flattened parallel download failed"]
    assert all_files(tmp_path) == []


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(MODES))
async def test_failure_with_overwrite_leaves_existing_destination_untouched(tmp_path, out_dir, sleeps, mode):
    run, _, _ = MODES[mode]
    out_dir.mkdir()
    dest = out_dir / FILENAME
    dest.write_bytes(b"previous good copy")
    server = RangeServer(always={"bytes=16-31": midstream(2)})
    client = make_client(tmp_path, server, overwrite_existing=True)

    result = await run(client, out_dir, overwrite_existing=True)

    if mode == "multipart":
        assert result is not None and result.status == DownloadStatus.FAILED
    else:
        assert result is None
    assert dest.read_bytes() == b"previous good copy"
    assert part_files(tmp_path) == []
    assert all_files(tmp_path) == [dest]


# --------------------------------------------------------------------------- #
# Salvage reached end-to-end
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(MODES))
async def test_salvage_promotes_complete_file_when_later_step_raises(tmp_path, out_dir, sleeps, monkeypatch, mode):
    run, _, _ = MODES[mode]

    def exploding_flush(self):
        raise RuntimeError("progress sink exploded")

    monkeypatch.setattr(parallel._ProgressReporter, "flush", exploding_flush)
    server = RangeServer()
    client = make_client(tmp_path, server)

    result = await run(client, out_dir)

    dest = out_dir / FILENAME
    assert result is not None
    assert result.status == DownloadStatus.COMPLETED
    assert result.error_message is None
    assert result.local_path == dest
    assert result.file_size == len(PAYLOAD)
    assert result.bytes_downloaded == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert part_files(tmp_path) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(MODES))
async def test_salvage_promote_oserror_is_logged_and_failure_returned(
    tmp_path, out_dir, sleeps, log, monkeypatch, mode
):
    run, _, _ = MODES[mode]
    real_replace = Path.replace

    def failing_replace(self, target):
        if self.name.endswith(C.TEMP_SUFFIX):
            raise OSError(28, "No space left on device")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)
    server = RangeServer()
    client = make_client(tmp_path, server)

    result = await run(client, out_dir)

    dest = out_dir / FILENAME
    temp = out_dir / (FILENAME + C.TEMP_SUFFIX)
    warnings = log.of("warning")
    assert warnings == [
        ("Salvage cleanup failed", {"temp_path": str(temp), "error": "[Errno 28] No space left on device"})
    ]
    if mode == "multipart":
        assert result is not None
        assert result.status == DownloadStatus.FAILED
        assert result.error_message == "OSError: [Errno 28] No space left on device"
        assert result.bytes_downloaded == len(PAYLOAD)
    else:
        assert result is None
        assert [e for e, _ in log.of("error")] == ["Flattened parallel download failed"]
    assert not dest.exists()
    # Promotion never happened, so the fully written temp file is left behind.
    assert temp.read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_salvage_unlink_oserror_after_range_failure_is_logged(tmp_path, out_dir, sleeps, log, monkeypatch):
    real_unlink = Path.unlink

    def failing_unlink(self, missing_ok=False):
        if self.name.endswith(C.TEMP_SUFFIX):
            raise PermissionError(13, "Permission denied")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    server = RangeServer(always={"bytes=0-15": CONNECT})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir)

    temp = out_dir / (FILENAME + C.TEMP_SUFFIX)
    assert result.status == DownloadStatus.FAILED
    assert result.error_message is not None and result.error_message.startswith("ClientConnectionError")
    assert log.of("warning") == [
        ("Salvage cleanup failed", {"temp_path": str(temp), "error": "[Errno 13] Permission denied"})
    ]
    assert not (out_dir / FILENAME).exists()


# --------------------------------------------------------------------------- #
# Failure reporting
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", list(MODES))
async def test_existing_destination_without_overwrite_reports_already_exists(tmp_path, out_dir, sleeps, mode):
    run, _, _ = MODES[mode]
    out_dir.mkdir()
    dest = out_dir / FILENAME
    dest.write_bytes(b"keep me")
    server = RangeServer()
    client = make_client(tmp_path, server)

    result = await run(client, out_dir)

    assert result is not None
    assert result.status == DownloadStatus.ALREADY_EXISTS
    assert result.error_message == f"FileExistsError: File already exists: {dest}"
    assert result.local_path == dest
    assert (result.file_size, result.bytes_downloaded) == (0, 0)
    assert [r["range"] for r in server.requests] == [PROBE]
    assert dest.read_bytes() == b"keep me"
    assert all_files(tmp_path) == [dest]


@pytest.mark.asyncio
async def test_parallel_invalid_file_datetime_raises_before_any_request(tmp_path, out_dir):
    server = RangeServer()
    client = make_client(tmp_path, server)

    with pytest.raises(ValueError, match="Invalid file-datetime format: '2024-01-02'"):
        await parallel.download_file_parallel(
            client, FILE_GROUP_ID, "2024-01-02", out_dir, num_parts=NUM_PARTS, global_semaphore=asyncio.Semaphore(8)
        )

    assert server.requests == []
    assert all_files(tmp_path) == []


# --------------------------------------------------------------------------- #
# Findings pinned as strict xfails (expected-correct behaviour)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:172 only backs off when the previous attempt wrote bytes; "
    "a failure before the first byte is retried immediately",
)
@pytest.mark.parametrize(
    ("faults", "expected_sleeps"),
    [
        ([CONNECT], [MULTIPART_DELAY]),
        ([503], [MULTIPART_DELAY]),
        ([midstream(4), CONNECT], [MULTIPART_DELAY, MULTIPART_DELAY * 2]),
    ],
    ids=["connect", "503", "midstream-then-connect"],
)
async def test_retry_backs_off_even_when_no_byte_was_written(tmp_path, out_dir, sleeps, faults, expected_sleeps):
    server = RangeServer(faults={"bytes=16-31": faults})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir)

    assert result.status == DownloadStatus.COMPLETED
    assert sleeps == expected_sleeps


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:322/443 use a bare asyncio.gather; a failed range does not cancel its siblings",
)
@pytest.mark.parametrize("mode", ["multipart", "parallel"])
async def test_failed_range_cancels_sibling_ranges(tmp_path, out_dir, sleeps, mode):
    server = RangeServer(always={"bytes=48-63": CONNECT})
    server.gates["bytes=0-15"] = asyncio.Event()
    client = make_client(tmp_path, server)
    start = start_multipart if mode == "multipart" else start_parallel

    try:
        result = await start(client, out_dir)
        if mode == "multipart":
            assert result is not None and result.status == DownloadStatus.FAILED
        else:
            assert result is None
        assert server.cancelled == ["bytes=0-15"]
    finally:
        server.gates["bytes=0-15"].set()
        await drain_background_tasks()


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:84/276 treat an unknown size ('bytes 0-0/*', no Content-Length) as 0 "
    "and report an empty file as COMPLETED",
)
@pytest.mark.parametrize("mode", list(MODES))
async def test_unknown_total_size_does_not_report_empty_file_as_completed(tmp_path, out_dir, sleeps, monkeypatch, mode):
    run, _, _ = MODES[mode]
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", 10 * 1024 * 1024)
    server = RangeServer(probe_total="*", probe_content_length=False)
    client = make_client(tmp_path, server)

    result = await run(client, out_dir)

    assert result is not None
    if result.status == DownloadStatus.COMPLETED:
        assert result.local_path.read_bytes() == PAYLOAD
    else:
        assert result.status == DownloadStatus.FAILED


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason="parallel.py:315 hard-codes throttler=None, so DownloadOptions.max_bandwidth_mbps is ignored by multipart",
)
async def test_multipart_honours_max_bandwidth_mbps(tmp_path, out_dir, sleeps, monkeypatch):
    throttled: list[int] = []

    async def record_throttle(self, n: int) -> None:
        throttled.append(n)

    monkeypatch.setattr(parallel.BandwidthThrottler, "throttle", record_throttle)
    server = RangeServer(faults={"bytes=16-31": [midstream(4)]})
    client = make_client(tmp_path, server)

    result = await run_multipart(client, out_dir, max_bandwidth_mbps=0.001)

    assert result.status == DownloadStatus.COMPLETED
    assert sum(throttled) >= len(PAYLOAD)
