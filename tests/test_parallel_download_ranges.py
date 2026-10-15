"""End-to-end coverage for the range-download paths in dataquery.download.parallel.

A real ``DataQueryClient`` is wired to an in-memory fake Files API that serves
HTTP byte ranges, so the probe, range math, preallocated temp file, retries
and salvage logic all run for real without network access.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from multidict import CIMultiDict

from dataquery.core.client import DataQueryClient
from dataquery.download import parallel
from dataquery.types.models import (
    BandwidthThrottler,
    ClientConfig,
    DownloadOptions,
    DownloadProgress,
    DownloadStatus,
)

PAYLOAD = bytes(range(256)) * 4  # 1024 bytes, every offset distinguishable
THRESHOLD = 64


class _FakeRangeResponse:
    def __init__(self, status, headers, body, fail_after=None, gate=None):
        self.status = status
        self.headers = CIMultiDict(headers)
        self.url = "https://files.example.com/group/file/download"
        self._body = body
        self._fail_after = fail_after
        self._gate = gate

    async def text(self):
        return ""

    @property
    def content(self):
        return self

    async def iter_chunked(self, n):
        if self._gate is not None:
            await self._gate.wait()
        body = self._body if self._fail_after is None else self._body[: self._fail_after]
        for i in range(0, len(body), n):
            yield body[i : i + n]
        if self._fail_after is not None:
            raise aiohttp.ClientPayloadError("connection reset mid-body")


class _Ctx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class FakeFilesApi:
    """Serves ``payload`` with Range support.

    ``failures`` maps a range start byte to a list of per-attempt behaviours:
    ``"connect"`` (raise before any byte), ``"midstream"`` (send part of the
    body then reset), ``"503"`` (server error status) or ``None`` (succeed).
    """

    def __init__(
        self,
        payload=PAYLOAD,
        *,
        content_range="default",
        probe_headers=None,
        probe_status=206,
        filename="report.csv",
        failures=None,
        gates=None,
    ):
        self.payload = payload
        self.content_range = content_range
        self.probe_headers = probe_headers
        self.probe_status = probe_status
        self.filename = filename
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.gates = gates or {}
        self.requests: list = []

    async def __call__(self, method, url, **kwargs):
        headers = kwargs.get("headers") or {}
        self.requests.append({"method": method, "url": url, **kwargs})
        start, end = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        if (start, end) == (0, 0):
            return _Ctx(self._probe())
        plan = self.failures.get(start)
        behaviour = plan.pop(0) if plan else None
        if behaviour == "connect":
            raise aiohttp.ClientConnectionError("connection refused")
        body = self.payload[start : end + 1]
        if behaviour == "503":
            return _Ctx(_FakeRangeResponse(503, {}, b""))
        fail_after = len(body) // 2 if behaviour == "midstream" else None
        return _Ctx(_FakeRangeResponse(206, {}, body, fail_after=fail_after, gate=self.gates.get(start)))

    def _probe(self):
        if self.probe_headers is not None:
            headers = self.probe_headers
        else:
            headers = {"Content-Disposition": f'attachment; filename="{self.filename}"', "Content-Length": "1"}
            if self.content_range == "default":
                headers["Content-Range"] = f"bytes 0-0/{len(self.payload)}"
            elif self.content_range is not None:
                headers["Content-Range"] = self.content_range
        return _FakeRangeResponse(self.probe_status, headers, self.payload[:1])

    def range_headers(self):
        return [r["headers"]["Range"] for r in self.requests if r["headers"]["Range"] != "bytes=0-0"]


def _client(tmp_path: Path, **config) -> DataQueryClient:
    config.setdefault("enable_range_downloads", True)
    cfg = ClientConfig(
        base_url="https://api.example.com",
        context_path="/research/dataquery-authe/api/v2",
        files_base_url="https://files.example.com",
        oauth_enabled=False,
        download_dir=str(tmp_path),
        **config,
    )
    return DataQueryClient(cfg)


@pytest.fixture
def small_threshold(monkeypatch):
    monkeypatch.setattr(parallel.C, "SMALL_FILE_THRESHOLD", THRESHOLD)


@pytest.fixture
def sleeps(monkeypatch):
    """Replace asyncio.sleep with a recorder that only yields to the loop."""
    recorded: list = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay, *args, **kwargs):
        recorded.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(parallel.asyncio, "sleep", fake_sleep)
    return recorded


def _options(tmp_path, **kw):
    kw.setdefault("chunk_size", 100)
    kw.setdefault("retry_delay", 0.5)
    kw.setdefault("max_retries", 2)
    return DownloadOptions(destination_path=tmp_path, **kw)


async def _multipart(client, api, tmp_path, num_parts=4, file_datetime=None, callback=None, **opts):
    client._make_authenticated_request = api  # type: ignore[method-assign]
    return await parallel.download_file_multipart(
        client=client,
        file_group_id="FG1",
        file_datetime=file_datetime,
        options=_options(tmp_path, **opts),
        num_parts=num_parts,
        progress_callback=callback,
    )


async def _parallel(client, api, tmp_path, num_parts=4, file_datetime=None, callback=None, semaphore=None):
    client._make_authenticated_request = api  # type: ignore[method-assign]
    return await parallel.download_file_parallel(
        client=client,
        file_group_id="FG1",
        file_datetime=file_datetime,
        destination_path=tmp_path,
        num_parts=num_parts,
        global_semaphore=semaphore or asyncio.Semaphore(8),
        progress_callback=callback,
    )


def _no_temp_files(tmp_path):
    return not list(tmp_path.glob("*.part"))


# --------------------------------------------------------------------------- #
# download_file_multipart
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multipart_downloads_all_ranges_into_destination(tmp_path, small_threshold):
    api = FakeFilesApi()
    seen = []
    result = await _multipart(_client(tmp_path), api, tmp_path, file_datetime="20240115", callback=seen.append)

    dest = tmp_path / "report.csv"
    assert result.status is DownloadStatus.COMPLETED
    assert result.local_path == dest
    assert result.file_size == result.bytes_downloaded == len(PAYLOAD)
    assert dest.read_bytes() == PAYLOAD
    assert _no_temp_files(tmp_path)
    assert sorted(api.range_headers()) == sorted(["bytes=0-255", "bytes=256-511", "bytes=512-767", "bytes=768-1023"])
    assert all(r["params"] == {"file-group-id": "FG1", "file-datetime": "20240115"} for r in api.requests)
    assert seen and seen[-1].bytes_downloaded == len(PAYLOAD)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api",
    [
        FakeFilesApi(content_range=None),  # server ignored the Range probe
        FakeFilesApi(payload=PAYLOAD[: THRESHOLD - 1]),  # below the parallel threshold
        FakeFilesApi(content_range="bytes 0-0/*"),  # unknown size, falls back to Content-Length=1
    ],
    ids=["no-content-range", "small-file", "unparseable-total"],
)
async def test_multipart_falls_back_to_single_stream(tmp_path, small_threshold, api):
    client = _client(tmp_path)
    client._download_file_single_stream = AsyncMock(return_value="single")  # type: ignore[method-assign]

    result = await _multipart(client, api, tmp_path)

    assert result == "single"
    client._download_file_single_stream.assert_awaited_once()
    assert client._download_file_single_stream.await_args.kwargs["file_group_id"] == "FG1"
    assert api.range_headers() == []


@pytest.mark.asyncio
async def test_multipart_reads_lowercase_content_range_header(tmp_path, small_threshold):
    api = FakeFilesApi(probe_headers={"content-range": f"bytes 0-0/{len(PAYLOAD)}"})
    result = await _multipart(_client(tmp_path), api, tmp_path, num_parts=2)

    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "FG1.bin").read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_multipart_refuses_to_overwrite_existing_file(tmp_path, small_threshold):
    dest = tmp_path / "report.csv"
    dest.write_bytes(b"keep me")
    api = FakeFilesApi()

    result = await _multipart(_client(tmp_path), api, tmp_path)

    assert result.status is DownloadStatus.ALREADY_EXISTS
    assert result.error_message.startswith("FileExistsError")
    assert dest.read_bytes() == b"keep me"
    assert api.range_headers() == []


@pytest.mark.asyncio
async def test_multipart_overwrites_when_allowed(tmp_path, small_threshold):
    dest = tmp_path / "report.csv"
    dest.write_bytes(b"old")
    result = await _multipart(_client(tmp_path), FakeFilesApi(), tmp_path, overwrite_existing=True)

    assert result.status is DownloadStatus.COMPLETED
    assert dest.read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_multipart_probe_http_error_returns_failed_result(tmp_path, small_threshold):
    api = FakeFilesApi(probe_status=404)
    result = await _multipart(_client(tmp_path), api, tmp_path)

    assert result.status is DownloadStatus.FAILED
    assert result.error_message.startswith("NotFoundError")
    assert result.bytes_downloaded == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_multipart_retries_midstream_failure_and_rewinds_progress(tmp_path, small_threshold, sleeps):
    api = FakeFilesApi(failures={256: ["midstream"]})
    result = await _multipart(_client(tmp_path), api, tmp_path, retry_delay=0.5)

    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "report.csv").read_bytes() == PAYLOAD
    # Bytes from the aborted attempt are rewound, so the count is not inflated.
    assert result.bytes_downloaded == len(PAYLOAD)
    assert api.range_headers().count("bytes=256-511") == 2
    assert sleeps == [0.5]


@pytest.mark.asyncio
async def test_multipart_backoff_doubles_between_range_attempts(tmp_path, small_threshold, sleeps):
    api = FakeFilesApi(failures={0: ["midstream", "midstream"]})
    result = await _multipart(_client(tmp_path), api, tmp_path, num_parts=2, retry_delay=0.5, max_retries=2)

    assert result.status is DownloadStatus.COMPLETED
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_multipart_retry_after_connect_error_does_not_back_off(tmp_path, small_threshold, sleeps):
    """Documents current behaviour (see PR Findings): backoff only applies after partial data."""
    api = FakeFilesApi(failures={0: ["connect", "503"]})
    result = await _multipart(_client(tmp_path), api, tmp_path, num_parts=2, retry_delay=0.5, max_retries=2)

    assert result.status is DownloadStatus.COMPLETED
    assert api.range_headers().count("bytes=0-511") == 3
    assert sleeps == []  # retried immediately, retry_delay ignored


@pytest.mark.asyncio
async def test_multipart_exhausted_range_retries_fail_and_clean_up(tmp_path, small_threshold, sleeps):
    api = FakeFilesApi(failures={512: ["midstream"] * 3})
    result = await _multipart(_client(tmp_path), api, tmp_path, max_retries=2)

    assert result.status is DownloadStatus.FAILED
    assert "ClientPayloadError" in result.error_message
    assert result.bytes_downloaded < len(PAYLOAD)
    assert api.range_headers().count("bytes=512-767") == 3
    assert not (tmp_path / "report.csv").exists()
    assert _no_temp_files(tmp_path)


@pytest.mark.asyncio
async def test_multipart_salvages_when_all_bytes_arrived_before_error(tmp_path, small_threshold, sleeps):
    """A progress callback that raises at 100% still yields the completed file."""

    def callback(progress):
        if progress.bytes_downloaded == len(PAYLOAD):
            raise RuntimeError("UI crashed")

    result = await _multipart(_client(tmp_path), FakeFilesApi(), tmp_path, num_parts=1, callback=callback)

    assert result.status is DownloadStatus.COMPLETED
    assert result.error_message is None
    assert (tmp_path / "report.csv").read_bytes() == PAYLOAD
    assert _no_temp_files(tmp_path)


@pytest.mark.asyncio
async def test_multipart_ignores_max_bandwidth_option(tmp_path, small_threshold, monkeypatch):
    """Documents current behaviour (see PR Findings): no throttler is built for multipart."""
    throttled: list = []

    async def spy(self, n):
        throttled.append(n)

    monkeypatch.setattr(BandwidthThrottler, "throttle", spy)
    result = await _multipart(_client(tmp_path), FakeFilesApi(), tmp_path, max_bandwidth_mbps=0.001)

    assert result.status is DownloadStatus.COMPLETED
    assert throttled == []


@pytest.mark.asyncio
async def test_multipart_unknown_size_without_content_length_completes_empty(tmp_path, small_threshold):
    """Documents current behaviour (see PR Findings): ``*/`` total + no length -> empty COMPLETED file."""
    api = FakeFilesApi(probe_headers={"Content-Range": "bytes 0-0/*"})
    result = await _multipart(_client(tmp_path), api, tmp_path)

    assert result.status is DownloadStatus.COMPLETED
    assert result.file_size == 0
    assert (tmp_path / "FG1.bin").read_bytes() == b""
    assert api.range_headers() == []


@pytest.mark.asyncio
async def test_multipart_failed_range_leaves_sibling_ranges_running(tmp_path, small_threshold, sleeps):
    """Documents current behaviour (see PR Findings): gather() does not cancel sibling ranges."""
    gate = asyncio.Event()
    api = FakeFilesApi(failures={0: ["connect"] * 3}, gates={512: gate})

    result = await _multipart(_client(tmp_path), api, tmp_path, num_parts=2, max_retries=2)

    assert result.status is DownloadStatus.FAILED
    orphans = [t for t in asyncio.all_tasks() if "_download_range" in repr(t.get_coro())]
    assert len(orphans) == 1 and not orphans[0].done()

    gate.set()
    await asyncio.wait_for(orphans[0], timeout=5)
    assert not (tmp_path / "report.csv").exists()


# --------------------------------------------------------------------------- #
# download_file_parallel
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_parallel_downloads_ranges_under_shared_semaphore(tmp_path, small_threshold):
    api = FakeFilesApi()
    seen = []
    result = await _parallel(
        _client(tmp_path),
        api,
        tmp_path,
        file_datetime="20240115T0930",
        callback=seen.append,
        semaphore=asyncio.Semaphore(1),
    )

    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "report.csv").read_bytes() == PAYLOAD
    assert result.bytes_downloaded == len(PAYLOAD)
    assert len(api.range_headers()) == 4
    assert all(r["params"]["file-datetime"] == "20240115T0930" for r in api.requests)
    assert seen[-1].bytes_downloaded == len(PAYLOAD)
    assert _no_temp_files(tmp_path)


@pytest.mark.asyncio
async def test_parallel_rejects_malformed_file_datetime(tmp_path):
    with pytest.raises(ValueError, match="Invalid file-datetime"):
        await _parallel(_client(tmp_path), FakeFilesApi(), tmp_path, file_datetime="2024-01-15")


@pytest.mark.asyncio
@pytest.mark.parametrize("num_parts", [0, -3])
async def test_parallel_non_positive_parts_use_single_stream(tmp_path, num_parts):
    client = _client(tmp_path)
    client.download_file_async = AsyncMock(return_value="single")  # type: ignore[method-assign]

    assert await _parallel(client, FakeFilesApi(), tmp_path, num_parts=num_parts) == "single"
    client.download_file_async.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "api",
    [
        FakeFilesApi(content_range=None),
        FakeFilesApi(payload=PAYLOAD[: THRESHOLD - 1]),
        FakeFilesApi(content_range="bytes 0-0/*"),
    ],
    ids=["no-content-range", "small-file", "unparseable-total"],
)
async def test_parallel_probe_fallback_uses_single_stream(tmp_path, small_threshold, api):
    client = _client(tmp_path)
    client.download_file_async = AsyncMock(return_value="single")  # type: ignore[method-assign]

    assert await _parallel(client, api, tmp_path, file_datetime="20240115") == "single"
    kwargs = client.download_file_async.await_args.kwargs
    assert kwargs["file_group_id"] == "FG1"
    assert kwargs["file_datetime"] == "20240115"
    assert kwargs["options"].destination_path == tmp_path
    assert api.range_headers() == []


@pytest.mark.asyncio
async def test_parallel_existing_file_reports_already_exists(tmp_path, small_threshold):
    dest = tmp_path / "report.csv"
    dest.write_bytes(b"keep me")
    result = await _parallel(_client(tmp_path), FakeFilesApi(), tmp_path)

    assert result.status is DownloadStatus.ALREADY_EXISTS
    assert dest.read_bytes() == b"keep me"


@pytest.mark.asyncio
async def test_parallel_overwrites_when_client_configured(tmp_path, small_threshold):
    dest = tmp_path / "report.csv"
    dest.write_bytes(b"old")
    result = await _parallel(_client(tmp_path, overwrite_existing=True), FakeFilesApi(), tmp_path)

    assert result.status is DownloadStatus.COMPLETED
    assert dest.read_bytes() == PAYLOAD


@pytest.mark.asyncio
async def test_parallel_probe_error_returns_none(tmp_path, small_threshold):
    assert await _parallel(_client(tmp_path), FakeFilesApi(probe_status=500), tmp_path) is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_parallel_exhausted_range_retries_return_none_and_clean_up(tmp_path, small_threshold, sleeps):
    api = FakeFilesApi(failures={256: ["midstream"] * 4})  # default DownloadOptions.max_retries == 3
    assert await _parallel(_client(tmp_path), api, tmp_path) is None
    assert api.range_headers().count("bytes=256-511") == 4
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_parallel_recovers_from_transient_range_failure(tmp_path, small_threshold, sleeps):
    api = FakeFilesApi(failures={768: ["midstream"]})
    result = await _parallel(_client(tmp_path), api, tmp_path)

    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "report.csv").read_bytes() == PAYLOAD
    assert result.bytes_downloaded == len(PAYLOAD)
    assert sleeps == [1.0]  # default DownloadOptions.retry_delay


@pytest.mark.asyncio
async def test_parallel_salvages_when_all_bytes_arrived_before_error(tmp_path, small_threshold, sleeps):
    def callback(progress):
        if progress.bytes_downloaded == len(PAYLOAD):
            raise RuntimeError("UI crashed")

    result = await _parallel(_client(tmp_path), FakeFilesApi(), tmp_path, num_parts=2, callback=callback)

    assert result.status is DownloadStatus.COMPLETED
    assert (tmp_path / "report.csv").read_bytes() == PAYLOAD


# --------------------------------------------------------------------------- #
# _download_range with a bandwidth throttler
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_download_range_throttles_every_chunk(tmp_path):
    temp = tmp_path / "blob.part"
    parallel._preallocate_file(temp, len(PAYLOAD))
    client = _client(tmp_path)
    client._make_authenticated_request = FakeFilesApi()  # type: ignore[method-assign]
    throttler = SimpleNamespace(calls=[])

    async def throttle(n):
        throttler.calls.append(n)

    throttler.throttle = throttle
    progress = DownloadProgress(file_group_id="FG1", total_bytes=len(PAYLOAD), start_time=datetime.now())
    reporter = parallel._ProgressReporter(progress, len(PAYLOAD), None, False, "FG1")
    ctx = parallel._RangeContext(
        client=client,
        url="https://files.example.com/group/file/download",
        params={"file-group-id": "FG1"},
        temp_path=temp,
        chunk_size=100,
        semaphore=asyncio.Semaphore(1),
        timeout=aiohttp.ClientTimeout(total=None),
        throttler=throttler,  # type: ignore[arg-type]
        reporter=reporter,
        max_retries=0,
        retry_delay=0.0,
    )

    await parallel._download_range(ctx, 100, 349)

    assert throttler.calls == [100, 100, 50]
    assert temp.read_bytes()[100:350] == PAYLOAD[100:350]
    assert reporter.bytes_downloaded == 250


# --------------------------------------------------------------------------- #
# _ProgressReporter dispatch rules
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_clock(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(parallel, "time", SimpleNamespace(time=lambda: clock.now))
    return clock


def _reporter(callback=None, show_progress=False, total=10_000_000):
    progress = DownloadProgress(file_group_id="FG1", total_bytes=total, start_time=datetime.now())
    return parallel._ProgressReporter(progress, total, callback, show_progress, "FG1")


def test_progress_reporter_throttles_by_bytes_and_time(fake_clock, monkeypatch):
    monkeypatch.setattr(parallel.C, "CALLBACK_BYTE_THRESHOLD", 1000)
    seen = []
    reporter = _reporter(callback=lambda p: seen.append(p.bytes_downloaded))

    reporter.add_bytes(10)
    assert seen == []  # below byte threshold, no time elapsed
    reporter.add_bytes(990)
    assert seen == [1000]  # byte threshold reached
    reporter.add_bytes(5)
    assert seen == [1000]
    fake_clock.now += parallel.C.CALLBACK_TIME_THRESHOLD
    reporter.add_bytes(5)
    assert seen == [1000, 1010]  # time threshold reached


def test_progress_reporter_logs_when_no_callback_and_show_progress(fake_clock, monkeypatch):
    logged = []
    monkeypatch.setattr(parallel, "logger", SimpleNamespace(debug=lambda msg, **kw: logged.append((msg, kw))))
    reporter = _reporter(show_progress=True, total=100)

    reporter.add_bytes(100)

    assert logged[0][0] == "Download progress (parallel)"
    assert logged[0][1]["file"] == "FG1"
    assert logged[0][1]["percentage"] == "100.0%"


def test_progress_reporter_silent_without_callback_or_show_progress(fake_clock, monkeypatch):
    logged = []
    monkeypatch.setattr(parallel, "logger", SimpleNamespace(debug=lambda *a, **kw: logged.append(a)))
    reporter = _reporter(show_progress=False, total=100)

    reporter.add_bytes(100)

    assert logged == []
    assert reporter._progress.bytes_downloaded == 100


# --------------------------------------------------------------------------- #
# _compute_ranges / _salvage edge cases
# --------------------------------------------------------------------------- #
def test_compute_ranges_more_parts_than_bytes_returns_no_ranges():
    """Documents current behaviour (see PR Findings): bytes are dropped instead of using fewer parts."""
    assert parallel._compute_ranges(3, 5) == []
    assert parallel._compute_ranges(0, 4) == []


def test_salvage_derives_destination_from_temp_name(tmp_path):
    temp = tmp_path / "out.csv.part"
    temp.write_bytes(b"0123456789")
    result = parallel._salvage(_client(tmp_path), "FG1", None, temp, 10, 10, 0.0)

    assert result.status is DownloadStatus.COMPLETED
    assert result.local_path == tmp_path / "out.csv"
    assert (tmp_path / "out.csv").read_bytes() == b"0123456789"


def test_salvage_logs_and_returns_none_when_promotion_fails(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(parallel, "logger", SimpleNamespace(warning=lambda msg, **kw: warnings.append(msg)))
    temp = tmp_path / "out.csv.part"
    temp.write_bytes(b"0123456789")
    blocking_dir = tmp_path / "out.csv"
    blocking_dir.mkdir()
    (blocking_dir / "child").write_text("x")  # non-empty dir: replace() raises OSError

    assert parallel._salvage(_client(tmp_path), "FG1", blocking_dir, temp, 10, 10, 0.0) is None
    assert warnings == ["Salvage cleanup failed"]
    assert temp.exists()


@pytest.mark.asyncio
async def test_stagger_sleeps_for_its_slot(monkeypatch, sleeps):
    async def fake_parallel(**kwargs):
        return SimpleNamespace(status=DownloadStatus.COMPLETED, file_group_id=kwargs["file_group_id"])

    monkeypatch.setattr(parallel, "download_file_parallel", fake_parallel)
    out = await parallel._download_one_with_stagger(
        client=object(),
        file_info={"file_group_id": "fg", "file_datetime": "20240101"},
        destination_dir=Path("/tmp"),
        num_parts=2,
        global_semaphore=asyncio.Semaphore(1),
        delay_seconds=0.25,
        progress_callback=None,
    )
    assert out.file_group_id == "fg"
    assert sleeps == [0.25]
