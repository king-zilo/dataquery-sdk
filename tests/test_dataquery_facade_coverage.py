"""Behavioural tests for the high-level :class:`DataQuery` facade.

Covers the sync proxy, lifecycle helpers, run_* reports, historical chunking and
rate-limit helpers. ``DataQueryClient`` and the parallel downloader are mocked
at the facade boundary, so no network or credentials are touched.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dataquery.constants.download import NO_FILES_FOUND_ERROR
from dataquery.core.client import DataQueryClient
from dataquery.dataquery import ConfigManager, DataQuery, ProgressTracker, _format_duration, _SyncProxy
from dataquery.transport.rate_limiter import RateLimitConfig
from dataquery.types.exceptions import ConfigurationError
from dataquery.types.models import (
    AvailabilityInfo,
    ClientConfig,
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
    FileInfo,
    OperationReport,
)

DOWNLOADER = "dataquery.download.parallel.download_files_with_retry"


def _cfg(**overrides: Any) -> ClientConfig:
    base: Dict[str, Any] = {"base_url": "https://api.example.com", "oauth_enabled": False, "bearer_token": "test-token"}
    base.update(overrides)
    return ClientConfig(**base)


def _rate_client(**rate_kwargs) -> SimpleNamespace:
    return SimpleNamespace(rate_limiter=SimpleNamespace(config=RateLimitConfig(**rate_kwargs)))


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep DataQuery.__init__ from reading any .env file on the developer machine."""
    monkeypatch.setattr("dataquery.dataquery.load_dotenv", lambda *a, **k: False)


@pytest.fixture
def dq():
    instance = DataQuery(_cfg())
    yield instance
    instance._sync_runner.close()


@pytest.fixture
def fake_client():
    client = MagicMock(name="DataQueryClient")
    client.connect = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def connected(dq, fake_client):
    """A DataQuery whose client is already 'connected' to a mock."""
    dq._client = fake_client
    return dq


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def test_format_duration_switches_to_minutes_at_sixty_seconds():
    assert _format_duration(0.44) == "0.4s"
    assert _format_duration(59.94) == "59.9s"
    assert _format_duration(60) == "1m 0s"
    assert _format_duration(125.7) == "2m 5s"


def test_progress_tracker_throttles_logging_by_interval(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr("dataquery.dataquery.time", SimpleNamespace(time=lambda: now["t"]))
    fake_logger = MagicMock()
    monkeypatch.setattr("dataquery.dataquery.logger", fake_logger)

    callback = ProgressTracker(log_interval=10).create_progress_callback()
    progress = SimpleNamespace(completed_files=1, total_files=4, percentage=25.0, current_file="a.csv")

    callback(progress)
    now["t"] = 105.0
    callback(progress)
    now["t"] = 111.0
    callback(progress)

    batch_logs = [c for c in fake_logger.info.call_args_list if c.args == ("Batch progress",)]
    assert len(batch_logs) == 2
    assert batch_logs[0].kwargs == {"completed": 1, "total": 4, "percentage": "25.0%", "current_file": "a.csv"}


# ---------------------------------------------------------------------------
# Configuration / construction
# ---------------------------------------------------------------------------


def test_config_manager_falls_back_to_defaults_when_env_config_invalid():
    no_credentials = ClientConfig(base_url="https://custom.example.com", oauth_enabled=False)
    with patch("dataquery.dataquery.EnvConfig.create_client_config", return_value=no_credentials):
        config = ConfigManager(Path("missing.env")).get_client_config()

    assert config.base_url == "https://api.dataquery.com"
    assert config.oauth_enabled is False


@pytest.mark.parametrize("env_file", ["creds.env", Path("creds.env")])
def test_dataquery_rejects_the_config_manager_default_fallback(env_file):
    """Finding: ConfigManager's fallback config has no credentials, so DataQuery always rejects it."""
    no_credentials = ClientConfig(base_url="https://custom.example.com", oauth_enabled=False)
    with patch("dataquery.dataquery.EnvConfig.create_client_config", return_value=no_credentials) as create:
        with pytest.raises(ConfigurationError, match="Either OAuth credentials or Bearer token"):
            DataQuery(env_file)

    create.assert_called_once_with(env_file=Path("creds.env"))


@pytest.mark.parametrize(
    "kwargs, message",
    [({"client_id": 123}, "client_id must be a string"), ({"client_secret": 123}, "client_secret must be a string")],
)
def test_non_string_credentials_are_rejected(kwargs, message):
    with pytest.raises(ConfigurationError, match=message):
        DataQuery(_cfg(), **kwargs)


def test_client_secret_alone_enables_oauth_and_derives_token_url():
    instance = DataQuery(
        _cfg(base_url="https://api.example.com/", client_id="existing-id", oauth_token_url=None), client_secret="s3cret"
    )

    cfg = instance.client_config
    assert cfg.oauth_enabled is True
    assert cfg.client_id == "existing-id"
    assert cfg.get_client_secret() == "s3cret"
    assert cfg.oauth_token_url == "https://api.example.com/oauth/token"


def test_explicit_oauth_token_url_is_preserved():
    instance = DataQuery(
        _cfg(oauth_token_url="https://auth.example.com/token"), client_id="cid", client_secret="secret"
    )
    assert instance.client_config.oauth_token_url == "https://auth.example.com/token"


def test_overrides_skip_unknown_keys_and_none_values():
    instance = DataQuery(_cfg(max_retries=2), timeout=12.0, not_a_config_field="x", max_retries=None)

    assert instance.client_config.timeout == 12.0
    assert instance.client_config.max_retries == 2
    assert not hasattr(instance.client_config, "not_a_config_field")


def test_overrides_are_not_type_coerced():
    """Finding: overrides use plain setattr (no validate_assignment), so 'no' stays a truthy string."""
    instance = DataQuery(_cfg(), overwrite_existing="no")
    assert instance.client_config.overwrite_existing == "no"


def test_override_that_breaks_validation_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="TIMEOUT must be positive"):
        DataQuery(_cfg(), timeout=-1)


# ---------------------------------------------------------------------------
# _SyncProxy / .sync
# ---------------------------------------------------------------------------


def test_sync_property_is_cached(dq):
    assert isinstance(dq.sync, _SyncProxy)
    assert dq.sync is dq.sync


def test_sync_proxy_runs_async_method_on_background_loop(connected, fake_client):
    fake_client.search_groups_async = AsyncMock(return_value=["g1"])

    wrapper = connected.sync.search_groups
    assert wrapper.__name__ == "search_groups"
    assert "search_groups_async" in wrapper.__doc__
    assert wrapper("fx", limit=5) == ["g1"]
    fake_client.search_groups_async.assert_awaited_once_with("fx", 5, None, page=None)


def test_sync_proxy_unknown_method_raises_attribute_error(dq):
    with pytest.raises(AttributeError, match="no async method 'does_not_exist_async'"):
        dq.sync.does_not_exist  # noqa: B018


def test_sync_proxy_rejects_non_callable_async_attribute(dq):
    dq.not_callable_async = 42
    with pytest.raises(AttributeError, match="not_callable_async"):
        dq.sync.not_callable  # noqa: B018


def test_sync_proxy_dir_lists_public_async_methods_without_suffix(dq):
    names = dir(dq.sync)
    assert {"list_groups", "health_check", "download_historical", "run_group_download"} <= set(names)
    assert not any(n.endswith("_async") or n.startswith("_") for n in names)


@pytest.mark.asyncio
async def test_sync_proxy_inside_running_loop_raises(connected, fake_client):
    fake_client.health_check_async = AsyncMock(return_value=True)
    with pytest.raises(RuntimeError, match="Use the async"):
        connected.sync.health_check()
    fake_client.health_check_async.assert_not_awaited()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_sync_context_manager_connects_and_closes(dq, fake_client):
    with patch("dataquery.dataquery.DataQueryClient", return_value=fake_client) as client_cls:
        with dq as entered:
            assert entered is dq
            assert dq._client is fake_client
    client_cls.assert_called_once_with(dq.client_config)
    fake_client.connect.assert_awaited_once()
    fake_client.close.assert_awaited_once()
    assert dq._client is None
    assert dq._sync_runner._loop is None


@pytest.mark.asyncio
async def test_close_async_without_client_is_noop(dq):
    await dq.close_async()
    await dq.cleanup_async()
    assert dq._client is None


@pytest.mark.asyncio
async def test_methods_raise_when_connect_leaves_no_client(dq):
    with pytest.raises(RuntimeError, match="Client not connected"):
        dq._ensure_client()
    with patch.object(dq, "connect_async", AsyncMock()):
        with pytest.raises(RuntimeError, match="Client not connected"):
            await dq.list_groups_async()


def test_cleanup_without_client_still_stops_runner(dq):
    dq._sync_runner._ensure_loop()
    dq.cleanup()
    assert dq._sync_runner._loop is None


def test_cleanup_with_client_closes_it(connected, fake_client):
    connected.cleanup()
    fake_client.close.assert_awaited_once()
    assert connected._client is None
    assert connected._sync_runner._loop is None


# ---------------------------------------------------------------------------
# Delegation to the client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_groups_without_limit_walks_all_pages(connected, fake_client):
    fake_client.list_all_groups_async = AsyncMock(return_value=["all"])
    fake_client.list_groups_async = AsyncMock()

    assert await connected.list_groups_async(limit=None) == ["all"]
    fake_client.list_groups_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_all_groups_and_iter_search_groups_delegate(connected, fake_client):
    fake_client.search_all_groups_async = AsyncMock(return_value=["a", "b"])

    async def _iter(keywords, limit=None, max_pages=1000):
        for item in (f"{keywords}-1", f"{keywords}-2"):
            yield item

    fake_client.iter_search_groups_async = MagicMock(side_effect=_iter)

    assert await connected.search_all_groups_async("fx", limit=10, max_pages=3, raise_on_cap=False) == ["a", "b"]
    fake_client.search_all_groups_async.assert_awaited_once_with("fx", limit=10, max_pages=3, raise_on_cap=False)
    assert [g async for g in connected.iter_search_groups_async("fx", limit=2, max_pages=4)] == ["fx-1", "fx-2"]
    fake_client.iter_search_groups_async.assert_called_once_with("fx", limit=2, max_pages=4)


@pytest.mark.asyncio
async def test_service_status_async_returns_client_status(connected, fake_client):
    status = {"up": False, "http_status": 503, "error": "maintenance"}
    fake_client.service_status_async = AsyncMock(return_value=status)
    assert await connected.service_status_async() == status


@pytest.mark.asyncio
async def test_download_file_async_builds_default_options_from_destination(connected, fake_client, tmp_path):
    result = DownloadResult(file_group_id="FG", status=DownloadStatus.COMPLETED)
    fake_client.download_file_async = AsyncMock(return_value=result)
    callback = MagicMock()

    assert (
        await connected.download_file_async("FG", "20240101", tmp_path, num_parts=3, progress_callback=callback)
        is result
    )

    args = fake_client.download_file_async.await_args.args
    assert args[0:2] == ("FG", "20240101")
    options = args[2]
    assert isinstance(options, DownloadOptions)
    assert options.destination_path == tmp_path
    assert options.create_directories is True
    assert options.overwrite_existing is True
    assert args[3:] == (3, callback)


@pytest.mark.asyncio
async def test_download_file_async_keeps_caller_options(connected, fake_client, tmp_path):
    fake_client.download_file_async = AsyncMock(return_value=DownloadResult(file_group_id="FG"))
    options = DownloadOptions(destination_path=tmp_path, overwrite_existing=False)

    await connected.download_file_async("FG", destination_path=tmp_path / "ignored", options=options)

    assert fake_client.download_file_async.await_args.args[2] is options


@pytest.mark.asyncio
async def test_overwrite_policy_differs_between_download_file_and_run_download(fake_client, tmp_path):
    """Finding: download_file_async(destination_path=...) always overwrites, ignoring config.overwrite_existing."""
    instance = DataQuery(_cfg(overwrite_existing=False))
    instance._client = fake_client
    fake_client.download_file_async = AsyncMock(
        return_value=DownloadResult(file_group_id="FG", status=DownloadStatus.COMPLETED)
    )

    await instance.download_file_async("FG", destination_path=tmp_path)
    await instance.run_download_async("FG", destination_path=tmp_path)

    direct_opts = fake_client.download_file_async.await_args_list[0].args[2]
    run_opts = fake_client.download_file_async.await_args_list[1].args[2]
    assert direct_opts.overwrite_existing is True
    assert run_opts.overwrite_existing is False
    instance._sync_runner.close()


@pytest.mark.asyncio
async def test_auto_download_async_forwards_all_arguments(connected, fake_client):
    manager = object()
    fake_client.auto_download_async = AsyncMock(return_value=manager)
    file_filter = MagicMock()

    result = await connected.auto_download_async("G", "/data", file_filter=file_filter, file_group_id=["A", "B"])

    assert result is manager
    kwargs = fake_client.auto_download_async.await_args.kwargs
    assert kwargs["group_id"] == "G"
    assert kwargs["destination_dir"] == "/data"
    assert kwargs["file_filter"] is file_filter
    assert kwargs["file_group_id"] == ["A", "B"]
    assert kwargs["initial_check"] is True
    assert kwargs["max_tracked_files"] == 10_000


def test_auto_download_sync_runs_through_runner(connected, fake_client):
    manager = object()
    fake_client.auto_download_async = AsyncMock(return_value=manager)

    assert connected.auto_download("G", "/data", max_retries=7, heartbeat_timeout=30.0) is manager
    kwargs = fake_client.auto_download_async.await_args.kwargs
    assert kwargs["max_retries"] == 7
    assert kwargs["heartbeat_timeout"] == 30.0


def test_sync_wrappers_for_search_all_groups_and_service_status(connected, fake_client):
    fake_client.search_all_groups_async = AsyncMock(return_value=["g"])
    fake_client.service_status_async = AsyncMock(return_value={"up": True})

    assert connected.search_all_groups("kw", limit=10, max_pages=2, raise_on_cap=False) == ["g"]
    fake_client.search_all_groups_async.assert_awaited_once_with("kw", limit=10, max_pages=2, raise_on_cap=False)
    assert connected.service_status() == {"up": True}


def test_download_historical_sync_forwards_arguments(dq, tmp_path):
    report = OperationReport(operation="historical_download")
    with patch.object(dq, "download_historical_async", AsyncMock(return_value=report)) as historical:
        assert dq.download_historical("G", "20240101", "20240331", tmp_path, num_parts=2, chunk_delay=0.0) is report

    historical.assert_awaited_once_with(
        group_id="G",
        start_date="20240101",
        end_date="20240331",
        destination_dir=tmp_path,
        max_concurrent=5,
        num_parts=2,
        delay_between_downloads=0.2,
        max_retries=3,
        progress_callback=None,
        chunk_delay=0.0,
    )


@pytest.mark.parametrize(
    "method", ["groups_to_dataframe", "files_to_dataframe", "instruments_to_dataframe", "time_series_to_dataframe"]
)
def test_dataframe_helpers_create_client_lazily_and_delegate(dq, method):
    lazy_client = MagicMock()
    getattr(lazy_client, method).return_value = "frame"
    with patch("dataquery.dataquery.DataQueryClient", return_value=lazy_client) as client_cls:
        assert getattr(dq, method)(["item"], include_metadata=True) == "frame"
        assert getattr(dq, method)(["item"]) == "frame"

    client_cls.assert_called_once_with(dq.client_config)
    getattr(lazy_client, method).assert_any_call(["item"], include_metadata=True)


def test_to_dataframe_creates_client_lazily_and_forwards_options(dq):
    lazy_client = MagicMock()
    lazy_client.to_dataframe.return_value = "frame"
    transforms = {"x": str}
    with patch("dataquery.dataquery.DataQueryClient", return_value=lazy_client):
        result = dq.to_dataframe({"a": 1}, flatten_nested=False, date_columns=["d"], custom_transformations=transforms)

    assert result == "frame"
    assert dq._client is lazy_client

    lazy_client.to_dataframe.assert_called_once_with(
        {"a": 1},
        flatten_nested=False,
        include_metadata=False,
        date_columns=["d"],
        numeric_columns=None,
        custom_transformations=transforms,
    )


def test_to_dataframe_reuses_existing_client(connected, fake_client):
    fake_client.to_dataframe.return_value = "frame"
    with patch("dataquery.dataquery.DataQueryClient") as client_cls:
        assert connected.to_dataframe([{"a": 1}]) == "frame"
    client_cls.assert_not_called()
    assert connected._client is fake_client


# ---------------------------------------------------------------------------
# run_* reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_group_files_collects_string_and_list_file_types(dq):
    files = [
        FileInfo(file_group_id="f1", file_type=["CSV"]),
        SimpleNamespace(file_type="PDF", model_dump=lambda: {"file_group_id": "f2"}),
        FileInfo(file_group_id="f3"),
    ]
    with patch.object(dq, "list_files_async", AsyncMock(return_value=files)):
        report = await dq.run_group_files_async("G")

    assert report.status == "success"
    assert report.counts == {"total_files": 3}
    assert report.details["file_types"] == ["CSV", "PDF"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, patched, args",
    [
        ("run_groups_async", "list_groups_async", ()),
        ("run_group_files_async", "list_files_async", ("G",)),
        ("run_availability_async", "check_availability_async", ("FG", "20240101")),
        ("run_download_async", "download_file_async", ("FG", "20240101")),
    ],
)
async def test_run_methods_propagate_errors(dq, method, patched, args):
    with patch.object(dq, patched, AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError, match="boom"):
            await getattr(dq, method)(*args)


@pytest.mark.asyncio
async def test_run_availability_reports_error_when_unavailable(dq):
    info = AvailabilityInfo(**{"file-datetime": "20240101", "is-available": False, "file-name": "f.csv"})
    with patch.object(dq, "check_availability_async", AsyncMock(return_value=info)):
        report = await dq.run_availability_async("FG", "20240101")

    assert report.status == "error"
    assert report.details["is_available"] is False
    assert report.details["file_name"] == "f.csv"


@pytest.mark.asyncio
async def test_run_download_reports_failed_result(dq):
    failed = DownloadResult(file_group_id="FG", status=DownloadStatus.FAILED, error_message="HTTP 404")
    with patch.object(dq, "download_file_async", AsyncMock(return_value=failed)) as download:
        report = await dq.run_download_async("FG", "20240101")

    assert report.status == "error"
    assert report.error == "HTTP 404"
    assert report.details == {"download_successful": False, "local_path": None, "file_size": None}
    download.assert_awaited_once_with("FG", "20240101", None, None)


# ---------------------------------------------------------------------------
# run_group_download_async
# ---------------------------------------------------------------------------


@pytest.fixture
def group_dq(connected, fake_client):
    fake_client.rate_limiter = SimpleNamespace(
        config=RateLimitConfig(requests_per_minute=600, burst_capacity=10, enable_queuing=True)
    )
    return connected


def _entry(fg, dt, key_style="dash"):
    if key_style == "dash":
        return {"file-group-id": fg, "file-datetime": dt, "is-available": True}
    return {"file_group_id": fg, "file_datetime": dt, "is_available": True}


@pytest.mark.asyncio
async def test_group_download_with_id_list_queries_each_id_and_deduplicates(group_dq, fake_client, tmp_path):
    a = _entry("FG1", "20240101")
    b = _entry("FG2", "20240102", key_style="snake")
    by_id = {"FG1": [a, dict(a)], "FG2": [dict(a), b], "FG3": None}
    fake_client.list_available_files_async = AsyncMock(side_effect=lambda g, fg, s, e: by_id[fg])
    successes = [
        DownloadResult(
            file_group_id="FG1", file_size=2 * 1024 * 1024, download_time=2.0, status=DownloadStatus.COMPLETED
        ),
        DownloadResult(file_group_id="FG2", status=DownloadStatus.ALREADY_EXISTS),
    ]
    downloader = AsyncMock(return_value=(successes, [], 0))

    with patch(DOWNLOADER, downloader):
        report = await group_dq.run_group_download_async(
            "G", "20240101", "20240131", destination_dir=tmp_path, file_group_id=["FG1", "", "FG2", "FG3"]
        )

    queried = [c.args[1] for c in fake_client.list_available_files_async.await_args_list]
    assert sorted(queried) == ["FG1", "FG2", "FG3"]
    assert downloader.await_args.kwargs["files"] == [a, b]
    assert report.status == "success"
    assert report.counts["total_files"] == 2
    timing = report.timing["per_file_timing"]
    assert timing["file_times"] == [
        {"file_group_id": "FG1", "download_time_seconds": 2.0, "file_size_bytes": 2 * 1024 * 1024, "speed_mbps": 1.0}
    ]
    assert timing["average_file_time_seconds"] == 1.0
    assert timing["min_file_time_seconds"] == timing["max_file_time_seconds"] == 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [[], ["", None]])
async def test_group_download_with_empty_id_list_falls_back_to_whole_group(group_dq, fake_client, tmp_path, ids):
    fake_client.list_available_files_async = AsyncMock(return_value=[])
    downloader = AsyncMock()

    with patch(DOWNLOADER, downloader):
        report = await group_dq.run_group_download_async(
            "G", "20240101", "20240131", destination_dir=tmp_path, file_group_id=ids
        )

    fake_client.list_available_files_async.assert_awaited_once_with("G", None, "20240101", "20240131")
    downloader.assert_not_awaited()
    assert report.status == "error"
    assert report.error == NO_FILES_FOUND_ERROR
    assert report.counts == {"total_files": 0, "successful_downloads": 0, "failed_downloads": 0}


@pytest.mark.asyncio
async def test_group_download_treats_malformed_entries_as_no_files(group_dq, fake_client, tmp_path):
    fake_client.list_available_files_async = AsyncMock(return_value=["not-a-dict", _entry("FG1", "20240101")])
    downloader = AsyncMock()

    with patch(DOWNLOADER, downloader):
        report = await group_dq.run_group_download_async("G", "20240101", "20240131", destination_dir=tmp_path)

    downloader.assert_not_awaited()
    assert report.error == NO_FILES_FOUND_ERROR


@pytest.mark.asyncio
async def test_group_download_partial_report_and_downloader_arguments(group_dq, fake_client, tmp_path):
    files = [_entry("a", "20240101"), _entry("b", "20240102")]
    fake_client.list_available_files_async = AsyncMock(return_value=files)
    ok = DownloadResult(file_group_id="a", file_size=10, download_time=0.5, status=DownloadStatus.COMPLETED)
    failed = [{**files[1], "error": "HTTP 500"}]
    downloader = AsyncMock(return_value=([ok], failed, 2))
    progress = MagicMock()

    with patch(DOWNLOADER, downloader):
        report = await group_dq.run_group_download_async(
            "G",
            "20240101",
            "20240102",
            destination_dir=tmp_path,
            max_concurrent=2,
            num_parts=2,
            max_retries=4,
            progress_callback=progress,
        )

    kwargs = downloader.await_args.kwargs
    assert kwargs["client"] is fake_client
    assert kwargs["destination_dir"] == tmp_path / "G"
    assert (tmp_path / "G").is_dir()
    assert kwargs["num_parts"] == 2
    assert kwargs["max_retries"] == 4
    assert kwargs["intelligent_delay"] == 0.2
    assert kwargs["progress_callback"] is progress
    assert isinstance(kwargs["global_semaphore"], asyncio.Semaphore)

    assert report.status == "partial"
    assert report.counts["retries_attempted"] == 2
    assert report.details["success_rate"] == 50.0
    assert report.details["downloaded_files"] == ["a"]
    assert report.details["failures"] == [{"file_group_id": "b", "file_datetime": "20240102", "error": "HTTP 500"}]
    assert report.details["total_concurrent_requests"] == 4
    assert report.details["rate_limit_recommendations"]["recommendations"] == []


@pytest.mark.asyncio
async def test_group_download_all_failed_reports_error(group_dq, fake_client, tmp_path):
    files = [_entry("a", "20240101")]
    fake_client.list_available_files_async = AsyncMock(return_value=files)
    downloader = AsyncMock(return_value=([], [{"file_group_id": "a", "error": "timeout"}], 3))

    with patch(DOWNLOADER, downloader):
        report = await group_dq.run_group_download_async("G", "20240101", "20240101", destination_dir=tmp_path)

    assert report.status == "error"
    assert report.details["success_rate"] == 0.0
    assert report.details["failed_files"] == ["a"]
    assert report.details["failures"][0]["file_datetime"] is None
    assert report.timing["per_file_timing"]["average_file_time_seconds"] == 0.0


@pytest.mark.asyncio
async def test_group_download_propagates_downloader_errors(group_dq, fake_client, tmp_path):
    fake_client.list_available_files_async = AsyncMock(return_value=[_entry("a", "20240101")])

    with patch(DOWNLOADER, AsyncMock(side_effect=RuntimeError("disk full"))):
        with pytest.raises(RuntimeError, match="disk full"):
            await group_dq.run_group_download_async("G", "20240101", "20240101", destination_dir=tmp_path)


# ---------------------------------------------------------------------------
# Monthly chunking / download_historical_async
# ---------------------------------------------------------------------------


def test_split_into_monthly_ranges_handles_year_boundary_and_leap_february():
    assert DataQuery._split_into_monthly_ranges("20231215", "20240305") == [
        ("20231215", "20231231"),
        ("20240101", "20240131"),
        ("20240201", "20240229"),
        ("20240301", "20240305"),
    ]


def test_split_into_monthly_ranges_empty_when_end_before_start():
    assert DataQuery._split_into_monthly_ranges("20240310", "20240301") == []


def _chunk_report(total, success, failed, status="success", error=None):
    return OperationReport(
        operation="group_download",
        status=status,
        error=error,
        counts={"total_files": total, "successful_downloads": success, "failed_downloads": failed},
    )


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr("dataquery.dataquery.asyncio.sleep", sleep)
    return sleep


@pytest.mark.asyncio
async def test_historical_download_aggregates_chunks_and_records_failures(dq, no_sleep, tmp_path):
    chunk_runner = AsyncMock(
        side_effect=[_chunk_report(3, 3, 0), _chunk_report(2, 1, 1, "partial"), RuntimeError("chunk boom")]
    )
    with patch.object(dq, "run_group_download_async", chunk_runner):
        report = await dq.download_historical_async(
            "G", "20240115", "20240310", destination_dir=tmp_path, num_parts=2, chunk_delay=2.5
        )

    windows = [(c.kwargs["start_date"], c.kwargs["end_date"]) for c in chunk_runner.await_args_list]
    assert windows == [("20240115", "20240131"), ("20240201", "20240229"), ("20240301", "20240310")]
    assert all(
        c.kwargs["destination_dir"] == tmp_path and c.kwargs["num_parts"] == 2 for c in chunk_runner.await_args_list
    )
    assert no_sleep.await_count == 2
    no_sleep.assert_awaited_with(2.5)

    assert report.operation == "historical_download"
    assert report.status == "partial"
    assert report.counts == {"monthly_chunks": 3, "total_files": 5, "successful_downloads": 4, "failed_downloads": 1}
    assert report.details["chunks_with_errors"] == ["20240301-20240310"]
    assert report.details["success_rate"] == 80.0
    assert report.data[2]["error"] == "chunk boom"
    assert report.data[0]["report"]["counts"]["total_files"] == 3


@pytest.mark.asyncio
async def test_historical_download_single_chunk_success_does_not_sleep(dq, no_sleep):
    with patch.object(dq, "run_group_download_async", AsyncMock(return_value=_chunk_report(4, 4, 0))):
        report = await dq.download_historical_async("G", "20240101", "20240131")

    no_sleep.assert_not_awaited()
    assert report.status == "success"
    assert report.details["success_rate"] == 100.0


@pytest.mark.asyncio
async def test_historical_download_all_chunks_failing_is_error(dq, no_sleep):
    with patch.object(dq, "run_group_download_async", AsyncMock(side_effect=RuntimeError("down"))):
        report = await dq.download_historical_async("G", "20240101", "20240229")

    assert report.status == "error"
    assert len(report.details["chunks_with_errors"]) == 2
    assert report.details["success_rate"] == 0.0


@pytest.mark.asyncio
async def test_historical_download_with_no_files_in_any_chunk_reports_success(dq, no_sleep):
    """Finding: chunks that return 'no files found' (status=error) roll up to status='success'."""
    empty = _chunk_report(0, 0, 0, status="error", error=NO_FILES_FOUND_ERROR)
    with patch.object(dq, "run_group_download_async", AsyncMock(return_value=empty)):
        report = await dq.download_historical_async("G", "20240101", "20240229")

    assert all(chunk["report"]["status"] == "error" for chunk in report.data)
    assert report.counts["total_files"] == 0
    assert report.status == "success"


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

UNLIMITED = {"requests_per_minute": 1000, "burst_capacity": 100, "safe_interval": 0.1}


def test_rate_limit_capacity_without_client_or_when_disabled(dq):
    assert dq._calculate_rate_limit_capacity() == UNLIMITED
    dq._client = _rate_client(enable_rate_limiting=False)
    assert dq._calculate_rate_limit_capacity() == UNLIMITED


def test_rate_limit_capacity_reflects_config(dq):
    dq._client = _rate_client(requests_per_minute=120, burst_capacity=10, enable_queuing=False)
    assert dq._calculate_rate_limit_capacity() == {
        "requests_per_minute": 120,
        "burst_capacity": 10,
        "safe_interval": 0.5,
        "queuing_enabled": False,
    }


def test_rate_limit_capacity_falls_back_to_conservative_defaults_on_error(dq):
    dq._client = SimpleNamespace()
    assert dq._calculate_rate_limit_capacity() == {
        "requests_per_minute": 100,
        "burst_capacity": 10,
        "safe_interval": 0.6,
    }


def test_intelligent_delay_uses_largest_of_base_and_rpm_interval(dq):
    capacity = {"requests_per_minute": 120, "burst_capacity": 10, "safe_interval": 0.5}
    assert dq._calculate_intelligent_delay(4, capacity, 0.2) == 0.5
    assert dq._calculate_intelligent_delay(4, capacity, 0.75) == 0.75


def test_intelligent_delay_spreads_requests_beyond_burst_capacity(dq):
    """Finding: the stagger shrinks as the overflow grows (1 over burst -> 5.0s, 100 over -> 0.1s)."""
    capacity = {"requests_per_minute": 6000, "burst_capacity": 10, "safe_interval": 0.01}
    assert dq._calculate_intelligent_delay(11, capacity, 0.0) == 5.0
    assert dq._calculate_intelligent_delay(60, capacity, 0.0) == 0.1
    assert dq._calculate_intelligent_delay(110, capacity, 0.0) == pytest.approx(0.1)


def test_intelligent_delay_returns_base_delay_on_bad_capacity(dq):
    assert dq._calculate_intelligent_delay(4, {}, 0.3) == 0.3


def test_safe_concurrency_without_client_or_when_disabled(dq):
    assert dq._calculate_safe_concurrency_limit(25) == 25
    dq._client = _rate_client(enable_rate_limiting=False)
    assert dq._calculate_safe_concurrency_limit(25) == 25


def test_safe_concurrency_caps_at_burst_capacity_with_queueing(dq):
    dq._client = _rate_client(requests_per_minute=300, burst_capacity=5, max_queue_size=1000, enable_queuing=True)
    assert dq._calculate_safe_concurrency_limit(20) == 5
    assert dq._calculate_safe_concurrency_limit(3) == 3


def test_safe_concurrency_without_queueing_uses_quarter_of_rpm(dq):
    dq._client = _rate_client(requests_per_minute=40, burst_capacity=20, enable_queuing=False)
    assert dq._calculate_safe_concurrency_limit(16) == 10


def test_safe_concurrency_conservative_default_on_error(dq):
    dq._client = SimpleNamespace()
    assert dq._calculate_safe_concurrency_limit(80) == 10
    assert dq._calculate_safe_concurrency_limit(8) == 2
    assert dq._calculate_safe_concurrency_limit(2) == 1


def test_rate_limit_recommendations_flag_every_shortfall(dq):
    assert dq._get_rate_limit_recommendations(10) == {}

    dq._client = _rate_client(requests_per_minute=60, burst_capacity=5, enable_queuing=False, max_queue_size=50)
    recs = dq._get_rate_limit_recommendations(10)

    assert recs["current_settings"] == {
        "requests_per_minute": 60,
        "burst_capacity": 5,
        "queuing_enabled": False,
        "max_queue_size": 50,
    }
    by_type = {r["type"]: r for r in recs["recommendations"]}
    assert set(by_type) == {"increase_burst_capacity", "increase_requests_per_minute", "enable_queuing"}
    assert by_type["increase_burst_capacity"]["recommended"] == 10
    assert by_type["increase_requests_per_minute"]["recommended"] == 150


def test_rate_limit_recommendations_empty_when_settings_suffice_or_on_error(dq):
    dq._client = _rate_client(requests_per_minute=300, burst_capacity=5, enable_queuing=True)
    assert dq._get_rate_limit_recommendations(2)["recommendations"] == []
    dq._client = SimpleNamespace()
    assert dq._get_rate_limit_recommendations(2) == {}


def test_rate_limit_info_reports_config_and_live_state(dq):
    assert dq.get_rate_limit_info() == {"error": "Client not connected"}

    dq._client = DataQueryClient(_cfg())
    info = dq.get_rate_limit_info()
    config = dq._client.rate_limiter.config
    assert info["configuration"]["requests_per_minute"] == config.requests_per_minute
    assert info["configuration"]["burst_capacity"] == config.burst_capacity
    assert info["current_state"]["queue_size"] == 0
    assert info["current_state"]["total_requests"] == 0

    dq._client = SimpleNamespace()
    assert dq.get_rate_limit_info()["error"].startswith("Failed to get rate limit info")


def test_optimize_concurrency_unchanged_without_client(dq):
    result = dq.optimize_concurrency_for_rate_limits(max_concurrent=4, num_parts=3)
    assert result["optimized"] == {"max_concurrent": 4, "num_parts": 3, "total_concurrency": 12}
    assert result["rate_limit_applied"] is False
    assert result["recommendations"] == {}


def test_optimize_concurrency_scales_down_to_safe_limit(dq):
    dq._client = _rate_client(requests_per_minute=300, burst_capacity=5, enable_queuing=True)
    result = dq.optimize_concurrency_for_rate_limits(max_concurrent=10, num_parts=1)

    assert result["safe_limit"] == 5
    assert result["rate_limit_applied"] is True
    assert result["optimized"] == {"max_concurrent": 5, "num_parts": 1, "total_concurrency": 5}


def test_optimize_concurrency_can_overshoot_the_reduction(dq):
    """Finding: per-dimension flooring turns a safe limit of 5 into 1x1 for a 6x6 request."""
    dq._client = _rate_client(requests_per_minute=300, burst_capacity=5, enable_queuing=True)
    result = dq.optimize_concurrency_for_rate_limits(max_concurrent=6, num_parts=6)

    assert result["safe_limit"] == 5
    assert result["optimized"] == {"max_concurrent": 1, "num_parts": 1, "total_concurrency": 1}


def test_pool_stats_and_stats_require_a_client(dq, fake_client):
    assert dq.get_pool_stats() == {"error": "Client not connected"}
    assert dq.get_stats() == {"status": "not_connected"}

    fake_client.get_pool_stats.return_value = {"connections": {"total": 2}}
    fake_client.get_stats.return_value = {"requests": 7}
    dq._client = fake_client

    assert dq.get_pool_stats() == {"connections": {"total": 2}}
    stats = dq.get_stats()
    assert stats["requests"] == 7
    assert stats["api_config"] == {
        "base_url": "https://api.example.com",
        "context_path": dq.client_config.context_path,
        "api_base_url": dq.client_config.api_base_url,
    }
