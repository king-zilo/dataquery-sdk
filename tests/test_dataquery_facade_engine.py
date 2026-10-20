"""Characterization tests for the ``DataQuery`` facade (``dataquery/dataquery.py``).

They pin the public API (signatures, type hints, package exports) and today's
behaviour, so the facade can be hardened and modernized without breaking callers.
Defects are recorded as strict xfails that describe the correct behaviour.
"""

from __future__ import annotations

import asyncio
import collections.abc as cabc
import inspect
import types
import typing
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import dataquery
import dataquery.dataquery as dq_module
from dataquery.constants.download import NO_FILES_FOUND_ERROR
from dataquery.dataquery import ConfigManager, DataQuery, ProgressTracker, _format_duration, _SyncProxy
from dataquery.types.exceptions import ConfigurationError
from dataquery.types.models import (
    AvailabilityInfo,
    ClientConfig,
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
    OperationReport,
)

# ---------------------------------------------------------------------------
# Public API pins
# ---------------------------------------------------------------------------

EXPECTED_SIGNATURES = {
    "DataQuery.__aenter__": "(self)",
    "DataQuery.__aexit__": "(self, exc_type, exc_val, exc_tb)",
    "DataQuery.__enter__": "(self)",
    "DataQuery.__exit__": "(self, exc_type, exc_val, exc_tb)",
    "DataQuery.__init__": "(self, config_or_env_file=None, client_id=None, client_secret=None, **overrides)",
    "DataQuery.auto_download": "(self, group_id, destination_dir='./downloads', file_filter=None, progress_callback=None, error_callback=None, max_retries=3, max_concurrent_downloads=5, initial_check=True, reconnect_delay=5.0, max_reconnect_delay=60.0, file_group_id=None, show_progress=True, enable_event_replay=True, heartbeat_timeout=0.0, max_tracked_files=10000, max_tracked_errors=1000)",
    "DataQuery.auto_download_async": "(self, group_id, destination_dir='./downloads', file_filter=None, progress_callback=None, error_callback=None, max_retries=3, max_concurrent_downloads=5, initial_check=True, reconnect_delay=5.0, max_reconnect_delay=60.0, file_group_id=None, show_progress=True, enable_event_replay=True, heartbeat_timeout=0.0, max_tracked_files=10000, max_tracked_errors=1000)",
    "DataQuery.check_availability": "(self, file_group_id, file_datetime)",
    "DataQuery.check_availability_async": "(self, file_group_id, file_datetime)",
    "DataQuery.cleanup": "(self)",
    "DataQuery.cleanup_async": "(self)",
    "DataQuery.close": "(self)",
    "DataQuery.close_async": "(self)",
    "DataQuery.connect": "(self)",
    "DataQuery.connect_async": "(self)",
    "DataQuery.create_progress_callback": "(self, log_interval=10)",
    "DataQuery.download_file": "(self, file_group_id, file_datetime=None, destination_path=None, options=None, num_parts=1, progress_callback=None)",
    "DataQuery.download_file_async": "(self, file_group_id, file_datetime=None, destination_path=None, options=None, num_parts=1, progress_callback=None)",
    "DataQuery.download_historical": "(self, group_id, start_date, end_date, destination_dir=PosixPath('downloads'), max_concurrent=5, num_parts=1, delay_between_downloads=0.2, max_retries=3, progress_callback=None, chunk_delay=2.0)",
    "DataQuery.download_historical_async": "(self, group_id, start_date, end_date, destination_dir=PosixPath('downloads'), max_concurrent=5, num_parts=1, delay_between_downloads=0.2, max_retries=3, progress_callback=None, chunk_delay=2.0)",
    "DataQuery.files_to_dataframe": "(self, files, include_metadata=False)",
    "DataQuery.get_expressions_time_series": "(self, expressions, format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', data='ALL', page=None)",
    "DataQuery.get_expressions_time_series_async": "(self, expressions, format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', data='ALL', page=None)",
    "DataQuery.get_grid_data": "(self, expr=None, grid_id=None, date=None)",
    "DataQuery.get_grid_data_async": "(self, expr=None, grid_id=None, date=None)",
    "DataQuery.get_group_attributes": "(self, group_id, instrument_id=None, page=None)",
    "DataQuery.get_group_attributes_async": "(self, group_id, instrument_id=None, page=None)",
    "DataQuery.get_group_filters": "(self, group_id, page=None)",
    "DataQuery.get_group_filters_async": "(self, group_id, page=None)",
    "DataQuery.get_group_time_series": "(self, group_id, attributes, filter=None, data='ALL', format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', page=None)",
    "DataQuery.get_group_time_series_async": "(self, group_id, attributes, filter=None, data='ALL', format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', page=None)",
    "DataQuery.get_instrument_time_series": "(self, instruments, attributes, data='ALL', format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', page=None)",
    "DataQuery.get_instrument_time_series_async": "(self, instruments, attributes, data='ALL', format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', page=None)",
    "DataQuery.get_next_page": "(self, page)",
    "DataQuery.get_next_page_async": "(self, page)",
    "DataQuery.get_pool_stats": "(self)",
    "DataQuery.get_rate_limit_info": "(self)",
    "DataQuery.get_stats": "(self)",
    "DataQuery.groups_to_dataframe": "(self, groups, include_metadata=False)",
    "DataQuery.health_check": "(self)",
    "DataQuery.health_check_async": "(self)",
    "DataQuery.instruments_to_dataframe": "(self, instruments, include_metadata=False)",
    "DataQuery.iter_search_groups_async": "(self, keywords, *, limit=None, max_pages=1000)",
    "DataQuery.list_available_files": "(self, group_id, file_group_id=None, start_date=None, end_date=None)",
    "DataQuery.list_available_files_async": "(self, group_id, file_group_id=None, start_date=None, end_date=None)",
    "DataQuery.list_files": "(self, group_id, file_group_id=None)",
    "DataQuery.list_files_async": "(self, group_id, file_group_id=None)",
    "DataQuery.list_files_page": "(self, group_id, file_group_id=None, *, page=None)",
    "DataQuery.list_files_page_async": "(self, group_id, file_group_id=None, *, page=None)",
    "DataQuery.list_groups": "(self, limit=100)",
    "DataQuery.list_groups_async": "(self, limit=100)",
    "DataQuery.list_groups_page": "(self, limit=None, *, page=None)",
    "DataQuery.list_groups_page_async": "(self, limit=None, *, page=None)",
    "DataQuery.list_instruments": "(self, group_id, instrument_id=None, page=None)",
    "DataQuery.list_instruments_async": "(self, group_id, instrument_id=None, page=None)",
    "DataQuery.optimize_concurrency_for_rate_limits": "(self, max_concurrent, num_parts)",
    "DataQuery.run_availability": "(self, file_group_id, file_datetime)",
    "DataQuery.run_availability_async": "(self, file_group_id, file_datetime)",
    "DataQuery.run_download": "(self, file_group_id, file_datetime=None, destination_path=None, max_concurrent=1)",
    "DataQuery.run_download_async": "(self, file_group_id, file_datetime=None, destination_path=None, max_concurrent=1)",
    "DataQuery.run_group_download": "(self, group_id, start_date, end_date, destination_dir=PosixPath('downloads'), max_concurrent=5, num_parts=1, progress_callback=None, delay_between_downloads=1.0, file_group_id=None)",
    "DataQuery.run_group_download_async": "(self, group_id, start_date, end_date, destination_dir=PosixPath('downloads'), max_concurrent=5, num_parts=1, progress_callback=None, delay_between_downloads=0.2, max_retries=3, file_group_id=None, on_file_complete=None)",
    "DataQuery.run_group_files": "(self, group_id, max_concurrent=5)",
    "DataQuery.run_group_files_async": "(self, group_id, max_concurrent=5)",
    "DataQuery.run_groups": "(self, max_concurrent=5)",
    "DataQuery.run_groups_async": "(self, max_concurrent=5)",
    "DataQuery.search": "(self, query)",
    "DataQuery.search_all_groups": "(self, keywords, *, limit=None, max_pages=1000, raise_on_cap=True)",
    "DataQuery.search_all_groups_async": "(self, keywords, *, limit=None, max_pages=1000, raise_on_cap=True)",
    "DataQuery.search_async": "(self, query)",
    "DataQuery.search_groups": "(self, keywords, limit=100, offset=None, *, page=None)",
    "DataQuery.search_groups_async": "(self, keywords, limit=100, offset=None, *, page=None)",
    "DataQuery.search_groups_page": "(self, keywords, limit=None, *, page=None)",
    "DataQuery.search_groups_page_async": "(self, keywords, limit=None, *, page=None)",
    "DataQuery.search_instruments": "(self, group_id, keywords, page=None)",
    "DataQuery.search_instruments_async": "(self, group_id, keywords, page=None)",
    "DataQuery.service_status": "(self)",
    "DataQuery.service_status_async": "(self)",
    "DataQuery.time_series_to_dataframe": "(self, time_series, include_metadata=False)",
    "DataQuery.to_dataframe": "(self, response_data, flatten_nested=True, include_metadata=False, date_columns=None, numeric_columns=None, custom_transformations=None)",
    "ConfigManager.__init__": "(self, env_file=None)",
    "ConfigManager.get_client_config": "(self)",
    "ProgressTracker.__init__": "(self, log_interval=10)",
    "ProgressTracker.create_progress_callback": "(self)",
    "_SyncProxy.__dir__": "(self)",
    "_SyncProxy.__getattr__": "(self, name)",
    "_SyncProxy.__init__": "(self, dq)",
}

EXPECTED_HINTS = {
    "ConfigManager.__init__": "env_file: Union[None, Path]",
    "ConfigManager.get_client_config": "return: ClientConfig",
    "DataQuery.__aenter__": "",
    "DataQuery.__aexit__": "",
    "DataQuery.__enter__": "",
    "DataQuery.__exit__": "",
    "DataQuery.__init__": "client_id: Union[None, str]; client_secret: Union[None, str]; config_or_env_file: Union[ClientConfig, None, Path, str]; overrides: Any",
    "DataQuery.auto_download": "destination_dir: str; enable_event_replay: bool; error_callback: Union[Callable, None]; file_filter: Union[Callable, None]; file_group_id: Union[None, list[str], str]; group_id: str; heartbeat_timeout: float; initial_check: bool; max_concurrent_downloads: int; max_reconnect_delay: float; max_retries: int; max_tracked_errors: int; max_tracked_files: int; progress_callback: Union[Callable, None]; reconnect_delay: float; show_progress: bool",
    "DataQuery.auto_download_async": "destination_dir: str; enable_event_replay: bool; error_callback: Union[Callable, None]; file_filter: Union[Callable, None]; file_group_id: Union[None, list[str], str]; group_id: str; heartbeat_timeout: float; initial_check: bool; max_concurrent_downloads: int; max_reconnect_delay: float; max_retries: int; max_tracked_errors: int; max_tracked_files: int; progress_callback: Union[Callable, None]; reconnect_delay: float; show_progress: bool",
    "DataQuery.check_availability": "file_datetime: str; file_group_id: str; return: AvailabilityInfo",
    "DataQuery.check_availability_async": "file_datetime: str; file_group_id: str; return: AvailabilityInfo",
    "DataQuery.cleanup": "",
    "DataQuery.cleanup_async": "",
    "DataQuery.close": "",
    "DataQuery.close_async": "",
    "DataQuery.connect": "",
    "DataQuery.connect_async": "",
    "DataQuery.create_progress_callback": "log_interval: int; return: Callable",
    "DataQuery.download_file": "destination_path: Union[None, Path]; file_datetime: Union[None, str]; file_group_id: str; num_parts: int; options: Union[DownloadOptions, None]; progress_callback: Union[Callable, None]; return: DownloadResult",
    "DataQuery.download_file_async": "destination_path: Union[None, Path]; file_datetime: Union[None, str]; file_group_id: str; num_parts: int; options: Union[DownloadOptions, None]; progress_callback: Union[Callable, None]; return: DownloadResult",
    "DataQuery.download_historical": "chunk_delay: float; delay_between_downloads: float; destination_dir: Path; end_date: str; group_id: str; max_concurrent: int; max_retries: int; num_parts: int; progress_callback: Union[Callable, None]; return: OperationReport; start_date: str",
    "DataQuery.download_historical_async": "chunk_delay: float; delay_between_downloads: float; destination_dir: Path; end_date: str; group_id: str; max_concurrent: int; max_retries: int; num_parts: int; progress_callback: Union[Callable, None]; return: OperationReport; start_date: str",
    "DataQuery.files_to_dataframe": "include_metadata: bool",
    "DataQuery.get_expressions_time_series": "calendar: str; conversion: str; data: str; end_date: Union[None, str]; expressions: list[str]; format: str; frequency: str; nan_treatment: str; page: Union[None, str]; return: TimeSeriesResponse; start_date: Union[None, str]",
    "DataQuery.get_expressions_time_series_async": "calendar: str; conversion: str; data: str; end_date: Union[None, str]; expressions: list[str]; format: str; frequency: str; nan_treatment: str; page: Union[None, str]; return: TimeSeriesResponse; start_date: Union[None, str]",
    "DataQuery.get_grid_data": "date: Union[None, str]; expr: Union[None, str]; grid_id: Union[None, str]; return: GridDataResponse",
    "DataQuery.get_grid_data_async": "date: Union[None, str]; expr: Union[None, str]; grid_id: Union[None, str]; return: GridDataResponse",
    "DataQuery.get_group_attributes": "group_id: str; instrument_id: Union[None, str]; page: Union[None, str]; return: AttributesResponse",
    "DataQuery.get_group_attributes_async": "group_id: str; instrument_id: Union[None, str]; page: Union[None, str]; return: AttributesResponse",
    "DataQuery.get_group_filters": "group_id: str; page: Union[None, str]; return: FiltersResponse",
    "DataQuery.get_group_filters_async": "group_id: str; page: Union[None, str]; return: FiltersResponse",
    "DataQuery.get_group_time_series": "attributes: list[str]; calendar: str; conversion: str; data: str; end_date: Union[None, str]; filter: Union[None, str]; format: str; frequency: str; group_id: str; nan_treatment: str; page: Union[None, str]; return: TimeSeriesResponse; start_date: Union[None, str]",
    "DataQuery.get_group_time_series_async": "attributes: list[str]; calendar: str; conversion: str; data: str; end_date: Union[None, str]; filter: Union[None, str]; format: str; frequency: str; group_id: str; nan_treatment: str; page: Union[None, str]; return: TimeSeriesResponse; start_date: Union[None, str]",
    "DataQuery.get_instrument_time_series": "attributes: list[str]; calendar: str; conversion: str; data: str; end_date: Union[None, str]; format: str; frequency: str; instruments: list[str]; nan_treatment: str; page: Union[None, str]; return: TimeSeriesResponse; start_date: Union[None, str]",
    "DataQuery.get_instrument_time_series_async": "attributes: list[str]; calendar: str; conversion: str; data: str; end_date: Union[None, str]; format: str; frequency: str; instruments: list[str]; nan_treatment: str; page: Union[None, str]; return: TimeSeriesResponse; start_date: Union[None, str]",
    "DataQuery.get_next_page": "page: ~P; return: Union[None, ~P]",
    "DataQuery.get_next_page_async": "page: ~P; return: Union[None, ~P]",
    "DataQuery.get_pool_stats": "return: dict[str, Any]",
    "DataQuery.get_rate_limit_info": "return: dict[str, Any]",
    "DataQuery.get_stats": "return: dict[str, Any]",
    "DataQuery.groups_to_dataframe": "include_metadata: bool",
    "DataQuery.health_check": "return: bool",
    "DataQuery.health_check_async": "return: bool",
    "DataQuery.instruments_to_dataframe": "include_metadata: bool",
    "DataQuery.iter_search_groups_async": "keywords: str; limit: Union[None, int]; max_pages: int",
    "DataQuery.list_available_files": "end_date: Union[None, str]; file_group_id: Union[None, str]; group_id: str; return: list[dict[str, Any]]; start_date: Union[None, str]",
    "DataQuery.list_available_files_async": "end_date: Union[None, str]; file_group_id: Union[None, str]; group_id: str; return: list[dict[str, Any]]; start_date: Union[None, str]",
    "DataQuery.list_files": "file_group_id: Union[None, str]; group_id: str; return: list[FileInfo]",
    "DataQuery.list_files_async": "file_group_id: Union[None, str]; group_id: str; return: list[FileInfo]",
    "DataQuery.list_files_page": "file_group_id: Union[None, str]; group_id: str; page: Union[None, str]; return: FileList",
    "DataQuery.list_files_page_async": "file_group_id: Union[None, str]; group_id: str; page: Union[None, str]; return: FileList",
    "DataQuery.list_groups": "limit: Union[None, int]; return: list[Group]",
    "DataQuery.list_groups_async": "limit: Union[None, int]; return: list[Group]",
    "DataQuery.list_groups_page": "limit: Union[None, int]; page: Union[None, str]; return: GroupList",
    "DataQuery.list_groups_page_async": "limit: Union[None, int]; page: Union[None, str]; return: GroupList",
    "DataQuery.list_instruments": "group_id: str; instrument_id: Union[None, str]; page: Union[None, str]; return: InstrumentsResponse",
    "DataQuery.list_instruments_async": "group_id: str; instrument_id: Union[None, str]; page: Union[None, str]; return: InstrumentsResponse",
    "DataQuery.optimize_concurrency_for_rate_limits": "max_concurrent: int; num_parts: int; return: dict[str, Any]",
    "DataQuery.run_availability": "file_datetime: str; file_group_id: str; return: OperationReport",
    "DataQuery.run_availability_async": "file_datetime: str; file_group_id: str; return: OperationReport",
    "DataQuery.run_download": "destination_path: Union[None, Path]; file_datetime: Union[None, str]; file_group_id: str; max_concurrent: int; return: OperationReport",
    "DataQuery.run_download_async": "destination_path: Union[None, Path]; file_datetime: Union[None, str]; file_group_id: str; max_concurrent: int; return: OperationReport",
    "DataQuery.run_group_download": "delay_between_downloads: float; destination_dir: Path; end_date: str; file_group_id: Union[None, list[str], str]; group_id: str; max_concurrent: int; num_parts: int; progress_callback: Union[Callable, None]; return: OperationReport; start_date: str",
    "DataQuery.run_group_download_async": "delay_between_downloads: float; destination_dir: Union[Path, str]; end_date: str; file_group_id: Union[None, list[str], str]; group_id: str; max_concurrent: int; max_retries: int; num_parts: int; on_file_complete: Union[Callable[[DownloadResult], Awaitable[None]], None]; progress_callback: Union[Callable, None]; return: OperationReport; start_date: str",
    "DataQuery.run_group_files": "group_id: str; max_concurrent: int; return: OperationReport",
    "DataQuery.run_group_files_async": "group_id: str; max_concurrent: int; return: OperationReport",
    "DataQuery.run_groups": "max_concurrent: int; return: OperationReport",
    "DataQuery.run_groups_async": "max_concurrent: int; return: OperationReport",
    "DataQuery.search": "query: str; return: dict[str, Any]",
    "DataQuery.search_all_groups": "keywords: str; limit: Union[None, int]; max_pages: int; raise_on_cap: bool; return: list[Group]",
    "DataQuery.search_all_groups_async": "keywords: str; limit: Union[None, int]; max_pages: int; raise_on_cap: bool; return: list[Group]",
    "DataQuery.search_async": "query: str; return: dict[str, Any]",
    "DataQuery.search_groups": "keywords: str; limit: Union[None, int]; offset: Union[None, int]; page: Union[None, str]; return: list[Group]",
    "DataQuery.search_groups_async": "keywords: str; limit: Union[None, int]; offset: Union[None, int]; page: Union[None, str]; return: list[Group]",
    "DataQuery.search_groups_page": "keywords: str; limit: Union[None, int]; page: Union[None, str]; return: GroupList",
    "DataQuery.search_groups_page_async": "keywords: str; limit: Union[None, int]; page: Union[None, str]; return: GroupList",
    "DataQuery.search_instruments": "group_id: str; keywords: str; page: Union[None, str]; return: InstrumentsResponse",
    "DataQuery.search_instruments_async": "group_id: str; keywords: str; page: Union[None, str]; return: InstrumentsResponse",
    "DataQuery.service_status": "return: dict[str, Any]",
    "DataQuery.service_status_async": "return: dict[str, Any]",
    "DataQuery.time_series_to_dataframe": "include_metadata: bool",
    "DataQuery.to_dataframe": "custom_transformations: Union[None, dict[str, Callable]]; date_columns: Union[None, list[str]]; flatten_nested: bool; include_metadata: bool; numeric_columns: Union[None, list[str]]",
    "ProgressTracker.__init__": "log_interval: int",
    "ProgressTracker.create_progress_callback": "return: Callable",
    "_SyncProxy.__dir__": "return: list[str]",
    "_SyncProxy.__getattr__": "name: str; return: Callable[..., Any]",
    "_SyncProxy.__init__": "dq: DataQuery; return: None",
}

EXPECTED_PACKAGE_ALL = [
    "DataQuery",
    "DataQueryClient",
    "ClientConfig",
    "Group",
    "GroupList",
    "Paginated",
    "Link",
    "FileInfo",
    "FileList",
    "AvailabilityInfo",
    "AvailableFilesResponse",
    "DownloadResult",
    "DownloadStatus",
    "DownloadOptions",
    "DownloadProgress",
    "DateRange",
    "DataQueryError",
    "APIResponseError",
    "AuthenticationError",
    "ValidationError",
    "NotFoundError",
    "PaginationError",
    "RateLimitError",
    "NetworkError",
    "ConfigurationError",
    "DownloadError",
    "AvailabilityError",
    "GroupNotFoundError",
    "FileNotFoundInGroupError",
    "DateRangeError",
    "FileTypeError",
    "WorkflowError",
    "EnvConfig",
    "NotificationDownloadManager",
    "SSEClient",
    "SSEEvent",
    "format_function_syntax",
    "get_function_categories",
    "get_function_param_counts",
    "get_function_registry",
    "list_functions_by_category",
    "lookup_function",
    "download_zip_async",
]

EXPECTED_PUBLIC_DATAQUERY_NAMES = [
    "auto_download",
    "auto_download_async",
    "check_availability",
    "check_availability_async",
    "cleanup",
    "cleanup_async",
    "close",
    "close_async",
    "connect",
    "connect_async",
    "create_progress_callback",
    "download_file",
    "download_file_async",
    "download_historical",
    "download_historical_async",
    "files_to_dataframe",
    "get_expressions_time_series",
    "get_expressions_time_series_async",
    "get_grid_data",
    "get_grid_data_async",
    "get_group_attributes",
    "get_group_attributes_async",
    "get_group_filters",
    "get_group_filters_async",
    "get_group_time_series",
    "get_group_time_series_async",
    "get_instrument_time_series",
    "get_instrument_time_series_async",
    "get_next_page",
    "get_next_page_async",
    "get_pool_stats",
    "get_rate_limit_info",
    "get_stats",
    "groups_to_dataframe",
    "health_check",
    "health_check_async",
    "instruments_to_dataframe",
    "iter_search_groups_async",
    "list_available_files",
    "list_available_files_async",
    "list_files",
    "list_files_async",
    "list_files_page",
    "list_files_page_async",
    "list_groups",
    "list_groups_async",
    "list_groups_page",
    "list_groups_page_async",
    "list_instruments",
    "list_instruments_async",
    "optimize_concurrency_for_rate_limits",
    "run_availability",
    "run_availability_async",
    "run_download",
    "run_download_async",
    "run_group_download",
    "run_group_download_async",
    "run_group_files",
    "run_group_files_async",
    "run_groups",
    "run_groups_async",
    "search",
    "search_all_groups",
    "search_all_groups_async",
    "search_async",
    "search_groups",
    "search_groups_async",
    "search_groups_page",
    "search_groups_page_async",
    "search_instruments",
    "search_instruments_async",
    "service_status",
    "service_status_async",
    "sync",
    "time_series_to_dataframe",
    "to_dataframe",
]


def _norm(t: Any) -> str:
    """Spelling-independent rendering of a type hint (``Optional[X]`` == ``X | None``, ``List`` == ``list``)."""
    if t is type(None):
        return "None"
    if t is typing.Any:
        return "Any"
    if t in (typing.Callable, cabc.Callable):
        return "Callable"
    if isinstance(t, typing.TypeVar):
        return f"~{t.__name__}"
    origin = typing.get_origin(t)
    if origin in (typing.Union, types.UnionType):
        return "Union[" + ", ".join(sorted(_norm(a) for a in typing.get_args(t))) + "]"
    if origin is not None:
        args = typing.get_args(t)
        name = "Callable" if origin is cabc.Callable else getattr(origin, "__name__", str(origin))
        if name == "Literal":
            return "Literal[" + ", ".join(repr(a) for a in args) + "]"

        def _arg(a: Any) -> str:
            if a is Ellipsis:
                return "..."
            if isinstance(a, list):
                return "[" + ", ".join(_norm(x) for x in a) + "]"
            return _norm(a)

        return name + "[" + ", ".join(_arg(a) for a in args) + "]"
    return getattr(t, "__qualname__", repr(t))


def _plain_signature(f: Any) -> str:
    sig = inspect.signature(f)
    params = [p.replace(annotation=inspect.Parameter.empty) for p in sig.parameters.values()]
    return str(sig.replace(parameters=params, return_annotation=inspect.Signature.empty))


def _resolve(qualname: str) -> Any:
    cls_name, attr = qualname.split(".")
    f = vars(getattr(dq_module, cls_name))[attr]
    return f.__func__ if isinstance(f, (staticmethod, classmethod)) else f


@pytest.mark.parametrize("qualname", sorted(EXPECTED_SIGNATURES))
def test_public_signature_is_unchanged(qualname: str) -> None:
    assert _plain_signature(_resolve(qualname)) == EXPECTED_SIGNATURES[qualname]


@pytest.mark.parametrize("qualname", sorted(EXPECTED_HINTS))
def test_public_type_hints_are_unchanged(qualname: str) -> None:
    hints = typing.get_type_hints(_resolve(qualname))
    rendered = "; ".join(f"{k}: {_norm(v)}" for k, v in sorted(hints.items()))
    assert rendered == EXPECTED_HINTS[qualname]


def test_public_dataquery_names_are_unchanged() -> None:
    assert sorted(n for n in vars(DataQuery) if not n.startswith("_")) == EXPECTED_PUBLIC_DATAQUERY_NAMES
    assert isinstance(vars(DataQuery)["sync"], property)
    assert isinstance(vars(DataQuery)["_split_into_monthly_ranges"], staticmethod)


def test_package_exports_are_unchanged() -> None:
    assert dataquery.__all__ == EXPECTED_PACKAGE_ALL
    for name in dataquery.__all__:
        assert hasattr(dataquery, name), name
    assert dataquery.DataQuery is DataQuery


def test_every_sync_wrapper_has_an_async_twin() -> None:
    public = set(EXPECTED_PUBLIC_DATAQUERY_NAMES)
    for name in public:
        if name.endswith("_async") and name != "iter_search_groups_async":
            assert name[: -len("_async")] in public, name
        assert not (name.endswith("_async") and not inspect.iscoroutinefunction(getattr(DataQuery, name))) or (
            name == "iter_search_groups_async"
        )
    assert inspect.isasyncgenfunction(DataQuery.iter_search_groups_async)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read a local .env and start every test without DATAQUERY_* settings."""
    monkeypatch.setattr(dq_module, "load_dotenv", lambda *a, **k: None)
    import os

    for key in list(os.environ):
        if key.startswith("DATAQUERY_") and key != "DATAQUERY_CONFIG_DIR":
            monkeypatch.delenv(key)


def _cfg(**kw: Any) -> ClientConfig:
    base: dict[str, Any] = {"base_url": "https://api.example.com", "oauth_enabled": False, "bearer_token": "t"}
    base.update(kw)
    return ClientConfig.model_validate(base)


def _facade(**kw: Any) -> tuple[DataQuery, MagicMock]:
    """A facade whose client is a mock, so ``connect_async`` is a no-op and no network is used."""
    dq = DataQuery(_cfg(**kw))
    client = MagicMock()
    dq._client = client
    return dq, client


def _last_call(mock: Any) -> Any:
    assert mock.await_args is not None
    return mock.await_args


def _result(**kw: Any) -> DownloadResult:
    return DownloadResult.model_validate(kw)


def _rate_client(**kw: Any) -> Any:
    from dataquery.transport.rate_limiter import RateLimitConfig, RateLimitState

    return types.SimpleNamespace(
        rate_limiter=types.SimpleNamespace(config=RateLimitConfig(**kw), state=RateLimitState(tokens=3.0))
    )


def _group_report(
    total: int, ok: int, failed: int, status: str = "success", error: str | None = None
) -> OperationReport:
    return OperationReport.model_validate(
        {
            "operation": "group_download",
            "status": status,
            "error": error,
            "counts": {"total_files": total, "successful_downloads": ok, "failed_downloads": failed},
        }
    )


# ---------------------------------------------------------------------------
# Construction and configuration
# ---------------------------------------------------------------------------


def test_explicit_config_is_used_and_validated() -> None:
    cfg = _cfg()
    dq = DataQuery(cfg)
    assert dq.client_config.base_url == "https://api.example.com"
    assert dq._client is None and dq._sync_proxy is None


def test_invalid_explicit_config_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="Either OAuth credentials or Bearer token"):
        DataQuery(ClientConfig(base_url="https://api.example.com", oauth_enabled=False))


def test_credentials_enable_oauth_and_wrap_secret() -> None:
    dq = DataQuery(_cfg(bearer_token=None), client_id="cid", client_secret="sec")
    assert dq.client_config.oauth_enabled is True
    assert dq.client_config.client_id == "cid"
    assert dq.client_config.get_client_secret() == "sec"


def test_token_url_is_derived_only_when_unset() -> None:
    dq = DataQuery(_cfg(oauth_token_url=None), client_id="cid", client_secret="sec")
    assert dq.client_config.oauth_token_url == "https://api.example.com/oauth/token"
    dq2 = DataQuery(_cfg(oauth_token_url="https://auth.example.com/t"), client_id="cid", client_secret="sec")
    assert dq2.client_config.oauth_token_url == "https://auth.example.com/t"


@pytest.mark.parametrize("kwargs", [{"client_id": 123}, {"client_secret": 456}])
def test_non_string_credentials_are_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError, match="must be a string"):
        DataQuery(_cfg(), **kwargs)


def test_valid_overrides_are_applied_and_none_or_unknown_ignored() -> None:
    dq = DataQuery(_cfg(), timeout=12.5, max_retries=None, not_a_field=1)
    assert dq.client_config.timeout == 12.5
    assert dq.client_config.max_retries == 3
    assert not hasattr(dq.client_config, "not_a_field")


def test_override_that_breaks_validation_raises() -> None:
    with pytest.raises(ConfigurationError, match="TIMEOUT must be positive"):
        DataQuery(_cfg(), timeout=-1)


def test_env_file_path_is_passed_to_config_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def fake(self: ConfigManager) -> ClientConfig:
        seen.append(self.env_file)
        return _cfg()

    monkeypatch.setattr(ConfigManager, "get_client_config", fake)
    DataQuery("some/.env")
    DataQuery(Path("other/.env"))
    DataQuery()
    assert seen == [Path("some/.env"), Path("other/.env"), None]


def test_config_manager_returns_valid_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAQUERY_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("DATAQUERY_CLIENT_ID", "cid")
    monkeypatch.setenv("DATAQUERY_CLIENT_SECRET", "sec")
    monkeypatch.setenv("DATAQUERY_TIMEOUT", "42")
    cfg = ConfigManager().get_client_config()
    assert cfg.base_url == "https://env.example.com" and cfg.timeout == 42.0
    dq = DataQuery()
    assert dq.client_config.base_url == "https://env.example.com"


def test_no_environment_and_no_credentials_raises() -> None:
    with pytest.raises(ConfigurationError):
        DataQuery()


def test_env_settings_survive_when_credentials_come_from_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAQUERY_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("DATAQUERY_TIMEOUT", "42")
    dq = DataQuery(client_id="cid", client_secret="sec")
    assert dq.client_config.base_url == "https://env.example.com"
    assert dq.client_config.timeout == 42.0


def test_fallback_without_base_url_uses_client_config_defaults() -> None:
    dq = DataQuery(client_id="cid", client_secret="sec")
    assert dq.client_config.base_url == ClientConfig().base_url


def test_validation_error_message_is_prefixed_once() -> None:
    with pytest.raises(ConfigurationError) as exc:
        DataQuery()
    assert str(exc.value).count("Configuration validation failed") == 1


def test_override_is_coerced_to_the_field_type() -> None:
    dq = DataQuery(_cfg(), timeout="30")
    assert dq.client_config.timeout == 30.0


@pytest.mark.parametrize("key, value", [("timeout", "abc"), ("max_retries", 2.5), ("oauth_enabled", "maybe")])
def test_invalid_override_raises_naming_the_key(key: str, value: Any) -> None:
    with pytest.raises(ConfigurationError, match=key):
        DataQuery(_cfg(), **{key: value})


def test_override_runs_field_validators() -> None:
    dq = DataQuery(_cfg(), base_url="https://other.example.com/")
    assert dq.client_config.base_url == "https://other.example.com"


def test_caller_config_is_not_mutated() -> None:
    cfg = _cfg(bearer_token=None, oauth_token_url=None)
    DataQuery(cfg, client_id="cid", client_secret="sec", timeout=5.0)
    assert cfg.timeout == 600.0
    assert cfg.client_id is None and cfg.oauth_enabled is False and cfg.oauth_token_url is None


def test_config_without_changes_is_used_as_is() -> None:
    cfg = _cfg()
    assert DataQuery(cfg).client_config is cfg


# ---------------------------------------------------------------------------
# Lifecycle, context managers, sync proxy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_creates_client_once_and_close_releases_it(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[Any] = []

    class FakeClient:
        def __init__(self, config: ClientConfig) -> None:
            created.append(config)
            self.connect = AsyncMock()
            self.close = AsyncMock()

    monkeypatch.setattr(dq_module, "DataQueryClient", FakeClient)
    dq = DataQuery(_cfg())
    async with dq as entered:
        assert entered is dq
        await dq.connect_async()
        client: Any = dq._ensure_client()
        assert len(created) == 1
        client.connect.assert_awaited_once()
    client.close.assert_awaited_once()
    assert dq._client is None
    await dq.cleanup_async()
    with pytest.raises(RuntimeError, match="Client not connected"):
        dq._ensure_client()


def test_sync_context_manager_and_close(monkeypatch: pytest.MonkeyPatch) -> None:
    closes: list[str] = []

    class FakeClient:
        def __init__(self, config: ClientConfig) -> None:
            self.connect = AsyncMock()
            self.close = AsyncMock(side_effect=lambda: closes.append("client"))

    monkeypatch.setattr(dq_module, "DataQueryClient", FakeClient)
    with DataQuery(_cfg()) as dq:
        assert dq._client is not None
    assert closes == ["client"] and dq._client is None
    dq.close()
    dq.cleanup()
    assert closes == ["client"]


def test_cleanup_closes_connected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeClient:
        def __init__(self, config: ClientConfig) -> None:
            self.connect = AsyncMock()
            self.close = AsyncMock()

    monkeypatch.setattr(dq_module, "DataQueryClient", FakeClient)
    dq = DataQuery(_cfg())
    dq.connect()
    client = dq._client
    dq.cleanup()
    assert dq._client is None
    assert isinstance(client, FakeClient)
    client.close.assert_awaited_once()


def test_sync_property_is_cached_and_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    dq = DataQuery(_cfg())
    assert dq.sync is dq.sync and isinstance(dq.sync, _SyncProxy)
    fake_list = AsyncMock(return_value=["g"])
    monkeypatch.setattr(dq, "list_groups_async", fake_list)
    fn = dq.sync.list_groups
    assert fn.__name__ == "list_groups"
    assert fn.__doc__ == "Synchronous wrapper for DataQuery.list_groups_async."
    assert fn(limit=3) == ["g"]
    fake_list.assert_awaited_once_with(limit=3)
    assert "list_groups" in dir(dq.sync) and "_run_sync" not in dir(dq.sync)
    with pytest.raises(AttributeError, match="no async method 'nope_async'"):
        dq.sync.nope
    dq.close()


def test_sync_method_inside_running_loop_raises() -> None:
    dq = DataQuery(_cfg())

    async def inner() -> None:
        with pytest.raises(RuntimeError, match="running"):
            dq.health_check()

    asyncio.run(inner())
    dq.close()


def test_sync_proxy_does_not_offer_async_generators() -> None:
    dq = DataQuery(_cfg())
    try:
        assert "iter_search_groups" not in dir(dq.sync)
        with pytest.raises(TypeError, match="iter_search_groups_async"):
            dq.sync.iter_search_groups("x")
    finally:
        dq.close()


# ---------------------------------------------------------------------------
# Thin delegation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_groups_uses_all_pages_when_limit_is_none() -> None:
    dq, client = _facade()
    client.list_all_groups_async = AsyncMock(return_value=["all"])
    client.list_groups_async = AsyncMock(return_value=["page"])
    assert await dq.list_groups_async(None) == ["all"]
    assert await dq.list_groups_async(7) == ["page"]
    client.list_groups_async.assert_awaited_once_with(limit=7)


@pytest.mark.asyncio
async def test_iter_search_groups_yields_from_client() -> None:
    dq, client = _facade()

    async def gen(keywords: str, limit: Any = None, max_pages: int = 0) -> Any:
        for g in ("a", "b"):
            yield g

    client.iter_search_groups_async = gen
    assert [g async for g in dq.iter_search_groups_async("kw")] == ["a", "b"]


@pytest.mark.asyncio
async def test_search_and_auto_download_delegate() -> None:
    dq, client = _facade()
    client.search_async = AsyncMock(return_value={"hits": 1})
    client.search_all_groups_async = AsyncMock(return_value=[])
    client.service_status_async = AsyncMock(return_value={"up": True})
    client.auto_download_async = AsyncMock(return_value="mgr")
    assert await dq.search_async("q") == {"hits": 1}
    assert await dq.search_all_groups_async("kw", limit=2, max_pages=3, raise_on_cap=False) == []
    client.search_all_groups_async.assert_awaited_once_with("kw", limit=2, max_pages=3, raise_on_cap=False)
    assert await dq.service_status_async() == {"up": True}
    assert await dq.auto_download_async("G", file_group_id=["F"]) == "mgr"
    kwargs = _last_call(client.auto_download_async).kwargs
    assert kwargs["group_id"] == "G" and kwargs["file_group_id"] == ["F"] and kwargs["max_tracked_files"] == 10_000


def test_sync_wrappers_route_through_run_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    dq = DataQuery(_cfg())
    calls: list[str] = []

    def fake_run(coro: Any) -> str:
        calls.append(coro.cr_code.co_name)
        coro.close()
        return "ok"

    monkeypatch.setattr(dq, "_run_sync", fake_run)
    dq.search_all_groups("k")
    dq.service_status()
    dq.search("q")
    dq.run_groups()
    dq.run_group_files("G")
    dq.run_availability("F", "20240101")
    dq.run_download("F")
    dq.run_group_download("G", "20240101", "20240102")
    dq.download_historical("G", "20240101", "20240102")
    dq.auto_download("G")
    dq.download_file("F")
    assert calls == [
        "search_all_groups_async",
        "service_status_async",
        "search_async",
        "run_groups_async",
        "run_group_files_async",
        "run_availability_async",
        "run_download_async",
        "run_group_download_async",
        "download_historical_async",
        "auto_download_async",
        "download_file_async",
    ]


def test_sync_run_group_download_forwards_its_own_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    dq = DataQuery(_cfg())
    fake = AsyncMock(return_value="r")
    monkeypatch.setattr(dq, "run_group_download_async", fake)
    assert dq.run_group_download("G", "20240101", "20240131") == "r"
    args = fake.await_args
    assert args is not None
    assert args.args == ("G", "20240101", "20240131", Path("./downloads"), 5, 1, None, 1.0)
    assert args.kwargs == {"file_group_id": None}
    dq.close()


@pytest.mark.parametrize(
    "method",
    [
        "to_dataframe",
        "groups_to_dataframe",
        "files_to_dataframe",
        "instruments_to_dataframe",
        "time_series_to_dataframe",
    ],
)
def test_dataframe_helpers_create_client_lazily_and_delegate(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    created: list[Any] = []

    class FakeClient:
        def __init__(self, config: ClientConfig) -> None:
            created.append(config)

        def __getattr__(self, name: str) -> Any:
            return lambda *a, **k: (name, a, k)

    monkeypatch.setattr(dq_module, "DataQueryClient", FakeClient)
    dq = DataQuery(_cfg())
    name, args, kwargs = getattr(dq, method)(["x"], include_metadata=True)
    getattr(dq, method)(["y"])
    assert name == method and args == (["x"],) and kwargs["include_metadata"] is True
    assert len(created) == 1


def test_stats_and_pool_stats() -> None:
    dq = DataQuery(_cfg())
    assert dq.get_pool_stats() == {"error": "Client not connected"}
    assert dq.get_stats() == {"status": "not_connected"}
    assert dq.get_rate_limit_info() == {"error": "Client not connected"}
    dq, client = _facade()
    client.get_pool_stats.return_value = {"active": 1}
    client.get_stats.return_value = {"requests": 2}
    assert dq.get_pool_stats() == {"active": 1}
    stats = dq.get_stats()
    assert stats["requests"] == 2
    assert stats["api_config"]["base_url"] == "https://api.example.com"


def test_rate_limit_info_reports_config_and_state() -> None:
    dq = DataQuery(_cfg())
    dq._client = _rate_client(requests_per_minute=120)
    info = dq.get_rate_limit_info()
    assert info["configuration"]["requests_per_minute"] == 120
    assert info["current_state"]["available_tokens"] == 3.0 and info["current_state"]["queue_size"] == 0
    dq._client = typing.cast(Any, types.SimpleNamespace())
    assert dq.get_rate_limit_info()["error"].startswith("Failed to get rate limit info")


# ---------------------------------------------------------------------------
# download_file_async / run_download_async
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_file_passes_options_through() -> None:
    dq, client = _facade()
    client.download_file_async = AsyncMock(return_value="res")
    opts = DownloadOptions(destination_path=Path("/tmp/a"))
    cb = MagicMock()
    assert await dq.download_file_async("F", "20240101", options=opts, num_parts=3, progress_callback=cb) == "res"
    client.download_file_async.assert_awaited_once_with("F", "20240101", opts, 3, cb)
    await dq.download_file_async("F")
    assert _last_call(client.download_file_async).args == ("F", None, None, 1, None)


@pytest.mark.asyncio
async def test_download_file_builds_options_from_destination() -> None:
    dq, client = _facade()
    client.download_file_async = AsyncMock()
    await dq.download_file_async("F", destination_path=Path("/tmp/d"))
    opts = _last_call(client.download_file_async).args[2]
    assert opts.destination_path == Path("/tmp/d")
    assert (opts.create_directories, opts.chunk_size, opts.max_retries, opts.timeout) == (True, 8192, 3, 600.0)
    assert opts.enable_range_requests is True and opts.show_progress is True


@pytest.mark.asyncio
async def test_download_file_respects_configured_overwrite() -> None:
    dq, client = _facade()
    client.download_file_async = AsyncMock()
    await dq.download_file_async("F", destination_path=Path("/tmp/d"))
    assert _last_call(client.download_file_async).args[2].overwrite_existing is False


@pytest.mark.asyncio
async def test_download_file_applies_destination_onto_options() -> None:
    dq, client = _facade()
    client.download_file_async = AsyncMock()
    opts = DownloadOptions(destination_path=Path("/tmp/other"), chunk_size=1234)
    await dq.download_file_async("F", destination_path=Path("/tmp/d"), options=opts)
    sent = _last_call(client.download_file_async).args[2]
    assert sent.destination_path == Path("/tmp/d") and sent.chunk_size == 1234
    assert opts.destination_path == Path("/tmp/other")


@pytest.mark.asyncio
async def test_run_download_reports_success_and_uses_configured_overwrite() -> None:
    dq, client = _facade(overwrite_existing=True)
    result = _result(
        file_group_id="F",
        status=DownloadStatus.COMPLETED,
        local_path=Path("/tmp/d/f.csv"),
        file_size=10,
        download_time=2.0,
    )
    client.download_file_async = AsyncMock(return_value=result)
    report = await dq.run_download_async("F", "20240101", Path("/tmp/d"))
    assert report.operation == "download" and report.status == "success"
    assert report.details == {"download_successful": True, "local_path": "/tmp/d/f.csv", "file_size": 10}
    assert report.subject == {"file_group_id": "F", "file_datetime": "20240101"}
    assert _last_call(client.download_file_async).args[2].overwrite_existing is True


@pytest.mark.asyncio
async def test_run_download_reports_error_and_reraises() -> None:
    dq, client = _facade()
    client.download_file_async = AsyncMock(
        return_value=_result(file_group_id="F", status=DownloadStatus.FAILED, error_message="boom")
    )
    report = await dq.run_download_async("F")
    assert report.status == "error" and report.error == "boom"
    assert _last_call(client.download_file_async).args[2] is None
    client.download_file_async = AsyncMock(side_effect=ValueError("x"))
    with pytest.raises(ValueError):
        await dq.run_download_async("F")


# ---------------------------------------------------------------------------
# run_* reports
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_groups_and_files_reraise() -> None:
    dq, client = _facade()
    client.list_groups_async = AsyncMock(side_effect=RuntimeError("down"))
    with pytest.raises(RuntimeError):
        await dq.run_groups_async()
    client.list_all_files_async = AsyncMock(side_effect=RuntimeError("down"))
    with pytest.raises(RuntimeError):
        await dq.run_group_files_async("G")


@pytest.mark.asyncio
async def test_run_group_files_collects_string_file_types() -> None:
    from dataquery.types.models import FileInfo

    dq, client = _facade()
    client.list_all_files_async = AsyncMock(
        return_value=[
            FileInfo.model_validate({"file-group-id": "a", "file-type": "csv"}),
            FileInfo.model_validate({"file-group-id": "b", "file-type": ["json", "csv"]}),
        ]
    )
    report = await dq.run_group_files_async("G")
    assert report.details["file_types"] == ["csv", "json"]


@pytest.mark.asyncio
async def test_run_availability_reports_and_reraises() -> None:
    dq, client = _facade()
    info = AvailabilityInfo.model_validate({"file-datetime": "20240101", "is-available": True, "file-name": "f.csv"})
    client.check_availability_async = AsyncMock(return_value=info)
    report = await dq.run_availability_async("F", "20240101")
    assert report.status == "success" and report.details["is_available"] is True
    client.check_availability_async = AsyncMock(side_effect=RuntimeError("x"))
    with pytest.raises(RuntimeError):
        await dq.run_availability_async("F", "20240101")


def _avail(fg: str, dt: str = "20240101", available: bool = True, dashed: bool = True) -> dict[str, Any]:
    if dashed:
        return {"file-group-id": fg, "file-datetime": dt, "is-available": available}
    return {"file_group_id": fg, "file_datetime": dt, "is_available": available}


@pytest.fixture
def batch(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the parallel batch downloader; returns a dict holding the call kwargs and the canned result."""
    state: dict[str, Any] = {"result": ([], [], 0)}

    async def fake(**kwargs: Any) -> Any:
        state["kwargs"] = kwargs
        return state["result"]

    import dataquery.download.parallel as parallel

    monkeypatch.setattr(parallel, "download_files_with_retry", fake)
    return state


@pytest.mark.asyncio
async def test_run_group_download_no_files(tmp_path: Path, batch: dict[str, Any]) -> None:
    dq, client = _facade()
    client.list_available_files_async = AsyncMock(return_value=[_avail("F", available=False)])
    report = await dq.run_group_download_async("G", "20240101", "20240131", destination_dir=tmp_path)
    assert report.status == "error" and report.error == NO_FILES_FOUND_ERROR
    assert report.counts == {"total_files": 0, "successful_downloads": 0, "failed_downloads": 0}
    assert "kwargs" not in batch and not (tmp_path / "G").exists()


@pytest.mark.asyncio
async def test_run_group_download_success_report(tmp_path: Path, batch: dict[str, Any]) -> None:
    dq, client = _facade()
    files = [_avail("A"), _avail("B", dashed=False)]
    client.list_available_files_async = AsyncMock(return_value=files)
    ok = [
        _result(file_group_id="A", status=DownloadStatus.COMPLETED, download_time=2.0, file_size=2_621_440),
        _result(file_group_id="B", status=DownloadStatus.COMPLETED, download_time=4.0),
    ]
    batch["result"] = (ok, [], 0)
    report = await dq.run_group_download_async("G", "20240101", "20240131", destination_dir=str(tmp_path), num_parts=2)
    assert (tmp_path / "G").is_dir()
    assert batch["kwargs"]["destination_dir"] == tmp_path / "G"
    assert batch["kwargs"]["num_parts"] == 2 and batch["kwargs"]["max_retries"] == 3
    assert report.status == "success" and report.counts["successful_downloads"] == 2
    timing = report.timing["per_file_timing"]
    assert timing["average_file_time_seconds"] == 3.0
    assert (timing["min_file_time_seconds"], timing["max_file_time_seconds"]) == (2.0, 4.0)
    assert timing["file_times"][0] == {
        "file_group_id": "A",
        "download_time_seconds": 2.0,
        "file_size_bytes": 2_621_440,
        "speed_mbps": 1.25,
    }
    assert report.details["downloaded_files"] == ["A", "B"] and report.details["success_rate"] == 100.0
    assert report.data == [{"file_group_id": "A"}, {"file_group_id": "B"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("n_ok, expected", [(1, "partial"), (0, "error")])
async def test_run_group_download_partial_and_error(
    tmp_path: Path, batch: dict[str, Any], n_ok: int, expected: str
) -> None:
    dq, client = _facade()
    client.list_available_files_async = AsyncMock(return_value=[_avail("A"), _avail("B")])
    ok = [_result(file_group_id="A", status=DownloadStatus.COMPLETED)][:n_ok]
    batch["result"] = (ok, [{"file-group-id": "B", "file-datetime": "20240101", "error": "x"}], 2)
    report = await dq.run_group_download_async("G", "20240101", "20240131", destination_dir=tmp_path)
    assert report.status == expected
    assert report.details["failures"] == [{"file_group_id": "B", "file_datetime": "20240101", "error": "x"}]
    assert report.counts["retries_attempted"] == 2


@pytest.mark.asyncio
async def test_run_group_download_merges_and_dedupes_file_group_ids(tmp_path: Path, batch: dict[str, Any]) -> None:
    dq, client = _facade()

    async def listing(group_id: str, file_group_id: str = "", start_date: Any = None, end_date: Any = None) -> Any:
        return {"A": [_avail("A"), _avail("S")], "B": [_avail("S"), _avail("B")]}[file_group_id]

    client.list_available_files_async = AsyncMock(side_effect=listing)
    await dq.run_group_download_async(
        "G", "20240101", "20240131", destination_dir=tmp_path, file_group_id=["A", "", "B"]
    )
    ids = [f["file-group-id"] for f in batch["kwargs"]["files"]]
    assert ids == ["A", "S", "B"]


@pytest.mark.asyncio
async def test_run_group_download_empty_id_list_lists_whole_group(tmp_path: Path, batch: dict[str, Any]) -> None:
    dq, client = _facade()
    client.list_available_files_async = AsyncMock(return_value=[])
    await dq.run_group_download_async(
        "G", "20240101", "20240131", destination_dir=tmp_path, file_group_id=typing.cast(Any, ["", None])
    )
    client.list_available_files_async.assert_awaited_once_with("G", None, "20240101", "20240131")


@pytest.mark.asyncio
async def test_run_group_download_listing_error_propagates(tmp_path: Path, batch: dict[str, Any]) -> None:
    dq, client = _facade()

    async def listing(group_id: str, file_group_id: Any = None, start_date: Any = None, end_date: Any = None) -> Any:
        if file_group_id == "B":
            raise ValueError("bad id")
        await asyncio.sleep(0)
        return [_avail("A")]

    client.list_available_files_async = AsyncMock(side_effect=listing)
    with pytest.raises(ValueError, match="bad id"):
        await dq.run_group_download_async(
            "G", "20240101", "20240131", destination_dir=tmp_path, file_group_id=["A", "B"]
        )


@pytest.mark.asyncio
async def test_run_group_download_malformed_listing_counts_as_no_files(tmp_path: Path, batch: dict[str, Any]) -> None:
    dq, client = _facade()
    client.list_available_files_async = AsyncMock(return_value=["not-a-dict"])
    report = await dq.run_group_download_async("G", "20240101", "20240131", destination_dir=tmp_path)
    assert report.error == NO_FILES_FOUND_ERROR


# ---------------------------------------------------------------------------
# Historical download
# ---------------------------------------------------------------------------


def test_split_into_monthly_ranges() -> None:
    split = DataQuery._split_into_monthly_ranges
    assert split("20231215", "20240302") == [
        ("20231215", "20231231"),
        ("20240101", "20240131"),
        ("20240201", "20240229"),
        ("20240301", "20240302"),
    ]
    assert split("20240110", "20240110") == [("20240110", "20240110")]
    assert split("20240301", "20240101") == []
    with pytest.raises(ValueError):
        split("2024-01-01", "20240101")


@pytest.mark.asyncio
async def test_historical_aggregates_chunks_and_sleeps_between(monkeypatch: pytest.MonkeyPatch) -> None:
    dq, _ = _facade()
    reports = [_group_report(2, 2, 0), _group_report(3, 3, 0)]
    run = AsyncMock(side_effect=reports)
    monkeypatch.setattr(dq, "run_group_download_async", run)
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    monkeypatch.setattr(dq_module.asyncio, "sleep", fake_sleep)
    report = await dq.download_historical_async("G", "20240115", "20240210", chunk_delay=1.5, max_retries=4)
    assert sleeps == [1.5]
    assert report.status == "success" and report.operation == "historical_download"
    assert report.counts == {"monthly_chunks": 2, "total_files": 5, "successful_downloads": 5, "failed_downloads": 0}
    assert report.details == {"success_rate": 100.0, "chunks_with_errors": []}
    assert [c["start_date"] for c in report.data] == ["20240115", "20240201"]
    assert run.await_args_list[0].kwargs["max_retries"] == 4
    assert run.await_args_list[1].kwargs["start_date"] == "20240201"


@pytest.mark.asyncio
async def test_historical_partial_when_a_chunk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    dq, _ = _facade()
    monkeypatch.setattr(
        dq, "run_group_download_async", AsyncMock(side_effect=[_group_report(1, 1, 0), RuntimeError("x")])
    )
    report = await dq.download_historical_async("G", "20240101", "20240229", chunk_delay=0)
    assert report.status == "partial"
    assert report.details["chunks_with_errors"] == ["20240201-20240229"]
    assert report.data[1]["error"] == "x"


@pytest.mark.asyncio
async def test_historical_error_when_nothing_succeeded(monkeypatch: pytest.MonkeyPatch) -> None:
    dq, _ = _facade()
    monkeypatch.setattr(dq, "run_group_download_async", AsyncMock(return_value=_group_report(2, 0, 2, "error")))
    report = await dq.download_historical_async("G", "20240101", "20240131", chunk_delay=0)
    assert report.status == "error" and report.counts["failed_downloads"] == 2


@pytest.mark.asyncio
async def test_historical_without_files_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    dq, _ = _facade()
    empty = _group_report(0, 0, 0, "error", NO_FILES_FOUND_ERROR)
    monkeypatch.setattr(dq, "run_group_download_async", AsyncMock(return_value=empty))
    report = await dq.download_historical_async("G", "20240101", "20240229", chunk_delay=0)
    assert report.status == "error" and report.error == NO_FILES_FOUND_ERROR


@pytest.mark.asyncio
async def test_historical_reversed_range_is_an_error() -> None:
    dq, _ = _facade()
    report = await dq.download_historical_async("G", "20240301", "20240101", chunk_delay=0)
    assert report.counts["monthly_chunks"] == 0
    assert report.status == "error" and report.error == NO_FILES_FOUND_ERROR


# ---------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------

DEFAULT_CAPACITY = {"requests_per_minute": 1000, "burst_capacity": 100, "safe_interval": 0.1}


def test_rate_limit_capacity_variants() -> None:
    dq = DataQuery(_cfg())
    assert dq._calculate_rate_limit_capacity() == DEFAULT_CAPACITY
    dq._client = _rate_client(enable_rate_limiting=False)
    assert dq._calculate_rate_limit_capacity() == DEFAULT_CAPACITY
    dq._client = _rate_client(requests_per_minute=120, burst_capacity=7, enable_queuing=False)
    assert dq._calculate_rate_limit_capacity() == {
        "requests_per_minute": 120,
        "burst_capacity": 7,
        "safe_interval": 0.5,
        "queuing_enabled": False,
    }
    zero_rpm = _rate_client()
    zero_rpm.rate_limiter.config.requests_per_minute = 0
    dq._client = zero_rpm
    assert dq._calculate_rate_limit_capacity() == {
        "requests_per_minute": 100,
        "burst_capacity": 10,
        "safe_interval": 0.6,
    }


CAP = {"requests_per_minute": 300, "burst_capacity": 5, "safe_interval": 0.2}


@pytest.mark.parametrize("total, base, expected", [(5, 0.2, 0.2), (3, 0.05, 0.2), (2, 1.0, 1.0), (1000, 0.2, 0.2)])
def test_intelligent_delay_values_that_hold(total: int, base: float, expected: float) -> None:
    dq = DataQuery(_cfg())
    assert dq._calculate_intelligent_delay(total, CAP, base) == pytest.approx(expected)


def test_intelligent_delay_bad_capacity_returns_base() -> None:
    dq = DataQuery(_cfg())
    assert dq._calculate_intelligent_delay(10, {}, 0.7) == 0.7


def test_intelligent_delay_is_monotonic_in_overflow() -> None:
    dq = DataQuery(_cfg())
    delays = [dq._calculate_intelligent_delay(n, CAP, 0.2) for n in (5, 6, 10, 30, 105)]
    assert delays == sorted(delays)
    assert delays[1] == pytest.approx(0.2)


def test_safe_concurrency_limit_variants() -> None:
    dq = DataQuery(_cfg())
    assert dq._calculate_safe_concurrency_limit(40) == 40
    dq._client = _rate_client(enable_rate_limiting=False)
    assert dq._calculate_safe_concurrency_limit(40) == 40
    dq._client = _rate_client(requests_per_minute=300, burst_capacity=50, max_queue_size=20)
    assert dq._calculate_safe_concurrency_limit(40) == 10
    dq._client = _rate_client(requests_per_minute=300, burst_capacity=50, enable_queuing=False)
    assert dq._calculate_safe_concurrency_limit(40) == 40
    dq._client = typing.cast(Any, types.SimpleNamespace())
    assert dq._calculate_safe_concurrency_limit(40) == 10
    assert dq._calculate_safe_concurrency_limit(2) == 1


def test_rate_limit_recommendations() -> None:
    dq = DataQuery(_cfg())
    assert dq._get_rate_limit_recommendations(10) == {}
    dq._client = _rate_client(requests_per_minute=100, burst_capacity=5, enable_queuing=False)
    recs = dq._get_rate_limit_recommendations(10)
    assert [r["type"] for r in recs["recommendations"]] == [
        "increase_burst_capacity",
        "increase_requests_per_minute",
        "enable_queuing",
    ]
    assert recs["recommendations"][0]["recommended"] == 10 and recs["recommendations"][1]["recommended"] == 150
    dq._client = _rate_client(requests_per_minute=1000, burst_capacity=50)
    assert dq._get_rate_limit_recommendations(10)["recommendations"] == []
    dq._client = typing.cast(Any, types.SimpleNamespace())
    assert dq._get_rate_limit_recommendations(10) == {}


def test_optimize_concurrency_without_reduction() -> None:
    dq = DataQuery(_cfg())
    result = dq.optimize_concurrency_for_rate_limits(4, 3)
    assert result["optimized"] == {"max_concurrent": 4, "num_parts": 3, "total_concurrency": 12}
    assert result["rate_limit_applied"] is False and result["safe_limit"] == 12 and result["recommendations"] == {}


@pytest.mark.parametrize("mc, parts, expected", [(10, 1, (5, 1)), (1, 10, (1, 5))])
def test_optimize_concurrency_single_axis(mc: int, parts: int, expected: tuple[int, int]) -> None:
    dq = DataQuery(_cfg())
    dq._client = _rate_client()
    result = dq.optimize_concurrency_for_rate_limits(mc, parts)
    assert (result["optimized"]["max_concurrent"], result["optimized"]["num_parts"]) == expected
    assert result["rate_limit_applied"] is True and result["safe_limit"] == 5


@pytest.mark.parametrize("mc, parts, expected", [(6, 6, (5, 1)), (3, 3, (2, 2))])
def test_optimize_concurrency_uses_the_safe_budget(mc: int, parts: int, expected: tuple[int, int]) -> None:
    dq = DataQuery(_cfg())
    dq._client = _rate_client()
    result = dq.optimize_concurrency_for_rate_limits(mc, parts)
    assert (result["optimized"]["max_concurrent"], result["optimized"]["num_parts"]) == expected
    assert result["optimized"]["total_concurrency"] <= result["safe_limit"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seconds, text", [(0.44, "0.4s"), (59.96, "60.0s"), (60, "1m 0s"), (125.7, "2m 5s")])
def test_format_duration(seconds: float, text: str) -> None:
    assert _format_duration(seconds) == text


def test_progress_tracker_logs_at_most_once_per_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(dq_module, "logger", types.SimpleNamespace(info=lambda msg, **kw: logged.append(kw)))
    now = [1000.0]
    monkeypatch.setattr(dq_module.time, "time", lambda: now[0])
    cb = DataQuery(_cfg()).create_progress_callback(log_interval=10)
    progress = types.SimpleNamespace(completed_files=1, total_files=4, percentage=25.0, current_file="a")
    cb(progress)
    now[0] += 5
    cb(progress)
    now[0] += 5
    cb(object())
    assert logged == [
        {"completed": 1, "total": 4, "percentage": "25.0%", "current_file": "a"},
        {"completed": 0, "total": 0, "percentage": "0.0%", "current_file": "unknown"},
    ]
    assert ProgressTracker().log_interval == 10


def test_unreadable_environment_falls_back_to_client_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAQUERY_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("DATAQUERY_TIMEOUT", "not-a-number")
    assert ConfigManager().get_client_config() == ClientConfig()
