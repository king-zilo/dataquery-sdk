"""Shared download utilities for the SSE-driven NotificationDownloadManager."""

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from ..types.models import DownloadOptions, DownloadProgress, DownloadStatus


class _SupportsAddContains(Protocol):
    """Set-like protocol used by :func:`download_and_track`."""

    def add(self, key: str) -> None: ...
    def __contains__(self, key: object) -> bool: ...


class _SupportsRetryCounter(Protocol):
    """Dict-like protocol for the per-file retry counter."""

    def get(self, key: str, default: int = ...) -> int: ...
    def pop(self, key: str, default: Any = ...) -> Any: ...
    def __setitem__(self, key: str, value: int) -> None: ...


logger = logging.getLogger(__name__)


def file_exists_locally(destination_dir: Path, file_group_id: str, date_str: str) -> bool:
    """Check if a file already exists in the destination directory (heuristic match)."""
    for file_path in destination_dir.glob("*"):
        if file_path.is_file() and file_group_id in file_path.name and date_str in file_path.name:
            return True
    return False


class _ProgressStatsWrapper:
    """Progress callback that adds each file's newly received bytes to ``stats`` once."""

    def __init__(
        self,
        stats: Dict[str, Any],
        user_callback: Optional[Callable],
        logger_instance: logging.Logger,
    ) -> None:
        self._stats = stats
        self._user_callback = user_callback
        self._logger = logger_instance
        self.bytes_counted = 0

    def __call__(self, progress: DownloadProgress) -> None:
        downloaded = int(getattr(progress, "bytes_downloaded", 0) or 0)
        if downloaded > self.bytes_counted:
            current_total = int(self._stats.get("total_bytes_downloaded", 0) or 0)
            self._stats["total_bytes_downloaded"] = current_total + downloaded - self.bytes_counted
            self.bytes_counted = downloaded
        if self._user_callback:
            try:
                if inspect.iscoroutinefunction(self._user_callback):
                    asyncio.create_task(self._user_callback(progress))
                else:
                    self._user_callback(progress)
            except Exception as e:
                self._logger.error("Error in progress callback: %s", e)


def create_progress_wrapper(
    stats: Dict[str, Any],
    user_callback: Optional[Callable] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> _ProgressStatsWrapper:
    """Create a progress callback that updates stats and optionally delegates to a user callback."""
    return _ProgressStatsWrapper(stats, user_callback, logger_instance or logger)


async def download_and_track(
    client: Any,
    file_group_id: str,
    date_str: str,
    file_key: str,
    download_options: DownloadOptions,
    stats: Dict[str, Any],
    downloaded_files: _SupportsAddContains,
    failed_files: _SupportsRetryCounter,
    progress_callback: Optional[Callable] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> None:
    """Download a file, updating stats and tracking sets on success/failure."""
    _logger = logger_instance or logger
    _logger.info("Downloading '%s' for %s", file_group_id, date_str)

    wrapper = create_progress_wrapper(stats, progress_callback, _logger)

    try:
        result = await client.download_file_async(
            file_group_id=file_group_id,
            file_datetime=date_str,
            options=download_options,
            progress_callback=wrapper,
        )

        succeeded = False
        already_exists = False
        if result is not None and getattr(result, "status", None) is not None:
            succeeded = result.status == DownloadStatus.COMPLETED
            already_exists = result.status == DownloadStatus.ALREADY_EXISTS

        if already_exists:
            _logger.info("File already exists: '%s' for %s — skipping", file_group_id, date_str)
            stats["files_skipped"] += 1
            downloaded_files.add(file_key)
            failed_files.pop(file_key, None)

        elif succeeded:
            _logger.info("Downloaded '%s' for %s", file_group_id, date_str)
            stats["files_downloaded"] += 1
            downloaded_files.add(file_key)
            failed_files.pop(file_key, None)
            file_size = getattr(result, "file_size", None)
            if file_size:
                try:
                    remaining = int(file_size) - wrapper.bytes_counted
                except (TypeError, ValueError):
                    remaining = 0
                if remaining > 0:
                    stats["total_bytes_downloaded"] += remaining

        else:
            error_msg = getattr(result, "error_message", "Unknown") if result else "No result"
            _logger.error("Download failed for '%s' for %s: %s", file_group_id, date_str, error_msg)
            failed_files[file_key] = failed_files.get(file_key, 0) + 1
            stats["download_failures"] += 1

    except Exception as exc:
        _logger.error("Exception downloading '%s' for %s: %s", file_group_id, date_str, exc)
        failed_files[file_key] = failed_files.get(file_key, 0) + 1
        stats["download_failures"] += 1
        raise
