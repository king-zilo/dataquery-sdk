"""Byte accounting in dataquery.download.utils."""

from types import SimpleNamespace

import pytest

from dataquery.download.utils import create_progress_wrapper, download_and_track
from dataquery.types.models import DownloadOptions, DownloadProgress, DownloadStatus


def _stats():
    return {"total_bytes_downloaded": 0, "files_downloaded": 0, "files_skipped": 0, "download_failures": 0}


class _FakeClient:
    def __init__(self, size, report_progress=True, file_size=None):
        self.size = size
        self.report_progress = report_progress
        self.file_size = size if file_size is None else file_size

    async def download_file_async(self, file_group_id, file_datetime, options, progress_callback):
        if self.report_progress:
            progress = DownloadProgress(file_group_id=file_group_id, total_bytes=self.size)
            for chunk in (self.size // 2, self.size):
                progress.update_progress(chunk)
                progress_callback(progress)
        return SimpleNamespace(status=DownloadStatus.COMPLETED, file_size=self.file_size)


@pytest.mark.asyncio
@pytest.mark.parametrize("report_progress", [True, False])
async def test_download_and_track_counts_each_file_once(report_progress):
    stats = _stats()
    for i in range(2):
        await download_and_track(
            _FakeClient(100, report_progress), f"F{i}", "20240101", f"k{i}", DownloadOptions(), stats, set(), {}
        )
    assert stats["total_bytes_downloaded"] == 200
    assert stats["files_downloaded"] == 2


@pytest.mark.asyncio
async def test_download_and_track_does_not_subtract_when_file_size_smaller_than_progress():
    stats = _stats()
    await download_and_track(_FakeClient(100, file_size=40), "F", "20240101", "k", DownloadOptions(), stats, set(), {})
    assert stats["total_bytes_downloaded"] == 100


def test_progress_wrappers_accumulate_concurrent_files():
    stats = _stats()
    first, second = create_progress_wrapper(stats), create_progress_wrapper(stats)
    p1 = DownloadProgress(file_group_id="A", total_bytes=100)
    p2 = DownloadProgress(file_group_id="B", total_bytes=50)
    p1.update_progress(30)
    first(p1)
    p2.update_progress(50)
    second(p2)
    p1.update_progress(100)
    first(p1)
    first(p1)
    assert stats["total_bytes_downloaded"] == 150
