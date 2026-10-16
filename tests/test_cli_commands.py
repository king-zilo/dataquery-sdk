"""Command-level tests for ``dataquery.cli``: every subcommand's output and exit code.

The SDK facade (``cli.DataQuery``) is replaced by an async context-manager
mock, so nothing here touches the network or real credentials.
"""

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dataquery import cli
from dataquery.sse.event_store import Subscription, build_event_id_store
from dataquery.types.exceptions import APIResponseError, AuthenticationError, ConfigurationError
from dataquery.types.models import (
    AvailabilityInfo,
    ClientConfig,
    DownloadResult,
    DownloadStatus,
    FileInfo,
    GridDataResponse,
    Group,
    OperationReport,
    TimeSeriesResponse,
)


def _parse(*argv):
    return cli.create_parser().parse_args(list(argv))


def _fake_dq(**methods):
    """A ``DataQuery`` stand-in: an async context manager exposing ``methods`` as AsyncMocks."""
    fake = MagicMock()
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=None)
    for name, value in methods.items():
        setattr(fake, name, value if isinstance(value, AsyncMock) else AsyncMock(return_value=value))
    return fake


@pytest.fixture
def use_dq(monkeypatch):
    """Install a fake DataQuery built from keyword methods; returns the fake."""

    def install(**methods):
        fake = _fake_dq(**methods)
        monkeypatch.setattr(cli, "DataQuery", MagicMock(return_value=fake))
        return fake

    return install


def _json_section(out):
    return json.loads(out.split("--- JSON ---", 1)[1])


class _Undumpable:
    """An item whose ``model_dump`` fails, so the CLI falls back to ``str()``."""

    def model_dump(self):
        raise TypeError("cannot dump")

    def __str__(self):
        return "<undumpable>"


def _ts(instrument_ids, next_link=None):
    return TimeSeriesResponse(
        **{
            "links": [{"self": "/x", "next": next_link}],
            "instruments": [
                {
                    "item": i,
                    "instrument-id": inst_id,
                    "attributes": [
                        {"attribute-id": "TR", "time-series": [["20240102", 1.0], ["20240101", 2.0]]},
                    ],
                }
                for i, inst_id in enumerate(instrument_ids)
            ],
        }
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def test_to_dict_normalizes_none_dicts_and_scalars():
    assert cli._to_dict(None) == {}
    assert cli._to_dict({"a": 1}) == {"a": 1}
    assert cli._to_dict(7) == {"value": 7}
    assert cli._to_dict(Group(**{"group-id": "G"}))["group-id"] == "G"


def test_count_timeseries_skips_points_that_are_not_date_value_pairs():
    data = {"instruments": [{"attributes": [{"time-series": [["20240105", 1], [], "junk", ["20240101", 2]]}]}]}
    assert cli._count_timeseries(data) == (1, 4, "20240101", "20240105")


def test_print_error_includes_the_suggestion(capsys):
    cli._print_error("boom", suggestion="try again")
    captured = capsys.readouterr()
    assert captured.err.strip() == "boom"
    assert _json_section(captured.out) == {
        "status": "error",
        "error_description": "boom",
        "suggestion": "try again",
    }


def test_next_page_token_falls_back_to_the_raw_link_without_a_page_param():
    resp = _ts(["A"], next_link="/group/time-series?cursor=abc")
    assert cli._next_page_token(resp) == "/group/time-series?cursor=abc"
    assert cli._next_page_token({"links": []}) is None


def test_mcp_extra_installed_reflects_fastmcp_availability():
    with patch.object(cli.importlib.util, "find_spec", return_value=None):
        assert cli._mcp_extra_installed() is False
    with patch.object(cli.importlib.util, "find_spec", return_value=object()):
        assert cli._mcp_extra_installed() is True


# ---------------------------------------------------------------------------
# groups / files / availability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groups_text_lists_id_and_name(use_dq, capsys):
    use_dq(list_groups_async=[Group(**{"group-id": "G1", "group-name": "Rates"})])

    assert await cli.cmd_groups(_parse("groups")) == 0
    assert capsys.readouterr().out.splitlines() == ["G1\tRates"]


@pytest.mark.asyncio
async def test_groups_search_flag_uses_the_search_endpoint(use_dq, capsys):
    fake = use_dq(search_groups_async=[Group(**{"group-id": "FX"})], list_groups_async=[])

    assert await cli.cmd_groups(_parse("groups", "--search", "fx", "--limit", "5")) == 0
    fake.search_groups_async.assert_awaited_once_with("fx", limit=5)
    fake.list_groups_async.assert_not_awaited()
    assert capsys.readouterr().out.startswith("FX\t")


@pytest.mark.asyncio
async def test_groups_falls_back_to_str_for_undumpable_items(use_dq, capsys):
    use_dq(list_groups_async=[_Undumpable()])

    await cli.cmd_groups(_parse("groups"))
    assert capsys.readouterr().out.strip() == "<undumpable>"

    await cli.cmd_groups(_parse("groups", "--json"))
    assert json.loads(capsys.readouterr().out) == ["<undumpable>"]


@pytest.mark.asyncio
async def test_files_json_and_undumpable_fallback(use_dq, capsys):
    use_dq(list_files_async=[FileInfo(**{"file-group-id": "FG1", "file-type": ["csv"]}), _Undumpable()])

    assert await cli.cmd_files(_parse("files", "--group-id", "G", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["file_group_id"] == "FG1"
    assert payload[1] == "<undumpable>"

    assert await cli.cmd_files(_parse("files", "--group-id", "G")) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["Found 2 files", "FG1\t['csv']", "<undumpable>"]


@pytest.mark.asyncio
@pytest.mark.parametrize("available, state", [(True, "available"), (False, "not available")])
async def test_availability_text_reports_state(use_dq, capsys, available, state):
    use_dq(
        check_availability_async=AvailabilityInfo(
            **{"file-group-id": "FG", "file-datetime": "20240101", "is-available": available}
        )
    )

    assert (
        await cli.cmd_availability(_parse("availability", "--file-group-id", "FG", "--file-datetime", "20240101")) == 0
    )
    assert capsys.readouterr().out.strip() == f"FG @ 20240101: {state}"


@pytest.mark.asyncio
async def test_availability_json_falls_back_to_str(use_dq, capsys):
    use_dq(check_availability_async=_Undumpable())

    args = _parse("availability", "--file-group-id", "FG", "--file-datetime", "20240101", "--json")
    assert await cli.cmd_availability(args) == 0
    assert capsys.readouterr().out.strip() == "<undumpable>"


# ---------------------------------------------------------------------------
# download / download-group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_download_requires_a_file_group_id(use_dq, capsys):
    fake = use_dq(download_file_async=None)

    assert await cli.cmd_download(_parse("download", "--file-datetime", "20240101")) == 1
    assert "--file-group-id is required" in capsys.readouterr().out
    fake.download_file_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_download_rejects_several_file_group_ids_outside_watch_mode(use_dq, capsys):
    fake = use_dq(download_file_async=None)

    rc = await cli.cmd_download(_parse("download", "--file-group-id", "A", "B", "--file-datetime", "20240101"))
    assert rc == 1
    assert "must be a single value" in capsys.readouterr().out
    fake.download_file_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, message",
    [(DownloadStatus.COMPLETED, "Downloaded to"), (DownloadStatus.ALREADY_EXISTS, "Already exists at")],
)
async def test_download_success_messages(use_dq, tmp_path, capsys, status, message):
    target = tmp_path / "FG_20240101.csv"
    fake = use_dq(download_file_async=DownloadResult(file_group_id="FG", local_path=target, status=status))

    args = _parse("download", "--file-group-id", "FG", "--file-datetime", "20240101", "--destination", str(tmp_path))
    assert await cli.cmd_download(args) == 0
    assert capsys.readouterr().out.strip() == f"{message} {target}"
    options = fake.download_file_async.await_args.kwargs["options"]
    assert options.destination_path == tmp_path
    assert options.chunk_size == 1048576
    assert fake.download_file_async.await_args.kwargs["num_parts"] == 5


class _Manager:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.stopped = False

    async def stop(self):
        self.stopped = True
        if self.stop_error:
            raise self.stop_error

    def get_stats(self):
        return {"files_downloaded": 2}


@pytest.fixture
def interrupted_sleep(monkeypatch):
    """Make the watch loop's first ``asyncio.sleep`` behave like Ctrl+C."""

    async def interrupt(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.asyncio, "sleep", interrupt)


@pytest.mark.asyncio
async def test_download_watch_reset_event_id_clears_the_persisted_id(use_dq, tmp_path, capsys, interrupted_sleep):
    config = ClientConfig(client_id="c", client_secret="s", token_storage_enabled=True, token_storage_dir=str(tmp_path))
    store = build_event_id_store(config, Subscription.from_user("G", ["FG2", "FG1"]))
    await store.save("42")
    assert store.load() == "42"

    mgr = _Manager()
    fake = use_dq(auto_download_async=mgr)
    fake._ensure_client = MagicMock(return_value=SimpleNamespace(config=config))

    args = _parse(
        "download",
        "--watch",
        "--group-id",
        "G",
        "--file-group-id",
        "FG1",
        "FG2",
        "--reset-event-id",
        "--no-event-replay",
    )
    assert await cli.cmd_download(args) == 0

    assert not store.file_path.exists()
    assert mgr.stopped
    fake.auto_download_async.assert_awaited_once_with(
        group_id="G", destination_dir="./downloads", file_group_id=["FG1", "FG2"], enable_event_replay=False
    )
    assert json.loads(capsys.readouterr().out) == {"files_downloaded": 2}


@pytest.mark.asyncio
async def test_download_watch_reset_event_id_without_a_state_dir(use_dq, interrupted_sleep):
    """No token store and no download dir: there is nothing to reset, and the watch still starts."""
    config = ClientConfig(client_id="c", client_secret="s", token_storage_enabled=False, download_dir="")
    fake = use_dq(auto_download_async=_Manager())
    fake._ensure_client = MagicMock(return_value=SimpleNamespace(config=config))

    args = _parse("download", "--watch", "--group-id", "G", "--reset-event-id", "--destination", "out")
    assert await cli.cmd_download(args) == 0
    assert fake.auto_download_async.await_args.kwargs["destination_dir"] == "out"


@pytest.mark.asyncio
async def test_download_watch_reports_a_failed_stop_but_still_exits_cleanly(use_dq, capsys, interrupted_sleep):
    mgr = _Manager(stop_error=RuntimeError("socket closed"))
    use_dq(auto_download_async=mgr)

    assert await cli.cmd_download(_parse("download", "--watch", "--group-id", "G")) == 0
    captured = capsys.readouterr()
    assert "Failed to stop notification manager: socket closed" in captured.err
    assert json.loads(captured.out) == {"files_downloaded": 2}


@pytest.mark.asyncio
async def test_download_group_lists_each_failure(use_dq, capsys):
    report = OperationReport(
        operation="group_download",
        status="partial",
        counts={"successful_downloads": 1, "failed_downloads": 1},
        details={"failures": [{"file_group_id": "FG", "file_datetime": "20240102", "error": "404"}]},
    )
    use_dq(run_group_download_async=report)

    args = _parse("download-group", "--group-id", "G", "--start-date", "20240101", "--end-date", "20240102")
    assert await cli.cmd_download_group(args) == 1
    assert capsys.readouterr().out.splitlines() == [
        "Downloaded 1 files to ./downloads",
        "Failed: 1",
        "  FG @ 20240102: 404",
    ]


@pytest.mark.asyncio
async def test_download_group_all_successful_prints_no_failure_lines(use_dq, capsys):
    report = OperationReport(operation="group_download", status="success", counts={"successful_downloads": 3})
    use_dq(run_group_download_async=report)

    args = _parse("download-group", "--group-id", "G", "--start-date", "20240101", "--end-date", "20240102")
    assert await cli.cmd_download_group(args) == 0
    assert capsys.readouterr().out.splitlines() == ["Downloaded 3 files to ./downloads"]


# ---------------------------------------------------------------------------
# search and catalog endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_json_prints_the_raw_result(use_dq, capsys):
    use_dq(search_async={"results": [{"id": 1}]})

    assert await cli.cmd_search(_parse("search", "--query", "rates", "--json")) == 0
    assert json.loads(capsys.readouterr().out) == {"results": [{"id": 1}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result, count",
    [
        ({"results": [1, 2]}, 2),
        ({"results": "n/a", "groups": [1, 2, 3]}, 3),
        ({"items": [1]}, 1),
        ({"message": "nothing"}, 0),
    ],
)
async def test_search_counts_the_first_list_in_a_dict_response(use_dq, capsys, result, count):
    use_dq(search_async=result)

    assert await cli.cmd_search(_parse("search", "--query", "q")) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"Search returned {count} result(s) for: q")
    assert _json_section(out)["data"] == result


@pytest.mark.asyncio
async def test_groups_search_summarizes_matches(use_dq, capsys):
    fake = use_dq(search_groups_async=[Group(**{"group-id": "FX"}), {"group-id": "FX2"}])

    assert await cli.cmd_groups_search(_parse("groups-search", "--keywords", "fx", "--page", "p2")) == 0
    fake.search_groups_async.assert_awaited_once_with("fx", page="p2")
    out = capsys.readouterr().out
    assert out.startswith("Found 2 dataset(s) matching 'fx'")
    assert [g["group-id"] for g in _json_section(out)["data"]["groups"]] == ["FX", "FX2"]


@pytest.mark.asyncio
async def test_instruments_reports_count_and_next_page(use_dq, capsys):
    from dataquery.types.models import InstrumentsResponse

    resp = InstrumentsResponse(
        **{
            "links": [{"next": "/group/instruments?group-id=G&page=n2"}],
            "instruments": [
                {"item": 1, "instrument-id": "I1", "instrument-name": "One"},
                {"item": 2, "instrument-id": "I2", "instrument-name": "Two"},
            ],
        }
    )
    fake = use_dq(list_instruments_async=resp)

    assert await cli.cmd_instruments(_parse("instruments", "--group-id", "G", "--instrument-id", "I1")) == 0
    fake.list_instruments_async.assert_awaited_once_with("G", "I1", None)
    out = capsys.readouterr().out
    assert out.startswith("Found 2 instrument(s) in G\nMore pages available; fetch the next one with --page n2")


@pytest.mark.asyncio
async def test_instruments_search_reports_matches(use_dq, capsys):
    fake = use_dq(search_instruments_async={"instruments": [{"instrument-id": "I1"}]})

    args = _parse("instruments-search", "--group-id", "G", "--keywords", "usd", "--page", "p")
    assert await cli.cmd_instruments_search(args) == 0
    fake.search_instruments_async.assert_awaited_once_with("G", "usd", "p")
    assert capsys.readouterr().out.startswith("Found 1 instrument(s) in G matching 'usd'\n\n--- JSON ---")


@pytest.mark.asyncio
async def test_filters_reports_dimension_count(use_dq, capsys):
    fake = use_dq(get_group_filters_async={"filters": [{"filter-name": "currency"}, {"filter-name": "country"}]})

    assert await cli.cmd_filters(_parse("filters", "--group-id", "G")) == 0
    fake.get_group_filters_async.assert_awaited_once_with("G", None)
    assert capsys.readouterr().out.startswith("Found 2 filter dimension(s) for G")


@pytest.mark.asyncio
async def test_attributes_reports_instrument_count(use_dq, capsys):
    fake = use_dq(get_group_attributes_async={"instruments": None})

    assert await cli.cmd_attributes(_parse("attributes", "--group-id", "G", "--instrument-id", "I")) == 0
    fake.get_group_attributes_async.assert_awaited_once_with("G", "I", None)
    assert capsys.readouterr().out.startswith("Attributes for 0 instrument(s) in G")


# ---------------------------------------------------------------------------
# time-series commands
# ---------------------------------------------------------------------------

_ALL_TS_FLAGS = (
    "--data",
    "ALL",
    "--start-date",
    "20240101",
    "--end-date",
    "20240131",
    "--calendar",
    "CAL_WEEKDAYS",
    "--frequency",
    "FREQ_WEEK",
    "--conversion",
    "CONV_LASTBUS_REL",
    "--nan-treatment",
    "NA_LAST",
    "--page",
    "p1",
)


@pytest.mark.asyncio
async def test_instrument_timeseries_forwards_every_option(use_dq, capsys):
    fake = use_dq(get_instrument_time_series_async=_ts(["I1"]), get_next_page_async=None)

    args = _parse("instrument-timeseries", "--instruments", "I1", "--attributes", "TR", *_ALL_TS_FLAGS)
    assert await cli.cmd_instrument_timeseries(args) == 0
    fake.get_instrument_time_series_async.assert_awaited_once_with(
        ["I1"],
        ["TR"],
        data="ALL",
        start_date="20240101",
        end_date="20240131",
        calendar="CAL_WEEKDAYS",
        frequency="FREQ_WEEK",
        conversion="CONV_LASTBUS_REL",
        nan_treatment="NA_LAST",
        page="p1",
    )
    fake.get_next_page_async.assert_not_awaited()
    assert capsys.readouterr().out.startswith(
        "Instrument time-series: 1 instrument(s), 2 data point(s) (20240101 to 20240102)"
    )


@pytest.mark.asyncio
async def test_expression_timeseries_csv_to_stdout(use_dq, capsys):
    fake = use_dq(get_expressions_time_series_async=_ts(["E1"]))

    args = _parse("expression-timeseries", "--expressions", "DB(X)", "--output-csv", "-")
    assert await cli.cmd_expression_timeseries(args) == 0
    fake.get_expressions_time_series_async.assert_awaited_once_with(["DB(X)"])
    out = capsys.readouterr().out
    csv_lines = out.split("Expression time-series:", 1)[0].strip().splitlines()
    assert len(csv_lines) == 3  # header + 2 points
    assert "CSV exported: stdout (2 rows)" in out
    envelope = _json_section(out)
    assert envelope["csv_exported"] == "stdout"
    assert envelope["csv_rows"] == 2


@pytest.mark.asyncio
async def test_timeseries_pagination_stops_when_the_next_page_is_empty(use_dq, capsys):
    """``get_next_page_async`` returning None ends the walk; the fetched page is still reported."""
    fake = use_dq(
        get_expressions_time_series_async=_ts(["E1"], next_link="/expressions/time-series?page=t2"),
        get_next_page_async=None,
    )

    assert await cli.cmd_expression_timeseries(_parse("expression-timeseries", "--expressions", "DB(X)")) == 0
    fake.get_next_page_async.assert_awaited_once()
    assert "Expression time-series: 1 instrument(s)" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_instrument_timeseries_marks_an_incomplete_walk(use_dq, capsys):
    use_dq(
        get_instrument_time_series_async=_ts(["I1"], next_link="/instruments/time-series?page=t2"),
        get_next_page_async=AsyncMock(side_effect=APIResponseError("[498] bad token")),
    )

    args = _parse("instrument-timeseries", "--instruments", "I1", "--attributes", "TR")
    assert await cli.cmd_instrument_timeseries(args) == 1
    assert "INCOMPLETE: page 2 failed ([498] bad token); kept 1 page(s), 1 of ? instrument(s)" in (
        capsys.readouterr().out
    )


# ---------------------------------------------------------------------------
# grid-data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grid_data_requires_expr_or_grid_id(use_dq, capsys):
    fake = use_dq(get_grid_data_async=None)

    assert await cli.cmd_grid_data(_parse("grid-data")) == 1
    captured = capsys.readouterr()
    assert "Provide --expr or --grid-id" in captured.err
    assert _json_section(captured.out)["suggestion"].startswith("Example: --expr")
    fake.get_grid_data_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_grid_data_summarizes_series_and_server_errors(use_dq, capsys):
    resp = GridDataResponse(
        **{
            "errorCode": "PARTIAL",
            "errorMessage": "some series failed",
            "series": [
                {"expr": "GOOD", "records": [{"k": 1}, {"k": 2}]},
                {"expr": "BAD", "errorCode": "E1", "errorMessage": "no data"},
            ],
        }
    )
    fake = use_dq(get_grid_data_async=resp)

    assert await cli.cmd_grid_data(_parse("grid-data", "--grid-id", "GRID1", "--date", "20240101")) == 0
    fake.get_grid_data_async.assert_awaited_once_with(expr=None, grid_id="GRID1", date="20240101")
    summary = capsys.readouterr().out.split("\n\n--- JSON ---")[0]
    assert summary.splitlines() == [
        "Grid: 2 series, 2 record(s)",
        "Server error: [PARTIAL] some series failed",
        "Server error for BAD: [E1] no data",
    ]


@pytest.mark.asyncio
async def test_grid_data_exports_csv(use_dq, tmp_path, capsys):
    out_csv = tmp_path / "grid.csv"
    use_dq(get_grid_data_async=GridDataResponse(series=[{"expr": "X", "records": [{"k": 1}, {"k": 2}]}]))

    assert await cli.cmd_grid_data(_parse("grid-data", "--expr", "X", "--output-csv", str(out_csv))) == 0
    assert out_csv.read_text().strip().splitlines() == ["expression,k", "X,1", "X,2"]
    assert f"CSV exported: {out_csv} (2 rows)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# function-help (local registry, no API)
# ---------------------------------------------------------------------------


def test_function_help_needs_an_option(capsys):
    assert cli.cmd_function_help(_parse("function-help")) == 1
    assert "Provide --name NAME, --list, or --category CATEGORY" in capsys.readouterr().out


def test_function_help_text_marks_optional_parameters(capsys):
    assert cli.cmd_function_help(_parse("function-help", "--name", "residual")) == 0
    out = capsys.readouterr().out
    assert out.startswith("RESIDUAL(")
    assert "  Parameters:" in out
    assert "(optional)" in out


def test_function_help_text_without_parameters(capsys):
    assert cli.cmd_function_help(_parse("function-help", "--name", "AGAVG")) == 0
    out = capsys.readouterr().out
    assert out.startswith("AGAVG() [Aggregate]")
    assert "Parameters:" not in out


def test_function_help_text_without_a_description(monkeypatch, capsys):
    from dataquery import function_registry

    spec = {"name": "NODESC", "category": "Misc", "description": "", "params": []}
    monkeypatch.setattr(function_registry, "get_function_registry", lambda: {"NODESC": spec})

    assert cli.cmd_function_help(_parse("function-help", "--name", "nodesc")) == 0
    assert capsys.readouterr().out.splitlines() == ["NODESC() [Misc]"]


def test_function_help_json(capsys):
    assert cli.cmd_function_help(_parse("function-help", "--name", "vol", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["function"] == "VOL"
    assert payload["syntax"] == "VOL(NDays, Expr)"
    assert [p["name"] for p in payload["parameters"]] == ["NDays", "Expr"]


@pytest.mark.parametrize("as_json", [False, True])
def test_function_help_unknown_name(capsys, as_json):
    argv = ["function-help", "--name", "nope"] + (["--json"] if as_json else [])
    assert cli.cmd_function_help(_parse(*argv)) == 1
    out = capsys.readouterr().out
    if as_json:
        assert json.loads(out)["error_description"] == "Unknown function: NOPE"
    else:
        assert out.splitlines() == ["Unknown function: NOPE", "Use --list to see all available functions."]


def test_function_help_lists_a_category(capsys):
    assert cli.cmd_function_help(_parse("function-help", "--category", "aggregate")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"Available functions: {len(lines) - 1}"
    assert len(lines) > 1
    assert all(line.endswith("\t[Aggregate]") for line in lines[1:])


def test_function_help_list_json(capsys):
    assert cli.cmd_function_help(_parse("function-help", "--list", "--json")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == len(payload["functions"]) > 100


def test_function_help_documented_fo_category_matches_nothing(capsys):
    """Finding: ``--help`` suggests 'F&O SPECIFIC', but the registry calls it 'Futures & Options'."""
    with pytest.raises(SystemExit):
        _parse("function-help", "--help")
    assert "F&O SPECIFIC" in capsys.readouterr().out

    assert cli.cmd_function_help(_parse("function-help", "--category", "F&O SPECIFIC")) == 0
    assert capsys.readouterr().out.strip() == "Available functions: 0"

    assert cli.cmd_function_help(_parse("function-help", "--category", "Futures & Options")) == 0
    assert capsys.readouterr().out.startswith("Available functions: 39")


# ---------------------------------------------------------------------------
# mcp-connect / mcp-install: env-file and missing-extra paths
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_dq_env(tmp_path):
    """Empty DATAQUERY_* env with the user config dir under tmp_path; restored afterwards."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("DATAQUERY_")}
    for key in saved:
        del os.environ[key]
    os.environ["DATAQUERY_CONFIG_DIR"] = str(tmp_path / "dq-home")
    yield tmp_path / "dq-home" / ".env"
    for key in [k for k in os.environ if k.startswith("DATAQUERY_")]:
        del os.environ[key]
    os.environ.update(saved)


@pytest.mark.asyncio
async def test_mcp_connect_without_the_mcp_extra(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "fastmcp", None)

    assert await cli.cmd_mcp_connect(_parse("mcp-connect")) == 1
    assert "pip install 'dataquery-sdk[mcp]'" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_mcp_connect_reads_credentials_from_env_file(clean_dq_env, tmp_path):
    pytest.importorskip("fastmcp")
    env_file = tmp_path / "creds.env"
    env_file.write_text("DATAQUERY_CLIENT_ID=file-id\nDATAQUERY_CLIENT_SECRET=file-secret\n")
    proxy = MagicMock(run_async=AsyncMock(return_value=None))

    args = _parse("--env-file", str(env_file), "mcp-connect", "--url", "https://mcp.example.com/mcp")
    with patch("fastmcp.FastMCP.as_proxy", return_value=proxy) as as_proxy:
        with patch("dataquery.transport.auth.TokenManager") as token_manager:
            assert await cli.cmd_mcp_connect(args) == 0

    config = token_manager.call_args.args[0]
    assert config.client_id == "file-id"
    assert as_proxy.call_args.kwargs["name"] == "dataquery-mcp"
    proxy.run_async.assert_awaited_once_with(transport="stdio", show_banner=False)


def test_mcp_install_reads_credentials_from_env_file(clean_dq_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_mcp_extra_installed", lambda: True)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    env_file = tmp_path / "creds.env"
    env_file.write_text("DATAQUERY_CLIENT_ID=file-id\nDATAQUERY_CLIENT_SECRET=file-secret\n")
    config_file = tmp_path / "mcp.json"

    args = _parse("--env-file", str(env_file), "mcp-install", "--config-file", str(config_file))
    assert cli.cmd_mcp_install(args) == 0
    assert "DATAQUERY_CLIENT_ID=file-id" in clean_dq_env.read_text()
    assert "dataquery" in json.loads(config_file.read_text())["mcpServers"]


def test_mcp_install_blank_prompt_answers_leave_it_without_credentials(clean_dq_env, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_mcp_extra_installed", lambda: True)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: "   ")
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt: "")
    config_file = tmp_path / "mcp.json"

    assert cli.cmd_mcp_install(_parse("mcp-install", "--config-file", str(config_file))) == 1
    assert "No credentials" in capsys.readouterr().err
    assert "DATAQUERY_CLIENT_ID" not in os.environ
    assert not config_file.exists()


def _token_manager(*headers):
    tm = MagicMock()
    tm.config.has_bearer_token = False
    tm.current_token = None
    tm.get_valid_token = AsyncMock(side_effect=list(headers))
    return tm


async def _post_through_authe_auth(tm, statuses):
    httpx = pytest.importorskip("httpx")
    seen = []
    replies = iter(statuses)

    def handler(request):
        seen.append(request.headers["Authorization"])
        return httpx.Response(next(replies))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=cli._authe_auth(tm)) as client:
        resp = await client.post("https://mcp.example.com/mcp", json={})
    return resp.status_code, seen


@pytest.mark.asyncio
async def test_authe_auth_without_a_token_raises_a_clear_error():
    from dataquery.types.exceptions import DataQueryError

    with pytest.raises(DataQueryError, match="Could not obtain an OAuth token"):
        await _post_through_authe_auth(_token_manager(None), [200])


@pytest.mark.asyncio
async def test_authe_auth_retries_a_401_without_clearing_a_token_it_no_longer_holds():
    """A concurrent request already dropped the token: retry with a fresh one, clear nothing."""
    tm = _token_manager("Bearer old", "Bearer new")

    assert await _post_through_authe_auth(tm, [401, 200]) == (200, ["Bearer old", "Bearer new"])
    tm.clear_token.assert_not_called()


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["dataquery", *argv])
    return cli.main()


@pytest.mark.parametrize(
    "argv, target",
    [
        (["config", "show"], "cmd_config_show"),
        (["config", "validate"], "cmd_config_validate"),
        (["config", "template", "--output", "x.env"], "cmd_config_template"),
        (["function-help", "--list"], "cmd_function_help"),
        (["skill-install"], "cmd_skill_install"),
    ],
)
def test_main_routes_sync_commands(monkeypatch, argv, target):
    handler = MagicMock(return_value=0)
    monkeypatch.setattr(cli, target, handler)

    assert _main(monkeypatch, *argv) == 0
    assert handler.call_args.args[0].command == argv[0]


def test_main_runs_auth_test(monkeypatch, use_dq, capsys):
    fake = use_dq(list_groups_async=[])

    assert _main(monkeypatch, "auth", "test") == 0
    fake.list_groups_async.assert_awaited_once_with(limit=100)
    assert capsys.readouterr().out.strip() == "Authentication successful."


def test_main_runs_async_commands_and_returns_their_code(monkeypatch, use_dq, capsys):
    use_dq(service_status_async={"up": False, "http_status": 503, "error": "maintenance"})

    assert _main(monkeypatch, "heartbeat") == 1
    assert capsys.readouterr().out.startswith("DataQuery is DOWN (HTTP 503)")


def test_main_turns_sdk_errors_into_a_json_error(monkeypatch, use_dq, capsys):
    err = AuthenticationError("token rejected")
    err.suggestion = "check DATAQUERY_CLIENT_ID"
    use_dq(list_groups_async=AsyncMock(side_effect=err))

    assert _main(monkeypatch, "groups") == 1
    captured = capsys.readouterr()
    assert captured.err.strip() == "token rejected"
    assert _json_section(captured.out) == {
        "status": "error",
        "error_description": "token rejected",
        "suggestion": "check DATAQUERY_CLIENT_ID",
    }


@pytest.mark.parametrize("argv", [["auth"], ["config"]])
def test_main_without_a_subcommand_exits_nonzero(monkeypatch, argv):
    assert _main(monkeypatch, *argv) == 1


def test_main_sync_ignores_non_config_commands():
    assert cli.main_sync(SimpleNamespace(command="groups")) == 0


def test_main_with_an_existing_env_file_proceeds(monkeypatch, tmp_path):
    env_file = tmp_path / "ok.env"
    env_file.write_text("")
    handler = MagicMock(return_value=0)
    monkeypatch.setattr(cli, "cmd_function_help", handler)

    assert _main(monkeypatch, "--env-file", str(env_file), "function-help", "--list") == 0
    assert handler.call_args.args[0].env_file == str(env_file)


# ---------------------------------------------------------------------------
# Findings: current behavior documented, not fixed (see PR description)
# ---------------------------------------------------------------------------


def test_finding_auth_test_failure_escapes_as_a_traceback(monkeypatch, use_dq):
    """``auth test`` runs outside main()'s DataQueryError handler, so rejected credentials
    raise instead of printing the JSON error envelope (exit 1) that every other command uses."""
    use_dq(list_groups_async=AsyncMock(side_effect=AuthenticationError("bad credentials")))

    with pytest.raises(AuthenticationError, match="bad credentials"):
        _main(monkeypatch, "auth", "test")


def test_finding_config_show_raises_on_invalid_settings(monkeypatch):
    """``config validate`` reports bad settings as "Configuration invalid" (exit 1);
    ``config show`` lets the same ConfigurationError escape uncaught."""
    monkeypatch.setenv("DATAQUERY_TIMEOUT", "abc")

    assert cli.cmd_config_validate(_parse("config", "validate")) == 1
    with pytest.raises(ConfigurationError, match="DATAQUERY_TIMEOUT"):
        _main(monkeypatch, "config", "show")


def test_finding_config_without_subcommand_is_silent(monkeypatch, capsys):
    """``dataquery config`` exits 1 with no usage or error message."""
    assert _main(monkeypatch, "config") == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
