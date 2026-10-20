"""Contract and characterization tests for the ``dataquery`` console CLI (``dataquery/cli.py``).

The CLI's public API is its argparse tree (commands, flags, defaults), its exit
codes and its output formats, plus the Python-level handlers other code imports.
Everything here drives ``cli.main()`` through ``sys.argv`` with ``cli.DataQuery``
replaced by an async context-manager mock returning real pydantic models, so no
test touches the network. ``DATAQUERY_*`` env vars and the user config dir are
isolated per test, because several commands export or save credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import subprocess
import sys
import tomllib
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dataquery import cli
from dataquery.types.exceptions import AuthenticationError, DataQueryError
from dataquery.types.models import (
    AttributesResponse,
    AvailabilityInfo,
    DownloadResult,
    DownloadStatus,
    FileInfo,
    FiltersResponse,
    GridDataResponse,
    Group,
    InstrumentsResponse,
    OperationReport,
    TimeSeriesResponse,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# Snapshot of every option: flags, dest, action, default, required, choices, nargs, type.
PARSER_SNAPSHOT: dict[str, list[str]] = {
    "groups": [
        "--json dest=json store_true",
        "--limit dest=limit store type=int",
        "--search dest=search store type=str",
    ],
    "files": [
        "--group-id dest=group_id store required",
        "--file-group-id dest=file_group_id store",
        "--limit dest=limit store type=int",
        "--json dest=json store_true",
    ],
    "availability": [
        "--file-group-id dest=file_group_id store required",
        "--file-datetime dest=file_datetime store required",
        "--json dest=json store_true",
    ],
    "available-files": [
        "--group-id dest=group_id store required",
        "--file-group-id dest=file_group_id store",
        "--start-date dest=start_date store",
        "--end-date dest=end_date store",
        "--json dest=json store_true",
    ],
    "download": [
        "--file-group-id dest=file_group_id store nargs=+",
        "--file-datetime dest=file_datetime store",
        "--destination dest=destination store type=str",
        "--watch dest=watch store_true",
        "--group-id dest=group_id store",
        "--json dest=json store_true",
        "--num-parts dest=num_parts store default=5 type=int",
        "--chunk-size dest=chunk_size store type=int",
        "--no-event-replay dest=no_event_replay store_true",
        "--reset-event-id dest=reset_event_id store_true",
    ],
    "download-group": [
        "--group-id dest=group_id store required",
        "--start-date dest=start_date store required",
        "--end-date dest=end_date store required",
        "--file-group-id dest=file_group_id store nargs=+",
        "--destination dest=destination store default='./downloads' type=str",
        "--max-concurrent dest=max_concurrent store default=3 type=int",
        "--num-parts dest=num_parts store default=5 type=int",
        "--json dest=json store_true",
    ],
    "config show": [],
    "config validate": [],
    "config template": [
        "--output dest=output store required type=str",
    ],
    "config": [],
    "auth test": [],
    "auth": [],
    "search": [
        "--query dest=query store required",
        "--json dest=json store_true",
    ],
    "function-help": [
        "--name dest=name store",
        "--category dest=category store",
        "--list dest=list store_true",
        "--json dest=json store_true",
    ],
    "groups-search": [
        "--keywords dest=keywords store required",
        "--page dest=page store",
    ],
    "instruments": [
        "--group-id dest=group_id store required",
        "--instrument-id dest=instrument_id store",
        "--page dest=page store",
    ],
    "instruments-search": [
        "--group-id dest=group_id store required",
        "--keywords dest=keywords store required",
        "--page dest=page store",
    ],
    "filters": [
        "--group-id dest=group_id store required",
        "--page dest=page store",
    ],
    "attributes": [
        "--group-id dest=group_id store required",
        "--instrument-id dest=instrument_id store",
        "--page dest=page store",
    ],
    "group-timeseries": [
        "--group-id dest=group_id store required",
        "--attributes dest=attributes append required",
        "--filter dest=filter store",
        "--data dest=data store choices=REFERENCE_DATA,NO_REFERENCE_DATA,ALL",
        "--start-date dest=start_date store",
        "--end-date dest=end_date store",
        "--calendar dest=calendar store",
        "--frequency dest=frequency store choices=FREQ_INTRA,FREQ_DAY,FREQ_WEEK,FREQ_MONTH,FREQ_QUARTER,FREQ_ANN",
        "--conversion dest=conversion store",
        "--nan-treatment dest=nan_treatment store choices=NA_NOTHING,NA_LAST,NA_NEXT,NA_INTERP",
        "--page dest=page store",
        "--output-csv dest=output_csv store",
    ],
    "instrument-timeseries": [
        "--instruments dest=instruments append required",
        "--attributes dest=attributes append required",
        "--data dest=data store choices=REFERENCE_DATA,NO_REFERENCE_DATA,ALL",
        "--start-date dest=start_date store",
        "--end-date dest=end_date store",
        "--calendar dest=calendar store",
        "--frequency dest=frequency store choices=FREQ_INTRA,FREQ_DAY,FREQ_WEEK,FREQ_MONTH,FREQ_QUARTER,FREQ_ANN",
        "--conversion dest=conversion store",
        "--nan-treatment dest=nan_treatment store choices=NA_NOTHING,NA_LAST,NA_NEXT,NA_INTERP",
        "--page dest=page store",
        "--output-csv dest=output_csv store",
    ],
    "expression-timeseries": [
        "--expressions dest=expressions append required",
        "--data dest=data store choices=REFERENCE_DATA,NO_REFERENCE_DATA,ALL",
        "--start-date dest=start_date store",
        "--end-date dest=end_date store",
        "--calendar dest=calendar store",
        "--frequency dest=frequency store choices=FREQ_INTRA,FREQ_DAY,FREQ_WEEK,FREQ_MONTH,FREQ_QUARTER,FREQ_ANN",
        "--conversion dest=conversion store",
        "--nan-treatment dest=nan_treatment store choices=NA_NOTHING,NA_LAST,NA_NEXT,NA_INTERP",
        "--page dest=page store",
        "--output-csv dest=output_csv store",
    ],
    "grid-data": [
        "--expr dest=expr store",
        "--grid-id dest=grid_id store",
        "--date dest=date store",
        "--output-csv dest=output_csv store",
    ],
    "heartbeat": [],
    "mcp-connect": [
        "--url dest=url store",
        "--name dest=name store default='dataquery-mcp'",
        "--client-id dest=client_id store",
        "--client-secret dest=client_secret store",
        "--bearer-token dest=bearer_token store",
        "--save-credentials dest=save_credentials store_true",
    ],
    "mcp-install": [
        "--app dest=app store default='claude-desktop' choices=claude-desktop,claude-code,chatgpt,cursor,vscode",
        "--config-file dest=config_file store type=Path",
        "--name dest=name store default='dataquery' type=_mcp_server_name",
        "--url dest=url store",
        "--client-id dest=client_id store",
        "--client-secret dest=client_secret store",
        "--bearer-token dest=bearer_token store",
    ],
    "skill-install": [
        "--app dest=app store default=['claude-code'] choices=claude-code,codex,vscode,cursor,all nargs=+",
        "--scope dest=scope store default='user' choices=user,project",
        "--project-dir dest=project_dir store type=Path",
        "--uninstall dest=uninstall store_true",
        "--force dest=force store_true",
    ],
    "": [
        "--env-file dest=env_file store type=str",
    ],
}


SUBCOMMANDS = [
    "groups",
    "files",
    "availability",
    "available-files",
    "download",
    "download-group",
    "config",
    "auth",
    "search",
    "function-help",
    "groups-search",
    "instruments",
    "instruments-search",
    "filters",
    "attributes",
    "group-timeseries",
    "instrument-timeseries",
    "expression-timeseries",
    "grid-data",
    "heartbeat",
    "mcp-connect",
    "mcp-install",
    "skill-install",
]

ASYNC_COMMANDS = [
    "groups",
    "files",
    "availability",
    "available-files",
    "download",
    "download-group",
    "search",
    "groups-search",
    "instruments",
    "instruments-search",
    "filters",
    "attributes",
    "group-timeseries",
    "instrument-timeseries",
    "expression-timeseries",
    "grid-data",
    "heartbeat",
    "mcp-connect",
]

HANDLERS: dict[str, bool] = {
    "cmd_attributes": True,
    "cmd_auth_test": True,
    "cmd_availability": True,
    "cmd_available_files": True,
    "cmd_config_show": False,
    "cmd_config_template": False,
    "cmd_config_validate": False,
    "cmd_download": True,
    "cmd_download_group": True,
    "cmd_expression_timeseries": True,
    "cmd_files": True,
    "cmd_filters": True,
    "cmd_function_help": False,
    "cmd_grid_data": True,
    "cmd_group_timeseries": True,
    "cmd_groups": True,
    "cmd_groups_search": True,
    "cmd_heartbeat": True,
    "cmd_instrument_timeseries": True,
    "cmd_instruments": True,
    "cmd_instruments_search": True,
    "cmd_mcp_connect": True,
    "cmd_mcp_install": False,
    "cmd_search": True,
    "cmd_skill_install": False,
}


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path) -> Iterator[Path]:
    """A clean ``DATAQUERY_*`` environment with the user config dir under tmp_path.

    Plain ``os.environ`` rather than monkeypatch: commands export credentials
    directly, and monkeypatch would restore (leak) those on undo.
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("DATAQUERY_")}
    for key in saved:
        del os.environ[key]
    config_dir = tmp_path / "dq-home"
    os.environ["DATAQUERY_CONFIG_DIR"] = str(config_dir)
    yield config_dir
    for key in [k for k in os.environ if k.startswith("DATAQUERY_")]:
        del os.environ[key]
    os.environ.update(saved)


def _fake_dq(**methods: Any) -> MagicMock:
    fake = MagicMock()
    fake.__aenter__ = AsyncMock(return_value=fake)
    fake.__aexit__ = AsyncMock(return_value=None)
    for name, value in methods.items():
        setattr(fake, name, value if isinstance(value, AsyncMock) else AsyncMock(return_value=value))
    return fake


@pytest.fixture
def use_dq(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(**methods: Any) -> MagicMock:
        fake = _fake_dq(**methods)
        monkeypatch.setattr(cli, "DataQuery", MagicMock(return_value=fake))
        return fake

    return install


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Any:
    """Run ``cli.main()`` with argv; return (exit code, stdout, stderr)."""

    def invoke(*argv: str) -> tuple[int, str, str]:
        monkeypatch.setattr(sys, "argv", ["dataquery", *argv])
        rc = cli.main()
        captured = capsys.readouterr()
        return rc, captured.out, captured.err

    return invoke


def _json_section(out: str) -> Any:
    return json.loads(out.split("--- JSON ---", 1)[1])


def _ts_response(instrument_ids: list[str], *, next_link: str | None = None, message: str | None = None) -> Any:
    attribute: dict[str, Any] = {"attribute-id": "TR", "expression": "DB(X)"}
    if message:
        attribute["message"] = message
    else:
        attribute["time-series"] = [["20240102", 1.0], ["20240101", 2.0]]
    return TimeSeriesResponse.model_validate(
        {
            "links": [{"self": "/x", "next": next_link}],
            "items": len(instrument_ids),
            "instruments": [
                {"item": i, "instrument-id": inst, "attributes": [dict(attribute)]}
                for i, inst in enumerate(instrument_ids)
            ],
        }
    )


def _grid(payload: dict[str, Any]) -> GridDataResponse:
    return GridDataResponse.model_validate(payload)


def _option_line(action: argparse.Action) -> str:
    kinds = {
        argparse._StoreAction: "store",
        argparse._StoreTrueAction: "store_true",
        argparse._AppendAction: "append",
    }
    parts = [",".join(action.option_strings), f"dest={action.dest}", kinds[type(action)]]
    if action.default not in (None, False):
        parts.append(f"default={action.default!r}")
    if action.required:
        parts.append("required")
    if action.choices:
        parts.append(f"choices={','.join(action.choices)}")
    if action.nargs not in (None, 0):
        parts.append(f"nargs={action.nargs}")
    type_name = getattr(action.type, "__name__", None)
    if type_name:
        parts.append(f"type={type_name}")
    return " ".join(parts)


def _subparsers(parser: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)


def _walk(parser: argparse.ArgumentParser, prefix: str, out: dict[str, list[str]]) -> dict[str, list[str]]:
    options: list[str] = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                _walk(sub, f"{prefix} {name}".strip(), out)
            continue
        options.append(_option_line(action))
    out[prefix] = options
    return out


def _hint_shape(hint: Any) -> Any:
    """Normalize ``List[str]``/``list[str]`` and ``Optional[X]``/``X | None`` to one comparable form."""
    origin = typing.get_origin(hint)
    if origin is None:
        return hint
    if origin is typing.Union or origin is getattr(__import__("types"), "UnionType"):
        return ("union", frozenset(_hint_shape(a) for a in typing.get_args(hint)))
    return (origin, tuple(_hint_shape(a) for a in typing.get_args(hint)))


# --------------------------------------------------------------------------- #
# Public API: argparse tree, handlers, entry points
# --------------------------------------------------------------------------- #
def test_parser_tree_is_unchanged() -> None:
    assert _walk(cli.create_parser(), "", {}) == PARSER_SNAPSHOT


def test_subcommand_names_and_order_are_unchanged() -> None:
    parser = cli.create_parser()
    top = _subparsers(parser)
    assert top is not None and top.dest == "command"
    assert list(top.choices) == SUBCOMMANDS
    config_sub = _subparsers(top.choices["config"])
    auth_sub = _subparsers(top.choices["auth"])
    assert config_sub is not None and config_sub.dest == "config_command"
    assert auth_sub is not None and auth_sub.dest == "auth_command"
    assert list(config_sub.choices) == ["show", "validate", "template"]
    assert list(auth_sub.choices) == ["test"]


@pytest.mark.parametrize(("name", "is_async"), sorted(HANDLERS.items()))
def test_command_handler_signature_is_unchanged(name: str, is_async: bool) -> None:
    fn = getattr(cli, name)
    assert inspect.iscoroutinefunction(fn) is is_async
    params = list(inspect.signature(fn).parameters.values())
    assert [(p.name, p.kind, p.default) for p in params] == [
        ("args", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty)
    ]
    assert typing.get_type_hints(fn) == {"args": argparse.Namespace, "return": int}


def test_entry_point_signatures_are_unchanged() -> None:
    assert list(inspect.signature(cli.main).parameters) == []
    assert typing.get_type_hints(cli.main) == {"return": int}
    assert list(inspect.signature(cli.main_sync).parameters) == ["ns"]
    assert typing.get_type_hints(cli.main_sync) == {"ns": argparse.Namespace, "return": int}
    assert list(inspect.signature(cli.create_parser).parameters) == []
    assert typing.get_type_hints(cli.create_parser) == {"return": argparse.ArgumentParser}


def test_imported_helper_signatures_are_unchanged() -> None:
    def shape(fn: Any) -> dict[str, Any]:
        return {k: _hint_shape(v) for k, v in typing.get_type_hints(fn).items()}

    assert list(inspect.signature(cli._export_mcp_credentials).parameters) == ["args"]
    assert shape(cli._export_mcp_credentials) == {"args": argparse.Namespace, "return": (list, (str,))}
    assert list(inspect.signature(cli._save_mcp_credentials).parameters) == ["keys"]
    assert shape(cli._save_mcp_credentials) == {"keys": (list, (str,)), "return": type(None)}
    assert list(inspect.signature(cli._authe_auth).parameters) == ["token_manager"]
    assert shape(cli._has_mcp_credentials) == {"keys": (list, (str,)), "return": bool}
    assert shape(cli._mcp_token_storage_dir) == {"config": Any, "return": Path}
    error_params = inspect.signature(cli._print_error).parameters
    assert [(p.name, p.kind.name, p.default) for p in error_params.values()] == [
        ("message", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("suggestion", "KEYWORD_ONLY", None),
    ]
    result_params = inspect.signature(cli._print_endpoint_result).parameters
    assert [(p.name, p.kind.name, p.default) for p in result_params.values()] == [
        ("summary", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("payload", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
        ("csv_info", "KEYWORD_ONLY", None),
    ]
    assert shape(cli._collect_all_pages)["return"] == (tuple, (Any, ("union", frozenset({str, type(None)}))))


def test_public_names_are_unchanged() -> None:
    assert not hasattr(cli, "__all__")
    defined = {
        name
        for name, obj in vars(cli).items()
        if not name.startswith("_") and callable(obj) and getattr(obj, "__module__", None) == "dataquery.cli"
    }
    assert defined == set(HANDLERS) | {"main", "main_sync", "create_parser"}
    assert cli.MCP_INSTALL_APPS == ("claude-desktop", "claude-code", "chatgpt", "cursor", "vscode")
    assert list(cli._ASYNC_COMMANDS) == ASYNC_COMMANDS
    assert {name: fn.__name__ for name, fn in cli._ASYNC_COMMANDS.items()} == {
        name: "cmd_" + name.replace("-", "_") for name in ASYNC_COMMANDS
    }


def test_console_script_entry_point_is_unchanged() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["dataquery"] == "dataquery.cli:main"


def test_module_runs_with_python_dash_m() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dataquery.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("usage: ")
    assert "Command Line Interface for the DataQuery SDK" in proc.stdout


# --------------------------------------------------------------------------- #
# main(): dispatch and top-level error handling
# --------------------------------------------------------------------------- #
def test_no_command_prints_help_and_exits_1(run: Any) -> None:
    rc, out, _ = run()
    assert rc == 1
    assert out.startswith("usage: dataquery")


def test_missing_env_file_is_an_error(run: Any, tmp_path: Path) -> None:
    rc, out, err = run("--env-file", str(tmp_path / "nope.env"), "groups")
    assert rc == 1
    assert err.strip() == f"--env-file not found: {tmp_path / 'nope.env'}"
    assert _json_section(out) == {
        "status": "error",
        "error_description": f"--env-file not found: {tmp_path / 'nope.env'}",
    }


def test_dataquery_error_becomes_an_error_envelope(run: Any, use_dq: Any) -> None:
    exc = DataQueryError("quota exceeded")
    exc.suggestion = "wait a minute"  # type: ignore[attr-defined]
    use_dq(list_groups_async=AsyncMock(side_effect=exc))
    rc, out, err = run("groups")
    assert rc == 1
    assert err.strip() == "quota exceeded"
    assert _json_section(out) == {
        "status": "error",
        "error_description": "quota exceeded",
        "suggestion": "wait a minute",
    }


def test_main_sync_returns_0_for_non_config_commands() -> None:
    assert cli.main_sync(argparse.Namespace(command="groups")) == 0
    assert cli.main_sync(argparse.Namespace(command="config", config_command=None)) == 1


def test_ctrl_c_exits_130_without_a_traceback(run: Any, use_dq: Any) -> None:
    use_dq(list_groups_async=AsyncMock(side_effect=KeyboardInterrupt()))
    try:
        rc, _, err = run("groups")
    except KeyboardInterrupt:
        pytest.fail("KeyboardInterrupt escaped main()")
    assert rc == 130
    assert err.strip() == "Interrupted"


# --------------------------------------------------------------------------- #
# Catalog listing commands
# --------------------------------------------------------------------------- #
def test_groups_text_and_search(run: Any, use_dq: Any) -> None:
    groups = [Group.model_validate({"group-id": "G1", "group-name": "One"})]
    fake = use_dq(list_groups_async=groups, search_groups_async=groups)
    assert run("groups", "--limit", "5") == (0, "G1\tOne\n", "")
    fake.list_groups_async.assert_awaited_once_with(limit=5)
    assert run("groups", "--search", "fx")[0] == 0
    fake.search_groups_async.assert_awaited_once_with("fx", limit=None)


def test_groups_json_uses_field_names(run: Any, use_dq: Any) -> None:
    use_dq(list_groups_async=[Group.model_validate({"group-id": "G1", "group-name": "One"})])
    rc, out, _ = run("groups", "--json")
    assert rc == 0
    payload = json.loads(out)
    assert payload[0]["group_id"] == "G1" and payload[0]["group_name"] == "One"


def test_groups_and_files_fall_back_to_str_for_undumpable_items(run: Any, use_dq: Any) -> None:
    class Undumpable:
        def model_dump(self) -> dict[str, Any]:
            raise TypeError("no")

        def __str__(self) -> str:
            return "<item>"

    use_dq(list_groups_async=[Undumpable()], list_files_async=[Undumpable()])
    assert run("groups") == (0, "<item>\n", "")
    assert json.loads(run("groups", "--json")[1]) == ["<item>"]
    assert run("files", "--group-id", "G") == (0, "Found 1 files\n<item>\n", "")
    assert json.loads(run("files", "--group-id", "G", "--json")[1]) == ["<item>"]


def test_files_text_prints_file_type_as_a_list_repr(run: Any, use_dq: Any) -> None:
    """Not fixed (finding 11): the file type is printed as a Python list repr."""
    fake = use_dq(list_files_async=[FileInfo.model_validate({"file-group-id": "F1", "file-type": "csv"})])
    assert run("files", "--group-id", "G", "--file-group-id", "F1") == (0, "Found 1 files\nF1\t['csv']\n", "")
    fake.list_files_async.assert_awaited_once_with("G", "F1")
    rc, out, _ = run("files", "--group-id", "G", "--json")
    assert rc == 0 and json.loads(out)[0]["file_group_id"] == "F1"


@pytest.mark.parametrize(("available", "state"), [(True, "available"), (False, "not available")])
def test_availability(run: Any, use_dq: Any, available: bool, state: str) -> None:
    info = AvailabilityInfo.model_validate({"file-datetime": "20240101", "is-available": available})
    use_dq(check_availability_async=info)
    assert run("availability", "--file-group-id", "FG", "--file-datetime", "20240101") == (
        0,
        f"FG @ 20240101: {state}\n",
        "",
    )
    rc, out, _ = run("availability", "--file-group-id", "FG", "--file-datetime", "20240101", "--json")
    assert rc == 0 and json.loads(out)["is_available"] is available


def test_available_files(run: Any, use_dq: Any) -> None:
    items = [
        {"file-group-id": "F1", "file-datetime": "20240101", "is-available": True},
        {"file-group-id": "F2", "file-datetime": "20240102", "is-available": "yes"},
    ]
    fake = use_dq(list_available_files_async=items)
    rc, out, _ = run("available-files", "--group-id", "G", "--start-date", "20240101", "--end-date", "20240102")
    assert rc == 0
    assert out == "Found 2 files (1 available)\nF1\t20240101\tavailable\nF2\t20240102\tunavailable\n"
    fake.list_available_files_async.assert_awaited_once_with(
        "G", file_group_id=None, start_date="20240101", end_date="20240102"
    )
    assert json.loads(run("available-files", "--group-id", "G", "--json")[1]) == items


# --------------------------------------------------------------------------- #
# download / download-group
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--watch"], "--group-id is required when using --watch"),
        ([], "--file-group-id is required for a single-file download"),
        (["--file-group-id", "A", "B"], "--file-group-id must be a single value for single-file download"),
    ],
)
def test_download_usage_errors_go_to_stdout(run: Any, use_dq: Any, argv: list[str], message: str) -> None:
    """Not fixed (finding 12): usage errors are printed on stdout, exit 1."""
    use_dq()
    assert run("download", *argv) == (1, message + "\n", "")


def test_download_single_file_success(run: Any, use_dq: Any, tmp_path: Path) -> None:
    result = DownloadResult.model_validate(
        {"file_group_id": "FG", "local_path": tmp_path / "f.csv", "status": DownloadStatus.COMPLETED}
    )
    fake = use_dq(download_file_async=result)
    rc, out, _ = run("download", "--file-group-id", "FG", "--file-datetime", "20240101", "--destination", "d")
    assert (rc, out) == (0, f"Downloaded to {tmp_path / 'f.csv'}\n")
    call = fake.download_file_async.await_args
    assert call.args == ("FG", "20240101")
    assert call.kwargs["num_parts"] == 5
    assert call.kwargs["options"].destination_path == Path("d")
    assert call.kwargs["options"].chunk_size == 1048576


def test_download_single_file_options_and_json(run: Any, use_dq: Any) -> None:
    result = DownloadResult.model_validate({"file_group_id": "FG", "status": DownloadStatus.ALREADY_EXISTS})
    fake = use_dq(download_file_async=result)
    rc, out, _ = run("download", "--file-group-id", "FG", "--num-parts", "2", "--chunk-size", "10", "--json")
    assert rc == 0
    assert json.loads(out)["status"] == "already_exists"
    call = fake.download_file_async.await_args
    assert call.args == ("FG", None)
    assert call.kwargs["num_parts"] == 2
    assert call.kwargs["options"].destination_path is None
    assert call.kwargs["options"].chunk_size == 10
    assert run("download", "--file-group-id", "FG") == (0, "Already exists at None\n", "")


def test_download_single_file_failure(run: Any, use_dq: Any) -> None:
    use_dq(download_file_async=DownloadResult.model_validate({"file_group_id": "FG", "error_message": "HTTP 404"}))
    rc, out, err = run("download", "--file-group-id", "FG", "--file-datetime", "20240101")
    assert (rc, out) == (1, "")
    assert err == "Download failed for FG @ 20240101: HTTP 404\n"
    use_dq(
        download_file_async=DownloadResult.model_validate({"file_group_id": "FG", "status": DownloadStatus.CANCELLED})
    )
    assert run("download", "--file-group-id", "FG")[2] == "Download failed for FG @ None: DownloadStatus.CANCELLED\n"


class _Manager:
    def __init__(self, *, stop_error: Exception | None = None) -> None:
        self.stop_error = stop_error
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True
        if self.stop_error:
            raise self.stop_error

    def get_stats(self) -> dict[str, Any]:
        from datetime import datetime

        return {"files_downloaded": 2, "start_time": datetime(2024, 1, 1)}


@pytest.fixture
def stop_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the watch loop's ``asyncio.sleep(60)`` behave like Ctrl+C."""
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args: Any) -> Any:
        if delay == 60:
            raise asyncio.CancelledError
        return await real_sleep(delay, *args)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


def test_download_watch_prints_stats(run: Any, use_dq: Any, stop_watch: None) -> None:
    mgr = _Manager()
    fake = use_dq(auto_download_async=mgr)
    rc, out, err = run("download", "--watch", "--group-id", "G", "--file-group-id", "F1", "F2")
    assert rc == 0 and mgr.stopped and err == ""
    assert json.loads(out) == {"files_downloaded": 2, "start_time": "2024-01-01 00:00:00"}
    fake.auto_download_async.assert_awaited_once_with(
        group_id="G", destination_dir="./downloads", file_group_id=["F1", "F2"], enable_event_replay=True
    )


def test_download_watch_reset_event_id_and_stop_failure(
    run: Any, use_dq: Any, stop_watch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MagicMock()
    builder = MagicMock(return_value=store)
    monkeypatch.setattr("dataquery.sse.event_store.build_event_id_store", builder)
    fake = use_dq(auto_download_async=_Manager(stop_error=RuntimeError("boom")))
    rc, _, err = run(
        "download", "--watch", "--group-id", "G", "--destination", "out", "--reset-event-id", "--no-event-replay"
    )
    assert rc == 0
    store.clear.assert_called_once_with()
    assert builder.call_args.args[1].group_id == "G"
    assert err == "Failed to stop notification manager: boom\n"
    assert fake.auto_download_async.await_args.kwargs == {
        "group_id": "G",
        "destination_dir": "out",
        "file_group_id": None,
        "enable_event_replay": False,
    }


def test_download_watch_reset_without_a_store(
    run: Any, use_dq: Any, stop_watch: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("dataquery.sse.event_store.build_event_id_store", MagicMock(return_value=None))
    use_dq(auto_download_async=_Manager())
    assert run("download", "--watch", "--group-id", "G", "--reset-event-id")[0] == 0


def test_download_group(run: Any, use_dq: Any) -> None:
    ok = OperationReport(operation="group", counts={"successful_downloads": 3})
    fake = use_dq(run_group_download_async=ok)
    argv = ["download-group", "--group-id", "G", "--start-date", "20240101", "--end-date", "20240102"]
    assert run(*argv) == (0, "Downloaded 3 files to ./downloads\n", "")
    assert fake.run_group_download_async.await_args.kwargs == {
        "group_id": "G",
        "start_date": "20240101",
        "end_date": "20240102",
        "destination_dir": "./downloads",
        "max_concurrent": 3,
        "num_parts": 5,
        "file_group_id": None,
    }
    bad = OperationReport(
        operation="group",
        counts={"successful_downloads": 1, "failed_downloads": 1},
        details={"failures": [{"file_group_id": "F", "file_datetime": "20240101", "error": "404"}]},
    )
    use_dq(run_group_download_async=bad)
    assert run(*argv) == (1, "Downloaded 1 files to ./downloads\nFailed: 1\n  F @ 20240101: 404\n", "")
    rc, out, _ = run(*argv, "--json")
    assert rc == 1 and json.loads(out)["counts"]["failed_downloads"] == 1


# --------------------------------------------------------------------------- #
# config / auth
# --------------------------------------------------------------------------- #
def test_config_show_prints_only_a_confirmation(run: Any) -> None:
    """Not fixed (finding 10): ``config show`` does not print the resolved config."""
    os.environ["DATAQUERY_BASE_URL"] = "https://dq.example.com"
    assert run("config", "show") == (0, "Configuration loaded\n", "")


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"DATAQUERY_BASE_URL": ""}, "DATAQUERY_BASE_URL environment variable is required"),
        ({"DATAQUERY_BASE_URL": "https://x", "DATAQUERY_TIMEOUT": "abc"}, "Invalid float value for DATAQUERY_TIMEOUT"),
    ],
)
def test_config_show_reports_configuration_errors(run: Any, env: dict[str, str], message: str) -> None:
    os.environ.update(env)
    rc, out, err = run("config", "show")
    assert rc == 1
    assert message in err
    assert _json_section(out)["status"] == "error"


def test_config_validate(run: Any, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATAQUERY_BASE_URL=https://dq.example.com\nDATAQUERY_CLIENT_ID=id\nDATAQUERY_CLIENT_SECRET=secret\n"
    )
    assert run("--env-file", str(env_file), "config", "validate") == (0, "Configuration valid\n", "")
    os.environ.pop("DATAQUERY_CLIENT_ID")
    os.environ.pop("DATAQUERY_CLIENT_SECRET")
    rc, out, _ = run("config", "validate")
    assert rc == 1 and out.startswith("Configuration invalid: ")


def test_config_template_writes_the_file(run: Any, tmp_path: Path) -> None:
    out_file = tmp_path / "sub" / "t.env"
    rc, out, _ = run("config", "template", "--output", str(out_file))
    assert rc == 0
    assert out.splitlines()[-1] == f"Template written to {out_file}"
    assert "DATAQUERY_BASE_URL=" in out_file.read_text()


def test_config_template_reports_unwritable_paths(run: Any, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    rc, _, err = run("config", "template", "--output", str(blocker / "t.env"))
    assert rc == 1
    assert "Could not write template" in err


def test_bare_config_prints_config_help(run: Any) -> None:
    rc, out, _ = run("config")
    assert rc == 1
    assert out.startswith("usage: dataquery config")


def test_bare_auth_prints_auth_help(run: Any) -> None:
    rc, out, _ = run("auth")
    assert rc == 1
    assert out.startswith("usage: dataquery auth")


def test_auth_test_success(run: Any, use_dq: Any) -> None:
    fake = use_dq(list_groups_async=[])
    assert run("auth", "test") == (0, "Authentication successful.\n", "")
    fake.list_groups_async.assert_awaited_once_with(limit=100)


def test_auth_test_failure_is_reported(run: Any, use_dq: Any) -> None:
    use_dq(list_groups_async=AsyncMock(side_effect=AuthenticationError("bad credentials")))
    rc, out, err = run("auth", "test")
    assert rc == 1
    assert err.strip() == "bad credentials"
    assert _json_section(out) == {"status": "error", "error_description": "bad credentials"}


def test_auth_test_without_credentials_is_reported(run: Any) -> None:
    os.environ["DATAQUERY_BASE_URL"] = "https://dq.example.com"
    rc, out, err = run("auth", "test")
    assert rc == 1
    assert "Either OAuth credentials or Bearer token must be configured" in err
    assert _json_section(out)["status"] == "error"


# --------------------------------------------------------------------------- #
# search and JSON-envelope endpoint commands
# --------------------------------------------------------------------------- #
def test_search(run: Any, use_dq: Any) -> None:
    use_dq(search_async={"results": [{"id": 1}, {"id": 2}]})
    rc, out, _ = run("search", "--query", "fx vol")
    assert rc == 0
    assert out.startswith("Search returned 2 result(s) for: fx vol\n\n--- JSON ---\n")
    assert _json_section(out) == {"status": "success", "data": {"results": [{"id": 1}, {"id": 2}]}}
    use_dq(search_async={"groups": "not-a-list", "items": [{"id": 1}]})
    assert run("search", "--query", "q")[1].startswith("Search returned 1 result(s)")
    use_dq(search_async=[{"id": 1}])
    assert run("search", "--query", "q", "--json") == (0, '[\n  {\n    "id": 1\n  }\n]\n', "")


def test_groups_search(run: Any, use_dq: Any) -> None:
    fake = use_dq(search_groups_async=[Group.model_validate({"group-id": "G"})])
    rc, out, _ = run("groups-search", "--keywords", "fx", "--page", "p2")
    assert rc == 0
    assert out.startswith("Found 1 dataset(s) matching 'fx'\n")
    assert _json_section(out)["data"]["groups"][0]["group-id"] == "G"
    fake.search_groups_async.assert_awaited_once_with("fx", page="p2")


def test_instruments_with_more_pages(run: Any, use_dq: Any) -> None:
    resp = InstrumentsResponse.model_validate(
        {
            "instruments": [{"item": 1, "instrument-id": "I1", "instrument-name": "One"}],
            "links": [{"next": "/instruments?page=tok2"}],
        }
    )
    fake = use_dq(list_instruments_async=resp)
    rc, out, _ = run("instruments", "--group-id", "G")
    assert rc == 0
    assert out.startswith("Found 1 instrument(s) in G\nMore pages available; fetch the next one with --page tok2\n")
    fake.list_instruments_async.assert_awaited_once_with("G", None, None)


def test_instruments_search(run: Any, use_dq: Any) -> None:
    fake = use_dq(search_instruments_async=InstrumentsResponse.model_validate({"instruments": []}))
    rc, out, _ = run("instruments-search", "--group-id", "G", "--keywords", "usd", "--page", "p")
    assert rc == 0
    assert out.startswith("Found 0 instrument(s) in G matching 'usd'\n\n--- JSON ---")
    fake.search_instruments_async.assert_awaited_once_with("G", "usd", "p")


def test_filters_and_attributes(run: Any, use_dq: Any) -> None:
    filters = FiltersResponse.model_validate({"filters": [{"filter-name": "currency"}]})
    attrs = AttributesResponse.model_validate(
        {"instruments": [{"item": 1, "instrument-id": "I"}, {"item": 2, "instrument-id": "J"}]}
    )
    fake = use_dq(get_group_filters_async=filters, get_group_attributes_async=attrs)
    assert run("filters", "--group-id", "G")[1].startswith("Found 1 filter dimension(s) for G\n")
    fake.get_group_filters_async.assert_awaited_once_with("G", None)
    assert run("attributes", "--group-id", "G", "--instrument-id", "I")[1].startswith(
        "Attributes for 2 instrument(s) in G\n"
    )
    fake.get_group_attributes_async.assert_awaited_once_with("G", "I", None)


def test_info_message_is_shown(run: Any, use_dq: Any) -> None:
    resp = FiltersResponse.model_validate({"info": {"code": 204, "description": "No content"}})
    use_dq(get_group_filters_async=resp)
    assert "Server info: [204] No content\n" in run("filters", "--group-id", "G")[1]


# --------------------------------------------------------------------------- #
# time-series commands
# --------------------------------------------------------------------------- #
TS_FLAGS = [
    "--data", "ALL", "--start-date", "20240101", "--end-date", "20240131", "--calendar", "CAL_X",
    "--frequency", "FREQ_DAY", "--conversion", "CONV_X", "--nan-treatment", "NA_LAST", "--page", "p1",
]  # fmt: skip
TS_KWARGS = {
    "data": "ALL",
    "start_date": "20240101",
    "end_date": "20240131",
    "calendar": "CAL_X",
    "frequency": "FREQ_DAY",
    "conversion": "CONV_X",
    "nan_treatment": "NA_LAST",
    "page": "p1",
}


def test_group_timeseries_passes_every_flag(run: Any, use_dq: Any) -> None:
    fake = use_dq(get_group_time_series_async=_ts_response(["A"]))
    rc, out, _ = run(
        "group-timeseries",
        "--group-id",
        "G",
        "--attributes",
        "TR,,LOC",
        "--attributes",
        "X",
        "--filter",
        "f",
        *TS_FLAGS,
    )
    assert rc == 0
    assert out.startswith("Group time-series: 1 instrument(s), 2 data point(s) (20240101 to 20240102)\n")
    call = fake.get_group_time_series_async.await_args
    assert call.args == ("G", ["TR,,LOC", "X"])
    assert call.kwargs == {"filter": "f", **TS_KWARGS}
    fake.get_next_page_async.assert_not_called()


def test_instrument_timeseries_follows_pages_and_exports_csv(run: Any, use_dq: Any, tmp_path: Path) -> None:
    first = _ts_response(["A"], next_link="/ts?page=2")
    fake = use_dq(get_instrument_time_series_async=first, get_next_page_async=_ts_response(["B"]))
    out_csv = tmp_path / "ts.csv"
    rc, out, _ = run("instrument-timeseries", "--instruments", "A", "--attributes", "TR", "--output-csv", str(out_csv))
    assert rc == 0
    assert out.startswith("Instrument time-series: 2 instrument(s), 4 data point(s)")
    assert f"CSV exported: {out_csv} (4 rows)" in out
    envelope = _json_section(out)
    assert (envelope["csv_exported"], envelope["csv_rows"]) == (str(out_csv), 4)
    assert out_csv.read_text().count("\n") == 5
    assert fake.get_instrument_time_series_async.await_args.kwargs == {}


def test_expression_timeseries_csv_to_stdout(run: Any, use_dq: Any) -> None:
    fake = use_dq(get_expressions_time_series_async=_ts_response(["A"]))
    rc, out, _ = run("expression-timeseries", "--expressions", "DB(X)", "--output-csv", "-")
    assert rc == 0
    assert out.startswith("date,value,instrument_id,")
    assert "CSV exported: stdout (2 rows)" in out
    assert fake.get_expressions_time_series_async.await_args.args == (["DB(X)"],)


def test_timeseries_server_messages_and_no_dates(run: Any, use_dq: Any) -> None:
    use_dq(get_expressions_time_series_async=_ts_response(["A"], message="Expression syntax error!"))
    rc, out, _ = run("expression-timeseries", "--expressions", "DB(X)")
    assert rc == 0
    assert out.startswith(
        "Expression time-series: 1 instrument(s), 0 data point(s) (no dates)\n"
        "Server message for DB(X): Expression syntax error!\n"
    )


def test_timeseries_keeps_fetched_pages_when_a_later_page_fails(run: Any, use_dq: Any) -> None:
    use_dq(
        get_group_time_series_async=_ts_response(["A"], next_link="/ts?page=2"),
        get_next_page_async=AsyncMock(side_effect=DataQueryError("HTTP 500")),
    )
    rc, out, _ = run("group-timeseries", "--group-id", "G", "--attributes", "TR")
    assert rc == 1
    assert "INCOMPLETE: page 2 failed (HTTP 500); kept 1 page(s), 1 of 1 instrument(s)" in out
    assert _json_section(out)["status"] == "success"


def test_timeseries_stops_when_next_page_is_none(run: Any, use_dq: Any) -> None:
    use_dq(get_group_time_series_async=_ts_response(["A"], next_link="/ts?page=2"), get_next_page_async=None)
    rc, out, _ = run("group-timeseries", "--group-id", "G", "--attributes", "TR")
    assert rc == 0
    assert "More pages available; fetch the next one with --page 2" in out


def test_timeseries_export_failure_still_shows_the_summary(run: Any, use_dq: Any, tmp_path: Path) -> None:
    use_dq(get_group_time_series_async=_ts_response(["A"]))
    rc, out, err = run(
        "group-timeseries", "--group-id", "G", "--attributes", "TR", "--output-csv", str(tmp_path / "no" / "x.csv")
    )
    assert rc == 1
    assert out.startswith("Group time-series: 1 instrument(s), 2 data point(s)")
    assert "Failed to write CSV to" in err
    assert _json_section(out)["status"] == "error"


# --------------------------------------------------------------------------- #
# grid-data / heartbeat
# --------------------------------------------------------------------------- #
def test_grid_data_requires_expr_or_grid_id(run: Any, use_dq: Any) -> None:
    use_dq()
    rc, out, err = run("grid-data")
    assert rc == 1
    assert err == "Provide --expr or --grid-id for grid data.\n"
    assert _json_section(out)["suggestion"].startswith("Example: --expr")


def test_grid_data_summary_and_csv(run: Any, use_dq: Any, tmp_path: Path) -> None:
    grid = _grid(
        {
            "series": [
                {"expr": "E1", "records": [{"a": 1}, {"a": 2}]},
                {"expr": "E2", "errorCode": "X", "errorMessage": "bad"},
            ]
        }
    )
    fake = use_dq(get_grid_data_async=grid)
    out_csv = tmp_path / "g.csv"
    rc, out, _ = run("grid-data", "--grid-id", "GR", "--date", "20240101", "--output-csv", str(out_csv))
    assert rc == 0
    assert out.startswith("Grid: 2 series, 2 record(s)\nServer error for E2: [X] bad\nCSV exported: ")
    fake.get_grid_data_async.assert_awaited_once_with(expr=None, grid_id="GR", date="20240101")


def test_grid_data_response_error_without_csv(run: Any, use_dq: Any) -> None:
    use_dq(get_grid_data_async=_grid({"errorCode": "E1", "errorMessage": "bad expr"}))
    rc, out, _ = run("grid-data", "--expr", "X")
    assert rc == 0
    assert out.startswith("Grid: 0 series, 0 record(s)\nServer error: [E1] bad expr\n")


def test_grid_data_export_failure_shows_the_server_error(run: Any, use_dq: Any, tmp_path: Path) -> None:
    use_dq(get_grid_data_async=_grid({"errorCode": "E1", "errorMessage": "bad expr"}))
    rc, out, err = run("grid-data", "--expr", "X", "--output-csv", str(tmp_path / "g.csv"))
    assert rc == 1
    assert out.startswith("Grid: 0 series, 0 record(s)\nServer error: [E1] bad expr\n")
    assert "No grid series found" in err
    assert _json_section(out)["status"] == "error"


@pytest.mark.parametrize(
    ("status", "summary", "rc"),
    [
        ({"up": True, "http_status": 200}, "DataQuery is UP", 0),
        ({"up": False, "http_status": 401}, "DataQuery authentication failed (HTTP 401)", 1),
        ({"up": False, "http_status": 503}, "DataQuery is DOWN (HTTP 503)", 1),
        ({"up": False, "http_status": None, "error": "timeout"}, "DataQuery is unreachable: timeout", 1),
    ],
)
def test_heartbeat(run: Any, use_dq: Any, status: dict[str, Any], summary: str, rc: int) -> None:
    use_dq(service_status_async=status)
    code, out, _ = run("heartbeat")
    assert code == rc
    assert out.startswith(summary + "\n")
    assert _json_section(out)["data"]["status"] == ("ok" if rc == 0 else "error")


# --------------------------------------------------------------------------- #
# function-help
# --------------------------------------------------------------------------- #
def test_function_help_needs_an_option(run: Any) -> None:
    assert run("function-help") == (1, "Provide --name NAME, --list, or --category CATEGORY\n", "")


def test_function_help_by_name(run: Any) -> None:
    rc, out, _ = run("function-help", "--name", "vol")
    assert rc == 0
    assert out.startswith("VOL(")
    assert "  Parameters:\n" in out
    rc, out, _ = run("function-help", "--name", "VOL", "--json")
    payload = json.loads(out)
    assert rc == 0 and payload["function"] == "VOL" and payload["syntax"].startswith("VOL(")


def test_function_help_unknown_name(run: Any) -> None:
    assert run("function-help", "--name", "nope") == (
        1,
        "Unknown function: NOPE\nUse --list to see all available functions.\n",
        "",
    )
    rc, out, _ = run("function-help", "--name", "nope", "--json")
    assert rc == 1 and json.loads(out)["error_description"] == "Unknown function: NOPE"


def test_function_help_list_and_category(run: Any) -> None:
    rc, out, _ = run("function-help", "--list")
    assert rc == 0 and out.startswith("Available functions: ")
    rc, out, _ = run("function-help", "--category", "futures & options", "--json")
    payload = json.loads(out)
    assert rc == 0 and payload["count"] == 39
    assert {f["category"] for f in payload["functions"]} == {"Futures & Options"}


def test_function_help_unknown_category_exits_0(run: Any) -> None:
    """Not fixed (finding 4b): an unknown category is not an error."""
    assert run("function-help", "--category", "NOPE") == (0, "Available functions: 0\n", "")


def test_function_help_category_help_names_real_categories() -> None:
    from dataquery.function_registry import get_function_categories, list_functions_by_category

    top = _subparsers(cli.create_parser())
    assert top is not None
    action = next(a for a in top.choices["function-help"]._actions if a.dest == "category")
    assert action.help is not None
    assert "F&O SPECIFIC" not in action.help
    for category in get_function_categories():
        assert category in action.help
        assert list_functions_by_category(category)


# --------------------------------------------------------------------------- #
# mcp-connect / mcp-install / skill-install dispatch
# --------------------------------------------------------------------------- #
def test_mcp_connect_without_the_extra(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastmcp", None)
    rc, out, err = run("mcp-connect")
    assert (rc, out) == (1, "")
    assert err.startswith("The MCP bridge requires the 'mcp' extra.")


def test_mcp_connect_startup_error_stays_off_stdout(run: Any) -> None:
    os.environ["DATAQUERY_BASE_URL"] = "https://dq.example.com"
    os.environ["DATAQUERY_TIMEOUT"] = "abc"
    rc, out, err = run("mcp-connect", "--url", "https://mcp.example.com/mcp")
    assert rc == 1
    assert out == ""
    assert "Invalid float value for DATAQUERY_TIMEOUT: abc" in err


def test_mcp_connect_missing_env_file_stays_off_stdout(run: Any, tmp_path: Path) -> None:
    missing = tmp_path / "nope.env"
    rc, out, err = run("--env-file", str(missing), "mcp-connect")
    assert (rc, out) == (1, "")
    assert err == f"--env-file not found: {missing}\n"


def test_main_dispatches_mcp_install_and_skill_install(run: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_mcp_extra_installed", lambda: False)
    rc, _, err = run("mcp-install")
    assert rc == 1 and err.startswith("The MCP bridge needs the 'mcp' extra.")
    rc, _, err = run("skill-install", "--project-dir", "x")
    assert rc == 1 and err == "skill-install: --project-dir needs --scope project\n"
