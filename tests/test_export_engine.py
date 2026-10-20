"""Characterization and reliability tests for ``dataquery.export``.

The two public exporters flatten time-series and grid responses (Pydantic models
or raw dicts) into CSV, either on disk or returned as ``content`` for ``-``.
These tests pin the public API and today's output, and describe the correct
behaviour for each confirmed defect.
"""

from __future__ import annotations

import csv
import inspect
import io
import typing
from pathlib import Path
from typing import Any

import pytest

import dataquery.export as export
from dataquery.export import export_grid_csv, export_timeseries_csv
from dataquery.types.exceptions import DataQueryError
from dataquery.types.models import GridDataResponse, TimeSeriesResponse

TS_HEADER = [
    "date",
    "value",
    "instrument_id",
    "instrument_name",
    "attribute_id",
    "attribute_name",
    "expression",
    "label",
    "last_published",
    "group_id",
    "group_name",
]


def _rows(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def _read(path: Path) -> list[list[str]]:
    return _rows(path.read_text(encoding="utf-8"))


def _ts_response() -> TimeSeriesResponse:
    return TimeSeriesResponse.model_validate(
        {
            "items": 2,
            "instruments": [
                {
                    "item": 1,
                    "instrument-id": "I1",
                    "instrument-name": "Inst One",
                    "group": {"group-id": "G1", "group-name": "Group One"},
                    "attributes": [
                        {
                            "attribute-id": "A1",
                            "attribute-name": "Close",
                            "expression": "DB(I1,A1)",
                            "label": "close",
                            "last-published": "2024-01-03T00:00:00Z",
                            "time-series": [["20240101", 1.5], ["20240102", None]],
                        },
                        {
                            "attribute-id": "A2",
                            "expression": "DB(I1,A2)",
                            "time-series": [["20240101", 2.0]],
                        },
                    ],
                },
                {
                    "item": 2,
                    "instrument-id": "I2",
                    "instrument-name": "Inst Two",
                    "attributes": [{"attribute-id": "A1", "time-series": [["20240105", -3.25]]}],
                },
            ],
        }
    )


def _ts_dict(**attr: Any) -> dict[str, Any]:
    return {
        "instruments": [
            {
                "instrument-id": "I",
                "attributes": [{"attribute-id": "A", "time-series": [["20240101", 1.0]], **attr}],
            }
        ]
    }


def _grid_response() -> GridDataResponse:
    return GridDataResponse.model_validate(
        {
            "series": [
                {"expr": "DBGRID(A)", "records": [{"strike": 90, "vol": 0.2}, {"strike": 100, "vol": 0.25}]},
                {"expr": "DBGRID(B)", "records": [{"tenor": "1M", "strike": 95}]},
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def test_public_functions_are_unchanged() -> None:
    public = sorted(
        name
        for name, obj in vars(export).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == export.__name__
    )
    assert public == ["export_grid_csv", "export_timeseries_csv"]
    assert not hasattr(export, "__all__")


@pytest.mark.parametrize("fn", [export_timeseries_csv, export_grid_csv], ids=lambda f: f.__name__)
def test_public_signature_is_unchanged(fn: Any) -> None:
    params = [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]
    assert params == [
        ("response", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("output_path", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
    ]


@pytest.mark.parametrize("fn", [export_timeseries_csv, export_grid_csv], ids=lambda f: f.__name__)
def test_public_type_hints_are_unchanged(fn: Any) -> None:
    hints = typing.get_type_hints(fn)
    assert set(hints) == {"response", "output_path", "return"}
    assert hints["response"] is Any
    assert hints["output_path"] is str
    assert typing.get_origin(hints["return"]) is dict
    assert typing.get_args(hints["return"]) == (str, Any)


def test_cli_imports_the_exporters_lazily() -> None:
    source = Path(export.__file__).with_name("cli.py").read_text(encoding="utf-8")
    assert "from .export import export_grid_csv, export_timeseries_csv" in source


# --------------------------------------------------------------------------- #
# Time-series exporter: today's behaviour
# --------------------------------------------------------------------------- #
def test_timeseries_model_to_file(tmp_path: Path) -> None:
    out = tmp_path / "ts.csv"
    result = export_timeseries_csv(_ts_response(), str(out))

    assert result == {"path": str(out), "rows": 4}
    assert _read(out) == [
        TS_HEADER,
        [
            "20240101",
            "1.5",
            "I1",
            "Inst One",
            "A1",
            "Close",
            "DB(I1,A1)",
            "close",
            "2024-01-03T00:00:00Z",
            "G1",
            "Group One",
        ],
        [
            "20240102",
            "",
            "I1",
            "Inst One",
            "A1",
            "Close",
            "DB(I1,A1)",
            "close",
            "2024-01-03T00:00:00Z",
            "G1",
            "Group One",
        ],
        ["20240101", "2.0", "I1", "Inst One", "A2", "", "DB(I1,A2)", "", "", "G1", "Group One"],
        ["20240105", "-3.25", "I2", "Inst Two", "A1", "", "", "", "", "", ""],
    ]


def test_timeseries_file_uses_crlf_and_utf8(tmp_path: Path) -> None:
    out = tmp_path / "ts.csv"
    export_timeseries_csv(_ts_dict(label="Zürich €"), str(out))
    raw = out.read_bytes()
    assert raw.startswith(b"date,value,instrument_id")
    assert raw.count(b"\r\n") == 2
    assert "Zürich €".encode() in raw


def test_timeseries_overwrites_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "ts.csv"
    out.write_text("old content\n" * 10)
    export_timeseries_csv(_ts_dict(), str(out))
    assert "old content" not in out.read_text()
    assert len(_read(out)) == 2


def test_timeseries_stdout_returns_content() -> None:
    result = export_timeseries_csv(_ts_response(), "-")
    assert set(result) == {"path", "rows", "content"}
    assert result["path"] == "stdout"
    assert result["rows"] == 4
    assert result["content"].startswith(",".join(TS_HEADER) + "\r\n")
    assert len(_rows(result["content"])) == 5


def test_timeseries_raw_dict_snake_case_fallbacks() -> None:
    data = {
        "instruments": [
            {
                "instrument_id": "I",
                "instrument_name": "N",
                "group": {"group_id": "G", "group_name": "GN"},
                "attributes": [
                    {
                        "attribute_id": "A",
                        "attribute_name": "AN",
                        "last_published": "LP",
                        "time_series": [["20240101", 7]],
                    }
                ],
            }
        ]
    }
    content = export_timeseries_csv(data, "-")["content"]
    assert _rows(content)[1] == ["20240101", "7", "I", "N", "A", "AN", "", "", "LP", "G", "GN"]


def test_timeseries_kebab_empty_string_falls_back_to_snake_key() -> None:
    data = {
        "instruments": [
            {
                "instrument-id": "",
                "instrument_id": "SNAKE",
                "attributes": [_ts_dict()["instruments"][0]["attributes"][0]],
            }
        ]
    }
    assert _rows(export_timeseries_csv(data, "-")["content"])[1][2] == "SNAKE"


def test_timeseries_points_shorter_than_two_or_not_lists_are_skipped() -> None:
    data = _ts_dict(**{"time-series": [["20240101"], [], "20240102,1", None, ["20240103", 3.0, "extra"]]})
    result = export_timeseries_csv(data, "-")
    assert result["rows"] == 1
    assert _rows(result["content"])[1][:2] == ["20240103", "3.0"]


def test_timeseries_instrument_without_attributes_or_group() -> None:
    data = {
        "instruments": [
            {"instrument-id": "EMPTY", "attributes": None, "group": None},
            _ts_dict()["instruments"][0],
        ]
    }
    result = export_timeseries_csv(data, "-")
    assert result["rows"] == 1


def test_model_coerces_numeric_time_series_to_float() -> None:
    """``Attribute.time_series`` (model behaviour, not this module) turns 5 into 5.0."""
    resp = TimeSeriesResponse.model_validate(
        {"instruments": [{"item": 1, "attributes": [{"time-series": [[20240101, 5]]}]}]}
    )
    assert _rows(export_timeseries_csv(resp, "-")["content"])[1][:2] == ["20240101.0", "5.0"]


@pytest.mark.parametrize(
    "payload", [{}, {"instruments": []}, {"instruments": None}, TimeSeriesResponse.model_validate({})]
)
def test_timeseries_without_instruments_raises(payload: Any) -> None:
    with pytest.raises(DataQueryError, match=r"^No instruments found in the response to export\.$"):
        export_timeseries_csv(payload, "-")


def test_timeseries_without_points_raises_plain_message() -> None:
    with pytest.raises(DataQueryError, match=r"^Response contains instruments but no time-series data points\.$"):
        export_timeseries_csv(_ts_dict(**{"time-series": []}), "-")


def test_timeseries_without_points_lists_server_messages_by_source() -> None:
    data = {
        "instruments": [
            {
                "instrument-id": "I",
                "attributes": [
                    {"expression": "DB(X)", "attribute-id": "A", "message": "by expression"},
                    {"attribute-id": "A2", "message": "by attribute"},
                    {"message": "by instrument"},
                    {"message": ""},
                ],
            }
        ]
    }
    with pytest.raises(DataQueryError) as info:
        export_timeseries_csv(data, "-")
    assert str(info.value) == (
        "Response contains instruments but no time-series data points. "
        "Server messages: DB(X): by expression; A2: by attribute; I: by instrument"
    )


@pytest.mark.parametrize("fn", [export_timeseries_csv, export_grid_csv], ids=lambda f: f.__name__)
@pytest.mark.parametrize("payload", [None, [1, 2], "text", 42])
def test_non_dict_payload_is_rejected(fn: Any, payload: Any) -> None:
    with pytest.raises(DataQueryError, match="response payload is not a dict or Pydantic model"):
        fn(payload, "-")


def test_timeseries_write_failure_is_wrapped(tmp_path: Path) -> None:
    with pytest.raises(DataQueryError, match=f"^Failed to write CSV to {tmp_path}: ") as info:
        export_timeseries_csv(_ts_dict(), str(tmp_path))  # a directory
    assert isinstance(info.value.__cause__, OSError)


def test_timeseries_has_no_messages_key_when_server_sent_none() -> None:
    assert "messages" not in export_timeseries_csv(_ts_dict(), "-")


# --------------------------------------------------------------------------- #
# Grid exporter: today's behaviour
# --------------------------------------------------------------------------- #
def test_grid_model_to_file_unions_columns_in_first_seen_order(tmp_path: Path) -> None:
    out = tmp_path / "grid.csv"
    result = export_grid_csv(_grid_response(), str(out))

    assert result == {"path": str(out), "rows": 3}
    assert _read(out) == [
        ["expression", "strike", "vol", "tenor"],
        ["DBGRID(A)", "90", "0.2", ""],
        ["DBGRID(A)", "100", "0.25", ""],
        ["DBGRID(B)", "95", "", "1M"],
    ]


def test_grid_stdout_returns_content() -> None:
    result = export_grid_csv(_grid_response(), "-")
    assert set(result) == {"path", "rows", "content"}
    assert result["path"] == "stdout"
    assert result["rows"] == 3
    assert result["content"].startswith("expression,strike,vol,tenor\r\n")


def test_grid_raw_dict_expression_key_fallback_and_non_dict_records_skipped() -> None:
    data = {"series": [{"expression": "E", "records": [{"a": 1}, "junk", None, [1, 2]]}, {"records": [{}]}]}
    result = export_grid_csv(data, "-")
    assert result["rows"] == 2
    assert _rows(result["content"]) == [["expression", "a"], ["E", "1"], ["", ""]]


def test_grid_nested_values_are_written_as_python_repr() -> None:
    """Finding 10 (listed, not fixed): dict/list cells are ``str()``, not JSON."""
    data = {"series": [{"expr": "X", "records": [{"v": {"a": 1}, "l": [1, 2]}]}]}
    assert _rows(export_grid_csv(data, "-")["content"])[1] == ["X", "{'a': 1}", "[1, 2]"]


@pytest.mark.parametrize("payload", [{}, {"series": []}, {"series": None}, GridDataResponse.model_validate({})])
def test_grid_without_series_raises(payload: Any) -> None:
    with pytest.raises(DataQueryError, match=r"^No grid series found in the response to export\.$"):
        export_grid_csv(payload, "-")


@pytest.mark.parametrize("records", [None, [], ["junk"]])
def test_grid_without_records_raises(records: Any) -> None:
    with pytest.raises(DataQueryError, match=r"^Grid series found but no records to export\.$"):
        export_grid_csv({"series": [{"expr": "X", "records": records}]}, "-")


def test_grid_write_failure_is_wrapped(tmp_path: Path) -> None:
    with pytest.raises(DataQueryError, match=f"^Failed to write CSV to {tmp_path}: ") as info:
        export_grid_csv(_grid_response(), str(tmp_path))
    assert isinstance(info.value.__cause__, OSError)


def test_grid_has_no_messages_key_when_server_sent_none() -> None:
    assert "messages" not in export_grid_csv(_grid_response(), "-")


# --------------------------------------------------------------------------- #
# Defects: each test describes the correct behaviour
# --------------------------------------------------------------------------- #
def test_f1_falsy_values_are_kept() -> None:
    data = {
        "instruments": [
            {
                "instrument-id": 0,
                "instrument-name": 0,
                "group": {"group-id": 0, "group-name": False},
                "attributes": [{"attribute-id": 0, "last-published": 0, "time-series": [["20240101", 0]]}],
            }
        ]
    }
    row = _rows(export_timeseries_csv(data, "-")["content"])[1]
    assert row == ["20240101", "0", "0", "0", "0", "", "", "", "0", "0", "False"]
    grid = export_grid_csv({"series": [{"expr": 0, "records": [{"v": 1}]}]}, "-")
    assert _rows(grid["content"])[1] == ["0", "1"]


@pytest.mark.parametrize(
    "payload",
    [
        {"errorCode": "E1", "errorMessage": "Bad expr", "series": []},
        GridDataResponse.model_validate({"errorCode": "E1", "errorMessage": "Bad expr"}),
    ],
    ids=["dict", "model"],
)
def test_f2_grid_response_error_is_reported(payload: Any) -> None:
    with pytest.raises(DataQueryError) as info:
        export_grid_csv(payload, "-")
    assert str(info.value) == "No grid series found in the response to export. Server error: [E1] Bad expr"


def test_f3_grid_series_errors_are_reported() -> None:
    payload = GridDataResponse.model_validate(
        {
            "series": [
                {"expr": "X", "errorCode": "E2", "errorMessage": "Invalid grid"},
                {"expr": "Y", "errorMessage": "No data", "records": []},
            ]
        }
    )
    with pytest.raises(DataQueryError) as info:
        export_grid_csv(payload, "-")
    assert str(info.value) == (
        "Grid series found but no records to export. Server errors: X: [E2] Invalid grid; Y: No data"
    )


@pytest.mark.parametrize("key", ["description", "message"])
def test_f4_timeseries_info_is_reported(key: str) -> None:
    payload = TimeSeriesResponse.model_validate({"instruments": [], "info": {"code": 204, key: "No content available"}})
    with pytest.raises(DataQueryError) as info:
        export_timeseries_csv(payload, "-")
    assert str(info.value) == "No instruments found in the response to export. Server info: [204] No content available"


def test_f5_non_dict_entries_are_skipped() -> None:
    good_attr = {"attribute-id": "A", "time-series": [["20240101", 1.0]]}
    data = {
        "instruments": [
            "junk",
            None,
            {"instrument-id": "S", "attributes": "abc"},
            {"instrument-id": "I", "group": "G", "attributes": ["junk", 3, good_attr]},
        ]
    }
    result = export_timeseries_csv(data, "-")
    assert result["rows"] == 1
    assert _rows(result["content"])[1] == ["20240101", "1.0", "I", "", "A", "", "", "", "", "", ""]

    grid = export_grid_csv({"series": ["junk", None, {"expr": "X", "records": [{"v": 1}]}]}, "-")
    assert grid["rows"] == 1


def test_f5_only_non_dict_entries_raise_dataquery_error() -> None:
    with pytest.raises(DataQueryError, match="^No instruments found"):
        export_timeseries_csv({"instruments": {"a": 1}}, "-")
    with pytest.raises(DataQueryError, match="no time-series data points"):
        export_timeseries_csv({"instruments": ["junk"]}, "-")
    with pytest.raises(DataQueryError, match="no records to export"):
        export_grid_csv({"series": ["junk"]}, "-")


def test_f6_record_expression_does_not_clobber_series_expr() -> None:
    data = {"series": [{"expr": "SERIES", "records": [{"expression": "REC", "v": 1}, {"v": 2}]}]}
    assert _rows(export_grid_csv(data, "-")["content"]) == [
        ["expression", "record_expression", "v"],
        ["SERIES", "REC", "1"],
        ["SERIES", "", "2"],
    ]


def test_f7_timeseries_messages_are_returned_with_rows(tmp_path: Path) -> None:
    data = {
        "instruments": [
            {
                "instrument-id": "I",
                "attributes": [
                    {"attribute-id": "A", "time-series": [["20240101", 1.0]]},
                    {"attribute-id": "B", "expression": "DB(B)", "message": "Expression syntax error!"},
                ],
            }
        ]
    }
    out = tmp_path / "ts.csv"
    assert export_timeseries_csv(data, str(out)) == {
        "path": str(out),
        "rows": 1,
        "messages": ["DB(B): Expression syntax error!"],
    }
    assert export_timeseries_csv(data, "-")["messages"] == ["DB(B): Expression syntax error!"]


def test_f7_grid_errors_are_returned_with_rows() -> None:
    data = {
        "series": [
            {"expr": "X", "records": [{"v": 1}]},
            {"expr": "Y", "errorCode": "E2", "errorMessage": "Invalid grid"},
        ]
    }
    result = export_grid_csv(data, "-")
    assert result["rows"] == 1
    assert result["messages"] == ["Y: [E2] Invalid grid"]


def test_f8_missing_parent_directories_are_created(tmp_path: Path) -> None:
    ts_out = tmp_path / "a" / "b" / "ts.csv"
    grid_out = tmp_path / "c" / "grid.csv"
    assert export_timeseries_csv(_ts_dict(), str(ts_out))["rows"] == 1
    assert export_grid_csv(_grid_response(), str(grid_out))["rows"] == 3
    assert len(_read(ts_out)) == 2
    assert len(_read(grid_out)) == 4


@pytest.mark.parametrize("which", ["timeseries", "grid"])
def test_f9_unencodable_value_raises_dataquery_error_and_keeps_file(tmp_path: Path, which: str) -> None:
    out = tmp_path / "out.csv"
    out.write_text("previous export\n", encoding="utf-8")
    if which == "timeseries":
        call = lambda: export_timeseries_csv(_ts_dict(label="bad \ud800"), str(out))  # noqa: E731
    else:
        call = lambda: export_grid_csv({"series": [{"expr": "X", "records": [{"v": "bad \ud800"}]}]}, str(out))  # noqa: E731
    with pytest.raises(DataQueryError, match=f"^Failed to write CSV to {out}: ") as info:
        call()
    assert isinstance(info.value.__cause__, UnicodeEncodeError)
    assert out.read_text(encoding="utf-8") == "previous export\n"
