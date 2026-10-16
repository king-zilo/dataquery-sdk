"""Unit tests for CSV export of time-series and grid-data responses (dataquery/export.py)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, List

import pytest

from dataquery import export
from dataquery.export import _to_dict, export_grid_csv, export_timeseries_csv
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


def _read_csv(path: Path) -> List[List[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _read_dicts(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ts_model(instruments: List[Dict[str, Any]]) -> TimeSeriesResponse:
    return TimeSeriesResponse.model_validate({"links": [{"self": "/x"}], "instruments": instruments})


def _two_instrument_model() -> TimeSeriesResponse:
    return _ts_model(
        [
            {
                "item": 1,
                "instrument-id": "INST-A",
                "instrument-name": "Alpha",
                "group": {"group-id": "G1", "group-name": "Group One"},
                "attributes": [
                    {
                        "attribute-id": "MID",
                        "attribute-name": "Mid",
                        "expression": "DB(A,MID)",
                        "label": "A mid",
                        "last-published": "2024-01-03T10:00:00Z",
                        "time-series": [["20240102", 1.5], ["20240101", 1.25]],
                    },
                    {
                        "attribute-id": "BID",
                        "attribute-name": "Bid",
                        "expression": "DB(A,BID)",
                        "time-series": [["20240101", 1.0]],
                    },
                ],
            },
            {
                "item": 2,
                "instrument-id": "INST-B",
                "instrument-name": "Beta",
                "attributes": [
                    {
                        "attribute-id": "MID",
                        "expression": "DB(B,MID)",
                        "time-series": [["20240101", 7.0]],
                    }
                ],
            },
        ]
    )


# --------------------------------------------------------------------------- _to_dict


def test_to_dict_dumps_pydantic_models_with_kebab_case_aliases():
    model = _ts_model([{"item": 1, "instrument-id": "X", "attributes": []}])

    data = _to_dict(model)

    assert isinstance(data, dict)
    inst = data["instruments"][0]
    assert inst["instrument-id"] == "X"
    assert "instrument_id" not in inst
    assert data["page-size"] is None
    assert data["links"] == [{"self": "/x", "next": None}]


def test_to_dict_returns_the_same_dict_object_unchanged():
    payload = {"instruments": [{"instrument-id": "X"}]}

    assert _to_dict(payload) is payload


def test_to_dict_uses_any_object_exposing_model_dump_with_by_alias():
    calls: List[Dict[str, Any]] = []

    class DuckModel:
        def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
            calls.append(kwargs)
            return {"series": []}

    assert _to_dict(DuckModel()) == {"series": []}
    assert calls == [{"by_alias": True}]


@pytest.mark.parametrize("payload", [None, [], [{"instrument-id": "X"}], "instruments", 42, ("a", "b")])
def test_to_dict_rejects_non_dict_non_model_payloads(payload):
    with pytest.raises(DataQueryError, match="response payload is not a dict or Pydantic model"):
        _to_dict(payload)


@pytest.mark.parametrize("exporter", [export_timeseries_csv, export_grid_csv])
def test_exporters_reject_unsupported_payloads_before_touching_the_filesystem(exporter, tmp_path):
    out = tmp_path / "out.csv"

    with pytest.raises(DataQueryError, match="not a dict or Pydantic model"):
        exporter(["not", "a", "response"], str(out))
    assert not out.exists()


# --------------------------------------------------------------------------- time series: layout


def test_timeseries_model_writes_header_and_rows_in_response_order(tmp_path):
    out = tmp_path / "ts.csv"

    summary = export_timeseries_csv(_two_instrument_model(), str(out))

    assert summary == {"path": str(out), "rows": 4}
    rows = _read_csv(out)
    assert rows[0] == TS_HEADER
    # Instrument order, then attribute order, then point order; dates are not re-sorted.
    assert rows[1:] == [
        [
            "20240102",
            "1.5",
            "INST-A",
            "Alpha",
            "MID",
            "Mid",
            "DB(A,MID)",
            "A mid",
            "2024-01-03T10:00:00Z",
            "G1",
            "Group One",
        ],
        [
            "20240101",
            "1.25",
            "INST-A",
            "Alpha",
            "MID",
            "Mid",
            "DB(A,MID)",
            "A mid",
            "2024-01-03T10:00:00Z",
            "G1",
            "Group One",
        ],
        ["20240101", "1.0", "INST-A", "Alpha", "BID", "Bid", "DB(A,BID)", "", "", "G1", "Group One"],
        ["20240101", "7.0", "INST-B", "Beta", "MID", "", "DB(B,MID)", "", "", "", ""],
    ]


def test_timeseries_file_bytes_use_csv_module_crlf_line_endings(tmp_path):
    out = tmp_path / "ts.csv"
    model = _ts_model(
        [{"item": 1, "instrument-id": "A", "attributes": [{"expression": "E", "time-series": [["20240101", 2.0]]}]}]
    )

    export_timeseries_csv(model, str(out))

    assert out.read_bytes() == (",".join(TS_HEADER).encode() + b"\r\n" + b"20240101,2.0,A,,,,E,,,,\r\n")


def test_timeseries_raw_dict_with_kebab_case_keys(tmp_path):
    out = tmp_path / "ts.csv"
    payload = {
        "instruments": [
            {
                "instrument-id": "A",
                "instrument-name": "Alpha",
                "group": {"group-id": "G", "group-name": "Grp"},
                "attributes": [
                    {
                        "attribute-id": "AT",
                        "attribute-name": "Attr",
                        "expression": "DB(A)",
                        "label": "lbl",
                        "last-published": "2024-02-01",
                        "time-series": [["20240101", 3]],
                    }
                ],
            }
        ]
    }

    summary = export_timeseries_csv(payload, str(out))

    assert summary == {"path": str(out), "rows": 1}
    # Raw dicts bypass model coercion, so an int value is written as-is.
    assert _read_dicts(out) == [
        {
            "date": "20240101",
            "value": "3",
            "instrument_id": "A",
            "instrument_name": "Alpha",
            "attribute_id": "AT",
            "attribute_name": "Attr",
            "expression": "DB(A)",
            "label": "lbl",
            "last_published": "2024-02-01",
            "group_id": "G",
            "group_name": "Grp",
        }
    ]


def test_timeseries_raw_dict_with_snake_case_keys_falls_back(tmp_path):
    out = tmp_path / "ts.csv"
    payload = {
        "instruments": [
            {
                "instrument_id": "A",
                "instrument_name": "Alpha",
                "group": {"group_id": "G", "group_name": "Grp"},
                "attributes": [
                    {
                        "attribute_id": "AT",
                        "attribute_name": "Attr",
                        "last_published": "2024-02-01",
                        "time_series": [["20240101", 1.0]],
                    }
                ],
            }
        ]
    }

    export_timeseries_csv(payload, str(out))

    row = _read_dicts(out)[0]
    assert (row["instrument_id"], row["instrument_name"]) == ("A", "Alpha")
    assert (row["attribute_id"], row["attribute_name"], row["last_published"]) == ("AT", "Attr", "2024-02-01")
    assert (row["group_id"], row["group_name"]) == ("G", "Grp")
    assert row["expression"] == "" and row["label"] == ""


def test_timeseries_kebab_case_key_wins_over_snake_case(tmp_path):
    out = tmp_path / "ts.csv"
    payload = {
        "instruments": [
            {
                "instrument-id": "KEBAB",
                "instrument_id": "SNAKE",
                "attributes": [{"time-series": [["d1", 1]], "time_series": [["d2", 2]]}],
            }
        ]
    }

    export_timeseries_csv(payload, str(out))

    assert [(r["date"], r["instrument_id"]) for r in _read_dicts(out)] == [("d1", "KEBAB")]


def test_timeseries_model_coerces_int_values_and_numeric_dates_to_float(tmp_path):
    out = tmp_path / "ts.csv"
    model = _ts_model(
        [{"item": 1, "instrument-id": "A", "attributes": [{"time-series": [[20240101, 5], ["20240102", 0]]}]}]
    )

    export_timeseries_csv(model, str(out))

    assert [(r["date"], r["value"]) for r in _read_dicts(out)] == [("20240101.0", "5.0"), ("20240102", "0.0")]


def test_timeseries_null_values_become_empty_cells_and_rows_are_kept(tmp_path):
    out = tmp_path / "ts.csv"
    model = _ts_model(
        [
            {
                "item": 1,
                "instrument-id": None,
                "attributes": [{"expression": "VOL(30,DB(A))", "time-series": [["20240101", None], ["20240102", 2.5]]}],
            }
        ]
    )

    summary = export_timeseries_csv(model, str(out))

    assert summary["rows"] == 2
    rows = _read_dicts(out)
    assert [(r["date"], r["value"]) for r in rows] == [("20240101", ""), ("20240102", "2.5")]
    # Null instrument id (computed expression) and None label/group become empty cells, not "None".
    assert all(r["instrument_id"] == "" and r["label"] == "" and r["group_id"] == "" for r in rows)
    assert "None" not in out.read_text(encoding="utf-8")


def test_timeseries_skips_short_and_non_list_points_and_ignores_extra_elements(tmp_path):
    out = tmp_path / "ts.csv"
    payload = {
        "instruments": [
            {
                "instrument-id": "A",
                "attributes": [
                    {
                        "time-series": [
                            ["only-date"],
                            [],
                            ("20240101", 1.0),
                            {"date": "20240101", "value": 1.0},
                            "20240101,1.0",
                            ["20240103", 3.0, "extra"],
                        ]
                    }
                ],
            }
        ]
    }

    summary = export_timeseries_csv(payload, str(out))

    assert summary["rows"] == 1
    rows = _read_csv(out)
    assert rows[0] == TS_HEADER
    assert rows[1][:3] == ["20240103", "3.0", "A"]
    assert len(rows) == 2


def test_timeseries_none_attributes_and_none_series_are_tolerated(tmp_path):
    out = tmp_path / "ts.csv"
    payload = {
        "instruments": [
            {"instrument-id": "NOATTR", "attributes": None},
            {"instrument-id": "NOKEY"},
            {"instrument-id": "A", "group": None, "attributes": [{"time-series": None}, {"time-series": [["d", 1]]}]},
        ]
    }

    summary = export_timeseries_csv(payload, str(out))

    assert summary == {"path": str(out), "rows": 1}
    assert [r["instrument_id"] for r in _read_dicts(out)] == ["A"]


def test_timeseries_non_ascii_text_is_written_as_utf8(tmp_path):
    out = tmp_path / "ts.csv"
    model = _ts_model(
        [
            {
                "item": 1,
                "instrument-id": "A",
                "instrument-name": 'Zürich € Index, "quoted"',
                "attributes": [{"time-series": [["20240101", 1.0]]}],
            }
        ]
    )

    export_timeseries_csv(model, str(out))

    assert _read_dicts(out)[0]["instrument_name"] == 'Zürich € Index, "quoted"'
    assert '"Zürich € Index, ""quoted"""'.encode("utf-8") in out.read_bytes()


# --------------------------------------------------------------------------- time series: errors


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"instruments": None},
        {"instruments": []},
        TimeSeriesResponse.model_validate({"instruments": []}),
        TimeSeriesResponse.model_validate({"info": {"code": "204", "description": "No content"}}),
    ],
)
def test_timeseries_without_instruments_raises(payload, tmp_path):
    out = tmp_path / "ts.csv"

    with pytest.raises(DataQueryError, match=r"^No instruments found in the response to export\.$"):
        export_timeseries_csv(payload, str(out))
    assert not out.exists()


def test_timeseries_instruments_without_points_and_without_messages_has_no_detail(tmp_path):
    out = tmp_path / "ts.csv"
    model = _ts_model([{"item": 1, "instrument-id": "A", "attributes": [{"expression": "E", "time-series": []}]}])

    with pytest.raises(DataQueryError) as excinfo:
        export_timeseries_csv(model, str(out))

    assert str(excinfo.value) == "Response contains instruments but no time-series data points."
    assert not out.exists()


def test_timeseries_no_points_error_joins_every_message_with_its_best_label(tmp_path):
    model = _ts_model(
        [
            {
                "item": 1,
                "instrument-id": "INST-A",
                "attributes": [
                    {"expression": "DB(A,X)", "attribute-id": "X", "message": "bad expr"},
                    {"attribute-id": "Y", "message": "no access"},
                    {"message": "unknown"},
                    {"expression": "DB(A,Z)", "message": None, "time-series": []},
                ],
            },
            {"item": 2, "attributes": [{"message": "orphan"}]},
        ]
    )

    with pytest.raises(DataQueryError) as excinfo:
        export_timeseries_csv(model, str(tmp_path / "ts.csv"))

    # Label preference: expression, then attribute id, then instrument id (empty when missing).
    assert str(excinfo.value) == (
        "Response contains instruments but no time-series data points."
        " Server messages: DB(A,X): bad expr; Y: no access; INST-A: unknown; : orphan"
    )


def test_timeseries_messages_do_not_block_export_when_other_attributes_have_data(tmp_path):
    out = tmp_path / "ts.csv"
    model = _ts_model(
        [
            {
                "item": 1,
                "instrument-id": "A",
                "attributes": [
                    {"expression": "DB(A,BAD)", "message": "Expression syntax error!"},
                    {"expression": "DB(A,OK)", "time-series": [["20240101", 1.0]]},
                ],
            }
        ]
    )

    summary = export_timeseries_csv(model, str(out))

    # The summary only carries path and row count; server messages are not part of it.
    assert summary == {"path": str(out), "rows": 1}
    assert [r["expression"] for r in _read_dicts(out)] == ["DB(A,OK)"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "export.py:35,41 call .get() on each instrument/attribute; a malformed raw-dict payload "
        "raises AttributeError instead of the DataQueryError the CLI's main() handler catches"
    ),
)
@pytest.mark.parametrize(
    "payload",
    [{"instruments": ["INST-A"]}, {"instruments": [{"instrument-id": "A", "attributes": ["DB(A)"]}]}],
    ids=["str-instrument", "str-attribute"],
)
def test_timeseries_malformed_entries_raise_dataquery_error(payload, tmp_path):
    with pytest.raises(DataQueryError):
        export_timeseries_csv(payload, str(tmp_path / "ts.csv"))


# --------------------------------------------------------------------------- time series: output target


def test_timeseries_stdout_returns_content_and_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _ts_model(
        [
            {
                "item": 1,
                "instrument-id": "A",
                "attributes": [{"expression": "E", "time-series": [["d1", 1.0], ["d2", None]]}],
            }
        ]
    )

    summary = export_timeseries_csv(model, "-")

    assert set(summary) == {"path", "rows", "content"}
    assert summary["path"] == "stdout"
    assert summary["rows"] == 2
    assert summary["content"] == ",".join(TS_HEADER) + "\r\nd1,1.0,A,,,,E,,,,\r\nd2,,A,,,,E,,,,\r\n"
    assert list(csv.reader(io.StringIO(summary["content"])))[0] == TS_HEADER
    assert list(tmp_path.iterdir()) == []


def test_timeseries_overwrites_an_existing_file(tmp_path):
    out = tmp_path / "ts.csv"
    out.write_text("stale,content\nthat,is\nlonger,than\nthe,export\n" * 10, encoding="utf-8")
    model = _ts_model([{"item": 1, "instrument-id": "A", "attributes": [{"time-series": [["d", 1.0]]}]}])

    export_timeseries_csv(model, str(out))

    rows = _read_csv(out)
    assert rows == [TS_HEADER, ["d", "1.0", "A", "", "", "", "", "", "", "", ""]]


def test_timeseries_does_not_create_missing_parent_directories(tmp_path):
    out = tmp_path / "missing" / "dir" / "ts.csv"
    model = _ts_model([{"item": 1, "instrument-id": "A", "attributes": [{"time-series": [["d", 1.0]]}]}])

    with pytest.raises(DataQueryError, match=r"^Failed to write CSV to .*missing/dir/ts\.csv: ") as excinfo:
        export_timeseries_csv(model, str(out))

    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert not (tmp_path / "missing").exists()


def test_timeseries_directory_as_output_path_raises_dataquery_error(tmp_path):
    model = _ts_model([{"item": 1, "instrument-id": "A", "attributes": [{"time-series": [["d", 1.0]]}]}])

    with pytest.raises(DataQueryError, match="Failed to write CSV to") as excinfo:
        export_timeseries_csv(model, str(tmp_path))

    assert isinstance(excinfo.value.__cause__, OSError)


def test_timeseries_open_uses_text_mode_utf8_and_no_newline_translation(tmp_path, monkeypatch):
    calls: List[Any] = []
    real_open = open

    def recording_open(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return real_open(*args, **kwargs)

    monkeypatch.setattr("builtins.open", recording_open)
    out = tmp_path / "ts.csv"
    model = _ts_model([{"item": 1, "instrument-id": "A", "attributes": [{"time-series": [["d", 1.0]]}]}])

    export_timeseries_csv(model, str(out))

    assert calls == [((str(out), "w"), {"newline": "", "encoding": "utf-8"})]


# --------------------------------------------------------------------------- grid


def _grid_model(series: List[Dict[str, Any]]) -> GridDataResponse:
    return GridDataResponse.model_validate({"series": series})


def test_grid_model_unions_columns_in_first_seen_order_and_fills_gaps(tmp_path):
    out = tmp_path / "grid.csv"
    model = _grid_model(
        [
            {
                "expr": "DBGRID(A)",
                "records": [{"date": "20240101", "strike": 100, "vol": 0.2}, {"date": "20240102", "vol": 0.25}],
            },
            {"expr": "DBGRID(B)", "records": [{"tenor": "1M", "date": "20240101", "vol": None}]},
        ]
    )

    summary = export_grid_csv(model, str(out))

    assert summary == {"path": str(out), "rows": 3}
    assert _read_csv(out) == [
        ["expression", "date", "strike", "vol", "tenor"],
        ["DBGRID(A)", "20240101", "100", "0.2", ""],
        ["DBGRID(A)", "20240102", "", "0.25", ""],
        ["DBGRID(B)", "20240101", "", "", "1M"],
    ]


def test_grid_file_bytes_are_exact(tmp_path):
    out = tmp_path / "grid.csv"

    export_grid_csv(_grid_model([{"expr": "X", "records": [{"a": 1, "b": "two, three"}]}]), str(out))

    assert out.read_bytes() == b'expression,a,b\r\nX,1,"two, three"\r\n'


def test_grid_raw_dict_uses_expression_fallback_and_skips_non_dict_records(tmp_path):
    out = tmp_path / "grid.csv"
    payload = {
        "series": [
            {"expression": "LEGACY", "records": [{"k": 1}, ["not", "a", "dict"], "text", None, {"k": 2}]},
            {"expr": "", "expression": "FALLBACK", "records": [{"k": 3}]},
            {"records": [{"k": 4}]},
        ]
    }

    summary = export_grid_csv(payload, str(out))

    assert summary["rows"] == 4
    assert _read_csv(out) == [["expression", "k"], ["LEGACY", "1"], ["LEGACY", "2"], ["FALLBACK", "3"], ["", "4"]]


def test_grid_series_without_records_are_skipped_when_others_have_records(tmp_path):
    out = tmp_path / "grid.csv"
    model = _grid_model(
        [
            {"expr": "EMPTY", "records": []},
            {"expr": "NONE"},
            {"expr": "FAILED", "errorCode": "E1", "errorMessage": "bad"},
            {"expr": "OK", "records": [{"v": 1}]},
        ]
    )

    summary = export_grid_csv(model, str(out))

    assert summary == {"path": str(out), "rows": 1}
    assert _read_csv(out) == [["expression", "v"], ["OK", "1"]]


def test_grid_record_key_named_expression_overrides_the_series_expr(tmp_path):
    out = tmp_path / "grid.csv"

    export_grid_csv(_grid_model([{"expr": "SERIES", "records": [{"expression": "RECORD", "v": 1}]}]), str(out))

    assert _read_csv(out) == [["expression", "v"], ["RECORD", "1"]]


def test_grid_does_not_mutate_the_input_records():
    records = [{"v": 1}]
    payload = {"series": [{"expr": "X", "records": records}]}

    export_grid_csv(payload, "-")

    assert records == [{"v": 1}]


@pytest.mark.parametrize(
    "payload",
    [{}, {"series": None}, {"series": []}, GridDataResponse.model_validate({})],
)
def test_grid_without_series_raises(payload, tmp_path):
    out = tmp_path / "grid.csv"

    with pytest.raises(DataQueryError, match=r"^No grid series found in the response to export\.$"):
        export_grid_csv(payload, str(out))
    assert not out.exists()


@pytest.mark.parametrize(
    "payload",
    [
        {"series": [{"expr": "X"}]},
        {"series": [{"expr": "X", "records": None}, {"expr": "Y", "records": []}]},
        {"series": [{"expr": "X", "records": [["a"], "b"]}]},
    ],
)
def test_grid_series_without_dict_records_raises(payload, tmp_path):
    out = tmp_path / "grid.csv"

    with pytest.raises(DataQueryError, match=r"^Grid series found but no records to export\.$"):
        export_grid_csv(payload, str(out))
    assert not out.exists()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "export.py:115-126 ignores the series errorMessage; unlike the time-series exporter it raises a "
        "generic 'no records' error, and cli.cmd_grid_data exports before printing its summary, "
        "so the server's reason is lost"
    ),
)
def test_grid_no_records_error_includes_series_error_message(tmp_path):
    model = _grid_model([{"expr": "DBGRID(BAD)", "errorCode": "E42", "errorMessage": "Invalid grid expression"}])

    with pytest.raises(DataQueryError, match="Invalid grid expression"):
        export_grid_csv(model, str(tmp_path / "grid.csv"))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "export.py:108-112 ignores the response-level errorMessage of a failed grid request and raises "
        "'No grid series found', so the server's reason never reaches the user"
    ),
)
def test_grid_no_series_error_includes_response_error_message(tmp_path):
    model = GridDataResponse.model_validate({"errorCode": "401", "errorMessage": "Entitlement missing"})

    with pytest.raises(DataQueryError, match="Entitlement missing"):
        export_grid_csv(model, str(tmp_path / "grid.csv"))


def test_grid_stdout_returns_content_and_writes_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = _grid_model([{"expr": "X", "records": [{"a": 1}, {"b": None}]}])

    summary = export_grid_csv(model, "-")

    assert summary == {"path": "stdout", "rows": 2, "content": "expression,a,b\r\nX,1,\r\nX,,\r\n"}
    assert list(tmp_path.iterdir()) == []


def test_grid_does_not_create_missing_parent_directories(tmp_path):
    out = tmp_path / "nope" / "grid.csv"

    with pytest.raises(DataQueryError, match=r"^Failed to write CSV to .*nope/grid\.csv: ") as excinfo:
        export_grid_csv(_grid_model([{"expr": "X", "records": [{"a": 1}]}]), str(out))

    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert not out.parent.exists()


def test_grid_write_failure_after_open_is_wrapped(tmp_path, monkeypatch):
    class FailingWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def writeheader(self) -> None:
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(export.csv, "DictWriter", FailingWriter)
    out = tmp_path / "grid.csv"

    with pytest.raises(DataQueryError, match="No space left on device") as excinfo:
        export_grid_csv(_grid_model([{"expr": "X", "records": [{"a": 1}]}]), str(out))

    assert str(excinfo.value).startswith(f"Failed to write CSV to {out}: ")
    assert isinstance(excinfo.value.__cause__, OSError)


def test_grid_overwrites_an_existing_file(tmp_path):
    out = tmp_path / "grid.csv"
    out.write_text("old\n" * 50, encoding="utf-8")

    export_grid_csv(_grid_model([{"expr": "X", "records": [{"a": 1}]}]), str(out))

    assert _read_csv(out) == [["expression", "a"], ["X", "1"]]
