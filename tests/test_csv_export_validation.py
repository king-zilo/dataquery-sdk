"""Validation tests for the CSV exporters in ``dataquery/export.py``."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

import pytest

from dataquery.export import _to_dict, export_grid_csv, export_timeseries_csv
from dataquery.types.exceptions import DataQueryError
from dataquery.types.models import GridDataResponse, TimeSeriesResponse

TS_HEADER = (
    "date,value,instrument_id,instrument_name,attribute_id,attribute_name,"
    "expression,label,last_published,group_id,group_name"
)
TS_FIELDS = TS_HEADER.split(",")


def _parse(content: str) -> List[List[str]]:
    return list(csv.reader(io.StringIO(content, newline="")))


def _csv_text(*lines: str) -> str:
    return "".join(f"{line}\r\n" for line in lines)


def _ts_model() -> TimeSeriesResponse:
    """A two-instrument page shaped like the live /instruments/time-series payload."""
    return TimeSeriesResponse.model_validate(
        {
            "links": [{"self": "/instruments/time-series", "next": None}],
            "items": 2,
            "page-size": 2,
            "instruments": [
                {
                    "item": 1,
                    "instrument-id": "CMDTY_GOLD",
                    "instrument-name": "Gold spot",
                    "group": {"group-id": "CMDTY", "group-name": "Commodities"},
                    "attributes": [
                        {
                            "attribute-id": "PX_LAST",
                            "attribute-name": "Last price",
                            "expression": "DB(CMDTY,GOLD,PX_LAST)",
                            "label": "Gold last",
                            "last-published": "2024-01-03T18:00:00Z",
                            "time-series": [["20240102", 2063.5], ["20240103", 2041.25]],
                        },
                        {
                            "attribute-id": "PX_OPEN",
                            "attribute-name": "Open price",
                            "expression": "DB(CMDTY,GOLD,PX_OPEN)",
                            "label": "Gold open",
                            "last-published": "2024-01-03T09:00:00Z",
                            "time-series": [["20240102", 2060.0]],
                        },
                    ],
                },
                {
                    "item": 2,
                    "instrument-id": "FX_EURUSD",
                    "instrument-name": "EUR/USD",
                    "group": {"group-id": "FX", "group-name": "Foreign exchange"},
                    "attributes": [
                        {
                            "attribute-id": "SPOT",
                            "attribute-name": "Spot rate",
                            "expression": "DB(FX,EURUSD,SPOT)",
                            "label": "EURUSD spot",
                            "last-published": "2024-01-02T17:00:00Z",
                            "time-series": [["20240102", 1.0942]],
                        }
                    ],
                },
            ],
        }
    )


TS_MODEL_CSV = _csv_text(
    TS_HEADER,
    '20240102,2063.5,CMDTY_GOLD,Gold spot,PX_LAST,Last price,"DB(CMDTY,GOLD,PX_LAST)",Gold last,'
    "2024-01-03T18:00:00Z,CMDTY,Commodities",
    '20240103,2041.25,CMDTY_GOLD,Gold spot,PX_LAST,Last price,"DB(CMDTY,GOLD,PX_LAST)",Gold last,'
    "2024-01-03T18:00:00Z,CMDTY,Commodities",
    '20240102,2060.0,CMDTY_GOLD,Gold spot,PX_OPEN,Open price,"DB(CMDTY,GOLD,PX_OPEN)",Gold open,'
    "2024-01-03T09:00:00Z,CMDTY,Commodities",
    '20240102,1.0942,FX_EURUSD,EUR/USD,SPOT,Spot rate,"DB(FX,EURUSD,SPOT)",EURUSD spot,'
    "2024-01-02T17:00:00Z,FX,Foreign exchange",
)


def _grid_model() -> GridDataResponse:
    """A grid payload shaped like the live /grid-data response (heterogeneous records)."""
    return GridDataResponse.model_validate(
        {
            "series": [
                {
                    "expr": "DBGRID(FXVOL,EURUSD)",
                    "records": [
                        {"tenor": "1M", "strike": "ATM", "vol": 7.25},
                        {"tenor": "3M", "strike": "ATM", "vol": 7.5},
                    ],
                },
                {
                    "expr": "DBGRID(FXVOL,USDJPY)",
                    "records": [{"tenor": "1M", "vol": 9.1, "delta": "25D"}],
                },
            ]
        }
    )


# --------------------------------------------------------------------------- _to_dict


def test_to_dict_dumps_pydantic_model_by_alias():
    model = _ts_model()
    data = _to_dict(model)

    assert data == model.model_dump(by_alias=True)
    inst = data["instruments"][0]
    assert inst["instrument-id"] == "CMDTY_GOLD"
    assert "instrument_id" not in inst
    assert inst["attributes"][0]["time-series"] == [["20240102", 2063.5], ["20240103", 2041.25]]
    assert data["page-size"] == 2


def test_to_dict_passes_raw_dict_through_unchanged():
    raw: Dict[str, Any] = {"instruments": [{"instrument_id": "X"}]}
    assert _to_dict(raw) is raw


@pytest.mark.parametrize("payload", [None, [{"instruments": []}], "instruments", 42, ("a", "b"), b"{}"])
def test_to_dict_rejects_other_types(payload):
    with pytest.raises(DataQueryError, match=r"^Cannot export: response payload is not a dict or Pydantic model\.$"):
        _to_dict(payload)


@pytest.mark.parametrize("exporter", [export_timeseries_csv, export_grid_csv])
def test_exporters_reject_non_dict_payload_before_touching_disk(exporter, tmp_path):
    target = tmp_path / "out.csv"
    with pytest.raises(DataQueryError, match="not a dict or Pydantic model"):
        exporter(["not", "a", "dict"], str(target))
    assert not target.exists()


# --------------------------------------------------------------------- time-series: valid


def test_timeseries_model_stdout_exact_csv():
    result = export_timeseries_csv(_ts_model(), "-")

    assert result == {"path": "stdout", "rows": 4, "content": TS_MODEL_CSV}
    rows = _parse(result["content"])
    assert rows[0] == TS_FIELDS
    assert [r[0] for r in rows[1:]] == ["20240102", "20240103", "20240102", "20240102"]
    assert [r[TS_FIELDS.index("group_id")] for r in rows[1:]] == ["CMDTY", "CMDTY", "CMDTY", "FX"]
    assert [r[TS_FIELDS.index("group_name")] for r in rows[1:]] == [
        "Commodities",
        "Commodities",
        "Commodities",
        "Foreign exchange",
    ]


def test_timeseries_raw_snake_case_dict():
    raw = {
        "instruments": [
            {
                "instrument_id": "RATES_UST10Y",
                "instrument_name": "UST 10Y",
                "group": {"group_id": "RATES", "group_name": "Rates"},
                "attributes": [
                    {
                        "attribute_id": "YLD",
                        "attribute_name": "Yield",
                        "expression": "DB(RATES,UST10Y,YLD)",
                        "label": "10Y yield",
                        "last_published": "2024-02-01",
                        "time_series": [["20240131", 3.91], ["20240201", 3.87]],
                    }
                ],
            }
        ]
    }
    result = export_timeseries_csv(raw, "-")

    assert result["rows"] == 2
    assert result["content"] == _csv_text(
        TS_HEADER,
        '20240131,3.91,RATES_UST10Y,UST 10Y,YLD,Yield,"DB(RATES,UST10Y,YLD)",10Y yield,2024-02-01,RATES,Rates',
        '20240201,3.87,RATES_UST10Y,UST 10Y,YLD,Yield,"DB(RATES,UST10Y,YLD)",10Y yield,2024-02-01,RATES,Rates',
    )


def test_timeseries_raw_kebab_case_dict_matches_model_output():
    raw = _ts_model().model_dump(by_alias=True)
    assert export_timeseries_csv(raw, "-")["content"] == TS_MODEL_CSV


def test_timeseries_mixed_kebab_and_snake_keys_in_one_response():
    raw = {
        "instruments": [
            {
                "instrument-id": "K1",
                "instrument_name": "snake name",
                "group": {"group_id": "G_SNAKE", "group-name": "Kebab group"},
                "attributes": [
                    {
                        "attribute_id": "a_snake",
                        "attribute-name": "A kebab",
                        "expression": "E1",
                        "last_published": "2024-01-01",
                        "time-series": [["20240101", 1.5]],
                    },
                    {
                        "attribute-id": "a_kebab",
                        "attribute_name": "A snake",
                        "expression": "E2",
                        "last-published": "2024-01-02",
                        "time_series": [["20240102", 2.5]],
                    },
                ],
            }
        ]
    }
    assert export_timeseries_csv(raw, "-")["content"] == _csv_text(
        TS_HEADER,
        "20240101,1.5,K1,snake name,a_snake,A kebab,E1,,2024-01-01,G_SNAKE,Kebab group",
        "20240102,2.5,K1,snake name,a_kebab,A snake,E2,,2024-01-02,G_SNAKE,Kebab group",
    )


def test_timeseries_writes_dates_and_values_as_given():
    raw = {
        "instruments": [
            {
                "instrument-id": "I",
                "attributes": [
                    {
                        "expression": "E",
                        "time-series": [
                            ["2024-01-01", 0],
                            ["20240102", 0.0],
                            ["20240103", -1.25e-05],
                            ["20240104", "N/A"],
                            ["20240105", 123456789012],
                        ],
                    }
                ],
            }
        ]
    }
    rows = _parse(export_timeseries_csv(raw, "-")["content"])
    assert [(r[0], r[1]) for r in rows[1:]] == [
        ("2024-01-01", "0"),
        ("20240102", "0.0"),
        ("20240103", "-1.25e-05"),
        ("20240104", "N/A"),
        ("20240105", "123456789012"),
    ]


def test_timeseries_extra_point_elements_are_dropped():
    raw = {"instruments": [{"attributes": [{"expression": "E", "time-series": [["20240101", 1.5, "extra", 9]]}]}]}
    assert export_timeseries_csv(raw, "-")["content"] == _csv_text(TS_HEADER, "20240101,1.5,,,,,E,,,,")


# ------------------------------------------------------------- time-series: missing values


def test_timeseries_none_values_and_missing_optional_fields_from_model():
    model = TimeSeriesResponse.model_validate(
        {
            "instruments": [
                {
                    "item": 1,
                    # Computed expressions come back with a null instrument id/name and no group.
                    "instrument-id": None,
                    "attributes": [
                        {
                            "expression": "VOL(30,DB(FX,EURUSD,SPOT))",
                            "time-series": [["20240102", None], ["20240103", 6.75]],
                        }
                    ],
                }
            ]
        }
    )
    result = export_timeseries_csv(model, "-")

    assert result["rows"] == 2
    assert result["content"] == _csv_text(
        TS_HEADER,
        '20240102,,,,,,"VOL(30,DB(FX,EURUSD,SPOT))",,,,',
        '20240103,6.75,,,,,"VOL(30,DB(FX,EURUSD,SPOT))",,,,',
    )


def test_timeseries_missing_optional_keys_in_raw_dict_are_empty_cells():
    raw = {
        "instruments": [
            {"instrument-id": "NO_GROUP", "attributes": [{"time-series": [["20240101", None]]}]},
            {
                "instrument-id": "NULL_GROUP",
                "group": None,
                "attributes": [{"attribute-id": "A", "label": None, "time-series": [["20240102", 2.0]]}],
            },
            {
                "instrument-id": "PARTIAL_GROUP",
                "group": {"group-id": "G"},
                "attributes": [{"attribute-id": "B", "time-series": [["20240103", 3.0]]}],
            },
        ]
    }
    assert export_timeseries_csv(raw, "-")["content"] == _csv_text(
        TS_HEADER,
        "20240101,,NO_GROUP,,,,,,,,",
        "20240102,2.0,NULL_GROUP,,A,,,,,,",
        "20240103,3.0,PARTIAL_GROUP,,B,,,,,G,",
    )


def test_timeseries_attributes_without_series_are_skipped():
    raw = {
        "instruments": [
            {"instrument-id": "NO_ATTRS"},
            {"instrument-id": "NULL_ATTRS", "attributes": None},
            {
                "instrument-id": "MIXED",
                "attributes": [
                    {"attribute-id": "ABSENT"},
                    {"attribute-id": "NULL", "time-series": None},
                    {"attribute-id": "EMPTY", "time-series": []},
                    {"attribute-id": "HAS", "time-series": [["20240101", 4.5]]},
                ],
            },
        ]
    }
    result = export_timeseries_csv(raw, "-")
    assert result["rows"] == 1
    assert result["content"] == _csv_text(TS_HEADER, "20240101,4.5,MIXED,,HAS,,,,,,")


@pytest.mark.parametrize(
    "bad_point",
    [
        "20240101,1.0",
        {"date": "20240101", "value": 1.0},
        ("20240101", 1.0),
        None,
        20240101,
        [],
        ["20240101"],
    ],
    ids=["str", "dict", "tuple", "none", "int", "empty-list", "one-element"],
)
def test_timeseries_malformed_points_are_skipped(bad_point):
    raw = {
        "instruments": [
            {
                "instrument-id": "I",
                "attributes": [{"attribute-id": "A", "time-series": [bad_point, ["20240102", 2.0], bad_point]}],
            }
        ]
    }
    result = export_timeseries_csv(raw, "-")
    assert result["rows"] == 1
    assert result["content"] == _csv_text(TS_HEADER, "20240102,2.0,I,,A,,,,,,")


# ------------------------------------------------------------------ time-series: errors


@pytest.mark.parametrize(
    "payload",
    [{}, {"instruments": None}, {"instruments": []}, TimeSeriesResponse.model_validate({"instruments": []})],
    ids=["absent", "none", "empty-list", "empty-model"],
)
def test_timeseries_no_instruments_raises(payload, tmp_path):
    target = tmp_path / "out.csv"
    with pytest.raises(DataQueryError, match=r"^No instruments found in the response to export\.$"):
        export_timeseries_csv(payload, str(target))
    assert not target.exists()


@pytest.mark.parametrize(
    "instruments",
    [
        [{"instrument-id": "A"}],
        [{"instrument-id": "A", "attributes": [{"attribute-id": "X", "time-series": []}]}],
        [{"instrument-id": "A", "attributes": [{"attribute-id": "X", "time-series": [["20240101"], "bad"]}]}],
        [{"instrument-id": "A", "attributes": [{"attribute-id": "X", "message": None, "time-series": None}]}],
    ],
    ids=["no-attributes", "empty-series", "only-malformed-points", "null-message"],
)
def test_timeseries_instruments_without_points_or_messages_raise(instruments, tmp_path):
    target = tmp_path / "out.csv"
    with pytest.raises(DataQueryError) as excinfo:
        export_timeseries_csv({"instruments": instruments}, str(target))
    assert str(excinfo.value) == "Response contains instruments but no time-series data points."
    assert not target.exists()


# ------------------------------------------------------------------------ grid: valid


def test_grid_model_stdout_exact_csv():
    result = export_grid_csv(_grid_model(), "-")

    assert result == {
        "path": "stdout",
        "rows": 3,
        "content": _csv_text(
            "expression,tenor,strike,vol,delta",
            '"DBGRID(FXVOL,EURUSD)",1M,ATM,7.25,',
            '"DBGRID(FXVOL,EURUSD)",3M,ATM,7.5,',
            '"DBGRID(FXVOL,USDJPY)",1M,,9.1,25D',
        ),
    }


def test_grid_raw_dict_with_expression_key_and_first_seen_column_order():
    raw = {
        "series": [
            {"expression": "E1", "records": [{"b": 1, "a": 2}]},
            {"expr": "E2", "records": [{"c": 3, "a": 4}, {"d": 5}]},
        ]
    }
    result = export_grid_csv(raw, "-")
    rows = _parse(result["content"])

    assert rows[0] == ["expression", "b", "a", "c", "d"]
    assert rows[1:] == [["E1", "1", "2", "", ""], ["E2", "", "4", "3", ""], ["E2", "", "", "", "5"]]
    assert result["rows"] == 3


def test_grid_expr_key_takes_precedence_over_expression():
    raw = {"series": [{"expr": "FROM_EXPR", "expression": "FROM_EXPRESSION", "records": [{"v": 1}]}]}
    assert export_grid_csv(raw, "-")["content"] == _csv_text("expression,v", "FROM_EXPR,1")


def test_grid_series_without_expression_gets_empty_cell():
    raw = {"series": [{"records": [{"v": 1}]}, {"expr": None, "expression": None, "records": [{"v": 2}]}]}
    # A None from .get("expression", "") is written by DictWriter as an empty cell too.
    assert export_grid_csv(raw, "-")["content"] == _csv_text("expression,v", ",1", ",2")


def test_grid_none_record_values_are_blank():
    raw = {"series": [{"expr": "E", "records": [{"v": None, "w": 0}]}]}
    assert export_grid_csv(raw, "-")["content"] == _csv_text("expression,v,w", "E,,0")


@pytest.mark.parametrize(
    "bad_record", [["1M", 7.25], "1M,7.25", None, 3, ("k", "v")], ids=["list", "str", "none", "int", "tuple"]
)
def test_grid_non_dict_records_are_skipped(bad_record):
    raw = {"series": [{"expr": "E", "records": [bad_record, {"tenor": "1M"}, bad_record]}]}
    result = export_grid_csv(raw, "-")
    assert result["rows"] == 1
    assert result["content"] == _csv_text("expression,tenor", "E,1M")


def test_grid_series_without_records_are_skipped():
    model = GridDataResponse.model_validate(
        {
            "series": [
                {"expr": "ERR", "errorCode": "E404", "errorMessage": "Unknown expression"},
                {"expr": "EMPTY", "records": []},
                {"expr": "OK", "records": [{"v": 1.5}]},
            ]
        }
    )
    result = export_grid_csv(model, "-")
    assert result == {"path": "stdout", "rows": 1, "content": _csv_text("expression,v", "OK,1.5")}


# ----------------------------------------------------------------------- grid: errors


@pytest.mark.parametrize(
    "payload",
    [{}, {"series": None}, {"series": []}, GridDataResponse.model_validate({"series": []})],
    ids=["absent", "none", "empty-list", "empty-model"],
)
def test_grid_no_series_raises(payload, tmp_path):
    target = tmp_path / "grid.csv"
    with pytest.raises(DataQueryError, match=r"^No grid series found in the response to export\.$"):
        export_grid_csv(payload, str(target))
    assert not target.exists()


@pytest.mark.parametrize(
    "series",
    [
        [{"expr": "E"}],
        [{"expr": "E", "records": None}],
        [{"expr": "E", "records": []}],
        [{"expr": "E", "records": [["a", 1], "b", None]}],
    ],
    ids=["absent", "none", "empty", "only-non-dict"],
)
def test_grid_series_without_dict_records_raise(series, tmp_path):
    target = tmp_path / "grid.csv"
    with pytest.raises(DataQueryError, match=r"^Grid series found but no records to export\.$"):
        export_grid_csv({"series": series}, str(target))
    assert not target.exists()


# ---------------------------------------------------------------------- output modes


@pytest.mark.parametrize(
    "exporter,payload",
    [(export_timeseries_csv, _ts_model()), (export_grid_csv, _grid_model())],
    ids=["timeseries", "grid"],
)
def test_stdout_mode_writes_no_file(exporter, payload, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = exporter(payload, "-")
    assert set(result) == {"path", "rows", "content"}
    assert result["path"] == "stdout"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "exporter,payload",
    [(export_timeseries_csv, _ts_model()), (export_grid_csv, _grid_model())],
    ids=["timeseries", "grid"],
)
def test_file_mode_writes_same_bytes_as_stdout_content(exporter, payload, tmp_path):
    target = tmp_path / "out.csv"
    stdout = exporter(payload, "-")
    result = exporter(payload, str(target))

    assert result == {"path": str(target), "rows": stdout["rows"]}
    assert target.read_bytes() == stdout["content"].encode("utf-8")


def test_file_mode_overwrites_existing_file(tmp_path):
    target = tmp_path / "out.csv"
    target.write_text("stale contents that are much longer than the new csv\n" * 50, encoding="utf-8")
    raw = {"series": [{"expr": "E", "records": [{"v": 1}]}]}

    assert export_grid_csv(raw, str(target)) == {"path": str(target), "rows": 1}
    assert target.read_bytes() == b"expression,v\r\nE,1\r\n"


TRICKY = ["Zürich – Ünïcödé €", "a,b", 'say "hi"', "line1\nline2", "  padded  ", "東京"]


def test_timeseries_file_round_trips_utf8_and_special_characters(tmp_path):
    target = tmp_path / "ts.csv"
    raw = {
        "instruments": [
            {
                "instrument-id": TRICKY[1],
                "instrument-name": TRICKY[0],
                "group": {"group-id": TRICKY[5], "group-name": TRICKY[4]},
                "attributes": [
                    {
                        "attribute-id": "A",
                        "attribute-name": TRICKY[2],
                        "expression": "DB(X,Y,Z)",
                        "label": TRICKY[3],
                        "last-published": "2024-01-01",
                        "time-series": [["20240101", TRICKY[2]]],
                    }
                ],
            }
        ]
    }
    assert export_timeseries_csv(raw, str(target)) == {"path": str(target), "rows": 1}

    data = target.read_bytes()
    assert "Zürich – Ünïcödé €".encode("utf-8") in data
    assert "東京".encode("utf-8") in data
    assert b'"a,b"' in data
    assert b'"say ""hi"""' in data
    assert b'"line1\nline2"' in data

    with open(target, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [
        TS_FIELDS,
        [
            "20240101",
            'say "hi"',
            "a,b",
            "Zürich – Ünïcödé €",
            "A",
            'say "hi"',
            "DB(X,Y,Z)",
            "line1\nline2",
            "2024-01-01",
            "東京",
            "  padded  ",
        ],
    ]


def test_grid_file_round_trips_utf8_and_special_characters(tmp_path):
    target = tmp_path / "grid.csv"
    model = GridDataResponse.model_validate(
        {"series": [{"expr": TRICKY[1], "records": [{"name": TRICKY[0], "note": TRICKY[3], TRICKY[2]: TRICKY[5]}]}]}
    )
    assert export_grid_csv(model, str(target)) == {"path": str(target), "rows": 1}

    with open(target, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows == [["expression", "name", "note", 'say "hi"'], ["a,b", TRICKY[0], "line1\nline2", "東京"]]
    assert target.read_bytes().startswith(b'expression,name,note,"say ""hi"""\r\n"a,b",')


@pytest.mark.parametrize(
    "exporter,payload",
    [(export_timeseries_csv, _ts_model()), (export_grid_csv, _grid_model())],
    ids=["timeseries", "grid"],
)
def test_unwritable_directory_path_wraps_oserror(exporter, payload, tmp_path):
    target = tmp_path / "a_directory"
    target.mkdir()
    with pytest.raises(DataQueryError) as excinfo:
        exporter(payload, str(target))

    assert str(excinfo.value).startswith(f"Failed to write CSV to {target}: ")
    assert isinstance(excinfo.value.__cause__, IsADirectoryError)
    assert str(excinfo.value.__cause__) in str(excinfo.value)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize(
    "exporter,payload",
    [(export_timeseries_csv, _ts_model()), (export_grid_csv, _grid_model())],
    ids=["timeseries", "grid"],
)
def test_missing_parent_directory_wraps_oserror(exporter, payload, tmp_path):
    target = tmp_path / "missing" / "nested" / "out.csv"
    with pytest.raises(DataQueryError) as excinfo:
        exporter(payload, str(target))

    assert str(excinfo.value).startswith(f"Failed to write CSV to {target}: ")
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert not (tmp_path / "missing").exists()


# --------------------------------------------------------- findings (documented, not fixed)


@pytest.mark.xfail(
    strict=True,
    reason="export.py:35-45 uses `or` fallbacks, so falsy-but-present ids/timestamps (0) are written as ''",
)
@pytest.mark.parametrize(
    "location,key,column",
    [
        ("instrument", "instrument-id", "instrument_id"),
        ("group", "group-id", "group_id"),
        ("attribute", "attribute-id", "attribute_id"),
        ("attribute", "last-published", "last_published"),
    ],
)
def test_finding_falsy_zero_fields_are_preserved(location, key, column):
    attr: Dict[str, Any] = {"time-series": [["20240101", 1.5]]}
    inst: Dict[str, Any] = {"attributes": [attr]}
    if location == "instrument":
        inst[key] = 0
    elif location == "group":
        inst["group"] = {key: 0}
    else:
        attr[key] = 0
    rows = _parse(export_timeseries_csv({"instruments": [inst]}, "-")["content"])
    assert rows[1][TS_FIELDS.index(column)] == "0"


def test_timeseries_empty_kebab_value_falls_back_to_snake_key():
    raw = {
        "instruments": [
            {"instrument-id": "", "instrument_id": "SNAKE", "attributes": [{"time-series": [["20240101", 1.5]]}]}
        ]
    }
    rows = _parse(export_timeseries_csv(raw, "-")["content"])
    assert rows[1][TS_FIELDS.index("instrument_id")] == "SNAKE"


@pytest.mark.xfail(
    strict=True,
    reason="export.py:108-125 ignores per-series errorMessage, so an all-error grid gives a generic error",
)
def test_finding_grid_error_messages_are_surfaced(tmp_path):
    model = GridDataResponse.model_validate(
        {"series": [{"expr": "DBGRID(BAD)", "errorCode": "E1", "errorMessage": "Unknown grid"}]}
    )
    with pytest.raises(DataQueryError, match="Unknown grid"):
        export_grid_csv(model, str(tmp_path / "grid.csv"))


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="export.py:35 / export.py:116 call .get() on entries without an isinstance check",
)
@pytest.mark.parametrize(
    "exporter,payload",
    [
        (export_timeseries_csv, {"instruments": ["CMDTY_GOLD"]}),
        (export_timeseries_csv, {"instruments": {"instrument-id": "X"}}),
        (export_grid_csv, {"series": ["DBGRID(X)"]}),
    ],
    ids=["ts-str-instrument", "ts-dict-instead-of-list", "grid-str-series"],
)
def test_finding_malformed_entries_raise_dataquery_error(exporter, payload, tmp_path):
    with pytest.raises(DataQueryError):
        exporter(payload, str(tmp_path / "out.csv"))


def test_stdout_content_uses_crlf_line_endings():
    content = export_grid_csv({"series": [{"expr": "E", "records": [{"v": 1}]}]}, "-")["content"]
    assert content == "expression,v\r\nE,1\r\n"
