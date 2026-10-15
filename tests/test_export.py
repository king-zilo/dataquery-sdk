"""Tests for dataquery.export (time-series and grid CSV export)."""

import csv
import io

import pytest

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


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _ts_response() -> TimeSeriesResponse:
    return TimeSeriesResponse.model_validate(
        {
            "instruments": [
                {
                    "item": 1,
                    "instrument-id": "INS1",
                    "instrument-name": "USD Swap 5Y",
                    "group": {"group-id": "G1", "group-name": "Rates"},
                    "attributes": [
                        {
                            "attribute-id": "MIDPRC",
                            "attribute-name": "Mid Price",
                            "expression": "DB(INS1,MIDPRC)",
                            "label": "mid",
                            "last-published": "2024-01-03T18:00:00Z",
                            "time-series": [["20240102", 1.5], ["20240103", None]],
                        }
                    ],
                },
                {
                    "item": 2,
                    "instrument-id": None,
                    "attributes": [
                        {"expression": "VOL(30, DB(INS1,MIDPRC))", "time-series": [["20240103", 0.25]]},
                    ],
                },
            ]
        }
    )


# --------------------------------------------------------------------------- #
# _to_dict
# --------------------------------------------------------------------------- #
def test_to_dict_dumps_pydantic_models_with_aliases():
    data = _to_dict(_ts_response())
    assert data["instruments"][0]["instrument-id"] == "INS1"
    assert data["instruments"][0]["attributes"][0]["time-series"][0] == ["20240102", 1.5]


def test_to_dict_passes_plain_dicts_through():
    payload = {"series": []}
    assert _to_dict(payload) is payload


@pytest.mark.parametrize("payload", [None, [], "instruments", 42])
def test_to_dict_rejects_other_payloads(payload):
    with pytest.raises(DataQueryError, match="not a dict or Pydantic model"):
        _to_dict(payload)


# --------------------------------------------------------------------------- #
# export_timeseries_csv
# --------------------------------------------------------------------------- #
def test_timeseries_export_flattens_model_to_csv(tmp_path):
    out = tmp_path / "ts.csv"
    result = export_timeseries_csv(_ts_response(), str(out))

    assert result == {"path": str(out), "rows": 3}
    rows = _read_csv(out)
    assert rows[0] == TS_HEADER
    assert rows[1] == [
        "20240102",
        "1.5",
        "INS1",
        "USD Swap 5Y",
        "MIDPRC",
        "Mid Price",
        "DB(INS1,MIDPRC)",
        "mid",
        "2024-01-03T18:00:00Z",
        "G1",
        "Rates",
    ]
    assert rows[2][:2] == ["20240103", ""]  # missing value becomes an empty cell
    # Computed expression: no instrument or group, the expression still identifies the row.
    assert rows[3] == ["20240103", "0.25", "", "", "", "", "VOL(30, DB(INS1,MIDPRC))", "", "", "", ""]


def test_timeseries_export_accepts_snake_case_keys(tmp_path):
    payload = {
        "instruments": [
            {
                "instrument_id": "INS2",
                "instrument_name": "EUR Swap",
                "group": {"group_id": "G2", "group_name": "Rates EU"},
                "attributes": [
                    {
                        "attribute_id": "BID",
                        "attribute_name": "Bid",
                        "last_published": "2024-02-01",
                        "time_series": [["20240201", 2.0]],
                    }
                ],
            }
        ]
    }
    out = tmp_path / "ts.csv"
    export_timeseries_csv(payload, str(out))

    row = dict(zip(TS_HEADER, _read_csv(out)[1]))
    assert row["instrument_id"] == "INS2"
    assert row["instrument_name"] == "EUR Swap"
    assert row["attribute_id"] == "BID"
    assert row["attribute_name"] == "Bid"
    assert row["last_published"] == "2024-02-01"
    assert row["group_id"] == "G2"
    assert row["group_name"] == "Rates EU"


def test_timeseries_export_skips_malformed_points_and_empty_attributes(tmp_path):
    payload = {
        "instruments": [
            {"instrument-id": "NOATTRS", "attributes": None},
            {
                "instrument-id": "INS1",
                "attributes": [
                    {"expression": "E1", "time-series": None},
                    {"expression": "E2", "time-series": [["20240101"], "junk", None, ["20240102", 3, "extra"]]},
                ],
            },
        ]
    }
    out = tmp_path / "ts.csv"
    result = export_timeseries_csv(payload, str(out))

    assert result["rows"] == 1
    assert _read_csv(out)[1][:3] == ["20240102", "3", "INS1"]


@pytest.mark.parametrize("payload", [{}, {"instruments": None}, {"instruments": []}])
def test_timeseries_export_requires_instruments(tmp_path, payload):
    with pytest.raises(DataQueryError, match="No instruments found"):
        export_timeseries_csv(payload, str(tmp_path / "ts.csv"))
    assert not (tmp_path / "ts.csv").exists()


def test_timeseries_export_without_points_reports_server_messages(tmp_path):
    payload = {
        "instruments": [
            {
                "instrument-id": "INS1",
                "attributes": [
                    {"expression": "DB(BAD)", "message": "Expression syntax error!"},
                    {"attribute-id": "ATTR2", "message": "No data for range"},
                    {"message": "Entitlement missing"},
                ],
            }
        ]
    }
    with pytest.raises(DataQueryError) as exc_info:
        export_timeseries_csv(payload, str(tmp_path / "ts.csv"))

    assert str(exc_info.value) == (
        "Response contains instruments but no time-series data points. Server messages: "
        "DB(BAD): Expression syntax error!; ATTR2: No data for range; INS1: Entitlement missing"
    )
    assert not (tmp_path / "ts.csv").exists()


def test_timeseries_export_without_points_or_messages(tmp_path):
    with pytest.raises(DataQueryError) as exc_info:
        export_timeseries_csv({"instruments": [{"attributes": [{"time-series": []}]}]}, str(tmp_path / "ts.csv"))
    assert str(exc_info.value) == "Response contains instruments but no time-series data points."


def test_timeseries_export_to_stdout_returns_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = export_timeseries_csv(_ts_response(), "-")

    assert result["path"] == "stdout"
    assert result["rows"] == 3
    rows = list(csv.reader(io.StringIO(result["content"])))
    assert rows[0] == TS_HEADER
    assert len(rows) == 4
    assert list(tmp_path.iterdir()) == []


def test_timeseries_export_wraps_write_errors(tmp_path):
    target = tmp_path / "missing-dir" / "ts.csv"
    with pytest.raises(DataQueryError, match="Failed to write CSV to") as exc_info:
        export_timeseries_csv(_ts_response(), str(target))
    assert isinstance(exc_info.value.__cause__, OSError)


# --------------------------------------------------------------------------- #
# export_grid_csv
# --------------------------------------------------------------------------- #
def _grid_response() -> GridDataResponse:
    return GridDataResponse.model_validate(
        {
            "series": [
                {"expr": "GRID(A)", "records": [{"date": "20240101", "tenor": "1Y", "value": 1.1}]},
                {"expr": "GRID(B)", "records": [{"date": "20240101", "strike": 95, "value": 2.2}]},
            ]
        }
    )


def test_grid_export_unions_columns_in_first_seen_order(tmp_path):
    out = tmp_path / "grid.csv"
    result = export_grid_csv(_grid_response(), str(out))

    assert result == {"path": str(out), "rows": 2}
    assert _read_csv(out) == [
        ["expression", "date", "tenor", "value", "strike"],
        ["GRID(A)", "20240101", "1Y", "1.1", ""],
        ["GRID(B)", "20240101", "", "2.2", "95"],
    ]


def test_grid_export_accepts_expression_key_and_skips_non_dict_records(tmp_path):
    payload = {
        "series": [
            {"expression": "GRID(C)", "records": [{"k": 1}, "junk", None]},
            {"expr": "GRID(D)", "records": None},
        ]
    }
    out = tmp_path / "grid.csv"
    assert export_grid_csv(payload, str(out))["rows"] == 1
    assert _read_csv(out) == [["expression", "k"], ["GRID(C)", "1"]]


def test_grid_export_record_expression_column_overrides_series_expr(tmp_path):
    """Documents current behaviour (see PR Findings): a record's own key wins."""
    payload = {"series": [{"expr": "GRID(A)", "records": [{"expression": "inner", "v": 1}]}]}
    result = export_grid_csv(payload, "-")
    assert result["content"].splitlines() == ["expression,v", "inner,1"]


@pytest.mark.parametrize("payload", [{}, {"series": None}, GridDataResponse(errorCode="E1", errorMessage="bad")])
def test_grid_export_requires_series(tmp_path, payload):
    with pytest.raises(DataQueryError, match="No grid series found"):
        export_grid_csv(payload, str(tmp_path / "grid.csv"))


def test_grid_export_requires_records(tmp_path):
    payload = {"series": [{"expr": "GRID(A)", "records": []}, {"expr": "GRID(B)", "records": ["x"]}]}
    with pytest.raises(DataQueryError, match="Grid series found but no records"):
        export_grid_csv(payload, str(tmp_path / "grid.csv"))
    assert not (tmp_path / "grid.csv").exists()


def test_grid_export_to_stdout_returns_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = export_grid_csv(_grid_response(), "-")

    assert result["path"] == "stdout"
    assert result["rows"] == 2
    assert result["content"].splitlines()[0] == "expression,date,tenor,value,strike"
    assert list(tmp_path.iterdir()) == []


def test_grid_export_wraps_write_errors(tmp_path):
    target = tmp_path / "missing-dir" / "grid.csv"
    with pytest.raises(DataQueryError, match="Failed to write CSV to") as exc_info:
        export_grid_csv(_grid_response(), str(target))
    assert isinstance(exc_info.value.__cause__, OSError)
