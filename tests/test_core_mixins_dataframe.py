"""Edge-case coverage for ``DataFrameMixin`` in ``dataquery.core._mixins``.

Complements ``tests/test_dataframe_mixin.py``; pure transformation logic, no
HTTP client. Skipped when pandas is not installed.
"""

import sys
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from dataquery.core import _mixins as mixins_mod  # noqa: E402
from dataquery.core._mixins import DataFrameMixin  # noqa: E402
from dataquery.types.models import TimeSeriesResponse  # noqa: E402


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)


class _DF(DataFrameMixin):
    def __init__(self):
        self.logger = RecordingLogger()


@pytest.fixture
def df():
    return _DF()


def test_typed_wrappers_accept_plain_lists(df):
    groups = df.groups_to_dataframe([{"group_id": "G1", "last_updated": "2024-01-15"}])
    assert pd.api.types.is_datetime64_any_dtype(groups["last_updated"])

    files = df.files_to_dataframe([{"file_group_id": "F1", "file_size": "2048"}])
    assert files.loc[0, "file_size"] == 2048

    insts = df.instruments_to_dataframe([{"instrument_id": "I1", "created_date": "2024-01-15"}])
    assert pd.api.types.is_datetime64_any_dtype(insts["created_date"])


def test_time_series_to_dataframe_requires_pandas(df, monkeypatch):
    monkeypatch.setattr(mixins_mod, "HAS_PANDAS", False)
    with pytest.raises(ImportError, match="pandas is required"):
        df.time_series_to_dataframe({"instruments": []})


def test_to_dataframe_raises_when_pandas_import_fails(df, monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ImportError, match="pandas is required for DataFrame conversion"):
        df.to_dataframe([{"a": 1}])


def test_transform_helpers_return_input_when_pandas_import_fails(df, monkeypatch):
    frame = pd.DataFrame({"created": ["2024-01-15"]})
    monkeypatch.setitem(sys.modules, "pandas", None)
    assert df._apply_data_transformations(frame, ["created"], [], {}) is frame
    assert df._auto_convert_columns(frame) is frame
    assert frame["created"].tolist() == ["2024-01-15"]


def test_time_series_from_response_model(df):
    resp = TimeSeriesResponse(
        **{
            "instruments": [
                {
                    "item": 1,
                    "instrument-id": "I1",
                    "attributes": [{"attribute-id": "A1", "time-series": [["20240115", 1.5], ["20240116", 2.5]]}],
                }
            ]
        }
    )
    out = df.time_series_to_dataframe(resp)
    assert out["value"].tolist() == [1.5, 2.5]
    assert out["date"].tolist() == [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-16")]
    assert set(out["attribute_id"]) == {"A1"}


def test_time_series_empty_with_metadata_has_all_columns(df):
    out = df.time_series_to_dataframe({"instruments": []}, include_metadata=True)
    assert out.empty
    assert list(out.columns) == DataFrameMixin._TS_CORE_COLUMNS + DataFrameMixin._TS_METADATA_COLUMNS


def test_time_series_none_returns_empty_typed_frame(df):
    out = df.time_series_to_dataframe(None)
    assert out.empty
    assert list(out.columns) == DataFrameMixin._TS_CORE_COLUMNS


def test_time_series_flat_records_without_date_or_value(df):
    out = df.time_series_to_dataframe({"series": [{"ticker": "X", "px": 1}]})
    assert out.to_dict("records") == [{"ticker": "X", "px": 1}]


def test_time_series_unwrapped_payload_passes_through(df):
    out = df.time_series_to_dataframe({"date": "2024-01-15", "value": "3.5"})
    assert len(out) == 1
    assert out.loc[0, "value"] == 3.5
    assert out.loc[0, "date"] == pd.Timestamp("2024-01-15")


def test_time_series_points_as_dicts_objects_and_short_lists(df):
    payload = {
        "instruments": [
            {
                "instrument-id": "I1",
                "attributes": [
                    {
                        "attribute-id": "A1",
                        "time-series": [
                            {"date": "20240115", "value": 1.0},
                            SimpleNamespace(date="20240116", value=2.0),
                            ["20240117"],
                            [],
                        ],
                    }
                ],
            }
        ]
    }
    out = df.time_series_to_dataframe(payload)
    assert len(out) == 4
    assert out["value"].tolist()[:2] == [1.0, 2.0]
    assert out["value"].isna().tolist() == [False, False, True, True]
    assert out.loc[2, "date"] == pd.Timestamp("2024-01-17")
    assert pd.isna(out.loc[3, "date"])


def test_to_dataframe_wraps_single_object(df):
    out = df.to_dataframe(SimpleNamespace(name="one", qty="5"))
    assert out.to_dict("records") == [{"name": "one", "qty": "5"}]


def test_to_dataframe_skips_none_items(df):
    assert df.to_dataframe([None, None]).empty
    out = df.to_dataframe([None, {"a": "x"}])
    assert out.to_dict("records") == [{"a": "x"}]


def test_to_dataframe_primitive_items_go_to_value_column(df):
    out = df.to_dataframe([1, 2, 3])
    assert out["value"].tolist() == [1, 2, 3]


class _BrokenDump:
    def __init__(self):
        self.name = "fallback"

    def model_dump(self):
        raise RuntimeError("cannot dump")


class _BrokenDumpSlots:
    __slots__ = ("name",)

    def model_dump(self):
        raise RuntimeError("cannot dump")


def test_to_dataframe_falls_back_to_dict_when_model_dump_fails(df):
    out = df.to_dataframe([_BrokenDump()])
    assert out.to_dict("records") == [{"name": "fallback"}]


def test_to_dataframe_drops_object_without_dict_when_model_dump_fails(df):
    assert df.to_dataframe([_BrokenDumpSlots()]).empty


def test_list_of_dicts_flattening_skips_non_dicts_and_caps_at_five(df):
    items = [{"k": i} for i in range(7)]
    items.insert(1, "not-a-dict")
    out = df.to_dataframe([{"rows": items}])
    assert [c for c in out.columns if c.startswith("rows_")] == [
        "rows_0_k",
        "rows_2_k",
        "rows_3_k",
        "rows_4_k",
    ]


def test_convert_value_model_dump(df):
    class Model:
        def model_dump(self):
            return {"x": 1}

    class BadModel:
        def model_dump(self):
            raise RuntimeError("boom")

        def __str__(self):
            return "BadModel()"

    assert df._convert_value(Model()) == "{'x': 1}"
    assert df._convert_value(BadModel()) == "BadModel()"


def test_requested_columns_missing_from_frame_are_ignored(df):
    out = df.to_dataframe(
        [{"a": "x"}],
        date_columns=["missing_date"],
        numeric_columns=["missing_num"],
        custom_transformations={"missing": str.upper},
    )
    assert out.to_dict("records") == [{"a": "x"}]
    assert df.logger.warnings == []


def test_date_and_numeric_conversion_failures_are_logged(df, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(pd, "to_datetime", boom)
    monkeypatch.setattr(pd, "to_numeric", boom)
    out = df.to_dataframe([{"when": "x", "amt": "y"}], date_columns=["when"], numeric_columns=["amt"])
    assert out.to_dict("records") == [{"when": "x", "amt": "y"}]
    assert any("'when' to datetime" in w for w in df.logger.warnings)
    assert any("'amt' to numeric" in w for w in df.logger.warnings)


def test_auto_convert_all_null_date_hint_column_left_alone(df):
    out = df._auto_convert_columns(pd.DataFrame({"created_count": [None, None]}, dtype=object))
    assert out["created_count"].dtype == object
    assert out["created_count"].isna().all()


def test_auto_convert_numeric_hint_tolerates_to_numeric_errors(df, monkeypatch):
    def boom(*a, **k):
        raise TypeError("bad")

    monkeypatch.setattr(pd, "to_numeric", boom)
    out = df._auto_convert_columns(pd.DataFrame({"price": ["10", "20"]}))
    assert out["price"].tolist() == ["10", "20"]
