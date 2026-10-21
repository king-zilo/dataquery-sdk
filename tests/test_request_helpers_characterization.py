"""Characterization tests pinning the wire behaviour of the catalog/time-series/group methods.

They record exactly what each public method sends (HTTP method, URL, ordered query
params) and how it validates input, so internal refactors can't change the API surface.
"""

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch

import pytest

from dataquery import utils
from dataquery.core.client import DataQueryClient
from dataquery.types.exceptions import ValidationError
from dataquery.types.models import (
    AttributesResponse,
    FiltersResponse,
    GridDataResponse,
    GroupList,
    InstrumentsResponse,
    TimeSeriesResponse,
)
from tests.test_client_api_methods import DummyResponse, make_client

API = "https://api.example.com/research/dataquery-authe/api/v2"

TS_DEFAULTS = [
    ("calendar", "CAL_USBANK"),
    ("frequency", "FREQ_DAY"),
    ("conversion", "CONV_LASTBUS_ABS"),
    ("nan-treatment", "NA_NOTHING"),
]


class Recorder:
    """Stands in for ``_make_authenticated_request`` and records every call."""

    def __init__(self, payloads: Optional[List[Any]] = None):
        self.calls: List[Tuple[str, str, Optional[List[Tuple[str, Any]]], Dict[str, Any]]] = []
        self._payloads = list(payloads or [])

    async def __call__(self, method: str, url: str, **kwargs: Any) -> DummyResponse:
        params = kwargs.pop("params", None)
        self.calls.append((method, url, None if params is None else list(params.items()), kwargs))
        payload = self._payloads.pop(0) if len(self._payloads) > 1 else (self._payloads or [{}])[0]
        resp = DummyResponse()

        async def json() -> Any:
            return payload

        resp.json = json  # type: ignore[method-assign]
        return resp


class EventLog:
    def __init__(self) -> None:
        self.events: List[Tuple[str, str, Dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.events.append(("info", event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.events.append(("warning", event, kw))

    def error(self, event: str, **kw: Any) -> None:
        self.events.append(("error", event, kw))

    def debug(self, event: str, **kw: Any) -> None:
        self.events.append(("debug", event, kw))

    def log_metric(self, name: str, value: Any, unit: str) -> None:
        self.events.append(("metric", name, {"value": value, "unit": unit}))


def _client(monkeypatch, payloads: Optional[List[Any]] = None) -> Tuple[DataQueryClient, Recorder, EventLog]:
    client = make_client(monkeypatch)
    rec = Recorder(payloads)
    log = EventLog()
    monkeypatch.setattr(client, "_make_authenticated_request", rec)
    client.logger = log  # type: ignore[assignment]
    client.logging_manager = log  # type: ignore[assignment]
    return client, rec, log


# --------------------------------------------------------------------------- signatures

EXPECTED_SIGNATURES = {
    "list_instruments_async": "group_id, instrument_id=None, page=None",
    "iter_instruments_async": "group_id, instrument_id=None, *max_pages=1000",
    "search_instruments_async": "group_id, keywords, page=None",
    "iter_search_instruments_async": "group_id, keywords, *max_pages=1000",
    "get_group_filters_async": "group_id, page=None",
    "iter_group_filters_async": "group_id, *max_pages=1000",
    "get_group_attributes_async": "group_id, instrument_id=None, page=None",
    "iter_group_attributes_async": "group_id, instrument_id=None, *max_pages=1000",
    "get_instrument_time_series_async": "instruments, attributes, data='ALL', format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', page=None",
    "get_expressions_time_series_async": "expressions, format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', data='ALL', page=None",
    "get_group_time_series_async": "group_id, attributes, filter=None, data='ALL', format='JSON', start_date=None, end_date=None, calendar='CAL_USBANK', frequency='FREQ_DAY', conversion='CONV_LASTBUS_ABS', nan_treatment='NA_NOTHING', page=None",
    "iter_instrument_time_series_async": "instruments, attributes, *data='ALL', *format='JSON', *start_date=None, *end_date=None, *calendar='CAL_USBANK', *frequency='FREQ_DAY', *conversion='CONV_LASTBUS_ABS', *nan_treatment='NA_NOTHING', *max_pages=1000",
    "iter_expressions_time_series_async": "expressions, *format='JSON', *start_date=None, *end_date=None, *calendar='CAL_USBANK', *frequency='FREQ_DAY', *conversion='CONV_LASTBUS_ABS', *nan_treatment='NA_NOTHING', *data='ALL', *max_pages=1000",
    "iter_group_time_series_async": "group_id, attributes, *filter=None, *data='ALL', *format='JSON', *start_date=None, *end_date=None, *calendar='CAL_USBANK', *frequency='FREQ_DAY', *conversion='CONV_LASTBUS_ABS', *nan_treatment='NA_NOTHING', *max_pages=1000",
    "get_grid_data_async": "expr=None, grid_id=None, date=None",
    "list_groups_async": "limit=None",
    "list_groups_page_async": "limit=None, *page=None",
    "list_all_groups_async": "*max_pages=1000, *raise_on_cap=True",
    "iter_groups_pages_async": "*max_pages=1000, *raise_on_cap=True",
    "iter_groups_async": "*max_pages=1000",
    "search_groups_async": "keywords, limit=None, offset=None, *page=None",
    "search_groups_page_async": "keywords, limit=None, *offset=None, *page=None",
    "iter_search_groups_pages_async": "keywords, *limit=None, *max_pages=1000, *raise_on_cap=True",
    "iter_search_groups_async": "keywords, *limit=None, *max_pages=1000",
    "search_all_groups_async": "keywords, *limit=None, *max_pages=1000, *raise_on_cap=True",
}


def _describe(func: Any) -> str:
    parts = []
    for p in inspect.signature(func).parameters.values():
        if p.name == "self":
            continue
        prefix = "*" if p.kind is p.KEYWORD_ONLY else ""
        default = "" if p.default is p.empty else f"={p.default!r}"
        parts.append(f"{prefix}{p.name}{default}")
    return ", ".join(parts)


@pytest.mark.parametrize("name", sorted(EXPECTED_SIGNATURES))
def test_public_signature_is_pinned(name):
    assert _describe(getattr(DataQueryClient, name)) == EXPECTED_SIGNATURES[name]


def test_validate_env_config_signature_is_pinned():
    assert _describe(utils.validate_env_config) == ""


# --------------------------------------------------------------------------- instruments / metadata

INSTRUMENT_CASES: List[Tuple[Any, ...]] = [
    ("list_instruments_async", ("G",), {}, "/group/instruments", [("group-id", "G")], InstrumentsResponse),
    (
        "list_instruments_async",
        ("G", "I", "p2"),
        {},
        "/group/instruments",
        [("group-id", "G"), ("instrument-id", "I"), ("page", "p2")],
        InstrumentsResponse,
    ),
    ("list_instruments_async", ("G", "", ""), {}, "/group/instruments", [("group-id", "G")], InstrumentsResponse),
    (
        "search_instruments_async",
        ("G", "rates"),
        {},
        "/group/instruments/search",
        [("group-id", "G"), ("keywords", "rates")],
        InstrumentsResponse,
    ),
    (
        "search_instruments_async",
        ("G", "rates", "p2"),
        {},
        "/group/instruments/search",
        [("group-id", "G"), ("keywords", "rates"), ("page", "p2")],
        InstrumentsResponse,
    ),
    (
        "search_instruments_async",
        ("G", "rates", ""),
        {},
        "/group/instruments/search",
        [("group-id", "G"), ("keywords", "rates")],
        InstrumentsResponse,
    ),
    ("get_group_filters_async", ("G",), {}, "/group/filters", [("group-id", "G")], FiltersResponse),
    (
        "get_group_filters_async",
        ("G", "p2"),
        {},
        "/group/filters",
        [("group-id", "G"), ("page", "p2")],
        FiltersResponse,
    ),
    ("get_group_filters_async", ("G", ""), {}, "/group/filters", [("group-id", "G")], FiltersResponse),
    ("get_group_attributes_async", ("G",), {}, "/group/attributes", [("group-id", "G")], AttributesResponse),
    (
        "get_group_attributes_async",
        ("G", "I", "p2"),
        {},
        "/group/attributes",
        [("group-id", "G"), ("instrument-id", "I"), ("page", "p2")],
        AttributesResponse,
    ),
    ("get_group_attributes_async", ("G", "", ""), {}, "/group/attributes", [("group-id", "G")], AttributesResponse),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,kwargs,path,params,model", INSTRUMENT_CASES)
async def test_catalog_requests(monkeypatch, name, args, kwargs, path, params, model):
    client, rec, _ = _client(monkeypatch, [{"instruments": [], "filters": []}])
    result = await getattr(client, name)(*args, **kwargs)
    assert type(result) is model
    assert rec.calls == [("GET", API + path, params, {})]


# --------------------------------------------------------------------------- time series

TS_CASES = [
    (
        "get_instrument_time_series_async",
        (["I1", "I2"], ["A"]),
        {},
        "/instruments/time-series",
        [("instruments", ["I1", "I2"]), ("attributes", ["A"]), ("data", "ALL"), ("format", "JSON"), *TS_DEFAULTS],
    ),
    (
        "get_instrument_time_series_async",
        (["I1"], ["A"]),
        {"start_date": "20240101", "end_date": "TODAY", "page": "", "data": "REFERENCE_DATA", "calendar": "CAL_X"},
        "/instruments/time-series",
        [
            ("instruments", ["I1"]),
            ("attributes", ["A"]),
            ("data", "REFERENCE_DATA"),
            ("format", "JSON"),
            ("calendar", "CAL_X"),
            *TS_DEFAULTS[1:],
            ("start-date", "20240101"),
            ("end-date", "TODAY"),
            ("page", ""),
        ],
    ),
    (
        "get_expressions_time_series_async",
        (["DB(a,b,c)"],),
        {},
        "/expressions/time-series",
        [("expressions", ["DB(a,b,c)"]), ("format", "JSON"), *TS_DEFAULTS, ("data", "ALL")],
    ),
    (
        "get_expressions_time_series_async",
        (["E"],),
        {"end_date": "TODAY-1M", "page": "p2", "nan_treatment": "NA_FILL_FORWARD"},
        "/expressions/time-series",
        [
            ("expressions", ["E"]),
            ("format", "JSON"),
            *TS_DEFAULTS[:3],
            ("nan-treatment", "NA_FILL_FORWARD"),
            ("data", "ALL"),
            ("end-date", "TODAY-1M"),
            ("page", "p2"),
        ],
    ),
    (
        "get_group_time_series_async",
        ("G", ["A"]),
        {},
        "/group/time-series",
        [("group-id", "G"), ("attributes", ["A"]), ("data", "ALL"), ("format", "JSON"), *TS_DEFAULTS],
    ),
    (
        "get_group_time_series_async",
        ("G", ["A"]),
        {"filter": "", "start_date": "", "page": "p2"},
        "/group/time-series",
        [
            ("group-id", "G"),
            ("attributes", ["A"]),
            ("data", "ALL"),
            ("format", "JSON"),
            *TS_DEFAULTS,
            ("filter", ""),
            ("start-date", ""),
            ("page", "p2"),
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,kwargs,path,params", TS_CASES)
async def test_time_series_requests(monkeypatch, name, args, kwargs, path, params):
    client, rec, _ = _client(monkeypatch, [{"instruments": []}])
    result = await getattr(client, name)(*args, **kwargs)
    assert type(result) is TimeSeriesResponse
    assert rec.calls == [("GET", API + path, params, {})]


TS_VALIDATION_CASES = [
    ("get_instrument_time_series_async", ([], ["A"]), {}, ValidationError, "instruments"),
    ("get_instrument_time_series_async", (["I"], []), {}, ValidationError, "Attributes list cannot be empty"),
    ("get_instrument_time_series_async", (["I"], ["A"]), {"start_date": "bad"}, ValidationError, "start-date"),
    ("get_instrument_time_series_async", (["I"], ["A"]), {"end_date": "bad"}, ValidationError, "end-date"),
    ("get_expressions_time_series_async", ([],), {}, ValueError, "'expressions' must be a non-empty list"),
    ("get_expressions_time_series_async", ("E",), {}, ValueError, "'expressions' must be a non-empty list"),
    ("get_expressions_time_series_async", ([" "],), {}, ValueError, "All expressions must be non-empty strings"),
    ("get_expressions_time_series_async", (["E", 1],), {}, ValueError, "All expressions must be non-empty strings"),
    ("get_expressions_time_series_async", (["E"],), {"start_date": "bad"}, ValidationError, "start-date"),
    ("get_group_time_series_async", ("G", []), {}, ValidationError, "Attributes list cannot be empty"),
    ("get_group_time_series_async", ("G", ["A"]), {"end_date": "bad"}, ValidationError, "end-date"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,kwargs,exc,match", TS_VALIDATION_CASES)
async def test_time_series_validation(monkeypatch, name, args, kwargs, exc, match):
    client, rec, _ = _client(monkeypatch)
    with pytest.raises(exc, match=match) as info:
        await getattr(client, name)(*args, **kwargs)
    assert type(info.value) is exc
    assert rec.calls == []


@pytest.mark.asyncio
async def test_time_series_iterators_send_first_page_without_page_param(monkeypatch):
    client, rec, _ = _client(monkeypatch, [{"instruments": []}])
    assert [x async for x in client.iter_instrument_time_series_async(["I"], ["A"], start_date="20240101")] == []
    assert [x async for x in client.iter_expressions_time_series_async(["E"], data="REFERENCE_DATA")] == []
    assert [x async for x in client.iter_group_time_series_async("G", ["A"], filter="F")] == []
    assert [c[2] for c in rec.calls] == [
        [("instruments", ["I"]), ("attributes", ["A"]), ("data", "ALL"), ("format", "JSON"), *TS_DEFAULTS]
        + [("start-date", "20240101")],
        [("expressions", ["E"]), ("format", "JSON"), *TS_DEFAULTS, ("data", "REFERENCE_DATA")],
        [("group-id", "G"), ("attributes", ["A"]), ("data", "ALL"), ("format", "JSON"), *TS_DEFAULTS]
        + [("filter", "F")],
    ]


# --------------------------------------------------------------------------- grid


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,params",
    [
        ({"expr": "DBGRID(x)"}, [("expr", "DBGRID(x)")]),
        ({"grid_id": "GR", "date": "20240101"}, [("gridId", "GR"), ("date", "20240101")]),
        ({"expr": "", "grid_id": "GR"}, [("expr", ""), ("gridId", "GR")]),
    ],
)
async def test_grid_requests(monkeypatch, kwargs, params):
    client, rec, _ = _client(monkeypatch, [{"series": []}])
    result = await client.get_grid_data_async(**kwargs)
    assert type(result) is GridDataResponse
    assert rec.calls == [("GET", API + "/grid-data", params, {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,exc,match",
    [
        ({"expr": "E", "grid_id": "G"}, ValueError, "Cannot specify both expr and grid_id"),
        ({}, ValueError, "Must specify either expr or grid_id"),
        ({"expr": "E", "date": "bad"}, ValidationError, "date"),
    ],
)
async def test_grid_validation(monkeypatch, kwargs, exc, match):
    client, rec, _ = _client(monkeypatch)
    with pytest.raises(exc, match=match):
        await client.get_grid_data_async(**kwargs)
    assert rec.calls == []


@pytest.mark.asyncio
async def test_grid_no_content_envelope_returns_empty_series(monkeypatch):
    client, _, _ = _client(monkeypatch, [{"info": {"code": 204, "message": "No content"}}])
    result = await client.get_grid_data_async(expr="E")
    assert result.series == []


# --------------------------------------------------------------------------- groups

GROUPS_PAGE_1 = {"groups": [{"group-id": "a"}], "links": [{"next": "/groups?page=2"}]}
GROUPS_PAGE_2 = {"groups": [{"group-id": "b"}, {"group-id": "c"}], "links": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,params",
    [
        ({}, []),
        ({"limit": 5, "page": "p2"}, [("limit", "5"), ("page", "p2")]),
        ({"limit": 0, "page": ""}, [("limit", "0"), ("page", "")]),
    ],
)
async def test_list_groups_page_request(monkeypatch, kwargs, params):
    client, rec, _ = _client(monkeypatch, [{"groups": []}])
    result = await client.list_groups_page_async(**kwargs)
    assert type(result) is GroupList
    assert rec.calls == [("GET", API + "/groups", params, {})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,params",
    [
        ({}, [("keywords", "fx")]),
        (
            {"limit": 5, "offset": 10, "page": "p2"},
            [("keywords", "fx"), ("limit", "5"), ("offset", "10"), ("page", "p2")],
        ),
        ({"limit": 0, "offset": 0, "page": ""}, [("keywords", "fx"), ("limit", "0"), ("offset", "0"), ("page", "")]),
    ],
)
async def test_search_groups_page_request(monkeypatch, kwargs, params):
    client, rec, _ = _client(monkeypatch, [{"groups": []}])
    result = await client.search_groups_page_async("fx", **kwargs)
    assert type(result) is GroupList
    assert rec.calls == [("GET", API + "/groups/search", params, {})]


@pytest.mark.asyncio
async def test_list_groups_logs_and_metric(monkeypatch):
    client, rec, log = _client(monkeypatch, [GROUPS_PAGE_2])
    groups = await client.list_groups_async(limit=7)
    assert [g.group_id for g in groups] == ["b", "c"]
    assert rec.calls[0][2] == [("limit", "7")]
    assert log.events == [
        ("info", "Groups listed", {"count": 2, "limit": 7}),
        ("metric", "groups_listed", {"value": 2, "unit": "count"}),
    ]


@pytest.mark.asyncio
async def test_search_groups_logs(monkeypatch):
    client, rec, log = _client(monkeypatch, [GROUPS_PAGE_2])
    groups = await client.search_groups_async("fx", 3, 4, page="p")
    assert [g.group_id for g in groups] == ["b", "c"]
    assert rec.calls[0][2] == [("keywords", "fx"), ("limit", "3"), ("offset", "4"), ("page", "p")]
    assert log.events == [("info", "Groups searched", {"keywords": "fx", "count": 2})]


@pytest.mark.asyncio
async def test_list_all_groups_walks_pages_with_logs_and_metrics(monkeypatch):
    client, rec, log = _client(monkeypatch, [GROUPS_PAGE_1, GROUPS_PAGE_2])
    groups = await client.list_all_groups_async()
    assert [g.group_id for g in groups] == ["a", "b", "c"]
    assert [(c[0], c[1], c[2]) for c in rec.calls] == [
        ("GET", API + "/groups", []),
        ("GET", API + "/groups?page=2", None),
    ]
    assert log.events == [
        ("info", "Groups page fetched", {"page": 1, "groups_in_page": 1, "total_groups": 1}),
        ("info", "Groups page fetched", {"page": 2, "groups_in_page": 2, "total_groups": 3}),
        ("info", "All groups fetched", {"total_groups": 3, "total_pages": 2}),
        ("metric", "groups_listed", {"value": 3, "unit": "count"}),
        ("metric", "groups_pages_fetched", {"value": 2, "unit": "count"}),
    ]


@pytest.mark.asyncio
async def test_search_all_groups_walks_pages_with_logs_and_metrics(monkeypatch):
    client, rec, log = _client(monkeypatch, [GROUPS_PAGE_1, GROUPS_PAGE_2])
    groups = await client.search_all_groups_async("fx", limit=1)
    assert [g.group_id for g in groups] == ["a", "b", "c"]
    assert [(c[0], c[1], c[2]) for c in rec.calls] == [
        ("GET", API + "/groups/search", [("keywords", "fx"), ("limit", "1")]),
        ("GET", API + "/groups?page=2", None),
    ]
    assert log.events == [
        ("info", "Search-groups page fetched", {"page": 1, "keywords": "fx", "groups_in_page": 1, "total_groups": 1}),
        ("info", "Search-groups page fetched", {"page": 2, "keywords": "fx", "groups_in_page": 2, "total_groups": 3}),
        ("info", "All matching groups fetched", {"keywords": "fx", "total_groups": 3, "total_pages": 2}),
        ("metric", "groups_searched", {"value": 3, "unit": "count"}),
        ("metric", "groups_search_pages_fetched", {"value": 2, "unit": "count"}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call,event,extra",
    [
        (lambda c: c.list_groups_async(), "Failed to list groups", {}),
        (lambda c: c.list_all_groups_async(), "Failed to list all groups", {}),
        (lambda c: c.search_groups_async("fx"), "Failed to search groups", {"keywords": "fx"}),
        (lambda c: c.search_all_groups_async("fx"), "Failed to walk all matching groups", {"keywords": "fx"}),
    ],
)
async def test_group_errors_are_logged_and_reraised(monkeypatch, call, event, extra):
    client, _, log = _client(monkeypatch)

    async def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(client, "_make_authenticated_request", boom)
    with pytest.raises(RuntimeError, match="boom"):
        await call(client)
    assert log.events == [("error", event, {**extra, "error": "boom"})]


@pytest.mark.asyncio
async def test_group_iterators(monkeypatch):
    client, rec, _ = _client(monkeypatch, [GROUPS_PAGE_1, GROUPS_PAGE_2, GROUPS_PAGE_1, GROUPS_PAGE_2])
    assert [g.group_id async for g in client.iter_groups_async()] == ["a", "b", "c"]
    assert [g.group_id async for g in client.iter_search_groups_async("fx", limit=2)] == ["a", "b", "c"]
    assert [c[2] for c in rec.calls] == [[], None, [("keywords", "fx"), ("limit", "2")], None]


# --------------------------------------------------------------------------- validate_env_config

BASE = {"DATAQUERY_BASE_URL": "https://api.example.com"}

ENV_CASES = [
    ({}, "DATAQUERY_BASE_URL is required"),
    ({"DATAQUERY_BASE_URL": "ftp://x"}, "DATAQUERY_BASE_URL is required"),
    ({"DATAQUERY_BASE_URL": ""}, "DATAQUERY_BASE_URL is required"),
    (BASE, None),
    ({**BASE, "DATAQUERY_TIMEOUT": "abc"}, "Invalid timeout value: abc"),
    ({**BASE, "DATAQUERY_TIMEOUT": ""}, None),
    ({**BASE, "DATAQUERY_TIMEOUT": "1e3"}, None),
    ({**BASE, "DATAQUERY_MAX_RETRIES": "2.5"}, "Invalid max retries value: 2.5"),
    ({**BASE, "DATAQUERY_MAX_RETRIES": "3"}, None),
    ({"DATAQUERY_TIMEOUT": "x", "DATAQUERY_MAX_RETRIES": "y"}, "Invalid timeout value: x"),
    ({"DATAQUERY_MAX_RETRIES": "y", "DATAQUERY_OAUTH_ENABLED": "maybe"}, "Invalid max retries value: y"),
    ({"DATAQUERY_OAUTH_ENABLED": "maybe"}, "Invalid OAuth enabled value: maybe"),
    ({**BASE, "DATAQUERY_OAUTH_ENABLED": ""}, None),
    ({**BASE, "DATAQUERY_OAUTH_ENABLED": "FALSE"}, None),
    ({**BASE, "DATAQUERY_OAUTH_ENABLED": "false", "DATAQUERY_BEARER_TOKEN": ""}, None),
    ({**BASE, "DATAQUERY_OAUTH_ENABLED": "True"}, "OAuth credentials are required"),
    ({**BASE, "DATAQUERY_OAUTH_ENABLED": "true", "DATAQUERY_CLIENT_ID": "id"}, "OAuth credentials are required"),
    (
        {**BASE, "DATAQUERY_OAUTH_ENABLED": "true", "DATAQUERY_CLIENT_ID": " ", "DATAQUERY_CLIENT_SECRET": "s"},
        "OAuth credentials are required",
    ),
    (
        {**BASE, "DATAQUERY_OAUTH_ENABLED": "TRUE", "DATAQUERY_CLIENT_ID": "id", "DATAQUERY_CLIENT_SECRET": "s"},
        None,
    ),
]


@pytest.mark.parametrize("env,error", ENV_CASES)
def test_validate_env_config_matrix(env, error):
    with patch.dict(os.environ, env, clear=True):
        if error is None:
            assert utils.validate_env_config() is None
        else:
            with pytest.raises(ValueError) as info:
                utils.validate_env_config()
            assert type(info.value) is ValueError
            assert str(info.value) == error
