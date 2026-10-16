"""Behavior coverage for the client API mixins in ``dataquery.core._mixins``.

Network is never touched: ``_make_authenticated_request`` is replaced by an
in-memory router that records each request and answers with canned JSON.
"""

import pytest

from dataquery.core._mixins import PaginationMixin, check_envelope
from dataquery.types.exceptions import APIResponseError, PaginationError, ValidationError
from dataquery.types.models import InstrumentsResponse, Paginated

from .test_client_api_methods import DummyResponse, FakeCtx, make_client

API = "https://api.example.com/research/dataquery-authe/api/v2"


class Router:
    """Records requests and replies with queued JSON payloads, in order."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.calls = []

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        payload = self.payloads.pop(0)
        resp = DummyResponse()

        async def json():
            return payload

        resp.json = json
        return FakeCtx(resp)

    @property
    def params(self):
        return [c.get("params") for c in self.calls]


def _serve(monkeypatch, *payloads):
    client = make_client(monkeypatch)
    router = Router(*payloads)
    monkeypatch.setattr(client, "_make_authenticated_request", router)
    return client, router


def _page(key, items, next_link=None):
    return {"items": len(items), "page-size": 50, "links": [{"next": next_link}], key: items}


def _inst(i):
    return {"item": i, "instrument-id": f"I{i}", "instrument-name": f"Inst {i}"}


def _ts_inst(i):
    return {"item": i, "instrument-id": f"I{i}", "attributes": []}


# --------------------------------------------------------------------------- #
# check_envelope / _build_page
# --------------------------------------------------------------------------- #
def test_check_envelope_rejects_non_dict_body():
    with pytest.raises(APIResponseError, match="Unrecognized response shape from grid-data") as ei:
        check_envelope(["not", "a", "dict"], ("series",), "grid-data")
    assert ei.value.details == {"type": "list"}


def test_check_envelope_accepts_single_error_object_with_message():
    with pytest.raises(APIResponseError) as ei:
        check_envelope({"error": {"message": "Token expired"}}, ("groups",))
    # No code: message is the bare description, taken from ``message``.
    assert str(ei.value.message) == "Token expired"
    assert ei.value.code is None
    assert ei.value.details == {"errors": [{"message": "Token expired"}]}


def test_check_envelope_non_dict_error_items_use_generic_message():
    with pytest.raises(APIResponseError) as ei:
        check_envelope({"errors": ["Rate limit exceeded"]}, ("groups",))
    assert ei.value.message == "API returned an error response"
    assert ei.value.details == {"errors": ["Rate limit exceeded"]}


def test_check_envelope_known_key_and_info():
    assert check_envelope({"groups": []}, ("groups",)) is False
    assert check_envelope({"info": {"code": 204}}, ("groups",)) is True


def test_build_page_non_dict_payload_raises_type_error():
    """Documents current behavior: a JSON array body bypasses ``check_envelope``.

    A dict body of the wrong shape raises ``APIResponseError``; a list body
    escapes as a raw ``TypeError`` from ``model_cls(**payload)``.
    """
    with pytest.raises(TypeError):
        PaginationMixin._build_page(InstrumentsResponse, [{"instrument-id": "I1"}])


@pytest.mark.asyncio
async def test_list_instruments_json_array_body_surfaces_type_error(monkeypatch):
    client, _ = _serve(monkeypatch, [_inst(1)])
    with pytest.raises(TypeError):
        await client.list_instruments_async("G1")


# --------------------------------------------------------------------------- #
# iter_pages
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_iter_pages_stops_silently_at_cap(monkeypatch):
    client, router = _serve(monkeypatch)

    async def first():
        return InstrumentsResponse(**_page("instruments", [_inst(1)], "group/instruments?page=2"))

    pages = [p async for p in client.iter_pages(first, max_pages=1, raise_on_cap=False)]
    assert len(pages) == 1
    assert router.calls == []


@pytest.mark.asyncio
async def test_iter_pages_loop_error_counts_zero_items_for_unknown_shape(monkeypatch):
    client, _ = _serve(monkeypatch, {"links": [{"next": "same?page=1"}]})

    async def first():
        return Paginated(links=[{"next": "same?page=1"}])

    with pytest.raises(PaginationError) as ei:
        async for _ in client.iter_pages(first):
            pass
    assert ei.value.details["pages_fetched"] == 2
    assert ei.value.details["items_collected"] == 0


# --------------------------------------------------------------------------- #
# Instruments / search
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_list_instruments_sends_optional_params(monkeypatch):
    client, router = _serve(monkeypatch, _page("instruments", [_inst(1)]))
    res = await client.list_instruments_async("G1", instrument_id="I1", page="tok")
    assert [i.instrument_id for i in res.instruments] == ["I1"]
    assert router.calls[0]["url"] == f"{API}/group/instruments"
    assert router.params[0] == {"group-id": "G1", "instrument-id": "I1", "page": "tok"}


@pytest.mark.asyncio
async def test_iter_instruments_follows_next_links(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_inst(1), _inst(2)], "group/instruments?group-id=G1&page=2"),
        _page("instruments", [_inst(3)]),
    )
    ids = [i.instrument_id async for i in client.iter_instruments_async("G1", "I1")]
    assert ids == ["I1", "I2", "I3"]
    assert router.params[0] == {"group-id": "G1", "instrument-id": "I1"}
    assert router.calls[1]["url"] == f"{API}/group/instruments?group-id=G1&page=2"


@pytest.mark.asyncio
async def test_search_instruments_sends_keywords_and_page(monkeypatch):
    client, router = _serve(monkeypatch, _page("instruments", [_inst(1)]))
    res = await client.search_instruments_async("G1", "treasury", page="p2")
    assert len(res.instruments) == 1
    assert router.calls[0]["url"] == f"{API}/group/instruments/search"
    assert router.params[0] == {"group-id": "G1", "keywords": "treasury", "page": "p2"}


@pytest.mark.asyncio
async def test_iter_search_instruments_walks_pages(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_inst(1)], "group/instruments/search?page=2"),
        _page("instruments", [_inst(2)]),
    )
    ids = [i.instrument_id async for i in client.iter_search_instruments_async("G1", "bond")]
    assert ids == ["I1", "I2"]
    assert router.params[0] == {"group-id": "G1", "keywords": "bond"}


# --------------------------------------------------------------------------- #
# Metadata: filters / attributes
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_group_filters_page_param_and_iteration(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("filters", [{"filter-name": "country"}]),
        _page("filters", [{"filter-name": "country"}], "group/filters?page=2"),
        _page("filters", [{"filter-name": "currency"}]),
    )
    res = await client.get_group_filters_async("G1", page="p1")
    assert router.params[0] == {"group-id": "G1", "page": "p1"}
    assert res.filters[0].filter_name == "country"

    names = [f.filter_name async for f in client.iter_group_filters_async("G1")]
    assert names == ["country", "currency"]
    assert router.params[1] == {"group-id": "G1"}


@pytest.mark.asyncio
async def test_group_attributes_params_and_iteration(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_ts_inst(1)]),
        _page("instruments", [_ts_inst(1)], "group/attributes?page=2"),
        _page("instruments", [_ts_inst(2)]),
    )
    res = await client.get_group_attributes_async("G1", instrument_id="I1", page="p1")
    assert router.calls[0]["url"] == f"{API}/group/attributes"
    assert router.params[0] == {"group-id": "G1", "instrument-id": "I1", "page": "p1"}
    assert res.instruments[0].instrument_id == "I1"

    ids = [i.instrument_id async for i in client.iter_group_attributes_async("G1", "I9")]
    assert ids == ["I1", "I2"]
    assert router.params[1] == {"group-id": "G1", "instrument-id": "I9"}


# --------------------------------------------------------------------------- #
# Time series
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_instrument_time_series_sends_dates_and_page(monkeypatch):
    client, router = _serve(monkeypatch, _page("instruments", [_ts_inst(1)]))
    await client.get_instrument_time_series_async(["I1"], ["A1"], start_date="20240101", end_date="20240131", page="p2")
    params = router.params[0]
    assert router.calls[0]["url"] == f"{API}/instruments/time-series"
    assert params["instruments"] == ["I1"] and params["attributes"] == ["A1"]
    assert (params["start-date"], params["end-date"], params["page"]) == ("20240101", "20240131", "p2")


@pytest.mark.asyncio
@pytest.mark.parametrize("kw", [{"start_date": "2024-13-45"}, {"end_date": "bogus"}])
async def test_instrument_time_series_rejects_bad_dates(monkeypatch, kw):
    client, router = _serve(monkeypatch)
    with pytest.raises(ValidationError):
        await client.get_instrument_time_series_async(["I1"], ["A1"], **kw)
    assert router.calls == []


@pytest.mark.asyncio
async def test_expressions_time_series_sends_dates_and_page(monkeypatch):
    client, router = _serve(monkeypatch, _page("instruments", []))
    await client.get_expressions_time_series_async(["DB(A,B)"], start_date="20240101", end_date="20240131", page="p3")
    params = router.params[0]
    assert router.calls[0]["url"] == f"{API}/expressions/time-series"
    assert (params["start-date"], params["end-date"], params["page"]) == ("20240101", "20240131", "p3")


@pytest.mark.asyncio
async def test_expressions_time_series_rejects_bad_end_date(monkeypatch):
    client, router = _serve(monkeypatch)
    with pytest.raises(ValidationError):
        await client.get_expressions_time_series_async(["DB(X)"], end_date="nope")
    assert router.calls == []


@pytest.mark.asyncio
async def test_group_time_series_sends_filter_dates_and_page(monkeypatch):
    client, router = _serve(monkeypatch, _page("instruments", []))
    await client.get_group_time_series_async(
        "G1", ["A1"], filter="country(USA)", start_date="20240101", end_date="20240131", page="p4"
    )
    params = router.params[0]
    assert router.calls[0]["url"] == f"{API}/group/time-series"
    assert params["group-id"] == "G1" and params["filter"] == "country(USA)"
    assert (params["start-date"], params["end-date"], params["page"]) == ("20240101", "20240131", "p4")


@pytest.mark.asyncio
async def test_group_time_series_rejects_bad_start_date(monkeypatch):
    client, router = _serve(monkeypatch)
    with pytest.raises(ValidationError):
        await client.get_group_time_series_async("G1", ["A1"], start_date="x")
    assert router.calls == []


@pytest.mark.asyncio
async def test_iter_instrument_time_series_forwards_options_across_pages(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_ts_inst(1)], "instruments/time-series?page=2"),
        _page("instruments", [_ts_inst(2)]),
    )
    ids = [
        i.instrument_id
        async for i in client.iter_instrument_time_series_async(
            ["I1", "I2"], ["A1"], start_date="20240101", frequency="FREQ_WEEK"
        )
    ]
    assert ids == ["I1", "I2"]
    assert router.params[0]["start-date"] == "20240101"
    assert router.params[0]["frequency"] == "FREQ_WEEK"
    assert router.calls[1]["url"] == f"{API}/instruments/time-series?page=2"


@pytest.mark.asyncio
async def test_iter_expressions_time_series_forwards_options_across_pages(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_ts_inst(1)], "expressions/time-series?page=2"),
        _page("instruments", [_ts_inst(2)]),
    )
    ids = [
        i.instrument_id
        async for i in client.iter_expressions_time_series_async(["DB(X)"], end_date="20240131", data="REFERENCE_DATA")
    ]
    assert ids == ["I1", "I2"]
    assert router.params[0]["expressions"] == ["DB(X)"]
    assert router.params[0]["end-date"] == "20240131"
    assert router.params[0]["data"] == "REFERENCE_DATA"


@pytest.mark.asyncio
async def test_iter_group_time_series_forwards_filter_across_pages(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_ts_inst(1)], "group/time-series?page=2"),
        _page("instruments", [_ts_inst(2)]),
    )
    ids = [i.instrument_id async for i in client.iter_group_time_series_async("G1", ["A1"], filter="currency(USD)")]
    assert ids == ["I1", "I2"]
    assert router.params[0]["filter"] == "currency(USD)"
    assert len(router.calls) == 2


@pytest.mark.asyncio
async def test_iter_time_series_respects_max_pages(monkeypatch):
    client, router = _serve(
        monkeypatch,
        _page("instruments", [_ts_inst(1)], "group/time-series?page=2"),
    )
    with pytest.raises(PaginationError, match="cap hit after 1 pages"):
        async for _ in client.iter_group_time_series_async("G1", ["A1"], max_pages=1):
            pass
    assert len(router.calls) == 1


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_grid_data_by_grid_id_and_date(monkeypatch):
    client, router = _serve(monkeypatch, {"series": [{"expr": "X", "records": []}]})
    res = await client.get_grid_data_async(grid_id="GRID1", date="20240115")
    assert router.calls[0]["url"] == f"{API}/grid-data"
    assert router.params[0] == {"gridId": "GRID1", "date": "20240115"}
    assert len(res.series) == 1


@pytest.mark.asyncio
async def test_grid_data_info_envelope_is_empty_series(monkeypatch):
    client, _ = _serve(monkeypatch, {"info": {"code": 204, "description": "No content"}})
    res = await client.get_grid_data_async(expr="DBGRID(X)")
    assert res.series == []
    assert res.error_code is None
    # The no-content envelope is not carried over as data on the model.
    assert res.model_extra == {}


@pytest.mark.asyncio
async def test_grid_data_empty_expr_is_sent_alongside_grid_id(monkeypatch):
    """Documents current behavior: the exclusivity check uses truthiness but params use ``is not None``.

    ``expr=""`` passes validation as "not given", yet is still sent as ``expr=``
    next to ``gridId``, so the server receives both parameters.
    """
    client, router = _serve(monkeypatch, {"series": []})
    await client.get_grid_data_async(expr="", grid_id="GRID1")
    assert router.params[0] == {"expr": "", "gridId": "GRID1"}
