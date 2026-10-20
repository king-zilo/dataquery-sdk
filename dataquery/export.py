"""CSV export for DataQuery time-series and grid-data responses."""

import csv
import io
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .types.exceptions import DataQueryError

_TIMESERIES_FIELDS: Final = (
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
)


@dataclass(frozen=True, slots=True)
class _CsvTable:
    """Flattened rows plus the server messages collected along the way."""

    fieldnames: tuple[str, ...]
    rows: list[dict[str, Any]]
    messages: list[str]


def _to_dict(payload: Any) -> dict[str, Any]:
    """Normalize a Pydantic response or raw dict into a plain dict."""
    if hasattr(payload, "model_dump"):
        return payload.model_dump(by_alias=True)
    if isinstance(payload, dict):
        return payload
    raise DataQueryError(
        "Cannot export: response payload is not a dict or Pydantic model.",
    )


def _field(mapping: Mapping[str, Any], *keys: str) -> Any:
    """Return the first value under ``keys`` that is not ``None`` or ``""``."""
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return ""


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list | tuple) else []


def _dicts(value: Any) -> Iterator[dict[str, Any]]:
    """Yield the dict entries of a list, skipping anything malformed."""
    return (item for item in _as_list(value) if isinstance(item, dict))


def _server_error(code: Any, message: Any) -> str:
    text = "" if message is None else str(message)
    if code is None or code == "":
        return text
    return f"[{code}] {text}".rstrip()


def _error_of(mapping: Mapping[str, Any]) -> str | None:
    """Format a mapping's ``errorCode``/``errorMessage``, or ``None`` if it has neither."""
    code = _field(mapping, "errorCode", "error_code")
    message = _field(mapping, "errorMessage", "error_message")
    if code == "" and message == "":
        return None
    return _server_error(code, message)


def _write_csv(table: _CsvTable, output_path: str) -> dict[str, Any]:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=table.fieldnames)
    writer.writeheader()
    writer.writerows(table.rows)
    content = buf.getvalue()

    result: dict[str, Any]
    if output_path == "-":
        result = {"path": "stdout", "rows": len(table.rows), "content": content}
    else:
        try:
            encoded = content.encode("utf-8")
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded)
        except (OSError, UnicodeEncodeError) as exc:
            raise DataQueryError(f"Failed to write CSV to {output_path}: {exc}") from exc
        result = {"path": output_path, "rows": len(table.rows)}
    if table.messages:
        result["messages"] = table.messages
    return result


def _flatten_timeseries(data: dict[str, Any]) -> _CsvTable:
    instruments = _as_list(data.get("instruments"))
    if not instruments:
        detail = ""
        info = data.get("info")
        if isinstance(info, dict) and (text := info.get("description") or info.get("message")):
            detail = f" Server info: {_server_error(info.get('code'), text)}"
        raise DataQueryError(f"No instruments found in the response to export.{detail}")

    rows: list[dict[str, Any]] = []
    messages: list[str] = []
    for inst in _dicts(instruments):
        inst_id = _field(inst, "instrument-id", "instrument_id")
        inst_name = _field(inst, "instrument-name", "instrument_name")
        group = inst.get("group")
        if not isinstance(group, dict):
            group = {}
        group_id = _field(group, "group-id", "group_id")
        group_name = _field(group, "group-name", "group_name")
        for attr in _dicts(inst.get("attributes")):
            attr_id = _field(attr, "attribute-id", "attribute_id")
            expression = attr.get("expression", "")
            if message := attr.get("message"):
                messages.append(f"{expression or attr_id or inst_id}: {message}")
            base = {
                "instrument_id": inst_id,
                "instrument_name": inst_name,
                "attribute_id": attr_id,
                "attribute_name": _field(attr, "attribute-name", "attribute_name"),
                "expression": expression,
                "label": attr.get("label", ""),
                "last_published": _field(attr, "last-published", "last_published"),
                "group_id": group_id,
                "group_name": group_name,
            }
            for point in attr.get("time-series") or attr.get("time_series") or []:
                if isinstance(point, list) and len(point) >= 2:
                    rows.append({"date": point[0], "value": point[1], **base})

    if not rows:
        detail = f" Server messages: {'; '.join(messages)}" if messages else ""
        raise DataQueryError(f"Response contains instruments but no time-series data points.{detail}")
    return _CsvTable(_TIMESERIES_FIELDS, rows, messages)


def _flatten_grid(data: dict[str, Any]) -> _CsvTable:
    series = _as_list(data.get("series"))
    if not series:
        error = _error_of(data)
        detail = f" Server error: {error}" if error is not None else ""
        raise DataQueryError(f"No grid series found in the response to export.{detail}")

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for s in _dicts(series):
        expr = _field(s, "expr", "expression")
        if (error := _error_of(s)) is not None:
            errors.append(f"{expr}: {error}")
        for record in s.get("records", []) or []:
            if isinstance(record, dict):
                row: dict[str, Any] = {"expression": expr}
                for key, value in record.items():
                    row["record_expression" if key == "expression" else key] = value
                rows.append(row)

    if not rows:
        detail = f" Server errors: {'; '.join(errors)}" if errors else ""
        raise DataQueryError(f"Grid series found but no records to export.{detail}")
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    return _CsvTable(fieldnames, rows, errors)


def export_timeseries_csv(response: Any, output_path: str) -> dict[str, Any]:
    """Flatten a time-series response into a CSV file (or stdout when path is ``-``)."""
    return _write_csv(_flatten_timeseries(_to_dict(response)), output_path)


def export_grid_csv(response: Any, output_path: str) -> dict[str, Any]:
    """Flatten a grid-data response into a CSV file (or stdout when path is ``-``)."""
    return _write_csv(_flatten_grid(_to_dict(response)), output_path)
