"""CSV export for DataQuery time-series and grid-data responses."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any, Dict, List

from .types.exceptions import DataQueryError


def _to_dict(payload: Any) -> Dict[str, Any]:
    """Normalize a Pydantic response or raw dict into a plain dict."""
    if hasattr(payload, "model_dump"):
        return payload.model_dump(by_alias=True)
    if isinstance(payload, dict):
        return payload
    raise DataQueryError(
        "Cannot export: response payload is not a dict or Pydantic model.",
    )


def _field(mapping: Dict[str, Any], *keys: str) -> Any:
    """Return the first value under ``keys`` that is not ``None`` or ``""``."""
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return ""


def _as_list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _server_error(code: Any, message: Any) -> str:
    text = "" if message is None else str(message)
    if code is None or code == "":
        return text
    return f"[{code}] {text}".rstrip()


def _render_csv(fieldnames: List[str], rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _write_file(output_path: str, content: str) -> None:
    try:
        encoded = content.encode("utf-8")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(encoded)
    except (OSError, UnicodeEncodeError) as exc:
        raise DataQueryError(f"Failed to write CSV to {output_path}: {exc}") from exc


def export_timeseries_csv(response: Any, output_path: str) -> Dict[str, Any]:
    """Flatten a time-series response into a CSV file (or stdout when path is ``-``)."""
    data = _to_dict(response)
    instruments = _as_list(data.get("instruments"))
    if not instruments:
        detail = ""
        info = data.get("info")
        if isinstance(info, dict) and (info.get("description") or info.get("message")):
            detail = f" Server info: {_server_error(info.get('code'), info.get('description') or info.get('message'))}"
        raise DataQueryError(
            f"No instruments found in the response to export.{detail}",
        )

    rows: List[Dict[str, Any]] = []
    messages: List[str] = []
    for inst in instruments:
        if not isinstance(inst, dict):
            continue
        inst_id = _field(inst, "instrument-id", "instrument_id")
        inst_name = _field(inst, "instrument-name", "instrument_name")
        group = inst.get("group")
        if not isinstance(group, dict):
            group = {}
        group_id = _field(group, "group-id", "group_id")
        group_name = _field(group, "group-name", "group_name")
        for attr in _as_list(inst.get("attributes")):
            if not isinstance(attr, dict):
                continue
            attr_id = _field(attr, "attribute-id", "attribute_id")
            attr_name = _field(attr, "attribute-name", "attribute_name")
            expression = attr.get("expression", "")
            label = attr.get("label", "")
            last_pub = _field(attr, "last-published", "last_published")
            if attr.get("message"):
                messages.append(f"{expression or attr_id or inst_id}: {attr['message']}")
            ts = attr.get("time-series") or attr.get("time_series") or []
            for point in ts:
                if isinstance(point, list) and len(point) >= 2:
                    rows.append(
                        {
                            "date": point[0],
                            "value": point[1],
                            "instrument_id": inst_id,
                            "instrument_name": inst_name,
                            "attribute_id": attr_id,
                            "attribute_name": attr_name,
                            "expression": expression,
                            "label": label,
                            "last_published": last_pub,
                            "group_id": group_id,
                            "group_name": group_name,
                        }
                    )

    if not rows:
        detail = f" Server messages: {'; '.join(messages)}" if messages else ""
        raise DataQueryError(
            f"Response contains instruments but no time-series data points.{detail}",
        )

    fieldnames = [
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

    content = _render_csv(fieldnames, rows)
    result: Dict[str, Any]
    if output_path == "-":
        result = {"path": "stdout", "rows": len(rows), "content": content}
    else:
        _write_file(output_path, content)
        result = {"path": output_path, "rows": len(rows)}
    if messages:
        result["messages"] = messages
    return result


def export_grid_csv(response: Any, output_path: str) -> Dict[str, Any]:
    """Flatten a grid-data response into a CSV file (or stdout when path is ``-``)."""
    data = _to_dict(response)
    series = _as_list(data.get("series"))
    if not series:
        detail = ""
        code = _field(data, "errorCode", "error_code")
        message = _field(data, "errorMessage", "error_message")
        if code != "" or message != "":
            detail = f" Server error: {_server_error(code, message)}"
        raise DataQueryError(
            f"No grid series found in the response to export.{detail}",
        )

    all_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    for s in series:
        if not isinstance(s, dict):
            continue
        expr = _field(s, "expr", "expression")
        code = _field(s, "errorCode", "error_code")
        message = _field(s, "errorMessage", "error_message")
        if code != "" or message != "":
            errors.append(f"{expr}: {_server_error(code, message)}")
        for record in s.get("records", []) or []:
            if isinstance(record, dict):
                row: Dict[str, Any] = {"expression": expr}
                for key, value in record.items():
                    row["record_expression" if key == "expression" else key] = value
                all_rows.append(row)

    if not all_rows:
        detail = f" Server errors: {'; '.join(errors)}" if errors else ""
        raise DataQueryError(
            f"Grid series found but no records to export.{detail}",
        )

    seen: Dict[str, None] = {}
    for row in all_rows:
        for key in row:
            seen.setdefault(key, None)
    fieldnames = list(seen)

    content = _render_csv(fieldnames, all_rows)
    result: Dict[str, Any]
    if output_path == "-":
        result = {"path": "stdout", "rows": len(all_rows), "content": content}
    else:
        _write_file(output_path, content)
        result = {"path": output_path, "rows": len(all_rows)}
    if errors:
        result["messages"] = errors
    return result
