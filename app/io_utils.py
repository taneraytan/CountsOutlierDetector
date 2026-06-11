"""File loading helpers for the supported input formats."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Union

import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

# Legacy binary .xls is intentionally unsupported: parsing it requires the
# legacy xlrd engine and the format is rare enough that the extra attack
# surface isn't worth it. Convert to .xlsx instead.
SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xlsx", ".parquet"}

# Cell prefixes that spreadsheet applications interpret as formulas (the
# OWASP "CSV injection" set, plus tab/CR which can smuggle a prefix through).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def load_dataframe(file: Union[str, Path, io.IOBase], filename: str | None = None,
                   sep: str | None = None, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a dataframe from a path or file-like object.

    Supports CSV, TXT (delimited), Excel (.xlsx) and Parquet. ``filename``
    must be supplied when ``file`` is a file-like object so the extension can
    be detected.
    """
    name = filename if filename is not None else str(file)
    ext = _ext(name)

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext in {".csv", ".txt"}:
        kwargs: dict[str, Any] = {"encoding": encoding}
        if sep:
            kwargs["sep"] = sep
        else:
            # Let pandas auto-detect delimiter for txt; default comma for csv.
            if ext == ".txt":
                kwargs["sep"] = None
                kwargs["engine"] = "python"
        return pd.read_csv(file, **kwargs)

    if ext == ".xlsx":
        return pd.read_excel(file)

    if ext == ".parquet":
        return pd.read_parquet(file)

    raise ValueError(f"Unsupported file type: {ext}")


def _neutralize_cell(value):
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_for_spreadsheet(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralize formula-injection payloads before a CSV/Excel export.

    String cells (and column names) starting with ``=``, ``+``, ``-``, ``@``
    or control characters are prefixed with a quote so spreadsheet software
    treats them as text rather than executing them as formulas.
    """
    out = df.copy()
    for col in out.columns:
        # Cover both the classic object dtype and pandas' newer string dtype.
        if is_object_dtype(out[col]) or is_string_dtype(out[col]):
            out[col] = out[col].map(_neutralize_cell)
    out.columns = [_neutralize_cell(str(c)) for c in out.columns]
    return out


def safe_filename(name: str, default: str = "results") -> str:
    """Reduce a user-supplied label to a safe download filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "").strip("._-")
    return cleaned or default


def export_dataframe(df: pd.DataFrame, fmt: str) -> tuple[bytes, str, str]:
    """Serialize ``df`` to bytes for download.

    Returns ``(data, mime, suggested_extension)``.
    """
    fmt = fmt.lower()
    if fmt == "csv":
        safe = sanitize_for_spreadsheet(df)
        return safe.to_csv(index=False).encode("utf-8"), "text/csv", "csv"
    if fmt == "xlsx":
        safe = sanitize_for_spreadsheet(df)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            safe.to_excel(writer, index=False, sheet_name="results")
        return buf.getvalue(), (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ), "xlsx"
    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        return buf.getvalue(), "application/octet-stream", "parquet"
    if fmt == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8"), "application/json", "json"
    raise ValueError(f"Unknown export format: {fmt}")
