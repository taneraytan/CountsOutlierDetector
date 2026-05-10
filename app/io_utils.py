"""File loading helpers for the supported input formats."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".txt", ".xls", ".xlsx", ".parquet"}


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def load_dataframe(file: Union[str, Path, io.IOBase], filename: str | None = None,
                   sep: str | None = None, encoding: str = "utf-8") -> pd.DataFrame:
    """Load a dataframe from a path or file-like object.

    Supports CSV, TXT (delimited), Excel (.xls / .xlsx) and Parquet. ``filename``
    must be supplied when ``file`` is a file-like object so the extension can be
    detected.
    """
    name = filename if filename is not None else str(file)
    ext = _ext(name)

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext in {".csv", ".txt"}:
        kwargs = {"encoding": encoding}
        if sep:
            kwargs["sep"] = sep
        else:
            # Let pandas auto-detect delimiter for txt; default comma for csv.
            if ext == ".txt":
                kwargs["sep"] = None
                kwargs["engine"] = "python"
        return pd.read_csv(file, **kwargs)

    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(file)

    if ext == ".parquet":
        return pd.read_parquet(file)

    raise ValueError(f"Unsupported file type: {ext}")


def export_dataframe(df: pd.DataFrame, fmt: str) -> tuple[bytes, str, str]:
    """Serialize ``df`` to bytes for download.

    Returns ``(data, mime, suggested_extension)``.
    """
    fmt = fmt.lower()
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv", "csv"
    if fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="results")
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
