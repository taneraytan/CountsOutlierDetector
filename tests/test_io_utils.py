"""Tests for file loading / exporting helpers."""

import io

import pandas as pd
import pytest

from app.io_utils import (
    SUPPORTED_EXTENSIONS,
    export_dataframe,
    load_dataframe,
    safe_filename,
    sanitize_for_spreadsheet,
)


def test_load_csv():
    buf = io.BytesIO(b"a,b\n1,x\n2,y\n")
    df = load_dataframe(buf, filename="data.csv")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_csv_custom_sep():
    buf = io.BytesIO(b"a;b\n1;x\n")
    df = load_dataframe(buf, filename="data.csv", sep=";")
    assert list(df.columns) == ["a", "b"]


def test_unsupported_extension_rejected():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(io.BytesIO(b""), filename="data.pdf")


def test_legacy_xls_rejected():
    assert ".xls" not in SUPPORTED_EXTENSIONS
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_dataframe(io.BytesIO(b""), filename="legacy.xls")


def test_parquet_roundtrip():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    data, mime, ext = export_dataframe(df, "parquet")
    assert ext == "parquet"
    loaded = load_dataframe(io.BytesIO(data), filename="data.parquet")
    pd.testing.assert_frame_equal(loaded, df)


def test_csv_formula_injection_neutralized():
    df = pd.DataFrame({
        "name": ["=cmd|' /C calc'!A0", "+SUM(A1:A9)", "@evil", "-1+1", "safe"],
        "n": [1, 2, 3, 4, 5],
    })
    data, _, _ = export_dataframe(df, "csv")
    text = data.decode("utf-8")
    assert "'=cmd" in text
    assert "'+SUM" in text
    assert "'@evil" in text
    assert "'-1+1" in text
    assert "\nsafe," in text  # untouched


def test_xlsx_formula_injection_neutralized():
    df = pd.DataFrame({"=bad_header": ["=HYPERLINK(\"http://evil\")", "ok"]})
    data, _, ext = export_dataframe(df, "xlsx")
    assert ext == "xlsx"
    loaded = pd.read_excel(io.BytesIO(data))
    assert loaded.columns[0].startswith("'=")
    assert loaded.iloc[0, 0].startswith("'=")


def test_sanitize_preserves_numbers():
    df = pd.DataFrame({"n": [-5, 1.5], "s": ["plain", "text"]})
    out = sanitize_for_spreadsheet(df)
    assert (out["n"] == df["n"]).all()
    assert (out["s"] == df["s"]).all()


def test_json_export_not_sanitized_structurally():
    df = pd.DataFrame({"a": ["=x"]})
    data, mime, ext = export_dataframe(df, "json")
    assert ext == "json"
    assert b"=x" in data  # JSON is not a spreadsheet format


@pytest.mark.parametrize("raw,expected", [
    ("Run @ 2026-06-11 10:30", "Run_2026-06-11_10_30"),
    ("../../etc/passwd", "etc_passwd"),
    ("", "results"),
    ("///", "results"),
])
def test_safe_filename(raw, expected):
    assert safe_filename(raw) == expected
