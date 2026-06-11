"""Tests for preprocessing and feature engineering."""

import numpy as np
import pandas as pd

from app.preprocessing import (
    FeatureEngineeringConfig,
    PreprocessConfig,
    apply_feature_engineering,
    apply_preprocessing,
    column_summary,
    encode_for_model,
    suggest_datetime_columns,
)


def _df():
    return pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 100.0],
        "cat": ["a", "b", None, "b", "b"],
        "const": [1, 1, 1, 1, 1],
        "when": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01"],
    })


def test_drop_columns_and_constants():
    cfg = PreprocessConfig(drop_columns=["when"], drop_constant_columns=True)
    out, log = apply_preprocessing(_df(), cfg)
    assert "when" not in out.columns
    assert "const" not in out.columns
    assert any("Dropped 1 user-selected columns" in line for line in log)


def test_numeric_median_impute():
    cfg = PreprocessConfig()
    out, _ = apply_preprocessing(_df(), cfg)
    assert not out["num"].isna().any()
    assert out.loc[2, "num"] == 3.0  # median of [1, 2, 4, 100]


def test_categorical_constant_impute():
    cfg = PreprocessConfig(categorical_impute="constant", categorical_fill_value="MISSING")
    out, _ = apply_preprocessing(_df(), cfg)
    assert out.loc[2, "cat"] == "MISSING"


def test_iqr_clip():
    cfg = PreprocessConfig(outlier_treatment="iqr_clip", iqr_factor=1.5)
    out, _ = apply_preprocessing(_df(), cfg)
    assert out["num"].max() < 100.0


def test_datetime_expansion():
    cfg = FeatureEngineeringConfig(parse_datetimes=["when"],
                                   datetime_parts=["year", "month"])
    out, log = apply_feature_engineering(_df(), cfg)
    assert "when_year" in out.columns
    assert "when_month" in out.columns
    assert "when" not in out.columns  # dropped by default
    assert out["when_month"].tolist() == [1, 2, 3, 4, 5]


def test_ratio_feature_handles_zero_denominator():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 4.0]})
    cfg = FeatureEngineeringConfig(ratios=[("a", "b")])
    out, _ = apply_feature_engineering(df, cfg)
    assert np.isnan(out.loc[0, "a_div_b"])
    assert out.loc[1, "a_div_b"] == 0.5


def test_suggest_datetime_columns():
    df = _df().rename(columns={"when": "event_date"})
    assert "event_date" in suggest_datetime_columns(df)
    # parsed datetime columns are suggested regardless of name
    df["when"] = pd.to_datetime(_df()["when"])
    assert "when" in suggest_datetime_columns(df)


def test_column_summary_shape():
    summary = column_summary(_df())
    assert len(summary) == _df().shape[1]
    assert "Missing %" in summary.columns


def test_encode_for_model_all_numeric():
    out = encode_for_model(_df())
    assert all(np.issubdtype(dt, np.number) for dt in out.dtypes)
    assert not out.isna().any().any()
    assert out.shape == _df().shape
