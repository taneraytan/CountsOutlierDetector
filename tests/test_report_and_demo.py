"""Tests for the demo dataset generator and the HTML report builder."""

import pandas as pd
import pytest

from app.demo import generate_demo_dataset
from app.report import build_html_report, format_explanation_items
from counts_outlier_detector import CountsOutlierDetector


def test_demo_dataset_shape_and_determinism():
    a = generate_demo_dataset(n_rows=300, seed=0)
    b = generate_demo_dataset(n_rows=300, seed=0)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 300
    assert set(a.columns) == {"Department", "Region", "Seniority", "Salary", "TenureYears"}


def test_demo_planted_pair_is_rare():
    df = generate_demo_dataset(n_rows=400, seed=0)
    pair = df[(df["Department"] == "Legal") & (df["Region"] == "APAC")]
    assert len(pair) == 2  # exactly the planted rows


def test_demo_rejects_tiny_n():
    with pytest.raises(ValueError):
        generate_demo_dataset(n_rows=10)


def test_format_explanation_items():
    cell = [[["ColA", "ColB"], ["x", "y"]], [["ColC"], ["z"]]]
    lines = format_explanation_items(cell)
    assert lines == ["ColA = x AND ColB = y", "ColC = z"]
    assert format_explanation_items("") == []
    assert format_explanation_items(None) == []
    assert format_explanation_items([["bad"]]) == []


def test_build_html_report_end_to_end():
    df = generate_demo_dataset(n_rows=300, seed=0)
    detector = CountsOutlierDetector(max_dimensions=2)
    results = detector.fit_predict(df)
    res = {
        "label": "Demo <script>alert(1)</script> run",
        "input_df": df,
        "scores": results["Scores"],
        "summary": results["Flagged Summary"],
        "flagged_all": results["Breakdown All Rows"],
        "most_flagged": detector.get_most_flagged_rows(),
        "run_summary": detector.run_summary,
        "params": {"threshold": 0.05, "max_dimensions": 2},
    }
    html_out = build_html_report(res)
    assert html_out.startswith("<!DOCTYPE html>")
    assert "data:image/png;base64," in html_out
    # user-controlled label must be escaped
    assert "<script>alert(1)</script>" not in html_out
    assert "threshold" in html_out


def test_build_html_report_empty_results():
    html_out = build_html_report({"label": "empty", "scores": pd.Series(dtype=int)})
    assert "<!DOCTYPE html>" in html_out
