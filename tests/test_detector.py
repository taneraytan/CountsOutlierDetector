"""Tests for the CountsOutlierDetector library."""

import numpy as np
import pandas as pd
import pytest

from app.demo import generate_demo_dataset, planted_outlier_positions
from counts_outlier_detector import CountsOutlierDetector


@pytest.fixture(scope="module")
def demo_df() -> pd.DataFrame:
    return generate_demo_dataset(n_rows=600, seed=0)


@pytest.fixture(scope="module")
def demo_results(demo_df):
    detector = CountsOutlierDetector(n_bins=7, max_dimensions=3, threshold=0.05)
    results = detector.fit_predict(demo_df)
    return detector, results


def test_planted_outliers_are_flagged(demo_df, demo_results):
    _, results = demo_results
    scores = results["Scores"]
    for pos in planted_outlier_positions(len(demo_df)):
        assert scores.iloc[pos] > 0, f"planted outlier at row {pos} was not flagged"


def test_flagged_fraction_is_small(demo_df, demo_results):
    _, results = demo_results
    scores = results["Scores"]
    assert 0 < (scores > 0).mean() < 0.10


def test_deterministic(demo_df):
    r1 = CountsOutlierDetector(max_dimensions=2).fit_predict(demo_df)
    r2 = CountsOutlierDetector(max_dimensions=2).fit_predict(demo_df)
    pd.testing.assert_series_equal(r1["Scores"], r2["Scores"])


def test_most_flagged_rows_sorted(demo_results):
    detector, _ = demo_results
    most = detector.get_most_flagged_rows()
    assert not most.empty
    assert most["TOTAL SCORE"].is_monotonic_decreasing
    assert (most["TOTAL SCORE"] > 0).all()


def test_explanations_use_original_category_labels(demo_results):
    detector, results = demo_results
    flagged_all = results["Breakdown All Rows"]
    labels_seen = set()
    for cell in flagged_all["2d Explanations"]:
        if isinstance(cell, str):
            continue
        for item in cell:
            cols, vals = item[0], item[1]
            for c, v in zip(cols, vals):
                if c in ("Department", "Region"):
                    labels_seen.add(str(v))
    # The planted Legal/APAC pair must surface with its original labels.
    assert "Legal" in labels_seen
    assert "APAC" in labels_seen


def test_date_string_column_does_not_crash():
    """Regression: ISO-date strings used to be misclassified as numeric and
    crash the astype(float) conversion."""
    n = 200
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "when": pd.date_range("2023-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "a": rng.choice(["x", "y", "z"], size=n),
        "b": rng.normal(size=n),
    })
    detector = CountsOutlierDetector(max_dimensions=2)
    results = detector.fit_predict(df)
    assert len(results["Scores"]) == n


def test_numeric_string_column_is_numeric():
    n = 200
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "a": [str(x) for x in rng.normal(100, 15, size=n).round(2)],
        "b": rng.choice(["u", "v"], size=n),
        "c": rng.normal(size=n),
    })
    detector = CountsOutlierDetector(max_dimensions=2)
    detector.fit_predict(df)
    assert "a" in (detector.numeric_col_names or [])


def test_num_combinations_scales_with_dimension(demo_df):
    """Regression: the estimate used a fixed exponent of 2, badly
    underestimating higher-dimensional work."""
    detector = CountsOutlierDetector(max_dimensions=2)
    detector.fit_predict(demo_df)
    estimate = detector._CountsOutlierDetector__get_num_combinations
    avg_unique = np.mean([len(x) for x in detector.unique_vals])
    if avg_unique > 1 and len(detector.data_df.columns) >= 4:
        assert estimate(dim=3) > estimate(dim=2)
        # the dim exponent must matter beyond the binomial factor
        import math
        nc = len(detector.data_df.columns)
        ratio = estimate(dim=3) / estimate(dim=2)
        binomial_ratio = math.comb(nc, 3) / math.comb(nc, 2)
        assert ratio > binomial_ratio  # extra factor of avg_unique


def test_no_combination_subsumption(demo_df):
    """A combination flagged at dimension d must not reappear as part of a
    flagged combination at a higher dimension."""
    detector = CountsOutlierDetector(n_bins=4, max_dimensions=6, threshold=0.2,
                                     max_num_combinations=10_000_000)
    results = detector.fit_predict(demo_df)
    flagged_all = results["Breakdown All Rows"]

    flagged_combos_by_dim: dict[int, set] = {d: set() for d in range(1, 7)}
    for d in range(1, 7):
        col = f"{d}d Explanations"
        for cell in flagged_all[col]:
            if isinstance(cell, str):
                continue
            for item in cell:
                cols = tuple(str(c) for c in item[0])
                vals = tuple(str(v) for v in item[1])
                flagged_combos_by_dim[d].add((cols, vals))

    def subsets(cols, vals, size):
        from itertools import combinations
        idx = range(len(cols))
        for pick in combinations(idx, size):
            yield (tuple(cols[i] for i in pick), tuple(vals[i] for i in pick))

    for d in range(2, 7):
        for cols, vals in flagged_combos_by_dim[d]:
            for lower in range(1, d):
                for sub in subsets(cols, vals, lower):
                    assert sub not in flagged_combos_by_dim[lower], (
                        f"{d}d combination {cols}/{vals} contains already-flagged "
                        f"{lower}d combination {sub}"
                    )


def test_time_budget_truncates(demo_df):
    # A microscopic (but non-zero — zero means unlimited) budget must truncate.
    detector = CountsOutlierDetector(max_dimensions=6, max_execution_seconds=1e-6,
                                     max_num_combinations=10_000_000)
    results = detector.fit_predict(demo_df)
    assert detector.truncated
    assert "time budget" in (detector.run_summary or "")
    # 1d results are always produced
    assert len(results["Scores"]) == len(demo_df)


def test_single_column_input():
    df = pd.DataFrame({"only": ["a", "b", "a", "b"] * 10})
    detector = CountsOutlierDetector()
    results = detector.fit_predict(df)
    assert (results["Scores"] == 0).all()


def test_parallel_matches_serial(demo_df):
    serial = CountsOutlierDetector(max_dimensions=3).fit_predict(demo_df)
    par_detector = CountsOutlierDetector(max_dimensions=3, run_parallel=True)
    parallel = par_detector.fit_predict(demo_df)
    pd.testing.assert_series_equal(serial["Scores"], parallel["Scores"])
