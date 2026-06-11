"""Synthetic demo dataset with planted outliers.

Used for onboarding (the "Load demo dataset" button), as a regression-test
fixture, and as a quick way to sanity-check parameter changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Row positions (from the end of the frame) that carry planted anomalies.
N_PLANTED = 3


def generate_demo_dataset(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Return a deterministic synthetic HR-style dataset.

    The last :data:`N_PLANTED` rows are planted outliers:

    * two rows with the (Department=Legal, Region=APAC) combination, which
      never occurs naturally — a 2-D rarity;
    * one row with an extreme salary — a 1-D rarity.
    """
    if n_rows < 50:
        raise ValueError("n_rows must be at least 50 for the demo to make sense")

    rng = np.random.default_rng(seed)

    dept = rng.choice(
        ["Sales", "Engineering", "Support", "Legal"],
        size=n_rows, p=[0.40, 0.30, 0.22, 0.08],
    )
    region = rng.choice(["NA", "EMEA", "APAC"], size=n_rows, p=[0.5, 0.3, 0.2])
    seniority = rng.choice(["Junior", "Mid", "Senior"], size=n_rows, p=[0.4, 0.4, 0.2])

    base = {"Sales": 60_000, "Engineering": 90_000, "Support": 50_000, "Legal": 85_000}
    salary = np.array([base[d] for d in dept], dtype=float)
    salary += rng.normal(0, 8_000, size=n_rows)
    tenure_years = np.clip(rng.gamma(2.0, 2.0, size=n_rows), 0, 25).round(1)

    # Keep (Legal, APAC) out of the natural data so the planted pair is rare.
    region = np.where(dept == "Legal",
                      rng.choice(["NA", "EMEA"], size=n_rows), region)

    df = pd.DataFrame({
        "Department": dept,
        "Region": region,
        "Seniority": seniority,
        "Salary": salary.round(0),
        "TenureYears": tenure_years,
    })

    # Planted 2-D outliers: Legal employees in APAC.
    for pos in (n_rows - 3, n_rows - 2):
        df.loc[pos, ["Department", "Region"]] = ["Legal", "APAC"]
    # Planted 1-D outlier: an extreme salary.
    df.loc[n_rows - 1, "Salary"] = 950_000.0

    return df


def planted_outlier_positions(n_rows: int = 600) -> list[int]:
    """Row positions of the planted outliers in :func:`generate_demo_dataset`."""
    return [n_rows - 3, n_rows - 2, n_rows - 1]
