"""Preprocessing and feature-engineering helpers used by the Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    drop_duplicates: bool = False
    drop_constant_columns: bool = False
    drop_high_missing: bool = False
    high_missing_threshold: float = 0.9  # drop column if missing fraction >= this
    drop_columns: list[str] = field(default_factory=list)

    # Missing values
    numeric_impute: str = "median"          # none / mean / median / zero / drop_rows
    categorical_impute: str = "mode"        # none / mode / constant / drop_rows
    categorical_fill_value: str = "Missing"

    # Outliers (numeric)
    outlier_treatment: str = "none"         # none / iqr_clip / zscore_clip
    iqr_factor: float = 1.5
    zscore_threshold: float = 3.0

    # Scaling (numeric) — only useful for downstream tools, the detector bins regardless
    scaling: str = "none"                   # none / standard / minmax / robust

    # Categorical encoding
    categorical_encoding: str = "none"      # none / onehot / ordinal


@dataclass
class FeatureEngineeringConfig:
    parse_datetimes: list[str] = field(default_factory=list)
    datetime_parts: list[str] = field(default_factory=lambda: ["year", "month", "day", "weekday"])
    drop_original_datetime: bool = True

    log_transform: list[str] = field(default_factory=list)
    sqrt_transform: list[str] = field(default_factory=list)
    bin_columns: list[str] = field(default_factory=list)
    bin_count: int = 5

    interactions: list[tuple[str, str]] = field(default_factory=list)  # multiplicative
    ratios: list[tuple[str, str]] = field(default_factory=list)        # a / b


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def apply_preprocessing(df: pd.DataFrame, cfg: PreprocessConfig) -> tuple[pd.DataFrame, list[str]]:
    """Apply preprocessing in place-safe form. Returns the new df + a log of actions."""
    log: list[str] = []
    df = df.copy()

    if cfg.drop_columns:
        keep = [c for c in cfg.drop_columns if c in df.columns]
        if keep:
            df = df.drop(columns=keep)
            log.append(f"Dropped {len(keep)} user-selected columns: {keep}")

    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        log.append(f"Dropped duplicate rows: {before - len(df)}")

    if cfg.drop_constant_columns:
        constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant:
            df = df.drop(columns=constant)
            log.append(f"Dropped constant columns: {constant}")

    if cfg.drop_high_missing:
        thresh = cfg.high_missing_threshold
        miss_frac = df.isna().mean()
        high_missing = miss_frac[miss_frac >= thresh].index.tolist()
        if high_missing:
            df = df.drop(columns=high_missing)
            log.append(
                f"Dropped {len(high_missing)} columns with >= {thresh:.0%} missing: {high_missing}"
            )

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Imputation
    if cfg.numeric_impute == "drop_rows" and numeric_cols:
        before = len(df)
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        log.append(f"Dropped {before - len(df)} rows with missing numeric values")
    elif cfg.numeric_impute in {"mean", "median", "zero"} and numeric_cols:
        for c in numeric_cols:
            if df[c].isna().any():
                if cfg.numeric_impute == "mean":
                    df[c] = df[c].fillna(df[c].mean())
                elif cfg.numeric_impute == "median":
                    df[c] = df[c].fillna(df[c].median())
                else:
                    df[c] = df[c].fillna(0)
        log.append(f"Numeric imputation: {cfg.numeric_impute}")

    if cfg.categorical_impute == "drop_rows" and categorical_cols:
        before = len(df)
        df = df.dropna(subset=categorical_cols).reset_index(drop=True)
        log.append(f"Dropped {before - len(df)} rows with missing categorical values")
    elif cfg.categorical_impute == "mode" and categorical_cols:
        for c in categorical_cols:
            if df[c].isna().any():
                modes = df[c].mode(dropna=True)
                if len(modes):
                    df[c] = df[c].fillna(modes.iloc[0])
        log.append("Categorical imputation: mode")
    elif cfg.categorical_impute == "constant" and categorical_cols:
        for c in categorical_cols:
            df[c] = df[c].fillna(cfg.categorical_fill_value)
        log.append(f"Categorical imputation: constant '{cfg.categorical_fill_value}'")

    # Outliers
    if cfg.outlier_treatment != "none" and numeric_cols:
        if cfg.outlier_treatment == "iqr_clip":
            for c in numeric_cols:
                q1 = df[c].quantile(0.25)
                q3 = df[c].quantile(0.75)
                iqr = q3 - q1
                lo = q1 - cfg.iqr_factor * iqr
                hi = q3 + cfg.iqr_factor * iqr
                df[c] = df[c].clip(lower=lo, upper=hi)
            log.append(f"Outlier clipping (IQR x{cfg.iqr_factor})")
        elif cfg.outlier_treatment == "zscore_clip":
            for c in numeric_cols:
                mu, sigma = df[c].mean(), df[c].std()
                if sigma and not np.isnan(sigma):
                    lo = mu - cfg.zscore_threshold * sigma
                    hi = mu + cfg.zscore_threshold * sigma
                    df[c] = df[c].clip(lower=lo, upper=hi)
            log.append(f"Outlier clipping (z-score |z| > {cfg.zscore_threshold})")

    # Scaling
    if cfg.scaling != "none" and numeric_cols:
        if cfg.scaling == "standard":
            for c in numeric_cols:
                mu, sigma = df[c].mean(), df[c].std()
                if sigma:
                    df[c] = (df[c] - mu) / sigma
        elif cfg.scaling == "minmax":
            for c in numeric_cols:
                lo, hi = df[c].min(), df[c].max()
                if hi - lo:
                    df[c] = (df[c] - lo) / (hi - lo)
        elif cfg.scaling == "robust":
            for c in numeric_cols:
                med = df[c].median()
                iqr = df[c].quantile(0.75) - df[c].quantile(0.25)
                if iqr:
                    df[c] = (df[c] - med) / iqr
        log.append(f"Scaling: {cfg.scaling}")

    # Encoding categoricals
    if cfg.categorical_encoding == "onehot" and categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)
        log.append(f"One-hot encoded {len(categorical_cols)} categorical column(s)")
    elif cfg.categorical_encoding == "ordinal" and categorical_cols:
        for c in categorical_cols:
            df[c] = pd.Categorical(df[c]).codes
        log.append(f"Ordinal-encoded {len(categorical_cols)} categorical column(s)")

    return df, log


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def apply_feature_engineering(df: pd.DataFrame,
                              cfg: FeatureEngineeringConfig) -> tuple[pd.DataFrame, list[str]]:
    log: list[str] = []
    df = df.copy()

    # Datetime parsing & expansion
    for col in cfg.parse_datetimes:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=False)
        for part in cfg.datetime_parts:
            new = f"{col}_{part}"
            try:
                if part == "year":
                    df[new] = parsed.dt.year
                elif part == "month":
                    df[new] = parsed.dt.month
                elif part == "day":
                    df[new] = parsed.dt.day
                elif part == "weekday":
                    df[new] = parsed.dt.weekday
                elif part == "hour":
                    df[new] = parsed.dt.hour
                elif part == "quarter":
                    df[new] = parsed.dt.quarter
            except Exception:
                continue
        if cfg.drop_original_datetime and col in df.columns:
            df = df.drop(columns=[col])
        log.append(f"Expanded datetime '{col}' → {cfg.datetime_parts}")

    # Numeric transforms
    for col in cfg.log_transform:
        if col in df.columns and is_numeric_dtype(df[col]):
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
            log.append(f"log1p('{col}')")

    for col in cfg.sqrt_transform:
        if col in df.columns and is_numeric_dtype(df[col]):
            df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
            log.append(f"sqrt('{col}')")

    # Quantile binning
    for col in cfg.bin_columns:
        if col in df.columns and is_numeric_dtype(df[col]):
            try:
                df[f"{col}_bin"] = pd.qcut(
                    df[col], q=cfg.bin_count, duplicates="drop", labels=False
                )
                log.append(f"qcut('{col}', q={cfg.bin_count})")
            except Exception:
                continue

    # Interactions and ratios
    for a, b in cfg.interactions:
        if a in df.columns and b in df.columns and is_numeric_dtype(df[a]) and is_numeric_dtype(df[b]):
            df[f"{a}_x_{b}"] = df[a] * df[b]
            log.append(f"interaction {a} * {b}")

    for a, b in cfg.ratios:
        if a in df.columns and b in df.columns and is_numeric_dtype(df[a]) and is_numeric_dtype(df[b]):
            denom = df[b].replace(0, np.nan)
            df[f"{a}_div_{b}"] = df[a] / denom
            log.append(f"ratio {a} / {b}")

    return df, log


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        col = df[c]
        rows.append({
            "Column": c,
            "Type": str(col.dtype),
            "Non-null": int(col.notna().sum()),
            "Missing": int(col.isna().sum()),
            "Missing %": round(col.isna().mean() * 100, 2),
            "Unique": int(col.nunique(dropna=True)),
            "Sample": _sample_str(col),
        })
    return pd.DataFrame(rows)


def _sample_str(col: pd.Series, n: int = 3) -> str:
    vals = col.dropna().head(n).astype(str).tolist()
    return ", ".join(vals)


def suggest_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Heuristic: any column whose name contains 'date'/'time' or that already
    parses as datetime."""
    candidates = []
    for c in df.columns:
        if is_datetime64_any_dtype(df[c]):
            candidates.append(c)
            continue
        lname = c.lower()
        if any(k in lname for k in ("date", "time", "dt", "timestamp")):
            candidates.append(c)
    return candidates
