"""SQLite-backed storage for datasets and analysis runs.

Schema (all stored in a single SQLite file):

    datasets        — uploaded data sources, identified by name + content hash
    runs            — one row per detector execution
    run_results     — flagged rows for each run (one row each)

Datasets and run results are persisted as parquet bytes in BLOB columns so the
exact dataframe can be round-tripped without column-type loss.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


DEFAULT_DB_PATH = Path(os.environ.get(
    "COUNTS_DB_PATH",
    Path.home() / ".counts_outlier_detector" / "data.db",
))


SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    n_rows       INTEGER NOT NULL,
    n_cols       INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    parquet      BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    label         TEXT,
    created_at    TEXT NOT NULL,
    params_json   TEXT NOT NULL,
    summary_json  TEXT NOT NULL,
    n_flagged     INTEGER NOT NULL,
    run_summary   TEXT
);

CREATE TABLE IF NOT EXISTS run_results (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    parquet   BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_dataset ON runs(dataset_id);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def init_db(path: Path | str = DEFAULT_DB_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(p) as conn:
        conn.executescript(SCHEMA)
    return p


@contextmanager
def connect(path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    p = init_db(path)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _parquet_bytes_to_df(blob: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(blob))


def _df_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(_df_to_parquet_bytes(df))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def save_dataset(df: pd.DataFrame, name: str,
                 path: Path | str = DEFAULT_DB_PATH) -> int:
    """Insert (or return existing id for) a dataset. De-duplicated by content hash."""
    digest = _df_hash(df)
    with connect(path) as conn:
        existing = conn.execute(
            "SELECT id FROM datasets WHERE content_hash = ?", (digest,)
        ).fetchone()
        if existing:
            return int(existing["id"])

        cursor = conn.execute(
            "INSERT INTO datasets (name, content_hash, n_rows, n_cols, "
            "columns_json, created_at, parquet) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                name,
                digest,
                len(df),
                df.shape[1],
                json.dumps(list(df.columns.astype(str))),
                datetime.utcnow().isoformat(timespec="seconds"),
                _df_to_parquet_bytes(df),
            ),
        )
        return int(cursor.lastrowid)


def list_datasets(path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT id, name, n_rows, n_cols, created_at FROM datasets "
            "ORDER BY created_at DESC"
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def load_dataset(dataset_id: int, path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT parquet FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"No dataset with id={dataset_id}")
    return _parquet_bytes_to_df(row["parquet"])


def delete_dataset(dataset_id: int, path: Path | str = DEFAULT_DB_PATH) -> None:
    with connect(path) as conn:
        conn.execute("DELETE FROM run_results WHERE run_id IN "
                     "(SELECT id FROM runs WHERE dataset_id = ?)", (dataset_id,))
        conn.execute("DELETE FROM runs WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def save_run(dataset_id: int, label: str, params: dict, summary: dict,
             flagged_df: pd.DataFrame, run_summary: str = "",
             path: Path | str = DEFAULT_DB_PATH) -> int:
    n_flagged = int((flagged_df.get("TOTAL SCORE", pd.Series(dtype=int)) > 0).sum()) \
        if isinstance(flagged_df, pd.DataFrame) else 0
    with connect(path) as conn:
        cursor = conn.execute(
            "INSERT INTO runs (dataset_id, label, created_at, params_json, "
            "summary_json, n_flagged, run_summary) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                dataset_id,
                label,
                datetime.utcnow().isoformat(timespec="seconds"),
                json.dumps(params, default=str),
                json.dumps(summary, default=str),
                n_flagged,
                run_summary,
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO run_results (run_id, parquet) VALUES (?, ?)",
            (run_id, _df_to_parquet_bytes(flagged_df)),
        )
        return run_id


def list_runs(dataset_id: Optional[int] = None,
              path: Path | str = DEFAULT_DB_PATH) -> pd.DataFrame:
    with connect(path) as conn:
        if dataset_id is None:
            rows = conn.execute(
                "SELECT r.id, r.label, r.created_at, r.n_flagged, "
                "       d.name AS dataset_name, r.dataset_id "
                "FROM runs r JOIN datasets d ON r.dataset_id = d.id "
                "ORDER BY r.created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT r.id, r.label, r.created_at, r.n_flagged, "
                "       d.name AS dataset_name, r.dataset_id "
                "FROM runs r JOIN datasets d ON r.dataset_id = d.id "
                "WHERE r.dataset_id = ? ORDER BY r.created_at DESC",
                (dataset_id,),
            ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def load_run(run_id: int, path: Path | str = DEFAULT_DB_PATH) -> dict:
    with connect(path) as conn:
        run = conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if run is None:
            raise KeyError(f"No run with id={run_id}")
        result = conn.execute(
            "SELECT parquet FROM run_results WHERE run_id = ?", (run_id,)
        ).fetchone()
    return {
        "id": run["id"],
        "dataset_id": run["dataset_id"],
        "label": run["label"],
        "created_at": run["created_at"],
        "params": json.loads(run["params_json"]),
        "summary": json.loads(run["summary_json"]),
        "n_flagged": run["n_flagged"],
        "run_summary": run["run_summary"],
        "results": _parquet_bytes_to_df(result["parquet"]) if result else pd.DataFrame(),
    }


def delete_run(run_id: int, path: Path | str = DEFAULT_DB_PATH) -> None:
    with connect(path) as conn:
        conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
