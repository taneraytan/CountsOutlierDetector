"""Tests for the SQLite persistence layer."""

import os
import sqlite3
import stat
import sys

import pandas as pd
import pytest

from app import database as db


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": ["w", "x", "y", "z"],
        "TOTAL SCORE": [0, 2, 0, 1],
    })


def test_dataset_roundtrip(db_path, sample_df):
    ds_id = db.save_dataset(sample_df, "sample.csv", path=db_path)
    loaded = db.load_dataset(ds_id, path=db_path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_dataset_dedupe_by_hash(db_path, sample_df):
    id1 = db.save_dataset(sample_df, "sample.csv", path=db_path)
    id2 = db.save_dataset(sample_df, "same content, other name.csv", path=db_path)
    assert id1 == id2
    assert len(db.list_datasets(path=db_path)) == 1


def test_run_roundtrip(db_path, sample_df):
    ds_id = db.save_dataset(sample_df, "sample.csv", path=db_path)
    run_id = db.save_run(
        dataset_id=ds_id, label="test run",
        params={"threshold": 0.05}, summary=[{"Percent Flagged": 50.0}],
        flagged_df=sample_df, run_summary="summary text", path=db_path,
    )
    run = db.load_run(run_id, path=db_path)
    assert run["label"] == "test run"
    assert run["params"] == {"threshold": 0.05}
    assert run["n_flagged"] == 2  # two rows with TOTAL SCORE > 0
    pd.testing.assert_frame_equal(run["results"], sample_df)


def test_delete_dataset_removes_runs(db_path, sample_df):
    ds_id = db.save_dataset(sample_df, "sample.csv", path=db_path)
    db.save_run(dataset_id=ds_id, label="r", params={}, summary=[],
                flagged_df=sample_df, path=db_path)
    db.delete_dataset(ds_id, path=db_path)
    assert db.list_datasets(path=db_path).empty
    assert db.list_runs(path=db_path).empty


def test_delete_run(db_path, sample_df):
    ds_id = db.save_dataset(sample_df, "sample.csv", path=db_path)
    run_id = db.save_run(dataset_id=ds_id, label="r", params={}, summary=[],
                         flagged_df=sample_df, path=db_path)
    db.delete_run(run_id, path=db_path)
    assert db.list_runs(path=db_path).empty
    with pytest.raises(KeyError):
        db.load_run(run_id, path=db_path)


def test_purge_all_and_vacuum(db_path, sample_df):
    ds_id = db.save_dataset(sample_df, "sample.csv", path=db_path)
    db.save_run(dataset_id=ds_id, label="r", params={}, summary=[],
                flagged_df=sample_df, path=db_path)
    db.purge_all(path=db_path)
    assert db.list_datasets(path=db_path).empty
    assert db.list_runs(path=db_path).empty
    db.vacuum(path=db_path)  # must not raise


def test_missing_ids_raise(db_path):
    with pytest.raises(KeyError):
        db.load_dataset(999, path=db_path)
    with pytest.raises(KeyError):
        db.load_run(999, path=db_path)


def test_schema_version_set(db_path):
    db.init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    assert version == db.SCHEMA_VERSION


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions only")
def test_db_file_permissions(db_path, sample_df):
    db.save_dataset(sample_df, "sample.csv", path=db_path)
    mode = stat.S_IMODE(os.stat(db_path).st_mode)
    assert mode == 0o600


def test_timestamps_are_timezone_aware(db_path, sample_df):
    db.save_dataset(sample_df, "sample.csv", path=db_path)
    created = db.list_datasets(path=db_path)["created_at"].iloc[0]
    assert "+00:00" in created
