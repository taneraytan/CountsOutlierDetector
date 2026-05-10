"""Streamlit application — Counts Outlier Detector Studio.

Run with:

    streamlit run app/ui.py
"""

from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Ensure project root is on the path when running via ``streamlit run app/ui.py``
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import database as db
from app.io_utils import SUPPORTED_EXTENSIONS, export_dataframe, load_dataframe
from app.preprocessing import (
    FeatureEngineeringConfig,
    PreprocessConfig,
    apply_feature_engineering,
    apply_preprocessing,
    column_summary,
    suggest_datetime_columns,
)
from counts_outlier_detector import CountsOutlierDetector


# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Counts Outlier Detector Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1, h2, h3 {color: #14304a;}
    section[data-testid="stSidebar"] {background-color: #0f1f33;}
    section[data-testid="stSidebar"] * {color: #f5f7fa !important;}
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.45rem 0.6rem; border-radius: 6px;
    }
    section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover label {
        background-color: rgba(255,255,255,0.08);
    }
    .metric-card {
        background: linear-gradient(135deg, #1f4068 0%, #2d6cb0 100%);
        color: #fff; padding: 1rem 1.25rem; border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card .label {font-size: 0.85rem; opacity: 0.85;}
    .metric-card .value {font-size: 1.6rem; font-weight: 600; margin-top: 0.25rem;}
    .stButton > button[kind="primary"] {
        background-color: #1f4068; border-color: #1f4068;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2d6cb0; border-color: #2d6cb0;
    }
    .step-header {
        border-left: 4px solid #2d6cb0; padding-left: 0.75rem;
        margin-bottom: 0.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def _ss_default(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value


_ss_default("raw_df", None)
_ss_default("raw_filename", None)
_ss_default("processed_df", None)
_ss_default("processing_log", [])
_ss_default("selected_features", None)
_ss_default("last_results", None)
_ss_default("last_run_id", None)


def _metric_card(label: str, value: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home():
    st.title("Counts Outlier Detector Studio")
    st.caption("A multidimensional outlier detection workbench for tabular data.")

    cols = st.columns(4)
    with cols[0]:
        _metric_card("Datasets stored",
                     str(len(db.list_datasets())))
    with cols[1]:
        _metric_card("Analyses saved",
                     str(len(db.list_runs())))
    with cols[2]:
        df = st.session_state.raw_df
        _metric_card("Loaded rows", f"{len(df):,}" if df is not None else "—")
    with cols[3]:
        df = st.session_state.processed_df if st.session_state.processed_df is not None \
            else st.session_state.raw_df
        _metric_card("Active features", f"{df.shape[1]:,}" if df is not None else "—")

    st.markdown("---")
    st.markdown(
        """
        ### How it works
        1. **Load Data** — upload a CSV / TXT / Excel / Parquet file or pick a previously-stored
           dataset from the local database.
        2. **Preprocess** — handle missing values, drop or clip outliers, optionally scale or
           encode categoricals.
        3. **Feature Engineering** — expand datetimes, add log/sqrt transforms, generate
           interactions or ratios.
        4. **Configure & Run** — pick which features to feed the detector, tune parameters,
           and execute the analysis.
        5. **Results** — explore flagged rows, download exports, and revisit previous runs from
           the local database.

        The detector flags rows whose values (or combinations of values across 2–6 columns) are
        substantially rarer than what would be expected under a uniform distribution.
        """
    )

    if st.session_state.last_results is not None:
        st.success("A recent analysis is available — head to **Results** to view it.")


def page_load_data():
    st.header("1 · Load Data", anchor=False)
    st.caption("Upload a file or pick a dataset stored locally.")

    tabs = st.tabs(["Upload file", "From local database"])

    with tabs[0]:
        ext_help = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
        uploaded = st.file_uploader(
            f"Supported formats: {ext_help}",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        )
        col_a, col_b = st.columns(2)
        sep = col_a.text_input(
            "CSV/TXT delimiter (leave blank to auto-detect)",
            value="", help="Only used for .csv and .txt files.",
        )
        encoding = col_b.text_input("Encoding", value="utf-8")
        save_to_db = st.checkbox(
            "Save to local database after loading", value=True,
            help="The dataset is de-duplicated by content hash.",
        )

        if uploaded is not None and st.button("Load file", type="primary"):
            try:
                df = load_dataframe(
                    uploaded, filename=uploaded.name,
                    sep=sep or None, encoding=encoding,
                )
            except Exception as exc:
                st.error(f"Failed to read file: {exc}")
                return
            st.session_state.raw_df = df
            st.session_state.raw_filename = uploaded.name
            st.session_state.processed_df = None
            st.session_state.selected_features = None
            if save_to_db:
                try:
                    ds_id = db.save_dataset(df, uploaded.name)
                    st.success(f"Loaded {len(df):,} rows · saved as dataset id {ds_id}")
                except Exception as exc:
                    st.warning(f"Loaded but failed to persist to DB: {exc}")
            else:
                st.success(f"Loaded {len(df):,} rows")

    with tabs[1]:
        catalogue = db.list_datasets()
        if catalogue.empty:
            st.info("No datasets saved yet — upload one in the other tab.")
        else:
            st.dataframe(catalogue, use_container_width=True, hide_index=True)
            ids = catalogue["id"].tolist()
            picked = st.selectbox(
                "Pick a dataset to load",
                ids,
                format_func=lambda i: f"{i} · {catalogue.loc[catalogue['id']==i,'name'].iloc[0]}",
            )
            col_l, col_d = st.columns([1, 1])
            if col_l.button("Load selected", type="primary"):
                df = db.load_dataset(int(picked))
                st.session_state.raw_df = df
                st.session_state.raw_filename = catalogue.loc[
                    catalogue['id'] == picked, 'name'
                ].iloc[0]
                st.session_state.processed_df = None
                st.session_state.selected_features = None
                st.success(f"Loaded {len(df):,} rows from dataset id {picked}")
            if col_d.button("Delete selected", help="Removes the dataset and any saved runs."):
                db.delete_dataset(int(picked))
                st.success(f"Deleted dataset {picked}")
                st.rerun()

    df = st.session_state.raw_df
    if df is None:
        return

    st.markdown("---")
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=False)

    with st.expander("Column summary", expanded=False):
        st.dataframe(column_summary(df), use_container_width=True, hide_index=True)


def page_preprocess():
    st.header("2 · Preprocess", anchor=False)
    df = st.session_state.raw_df
    if df is None:
        st.info("Load a dataset first.")
        return

    st.caption("Apply data cleaning before feature engineering and detection.")
    cfg = PreprocessConfig()

    with st.expander("Drop columns / rows", expanded=True):
        cfg.drop_columns = st.multiselect("Columns to drop", df.columns.tolist())
        col_a, col_b = st.columns(2)
        cfg.drop_duplicates = col_a.checkbox("Drop duplicate rows", value=False)
        cfg.drop_constant_columns = col_b.checkbox("Drop constant columns", value=True)
        cfg.drop_high_missing = st.checkbox(
            "Drop columns with high missingness", value=False,
        )
        if cfg.drop_high_missing:
            cfg.high_missing_threshold = st.slider(
                "Missingness threshold", 0.1, 1.0, 0.9, 0.05,
            )

    with st.expander("Missing values", expanded=True):
        col_a, col_b = st.columns(2)
        cfg.numeric_impute = col_a.selectbox(
            "Numeric columns",
            ["median", "mean", "zero", "drop_rows", "none"],
            index=0,
        )
        cfg.categorical_impute = col_b.selectbox(
            "Categorical columns",
            ["mode", "constant", "drop_rows", "none"],
            index=0,
        )
        if cfg.categorical_impute == "constant":
            cfg.categorical_fill_value = st.text_input(
                "Constant fill value", value="Missing",
            )

    with st.expander("Outlier treatment (numeric)"):
        cfg.outlier_treatment = st.selectbox(
            "Method", ["none", "iqr_clip", "zscore_clip"], index=0,
        )
        if cfg.outlier_treatment == "iqr_clip":
            cfg.iqr_factor = st.slider("IQR multiplier", 1.0, 5.0, 1.5, 0.1)
        elif cfg.outlier_treatment == "zscore_clip":
            cfg.zscore_threshold = st.slider("Z-score threshold", 1.5, 6.0, 3.0, 0.1)

    with st.expander("Scaling & encoding"):
        col_a, col_b = st.columns(2)
        cfg.scaling = col_a.selectbox(
            "Scaling (numeric)",
            ["none", "standard", "minmax", "robust"], index=0,
        )
        cfg.categorical_encoding = col_b.selectbox(
            "Categorical encoding",
            ["none", "ordinal", "onehot"], index=0,
            help="The detector handles categoricals natively — only set this if you "
                 "want to feed encoded features into the detector instead.",
        )

    if st.button("Apply preprocessing", type="primary"):
        try:
            new_df, log = apply_preprocessing(df, cfg)
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")
            st.code(traceback.format_exc())
            return
        st.session_state.processed_df = new_df
        st.session_state.processing_log = log
        st.session_state.selected_features = None
        st.success(f"Preprocessing applied — {new_df.shape[0]:,} rows × {new_df.shape[1]:,} columns")

    if st.session_state.processed_df is not None:
        st.markdown("---")
        st.subheader("Result")
        if st.session_state.processing_log:
            with st.expander("Processing log", expanded=False):
                for line in st.session_state.processing_log:
                    st.markdown(f"- {line}")
        st.dataframe(st.session_state.processed_df.head(50),
                     use_container_width=True)


def page_feature_engineering():
    st.header("3 · Feature Engineering", anchor=False)
    df = st.session_state.processed_df
    if df is None:
        df = st.session_state.raw_df
    if df is None:
        st.info("Load a dataset first.")
        return

    cfg = FeatureEngineeringConfig()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()

    with st.expander("Datetime expansion", expanded=True):
        cfg.parse_datetimes = st.multiselect(
            "Columns to parse as dates",
            all_cols,
            default=suggest_datetime_columns(df),
        )
        cfg.datetime_parts = st.multiselect(
            "Components to extract",
            ["year", "month", "day", "weekday", "hour", "quarter"],
            default=["year", "month", "weekday"],
        )
        cfg.drop_original_datetime = st.checkbox(
            "Drop original datetime column after extraction", value=True,
        )

    with st.expander("Numeric transforms"):
        col_a, col_b = st.columns(2)
        cfg.log_transform = col_a.multiselect(
            "log1p(x)", numeric_cols,
        )
        cfg.sqrt_transform = col_b.multiselect(
            "sqrt(x)", numeric_cols,
        )
        cfg.bin_columns = st.multiselect(
            "Quantile-bin (qcut)", numeric_cols,
        )
        cfg.bin_count = st.slider("Bins for qcut", 2, 20, 5)

    with st.expander("Pairwise features"):
        st.caption("Pick numeric columns to pair up.")
        col_a, col_b = st.columns(2)
        x_cols = col_a.multiselect("Interaction X (a × b)", numeric_cols, key="fe_x")
        y_cols = col_b.multiselect("Interaction Y", numeric_cols, key="fe_y")
        if x_cols and y_cols and len(x_cols) == len(y_cols):
            cfg.interactions = list(zip(x_cols, y_cols))
        elif x_cols or y_cols:
            st.caption("Pick the same number of columns on each side to enable interactions.")

        col_c, col_d = st.columns(2)
        rx = col_c.multiselect("Ratio numerator (a)", numeric_cols, key="fe_rx")
        ry = col_d.multiselect("Ratio denominator (b)", numeric_cols, key="fe_ry")
        if rx and ry and len(rx) == len(ry):
            cfg.ratios = list(zip(rx, ry))

    if st.button("Generate features", type="primary"):
        try:
            new_df, log = apply_feature_engineering(df, cfg)
        except Exception as exc:
            st.error(f"Feature engineering failed: {exc}")
            st.code(traceback.format_exc())
            return
        st.session_state.processed_df = new_df
        st.session_state.processing_log = (st.session_state.processing_log or []) + log
        st.session_state.selected_features = None
        st.success(
            f"{len(log)} feature operation(s) applied · now "
            f"{new_df.shape[1]:,} column(s)."
        )

    df = st.session_state.processed_df
    if df is not None:
        st.markdown("---")
        st.subheader("Current frame")
        st.dataframe(df.head(50), use_container_width=True)


def page_configure_run():
    st.header("4 · Configure & Run", anchor=False)
    df = st.session_state.processed_df
    if df is None:
        df = st.session_state.raw_df
    if df is None:
        st.info("Load a dataset first.")
        return

    st.caption("Pick which features the detector should consider.")
    all_cols = df.columns.tolist()

    default_selection = (
        st.session_state.selected_features
        if st.session_state.selected_features
        else all_cols
    )
    selected = st.multiselect(
        "Features to include in the analysis",
        all_cols,
        default=[c for c in default_selection if c in all_cols],
    )
    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Select all"):
        selected = all_cols
        st.session_state.selected_features = all_cols
        st.rerun()
    if col_b.button("Numeric only"):
        selected = df.select_dtypes(include="number").columns.tolist()
        st.session_state.selected_features = selected
        st.rerun()
    if col_c.button("Clear"):
        st.session_state.selected_features = []
        st.rerun()

    st.session_state.selected_features = selected

    st.markdown("---")
    st.subheader("Detector parameters")

    col1, col2, col3 = st.columns(3)
    n_bins = col1.slider("Numeric bins", 2, 12, 7)
    max_dim = col2.slider("Max dimensions to inspect", 1, 6, 3)
    threshold = col3.slider(
        "Rarity threshold", 0.001, 0.5, 0.05, 0.005,
        help="Lower values flag only the most extreme combinations.",
    )

    col4, col5, col6 = st.columns(3)
    min_uv = col4.number_input("Min unique values per categorical", 2, 100, 2)
    max_uv = col5.number_input("Max unique values per categorical", 3, 1000, 25)
    max_combos = col6.number_input(
        "Max combinations to evaluate", 1_000, 10_000_000, 100_000, step=10_000,
    )
    col7, col8 = st.columns(2)
    check_marginal = col7.checkbox(
        "Use marginal probabilities", value=False,
        help="Only flag combinations rare both in absolute terms and given marginal distributions.",
    )
    run_parallel = col8.checkbox("Run in parallel", value=False)

    label = st.text_input("Run label", value=f"Run @ {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    save_run = st.checkbox("Save run to local database", value=True)

    if not selected:
        st.warning("Select at least one feature to continue.")
        return

    if st.button("Run analysis", type="primary"):
        sub = df[selected].copy()
        detector = CountsOutlierDetector(
            n_bins=int(n_bins),
            max_dimensions=int(max_dim),
            threshold=float(threshold),
            check_marginal_probs=bool(check_marginal),
            max_num_combinations=int(max_combos),
            min_values_per_column=int(min_uv),
            max_values_per_column=int(max_uv),
            run_parallel=bool(run_parallel),
        )
        with st.spinner("Running detector — this may take a while for large feature sets…"):
            try:
                results = detector.fit_predict(sub)
            except Exception as exc:
                st.error(f"Detector failed: {exc}")
                st.code(traceback.format_exc())
                return

        flagged_all = results["Breakdown All Rows"]
        flagged_only = results["Breakdown Flagged Rows"]
        summary_df = results["Flagged Summary"]

        # Build a compact display dataframe joining original rows with scores
        most = detector.get_most_flagged_rows()
        st.session_state.last_results = {
            "label": label,
            "input_df": sub,
            "scores": results["Scores"],
            "summary": summary_df,
            "flagged_all": flagged_all,
            "flagged_only": flagged_only,
            "most_flagged": most,
            "run_summary": detector.run_summary or "",
            "params": {
                "features": selected,
                "n_bins": n_bins,
                "max_dimensions": max_dim,
                "threshold": threshold,
                "min_values_per_column": min_uv,
                "max_values_per_column": max_uv,
                "max_num_combinations": max_combos,
                "check_marginal_probs": check_marginal,
                "run_parallel": run_parallel,
            },
        }

        if save_run:
            try:
                ds_id = db.save_dataset(
                    df,
                    name=st.session_state.raw_filename or "in_memory_dataset",
                )
                run_id = db.save_run(
                    dataset_id=ds_id,
                    label=label,
                    params=st.session_state.last_results["params"],
                    summary=summary_df.to_dict(orient="records"),
                    flagged_df=flagged_all.assign(
                        TOTAL_SCORE=flagged_all["TOTAL SCORE"]
                    ),
                    run_summary=detector.run_summary or "",
                )
                st.session_state.last_run_id = run_id
                st.success(f"Run complete · saved as run id {run_id}. Open the Results page.")
            except Exception as exc:
                st.warning(f"Run completed but DB save failed: {exc}")
        else:
            st.success("Run complete. Open the Results page.")


def _altair_score_distribution(scores: pd.Series) -> alt.Chart:
    counts = scores.value_counts().sort_index().reset_index()
    counts.columns = ["score", "rows"]
    return (
        alt.Chart(counts)
        .mark_bar(color="#2d6cb0")
        .encode(
            x=alt.X("score:O", title="Total score"),
            y=alt.Y("rows:Q", title="Number of rows"),
            tooltip=["score", "rows"],
        )
        .properties(height=260)
    )


def page_results():
    st.header("5 · Results", anchor=False)

    res = st.session_state.last_results
    if res is None:
        st.info("Run an analysis on the previous page, or load a saved one from "
                "the **History** tab below.")
        with st.expander("Load saved run"):
            history_picker(load_into_state=True)
        return

    cols = st.columns(4)
    scores = res["scores"]
    flagged_count = int((scores > 0).sum())
    with cols[0]:
        _metric_card("Rows analysed", f"{len(res['input_df']):,}")
    with cols[1]:
        _metric_card("Rows flagged", f"{flagged_count:,}")
    with cols[2]:
        _metric_card("Flagged %", f"{flagged_count/max(len(scores),1)*100:.2f}%")
    with cols[3]:
        _metric_card("Top score", f"{int(scores.max() or 0)}")

    st.markdown("### Score distribution")
    st.altair_chart(_altair_score_distribution(scores), use_container_width=True)

    st.markdown("### Flagged rows (highest score first)")
    most = res["most_flagged"]
    if most is None or most.empty:
        st.info("No rows were flagged with the current parameters.")
    else:
        st.dataframe(most, use_container_width=True)

    with st.expander("Detector summary"):
        st.code(res["run_summary"] or "(no summary)", language="text")
        st.dataframe(res["summary"], use_container_width=True, hide_index=True)

    with st.expander("Per-row breakdown (all rows, with explanations)"):
        st.dataframe(res["flagged_all"], use_container_width=True)

    st.markdown("### Export")
    fmt = st.radio("Format", ["csv", "xlsx", "parquet", "json"],
                   index=0, horizontal=True)
    target = st.radio(
        "What to export",
        ["Flagged rows only", "All rows with scores", "Detector summary"],
        horizontal=True,
    )
    if target == "Flagged rows only":
        export_df = most if most is not None else pd.DataFrame()
    elif target == "All rows with scores":
        export_df = res["flagged_all"]
    else:
        export_df = res["summary"]

    if export_df is None or export_df.empty:
        st.warning("Nothing to export.")
    else:
        try:
            data, mime, ext = export_dataframe(export_df, fmt)
        except Exception as exc:
            st.error(f"Export failed: {exc}")
        else:
            label_safe = (res["label"] or "results").replace(" ", "_")
            st.download_button(
                f"Download {target.lower()} ({fmt})",
                data=data, mime=mime,
                file_name=f"{label_safe}.{ext}",
                type="primary",
            )

    st.markdown("---")
    with st.expander("Run history"):
        history_picker(load_into_state=True)


def history_picker(load_into_state: bool = False):
    runs = db.list_runs()
    if runs.empty:
        st.info("No runs saved yet.")
        return
    st.dataframe(runs, use_container_width=True, hide_index=True)
    rid = st.selectbox(
        "Pick a run",
        runs["id"].tolist(),
        format_func=lambda i: (
            f"#{i} · " + runs.loc[runs['id'] == i, 'label'].iloc[0]
        ),
    )
    col_a, col_b = st.columns(2)
    if col_a.button("Load run", key="load_run"):
        run = db.load_run(int(rid))
        flagged_all = run["results"]
        scores = flagged_all.get("TOTAL SCORE")
        if scores is None and "TOTAL_SCORE" in flagged_all.columns:
            scores = flagged_all["TOTAL_SCORE"]
        try:
            input_df = db.load_dataset(run["dataset_id"])
        except KeyError:
            input_df = pd.DataFrame()

        st.session_state.last_results = {
            "label": run["label"],
            "input_df": input_df,
            "scores": scores if scores is not None else pd.Series([0]),
            "summary": pd.DataFrame(run["summary"]),
            "flagged_all": flagged_all,
            "flagged_only": flagged_all[flagged_all.get("Any Scored", False)] \
                if "Any Scored" in flagged_all.columns else flagged_all,
            "most_flagged": (
                flagged_all.sort_values("TOTAL SCORE", ascending=False)
                if "TOTAL SCORE" in flagged_all.columns else flagged_all
            ),
            "run_summary": run["run_summary"],
            "params": run["params"],
        }
        st.success(f"Loaded run {rid}")
        st.rerun()
    if col_b.button("Delete run", key="del_run"):
        db.delete_run(int(rid))
        st.success(f"Deleted run {rid}")
        st.rerun()


def page_history():
    st.header("History", anchor=False)
    st.caption("All datasets and analyses persisted to the local SQLite database.")
    st.subheader("Datasets")
    st.dataframe(db.list_datasets(), use_container_width=True, hide_index=True)
    st.subheader("Runs")
    st.dataframe(db.list_runs(), use_container_width=True, hide_index=True)
    st.markdown("---")
    history_picker(load_into_state=True)


# ---------------------------------------------------------------------------
# Sidebar / router
# ---------------------------------------------------------------------------

PAGES = {
    "Home": page_home,
    "1 · Load Data": page_load_data,
    "2 · Preprocess": page_preprocess,
    "3 · Feature Engineering": page_feature_engineering,
    "4 · Configure & Run": page_configure_run,
    "5 · Results": page_results,
    "History": page_history,
}


def main():
    db.init_db()
    with st.sidebar:
        st.markdown("### Counts Outlier Studio")
        choice = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        if st.session_state.raw_filename:
            st.caption(f"**Loaded:** `{st.session_state.raw_filename}`")
        if st.session_state.raw_df is not None:
            df = (st.session_state.processed_df
                  if st.session_state.processed_df is not None
                  else st.session_state.raw_df)
            st.caption(f"{len(df):,} rows × {df.shape[1]:,} cols")
        st.caption(f"DB: `{db.DEFAULT_DB_PATH}`")

    PAGES[choice]()


if __name__ == "__main__":
    main()
