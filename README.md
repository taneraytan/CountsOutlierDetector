# Counts Outlier Detector Studio

A local desktop-style app that wraps the **Counts Outlier Detector** — a
multidimensional, interpretable outlier detector for tabular data — with a
professional Streamlit UI for the full workflow:

1. **Load Data** — upload `.csv`, `.txt`, `.xls`, `.xlsx`, or `.parquet`, or
   load a previously-saved dataset from the local SQLite database.
2. **Preprocess** — drop columns, handle missing values, clip outliers, scale
   numerics, encode categoricals.
3. **Feature Engineering** — datetime expansion, log/sqrt transforms,
   quantile binning, multiplicative interactions and ratios.
4. **Configure & Run** — choose which features to include, tune detector
   parameters, run the analysis.
5. **Results** — score distribution chart, flagged-row table, run summary,
   and one-click export to CSV / Excel / Parquet / JSON.

Every dataset and every analysis run is persisted to a local SQLite database
(`~/.counts_outlier_detector/data.db` by default) so you can revisit prior
results without re-running anything.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## Run

```bash
./run.sh                    # macOS / Linux
run.bat                     # Windows
# or directly:
python -m streamlit run app/ui.py
```

Streamlit will open the app in your browser at
[http://localhost:8501](http://localhost:8501).

## Project layout

```
counts_outlier_detector/    # Detector library (algorithm, unchanged API)
    detector.py
app/
    ui.py                   # Streamlit application entry point
    preprocessing.py        # Preprocessing & feature engineering
    io_utils.py             # Multi-format file loading / exporting
    database.py             # SQLite persistence (datasets + runs)
requirements.txt
run.sh / run.bat            # Convenience launchers
```

## Configuration

* **Database location** — set `COUNTS_DB_PATH` to override the default path.
* **Performance** — for wide datasets, raise *max combinations* and enable
  *Run in parallel* in the **Configure & Run** screen.

## Library use (without the UI)

```python
import pandas as pd
from counts_outlier_detector import CountsOutlierDetector

df = pd.read_csv("my_data.csv")
detector = CountsOutlierDetector(n_bins=7, max_dimensions=3, threshold=0.05)
results = detector.fit_predict(df)
flagged = detector.get_most_flagged_rows()
```
