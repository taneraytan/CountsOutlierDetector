# Counts Outlier Detector Studio

A local desktop-style app that wraps the **Counts Outlier Detector** — a
multidimensional, interpretable outlier detector for tabular data — with a
professional Streamlit UI for the full workflow:

1. **Load Data** — upload `.csv`, `.txt`, `.xlsx`, or `.parquet`, load a
   previously-saved dataset from the local SQLite database, or start with the
   built-in **demo dataset** (synthetic data with planted outliers).
2. **Preprocess** — drop columns, handle missing values, clip outliers, scale
   numerics, encode categoricals.
3. **Feature Engineering** — datetime expansion, log/sqrt transforms,
   quantile binning, multiplicative interactions and ratios.
4. **Configure & Run** — choose which features to include, tune detector
   parameters, set an optional wall-clock time budget, run the analysis —
   optionally cross-checked against an IsolationForest baseline. A quick
   **threshold sweep** helps you pick the rarity threshold empirically.
5. **Results** — score distribution chart, flagged-row table, a per-row
   **inspector** that explains *why* each row was flagged (with contingency
   tables for 2-D rarities), one-click export to CSV / Excel / Parquet /
   JSON, and a self-contained **HTML audit report** for sharing.
6. **History** — revisit, **compare**, and manage previous runs; purge and
   vacuum the local database.

Every dataset and every analysis run can be persisted to a local SQLite
database (`~/.counts_outlier_detector/data.db` by default) so you can revisit
prior results without re-running anything. If you opt out of persistence when
loading a dataset, the app will not store it later without asking.

## Install

```bash
pip install -r requirements.lock   # reproducible, fully pinned
# or, if you prefer floor pins:
pip install -r requirements.txt
```

Python 3.10 or newer is required.

## Run

```bash
./run.sh                    # macOS / Linux
run.bat                     # Windows
# or directly:
python -m streamlit run app/ui.py
```

Streamlit will open the app in your browser at
[http://localhost:8501](http://localhost:8501).

## Security notes

* **Local only by default.** The app has no authentication, so
  `.streamlit/config.toml` (and the launch scripts) bind the server to
  `127.0.0.1`. Do not expose it to a network without putting it behind an
  authenticating reverse proxy.
* **Data at rest.** The SQLite database file is created with `0600`
  permissions. Use **History → Maintenance** to purge all stored data and
  vacuum the file (SQLite otherwise leaves deleted data in free pages).
* **Export safety.** CSV/Excel exports neutralize spreadsheet formula
  injection (cells starting with `=`, `+`, `-`, `@`).
* **Legacy `.xls` files are not supported** — convert to `.xlsx`. This avoids
  shipping the legacy `xlrd` parser.

## Project layout

```
counts_outlier_detector/    # Detector library (algorithm; pip-installable package)
    detector.py
app/
    ui.py                   # Streamlit application entry point
    preprocessing.py        # Preprocessing & feature engineering
    io_utils.py             # Multi-format file loading / exporting (with sanitization)
    database.py             # SQLite persistence (datasets + runs + maintenance)
    demo.py                 # Synthetic demo dataset with planted outliers
    report.py               # Self-contained HTML audit report builder
tests/                      # pytest suite (unit + Streamlit AppTest end-to-end)
.streamlit/config.toml      # Server hardening (loopback bind, upload cap)
.github/workflows/ci.yml    # CI: ruff + mypy + pytest on Python 3.10–3.12
pyproject.toml              # Packaging (library + optional [studio] extra) & tool config
requirements.txt            # Direct dependencies (floors)
requirements.lock           # Fully pinned, reproducible install
run.sh / run.bat            # Convenience launchers (loopback-bound)
```

## Configuration

* **Database location** — set `COUNTS_DB_PATH` to override the default path.
* **Performance** — for wide datasets, raise *max combinations* and enable
  *Run in parallel* in the **Configure & Run** screen. Set a **time budget**
  to cap runaway analyses; the detector keeps whatever it finished and notes
  the truncation in the run summary.

## Library use (without the UI)

The detector is packaged independently of the UI
(`pip install .` for the library alone, `pip install .[studio]` to include
the Streamlit app's dependencies):

```python
import pandas as pd
from counts_outlier_detector import CountsOutlierDetector

df = pd.read_csv("my_data.csv")
detector = CountsOutlierDetector(n_bins=7, max_dimensions=3, threshold=0.05,
                                 max_execution_seconds=120)
results = detector.fit_predict(df)
flagged = detector.get_most_flagged_rows()
```

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .                          # lint
mypy app counts_outlier_detector      # type-check
pytest                                # unit + end-to-end app tests
```

CI runs all three on every push and pull request.
