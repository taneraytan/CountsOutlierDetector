#!/usr/bin/env bash
# Launch the Counts Outlier Detector Studio locally.
set -euo pipefail
cd "$(dirname "$0")"
exec python -m streamlit run app/ui.py "$@"
