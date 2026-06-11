#!/usr/bin/env bash
# Launch the Counts Outlier Detector Studio locally.
set -euo pipefail
cd "$(dirname "$0")"
# Bind to loopback explicitly (defense in depth on top of .streamlit/config.toml):
# the app has no authentication and must not be exposed to the network.
exec python -m streamlit run app/ui.py --server.address=127.0.0.1 "$@"
