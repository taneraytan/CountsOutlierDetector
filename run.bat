@echo off
REM Launch the Counts Outlier Detector Studio locally.
REM Bind to loopback explicitly (defense in depth on top of .streamlit\config.toml):
REM the app has no authentication and must not be exposed to the network.
cd /d "%~dp0"
python -m streamlit run app\ui.py --server.address=127.0.0.1 %*
