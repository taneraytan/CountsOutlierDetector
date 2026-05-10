@echo off
REM Launch the Counts Outlier Detector Studio locally.
cd /d "%~dp0"
python -m streamlit run app\ui.py %*
