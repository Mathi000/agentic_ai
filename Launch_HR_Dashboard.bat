@echo off
title HR Workforce Dashboard
echo ====================================================
echo Starting Agentic AI - HR Workforce Intelligence...
echo DO NOT CLOSE this window while using the dashboard.
echo ====================================================
cd /d "%~dp0"
python -m streamlit run dashboard/app.py
pause
