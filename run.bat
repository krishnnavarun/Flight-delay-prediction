@echo off
title AeroPredict - Flight Delay Prediction System
color 0f
echo =========================================================================
echo   Starting AeroPredict: Tamil Nadu & Indian Flight Delay Prediction System
echo =========================================================================
echo.
cd /d "%~dp0"
echo Starting application on http://127.0.0.1:5000 ...
echo Press Ctrl+C in this terminal window to stop the server anytime.
echo.
".\venv\Scripts\python.exe" app.py
pause

