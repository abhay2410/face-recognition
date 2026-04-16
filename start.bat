@echo off
TITLE Face Access System
:: 1. Force ONNX and Python to silent mode globally for this terminal
SET ORT_LOGGING_LEVEL=3
SET PYTHONWARNINGS=ignore
SET ONNXRUNTIME_LOGGING_LEVEL=3

echo ============================================================
echo   Starting Face Access System in Silent Mode
echo ============================================================

:: 2. Run the server
.\venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000

pause
