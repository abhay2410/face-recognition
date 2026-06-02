@echo off
title Face System Remote Tunnel Launcher
cd /d "%~dp0"
echo Starting Face System Remote Tunnel...
venv\Scripts\python.exe start_tunnel_telegram.py
if %ERRORLEVEL% neq 0 (
    python start_tunnel_telegram.py
)
pause
