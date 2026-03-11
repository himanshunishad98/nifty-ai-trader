@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "launcher_gui.py"
) else (
    start "" ".venv\Scripts\python.exe" "launcher_gui.py"
)
