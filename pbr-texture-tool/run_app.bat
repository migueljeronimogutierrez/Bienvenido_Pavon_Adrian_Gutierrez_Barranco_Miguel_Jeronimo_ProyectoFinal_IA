@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo  PBR Texture Generation and Aging Tool
echo  Launching Streamlit application

echo ==============================================

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment was not found.
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

echo.
echo Starting application...
call ".venv\Scripts\python.exe" -m streamlit run app.py
