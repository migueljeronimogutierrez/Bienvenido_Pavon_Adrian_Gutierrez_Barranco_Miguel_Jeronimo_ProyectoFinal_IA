@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo  PBR Texture Generation and Aging Tool
echo  Environment setup

echo ==============================================

echo.
echo [1/4] Checking Python installation...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.12 was not found using the "py" launcher.
    echo Please install Python 3.12.x and make sure the launcher is available.
    pause
    exit /b 1
)

echo.
echo [2/4] Creating virtual environment...
if not exist ".venv" (
    py -3.12 -m venv .venv
) else (
    echo Virtual environment already exists. Reusing .venv
)

echo.
echo [3/4] Upgrading pip...
call ".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo [4/4] Installing dependencies...
call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Environment setup completed successfully.
echo You can now run the application using run_app.bat
pause
