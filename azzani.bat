@echo off
setlocal enabledelayedexpansion

REM AZZANI Image Enhancements -- Windows 1-Click & Drag-and-Drop Auto-GPU Launcher (MIT)
cd /d "%~dp0"

set "PY_CMD="
where py >nul 2>&1 && (
    py -3 --version >nul 2>&1 && set "PY_CMD=py -3"
)
if "%PY_CMD%"=="" (
    where python >nul 2>&1 && set "PY_CMD=python"
)
if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.10+ was not found. Install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

set "AZZANI_AUTO_SETUP=1"

if "%~1"=="" (
    echo Starting AZZANI Image Enhancements Web Studio ^(Auto-GPU^)...
    %PY_CMD% "%~dp0enhance.py" --ui
    exit /b %ERRORLEVEL%
)

set "FIRST_ARG=%~1"
set "SECOND_ARG=%~2"
if "!FIRST_ARG:~0,1!"=="-" (
    %PY_CMD% "%~dp0enhance.py" %*
    exit /b %ERRORLEVEL%
)
if not "%~2"=="" if "!SECOND_ARG:~0,1!"=="-" (
    %PY_CMD% "%~dp0enhance.py" %*
    exit /b %ERRORLEVEL%
)

REM Multiple files/folders dragged onto azzani.bat in Explorer
for %%F in (%*) do (
    echo Enhancing: %%~F
    %PY_CMD% "%~dp0enhance.py" "%%~F"
)
echo.
echo All done! Press any key to close.
pause >nul
