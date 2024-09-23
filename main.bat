@echo off
chcp 65001 > NUL

pushd %~dp0

openfiles >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo This script requires administrator privileges.
    echo Please run this script as Administrator.
    pause
    popd
    exit /b 1
)

echo Activating Conda environment...
call conda activate AniTTS_Builder_webUI
if %ERRORLEVEL% neq 0 (
    echo Failed to activate the Conda environment.
    pause
    popd
    exit /b 1
)

echo Running main.py with administrator privileges...
python main.py

if %ERRORLEVEL% neq 0 (
    echo  main.py failed to run.
    pause
    popd
    exit /b %ERRORLEVEL%
)

popd
pause
