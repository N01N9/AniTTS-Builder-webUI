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

REM Add ffmpeg path to PATH temporarily
for %%I in ("%~dp0..\lib\ffmpeg\ffmpeg-7.1-essentials_build\bin") do set "ffmpeg_path=%%~fI"
set "PATH=%ffmpeg_path%;%PATH%"

for %%I in ("%~dp0module\MSST_WebUI") do set "msst_webui_path=%%~fI"
set "PYTHONPATH=%msst_webui_path%;%PYTHONPATH%"

echo Activating Conda environment...
call conda activate "../lib/Miniconda3/envs/AniTTS-Builder2-webUI"
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
