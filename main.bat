@echo off
chcp 65001 > NUL

:: 관리자 권한 확인
openfiles >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo This script requires administrator privileges. Restarting as Administrator...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:: 가상환경 경로 설정 (main.py와 같은 디렉터리에 venv 폴더가 있을 경우)
set VENV_DIR=%~dp0venv
set PYTHON_CMD=%VENV_DIR%\Scripts\python.exe

:: main.py 경로 설정 (main.py가 이 배치 파일과 동일한 디렉터리에 있다고 가정)
set MAIN_SCRIPT=%~dp0main.py

:: 가상환경 활성화
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    pause
    exit /b 1
)

echo Current batch file directory: %~dp0

:: main.py 실행
echo Running main.py as Administrator...
"%PYTHON_CMD%" "%MAIN_SCRIPT%"

:: 종료 전 대기
pause
