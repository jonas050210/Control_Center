@echo off
setlocal

rem SandboxAI desktop Control Center launcher (Windows).
rem
rem Usage:
rem   tools\windows\start_control_center.bat
rem   tools\windows\start_control_center.bat --output-root D:\sandboxai-runs
rem
rem This is a thin wrapper around "python start.py" in the repository root,
rem which runs everything with the project's own .venv (created by
rem "python install.py") and offers to run the setup if it is missing.
rem Extra arguments are forwarded to "sandboxai control-center-desktop".

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\.."

pushd "%ROOT%" || (
    echo [SandboxAI] Could not locate the repository root next to this script.
    exit /b 1
)

set "PYTHON=python"
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo [SandboxAI] No Python found on PATH.
        echo             Install Python 3.10+ from https://www.python.org/downloads/windows/
        echo             ^(Tkinter is included by default^), then run:  python install.py
        popd
        exit /b 1
    )
    set "PYTHON=py -3"
)

echo [SandboxAI] Launching desktop Control Center...
%PYTHON% start.py %*
set "EXIT_CODE=%ERRORLEVEL%"

popd
exit /b %EXIT_CODE%
