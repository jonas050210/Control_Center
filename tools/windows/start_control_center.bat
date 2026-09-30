@echo off
setlocal

rem SandboxAI desktop Control Center launcher (Windows).
rem
rem Usage:
rem   tools\windows\start_control_center.bat
rem   tools\windows\start_control_center.bat --output-root D:\sandboxai-runs
rem
rem Any extra arguments are forwarded to
rem "sandboxai control-center-desktop" (see --help), e.g. --output-root to
rem point the Runs/Checkpoints/Evaluations/Benchmarks pages at a different
rem directory than <project>\training.
rem
rem Prerequisites (see README.md / docs/ADAPTER_AND_DESKTOP_CONTROL_CENTER.md):
rem   - Python 3.10+ on PATH (Tkinter ships with the standard python.org
rem     Windows installer; no extra download is required).
rem   - SandboxAI installed into that Python environment, e.g.:
rem       python -m pip install -e .[training]
rem     (drop ".[training]" for an inspection-only setup without PyTorch).
rem
rem If a virtual environment exists at <repo>\.venv it is activated
rem automatically; otherwise whatever "python" resolves to on PATH is used,
rem so a Conda/system install works the same way.

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%..\.."

pushd "%ROOT%" || (
    echo [SandboxAI] Could not locate the repository root next to this script.
    exit /b 1
)

if exist ".venv\Scripts\activate.bat" (
    echo [SandboxAI] Activating .venv
    call ".venv\Scripts\activate.bat"
)

where python >nul 2>nul
if errorlevel 1 (
    echo [SandboxAI] No "python" found on PATH.
    echo             Install Python 3.10+ from https://www.python.org/downloads/windows/
    echo             and/or activate the project's virtual environment first.
    popd
    exit /b 1
)

python -c "import sandboxai" >nul 2>nul
if errorlevel 1 (
    echo [SandboxAI] The "sandboxai" package is not importable from this Python.
    echo             Run:  python -m pip install -e .[training]
    echo             (from this repository root) and try again.
    popd
    exit /b 1
)

python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo [SandboxAI] Tkinter is not available in this Python installation.
    echo             Reinstall Python from python.org with the default
    echo             options ^(Tkinter is included^), or repair the install
    echo             via the Windows "Modify" installer option.
    popd
    exit /b 1
)

echo [SandboxAI] Launching desktop Control Center...
python -m sandboxai control-center-desktop --project-path "%ROOT%" %*
set "EXIT_CODE=%ERRORLEVEL%"

popd
exit /b %EXIT_CODE%
