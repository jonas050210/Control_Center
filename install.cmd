@echo off
REM RocketAI einrichten: venv, PyTorch (CPU), RocketAI, RLBotServer.
REM Einfach doppelklicken. Fuer die CUDA-Variante: install.cmd --cuda
setlocal
cd /d "%~dp0"

set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=py -3.12"

%PY% -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)" >nul 2>nul
if errorlevel 1 (
  echo Bitte Python 3.12 installieren ^(3.11 bis 3.13 gehen auch^):
  echo   https://www.python.org/downloads/
  echo Beim Installieren "Add python.exe to PATH" ankreuzen.
  pause
  exit /b 1
)

%PY% install.py %*
if errorlevel 1 (
  echo.
  echo Es gab einen Fehler. Die Ausgabe oben hilft weiter.
  pause
  exit /b 1
)

echo.
echo Fertig. Ab jetzt start.cmd doppelklicken.
pause
