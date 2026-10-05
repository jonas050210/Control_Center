@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
  py -3 start.py
  if not errorlevel 1 goto :ende
)

where python >nul 2>nul
if not errorlevel 1 (
  python start.py
  if not errorlevel 1 goto :ende
)

echo.
echo Python 3 wurde nicht gefunden. Bitte installieren: https://www.python.org/downloads/
echo.
pause

:ende
endlocal
