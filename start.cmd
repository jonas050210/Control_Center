@echo off
REM RocketAI starten: Web-App im Browser. Einfach doppelklicken.
REM Zusaetzliche Optionen werden durchgereicht, z. B.: start.cmd --port 9000
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo RocketAI ist noch nicht eingerichtet.
  echo Bitte zuerst install.cmd doppelklicken.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" start.py %*
if errorlevel 1 (
  echo.
  echo Es gab einen Fehler. Die Ausgabe oben hilft weiter.
  pause
)
