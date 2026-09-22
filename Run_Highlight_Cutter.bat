@echo off
setlocal

cd /d "%~dp0"
set "PYTHON=C:\Users\Furkan Mert\AppData\Local\Python\pythoncore-3.14-64\python.exe"

if not exist "%PYTHON%" (
    echo Python wurde nicht gefunden:
    echo %PYTHON%
    echo.
    pause
    exit /b 1
)

if not exist "app\webui\server.py" (
    echo app\webui\server.py wurde nicht gefunden.
    echo Bitte diese Datei direkt im Projektordner ausfuehren.
    echo.
    pause
    exit /b 1
)

echo Highlight Cutter wird gestartet...
echo Browser: http://127.0.0.1:8765
"%PYTHON%" "app\webui\server.py"

if errorlevel 1 (
    echo.
    echo Die Anwendung wurde mit einem Fehler beendet.
    pause
)
endlocal
