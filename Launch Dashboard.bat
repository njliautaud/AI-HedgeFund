@echo off
echo Starting AI Hedge Fund Dashboard...
echo.

:: Run the Python launcher script
python launch_dashboard.py

:: If there was an error, pause so the user can see it
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error launching dashboard. See error message above.
    pause
)