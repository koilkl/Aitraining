@echo off
echo Starting TFLiteTraining for debugging...
cd /d "%~dp0"
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Running desktop_launcher.py...
echo.
python desktop_launcher.py
echo.
echo If you see this, the program exited. Press any key to exit...
pause
pause
