@echo off
REM Phase 1 Project 3 setup: reuse the logdet2 venv from Project 2
REM (already has torch+CUDA working on this machine) and add the
REM Hugging Face fine-tuning packages needed here.
REM
REM Double-click this file, or run it from a cmd/PowerShell prompt.

call "C:\Users\santo\venvs\logdet2\Scripts\activate.bat"
if errorlevel 1 (
    echo Could not activate C:\Users\santo\venvs\logdet2 - check the path exists.
    pause
    exit /b 1
)

cd /d "C:\Users\santo\OneDrive\Desktop\Projects\security-alert-triage"
pip install -r requirements.txt

echo.
echo Done. Venv logdet2 now has the Project 3 dependencies too.
pause
