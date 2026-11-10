@echo off
setlocal

cd /d "%~dp0"
for /f "delims=" %%I in ('where conda 2^>nul') do if not defined CONDA_EXE set "CONDA_EXE=%%I"
if not defined CONDA_EXE (
    echo Could not find Conda on PATH.
    pause
    exit /b 1
)
for %%I in ("%CONDA_EXE%") do set "CONDA_ROOT=%%~dpI.."
call "%CONDA_ROOT%\condabin\conda.bat" activate python311
if errorlevel 1 (
    echo Failed to activate the Conda environment "python311".
    echo Make sure the environment exists and Conda is available on PATH.
    pause
    exit /b 1
)

python elastica_gui.py
if errorlevel 1 (
    echo.
    echo The GUI exited with an error.
    pause
    exit /b 1
)

endlocal