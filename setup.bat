@echo off
setlocal EnableExtensions

REM ===========================================================================
REM  AI Interview Notetaker - one-shot environment setup for a new PC.
REM  Safe to re-run: every step is skipped when it is already done.
REM  Needs an internet connection (Python/Ollama installers, pip, model files).
REM  Set NOPAUSE=1 to run without the "press any key" prompts.
REM ===========================================================================

cd /d "%~dp0"

echo ================================================
echo   AI Interview Notetaker - Environment Setup
echo ================================================
echo.

set "VPY=.venv\Scripts\python.exe"
set "OLLAMA_DIR=%LOCALAPPDATA%\Programs\Ollama"
set "LLM_MODEL=llama3.1"
set "WHISPER_MODEL=medium"

REM ---------------------------------------------------------------- 1. Python
echo [1/7] Looking for Python 3.10 - 3.13...
call :find_python
if not defined PY (
    echo       Not found. Installing Python 3.13 with winget...
    winget install -e --id Python.Python.3.13 --accept-source-agreements --accept-package-agreements
    call :find_python
)
if not defined PY (
    echo.
    echo Python was installed or is missing from PATH for this window.
    echo Close this window and run setup.bat again.
    echo If it keeps failing, install Python 3.13 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH".
    goto :fail
)
echo       Using: %PY%

REM ----------------------------------------------------------------- 2. venv
if not exist "%VPY%" (
    echo [2/7] Creating virtual environment in .venv ...
    %PY% -m venv .venv || goto :fail
) else (
    echo [2/7] Virtual environment .venv already exists, skipping creation.
)

REM ------------------------------------------------------ 3. Python packages
echo [3/7] Installing Python packages from requirements.txt ...
"%VPY%" -m pip install --upgrade pip || goto :fail
"%VPY%" -m pip install -r requirements.txt || goto :fail

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       No NVIDIA GPU detected - skipping CUDA libraries, Whisper will use the CPU.
) else (
    echo       NVIDIA GPU detected - installing CUDA libraries for GPU transcription ...
    "%VPY%" -m pip install -r requirements-gpu.txt || goto :fail
)

REM ------------------------------------------------------------- 4. Ollama
echo [4/7] Looking for Ollama...
set "OLLAMA="
where ollama >nul 2>&1 && set "OLLAMA=ollama"
if not defined OLLAMA if exist "%OLLAMA_DIR%\ollama.exe" set "OLLAMA=%OLLAMA_DIR%\ollama.exe"
if not defined OLLAMA (
    echo       Not found. Installing Ollama with winget...
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    if exist "%OLLAMA_DIR%\ollama.exe" set "OLLAMA=%OLLAMA_DIR%\ollama.exe"
)
if not defined OLLAMA (
    echo.
    echo Could not install Ollama automatically.
    echo Install it from https://ollama.com/download and run setup.bat again.
    goto :fail
)

REM Make sure the Ollama server is running so the model can be pulled.
curl -s -m 2 http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo       Starting Ollama...
    if exist "%OLLAMA_DIR%\ollama app.exe" (
        start "" "%OLLAMA_DIR%\ollama app.exe"
    ) else (
        start "" /b "%OLLAMA%" serve
    )
    set /a TRIES=0
    call :wait_ollama
    if errorlevel 1 (
        echo Ollama did not start. Start it manually and run setup.bat again.
        goto :fail
    )
)

REM -------------------------------------------------- 5. Summarization model
echo [5/7] Checking the %LLM_MODEL% model ^(about 5 GB on first download^)...
"%OLLAMA%" list | findstr /i "%LLM_MODEL%" >nul
if errorlevel 1 (
    "%OLLAMA%" pull %LLM_MODEL% || goto :fail
) else (
    echo       %LLM_MODEL% already downloaded.
)

REM ----------------------------------------------------- 6. Whisper model
echo [6/7] Checking the Whisper '%WHISPER_MODEL%' model ^(about 1.5 GB on first download^)...
"%VPY%" -c "from faster_whisper import download_model; download_model('%WHISPER_MODEL%'); print('      Whisper model ready.')" || goto :fail

REM ------------------------------------------------------------ 7. Launcher
echo [7/7] Building the launcher exe...
call tools\build_launcher.bat
if errorlevel 1 echo       Launcher build failed - you can still start the app with: .venv\Scripts\python.exe main.py

echo.
echo ================================================
echo   Setup complete.
echo   Start the app with "AI Interview Notetaker.exe"
echo ================================================
echo.
if not defined NOPAUSE pause
exit /b 0


REM ===========================================================================
:find_python
REM Sets PY to a usable Python 3.10 - 3.13 command, or leaves it undefined.
set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PY (
        py -%%V --version >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
exit /b 0

:wait_ollama
REM Polls the Ollama API for up to ~60 s. Returns errorlevel 1 on timeout.
curl -s -m 2 http://localhost:11434/api/version >nul 2>&1 && exit /b 0
set /a TRIES+=1
if %TRIES% GEQ 30 exit /b 1
timeout /t 2 /nobreak >nul
goto :wait_ollama

:fail
echo.
echo Setup FAILED - see the messages above.
if not defined NOPAUSE pause
exit /b 1
