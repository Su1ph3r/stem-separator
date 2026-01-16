@echo off
setlocal enabledelayedexpansion

echo Installing Stem Separator...
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Found Python %PYTHON_VERSION%

:: Check for FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo FFmpeg not found. Installing with winget...
    winget install FFmpeg.FFmpeg
    if errorlevel 1 (
        echo.
        echo Could not install FFmpeg automatically.
        echo Please install manually: winget install FFmpeg.FFmpeg
        echo Or download from: https://ffmpeg.org/download.html
    )
) else (
    echo FFmpeg found.
)

echo.
echo Installing Python dependencies...

:: Upgrade pip
python -m pip install --upgrade pip

:: Install main dependencies
pip install demucs yt-dlp soundfile scipy pyyaml rich

if errorlevel 1 (
    echo.
    echo Installation failed. Make sure Python is installed correctly.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Core installation complete!
echo ========================================
echo.

:: Optional: Spotify support
echo.
set /p SPOTIFY="Install Spotify support (spotdl)? [y/N]: "
if /i "%SPOTIFY%"=="y" (
    pip install spotdl
    echo Spotify support installed!
)

:: Optional: Audio preview
echo.
set /p PREVIEW="Install audio preview support (sounddevice)? [y/N]: "
if /i "%PREVIEW%"=="y" (
    pip install sounddevice
    echo Audio preview support installed!
)

:: Optional: Metadata support
echo.
set /p METADATA="Install enhanced metadata support (mutagen)? [y/N]: "
if /i "%METADATA%"=="y" (
    pip install mutagen
    echo Metadata support installed!
)

echo.
echo ========================================
echo  Installation complete!
echo ========================================
echo.
echo Usage:
echo   python stem_separator.py song.mp3
echo   python stem_separator.py "https://youtube.com/watch?v=..."
echo   python -m stem_separator song.mp3
echo.
echo New features in v2.0:
echo   --batch         Process directory of files
echo   --playlist      Process YouTube/Spotify playlists
echo   --remix         Mix stems with volume control
echo   --preview       Preview stems interactively
echo   --normalize     Normalize audio levels
echo   --quality       Show separation quality analysis
echo.
echo For GPU support (NVIDIA):
echo   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo For RTX 5090 / Blackwell GPUs:
echo   pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
echo.
echo First run downloads AI models (~1GB), then it's fast.
echo.
pause
