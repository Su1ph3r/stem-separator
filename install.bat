@echo off
echo Installing Stem Separator...
echo.

pip install demucs yt-dlp soundfile scipy

if errorlevel 1 (
    echo.
    echo Installation failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Installation complete!
echo ========================================
echo.
echo Make sure FFmpeg is installed:
echo   winget install FFmpeg.FFmpeg
echo.
echo For RTX 5090 / Blackwell GPU support:
echo   pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
echo.
echo Usage:
echo   python stem_separator.py song.mp3
echo   python stem_separator.py "https://youtube.com/watch?v=..."
echo.
echo First run downloads AI models (~1GB), then it's fast.
echo.
pause
