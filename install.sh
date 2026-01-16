#!/bin/bash
# Stem Separator Installation Script for Linux/macOS

set -e

echo "Installing Stem Separator..."
echo

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM=linux;;
    Darwin*)    PLATFORM=macos;;
    *)          PLATFORM=unknown;;
esac

echo "Detected platform: ${PLATFORM}"
echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo
    if [ "$PLATFORM" = "macos" ]; then
        echo "Install Python with: brew install python3"
    else
        echo "Install Python with: sudo apt install python3 python3-pip"
    fi
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Found Python ${PYTHON_VERSION}"

# Check for pip
if ! command -v pip3 &> /dev/null && ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip is not installed."
    echo "Install with: python3 -m ensurepip --upgrade"
    exit 1
fi

# Install FFmpeg if not present
if ! command -v ffmpeg &> /dev/null; then
    echo
    echo "FFmpeg not found. Installing..."

    if [ "$PLATFORM" = "macos" ]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "Please install Homebrew first: https://brew.sh"
            echo "Then run: brew install ffmpeg"
            exit 1
        fi
    elif [ "$PLATFORM" = "linux" ]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        elif command -v pacman &> /dev/null; then
            sudo pacman -S ffmpeg
        else
            echo "Please install FFmpeg manually for your distribution"
            exit 1
        fi
    fi
else
    echo "FFmpeg found: $(ffmpeg -version 2>&1 | head -n1)"
fi

echo
echo "Installing Python dependencies..."

# Determine pip command
if command -v pip3 &> /dev/null; then
    PIP="pip3"
else
    PIP="python3 -m pip"
fi

# Install main dependencies
$PIP install --upgrade pip
$PIP install demucs yt-dlp soundfile scipy pyyaml rich

# Optional: spotdl for Spotify support
echo
read -p "Install Spotify support (spotdl)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $PIP install spotdl
    echo "Spotify support installed!"
fi

# Optional: sounddevice for audio preview
echo
read -p "Install audio preview support (sounddevice)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $PIP install sounddevice
    echo "Audio preview support installed!"
fi

# Optional: mutagen for better metadata handling
echo
read -p "Install enhanced metadata support (mutagen)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $PIP install mutagen
    echo "Metadata support installed!"
fi

echo
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo
echo "Usage:"
echo "  python3 stem_separator.py song.mp3"
echo "  python3 stem_separator.py \"https://youtube.com/watch?v=...\""
echo "  python3 -m stem_separator song.mp3"
echo
echo "For GPU acceleration (NVIDIA):"
echo "  pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
echo
echo "For Apple Silicon GPU (M1/M2/M3):"
echo "  pip3 install torch torchaudio"
echo "  (MPS backend is used automatically)"
echo
echo "First run downloads AI models (~1GB), then it's fast."
echo
