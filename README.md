# Stem Separator

Separates audio into vocals, drums, bass, and other instruments using [Demucs](https://github.com/facebookresearch/demucs) by Meta Research.

## Installation

1. **Install FFmpeg** (if not already installed):
   ```
   winget install FFmpeg.FFmpeg
   ```

2. **Install Python packages**:
   ```
   pip install demucs yt-dlp soundfile scipy
   ```
   Or run `install.bat`

## Usage

```powershell
# Local file
python stem_separator.py song.mp3

# YouTube URL
python stem_separator.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output folder
python stem_separator.py song.mp3 -o C:\Music\Stems

# Force CPU mode (skip GPU attempt)
python stem_separator.py song.mp3 --cpu
```

## Output

Creates a folder named `{song}_stems` containing:
- `vocals.wav` - Singing/voice
- `drums.wav` - Drums and percussion  
- `bass.wav` - Bass
- `other.wav` - Guitar, piano, synths, etc.

## GPU Support

The script automatically tries GPU first and falls back to CPU if needed.

**RTX 5090 / Blackwell users:** PyTorch stable doesn't support sm_120 yet. 
Install PyTorch nightly with CUDA 12.8:

```powershell
pip uninstall torch torchaudio -y
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

Then the script will use your GPU automatically (~30 seconds vs 2-4 minutes on CPU).

## Notes

- First run downloads ~1GB of AI models (one time only)
- CPU processing takes 2-4 minutes per song
- GPU processing takes ~30 seconds per song
