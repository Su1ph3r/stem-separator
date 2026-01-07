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
# Basic usage
python stem_separator.py song.mp3

# YouTube URL
python stem_separator.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output folder
python stem_separator.py song.mp3 -o C:\Music\Stems

# Use 6-stem model (adds guitar + piano separation)
python stem_separator.py song.mp3 --model htdemucs_6s

# Export as MP3 instead of WAV
python stem_separator.py song.mp3 --format mp3

# Extract only specific stems
python stem_separator.py song.mp3 --stems vocals,drums

# Create karaoke version (everything except vocals)
python stem_separator.py song.mp3 --stems karaoke --format mp3

# Extract acapella (vocals only)
python stem_separator.py song.mp3 --stems acapella

# Combine options
python stem_separator.py song.mp3 --model htdemucs_6s --format flac --stems guitar,piano
```

## Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory (default: current) |
| `--model` | AI model: `htdemucs` (default), `htdemucs_ft` (better quality), `htdemucs_6s` (6 stems) |
| `--format` | Output format: `wav` (default), `mp3`, `flac`, `ogg`, `aac` |
| `--stems` | Stems to export: comma-separated list or preset |
| `--cpu` | Force CPU mode (skip GPU) |
| `--browser` | Use browser cookies for YouTube (chrome, firefox, edge, safari) |

## Models

| Model | Stems | Description |
|-------|-------|-------------|
| `htdemucs` | 4 | Default model (vocals, drums, bass, other) |
| `htdemucs_ft` | 4 | Fine-tuned version (better quality) |
| `htdemucs_6s` | 6 | Adds guitar and piano separation |

## Stem Presets

| Preset | Description |
|--------|-------------|
| `all` | All stems (default) |
| `karaoke` | Everything except vocals (instrumental) |
| `acapella` | Vocals only |
| `instrumental` | Same as karaoke |

## Output

Creates a folder named `{song}_stems` containing the separated stems.

**4-stem model output:**
- `vocals` - Singing/voice
- `drums` - Drums and percussion
- `bass` - Bass
- `other` - Guitar, piano, synths, etc.

**6-stem model output** (htdemucs_6s):
- All of the above, plus:
- `guitar` - Guitar
- `piano` - Piano

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
- MP3 output uses 320kbps for highest quality
- FLAC provides lossless compression (~50% smaller than WAV)
