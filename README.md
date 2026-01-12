# Stem Separator

Separates audio into vocals, drums, bass, and other instruments using [Demucs](https://github.com/facebookresearch/demucs) by Meta Research.

## Features

- **Batch Processing** - Process multiple files, glob patterns, or batch files
- **YouTube Playlist Support** - Download and process entire playlists
- **BPM & Key Detection** - Analyze tempo and musical key
- **Audio Normalization** - LUFS loudness normalization
- **DAW Project Export** - Export to Audacity-compatible formats
- **Configuration Files** - Save default settings in YAML
- **API Server Mode** - REST API for integration with other apps
- **Progress Bars** - Visual feedback during processing

## Installation

1. **Install FFmpeg** (if not already installed):
   ```
   winget install FFmpeg.FFmpeg
   ```

2. **Install Python packages**:
   ```
   pip install -r requirements.txt
   ```
   Or run `install.bat`

## Quick Start

```bash
# Basic usage
python stem_separator.py song.mp3

# YouTube URL
python stem_separator.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Multiple files (batch processing)
python stem_separator.py *.mp3 --format mp3

# YouTube playlist
python stem_separator.py "https://youtube.com/playlist?list=PLxxxx"
```

## Usage Examples

### Basic Operations

```bash
# Specify output folder
python stem_separator.py song.mp3 -o ~/Music/Stems

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
```

### Batch Processing

```bash
# Process all MP3s in current directory
python stem_separator.py *.mp3 --format mp3

# Process multiple specific files
python stem_separator.py song1.mp3 song2.wav song3.flac

# Process from a batch file (one path/URL per line)
python stem_separator.py @songs.txt

# Process YouTube playlist
python stem_separator.py "https://youtube.com/playlist?list=PLxxxx" --format mp3
```

### Audio Analysis

```bash
# Analyze BPM and musical key
python stem_separator.py song.mp3 --analyze

# Output example:
# [Analysis] song
#   Duration: 3:45
#   BPM: 128.0
#   Key: A minor
```

### Audio Normalization

```bash
# Normalize to -14 LUFS (default, streaming standard)
python stem_separator.py song.mp3 --normalize

# Normalize to custom level
python stem_separator.py song.mp3 --normalize --normalize-level -16.0
```

### DAW Export

```bash
# Export with Audacity project files
python stem_separator.py song.mp3 --export-daw audacity

# Export with Studio One project file
python stem_separator.py song.mp3 --export-daw studioone
```

**Audacity** creates:
- `*.lof` - Open in Audacity → File → Import → Audio
- `*_project.json` - Project metadata

**Studio One** creates:
- `*.song` - Native Studio One project file (open directly)

### Configuration File

```bash
# Save current settings to config file
python stem_separator.py --save-config ~/.stem_separator.yaml

# Use specific config file
python stem_separator.py song.mp3 --config myconfig.yaml
```

Example config file (`~/.stem_separator.yaml`):
```yaml
model: htdemucs_6s
format: flac
output: ~/Music/Stems
normalize: true
normalize_level: -14.0
analyze: true
```

### API Server Mode

```bash
# Start API server
python stem_separator.py --server

# Custom host/port
python stem_separator.py --server --host 0.0.0.0 --port 8080
```

API Endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/models` | GET | List available models |
| `/formats` | GET | List output formats |
| `/separate` | POST | Submit file for separation |
| `/jobs/{id}` | GET | Check job status |
| `/jobs/{id}/download/{stem}` | GET | Download a stem |

Example API usage:
```bash
# Submit file
curl -X POST -F "file=@song.mp3" -F "model=htdemucs" http://localhost:8000/separate

# Check status
curl http://localhost:8000/jobs/abc123

# Download stem
curl http://localhost:8000/jobs/abc123/download/vocals.wav -o vocals.wav
```

## All Options

| Option | Description |
|--------|-------------|
| `-o, --output` | Output directory (default: current) |
| `--model` | AI model: `htdemucs`, `htdemucs_ft`, `htdemucs_6s` |
| `--format` | Output format: `wav`, `mp3`, `flac`, `ogg`, `aac` |
| `--stems` | Stems to export: comma-separated or preset |
| `--cpu` | Force CPU mode (skip GPU) |
| `--browser` | Browser cookies for YouTube (chrome, firefox, edge, safari, opera, brave) |
| `--normalize` | Normalize audio loudness |
| `--normalize-level` | Target LUFS level (default: -14.0) |
| `--analyze` | Analyze BPM and musical key |
| `--export-daw` | Export DAW project: `audacity`, `studioone` |
| `--config` | Load settings from YAML file |
| `--save-config` | Save current settings to file |
| `--server` | Start API server mode |
| `--host` | API server host (default: 127.0.0.1) |
| `--port` | API server port (default: 8000) |

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
| `karaoke` | Everything except vocals |
| `acapella` | Vocals only |
| `instrumental` | Same as karaoke |

## Output

Creates a folder named `{song}_stems` containing:

**4-stem model:**
- `vocals` - Singing/voice
- `drums` - Drums and percussion
- `bass` - Bass
- `other` - Guitar, piano, synths, etc.

**6-stem model** (htdemucs_6s):
- All of the above, plus:
- `guitar` - Guitar
- `piano` - Piano

## GPU Support

The script automatically tries GPU first and falls back to CPU if needed.

**RTX 5090 / Blackwell users:** Install PyTorch nightly:
```bash
pip uninstall torch torchaudio -y
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Performance

| Hardware | Time per song |
|----------|---------------|
| CPU | 2-4 minutes |
| GPU | ~30 seconds |

## Notes

- First run downloads ~1GB of AI models (one time)
- MP3 output uses 320kbps for highest quality
- FLAC provides lossless compression (~50% smaller than WAV)
- Normalization uses LUFS standard (same as Spotify, YouTube)
- BPM detection works best on rhythmic music
