# Stem Separator Docker Guide

This guide explains how to run Stem Separator in a Docker container with both CPU and GPU support, including the web UI.

## Quick Start

### CPU Mode (Works on any machine)

```bash
# Build and run
docker compose --profile cpu up -d

# Access the web UI
open http://localhost:8080
```

### GPU Mode (Requires NVIDIA GPU)

Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

```bash
# Build and run
docker compose --profile gpu up -d

# Access the web UI
open http://localhost:8080
```

## Building the Image

### CPU Version

```bash
docker build -t stem-separator:cpu --build-arg USE_GPU=false .
```

### GPU Version

```bash
docker build -t stem-separator:gpu --build-arg USE_GPU=true .
```

## Running the Container

### Using Docker Compose (Recommended)

The `docker-compose.yml` file provides pre-configured setups for both CPU and GPU modes.

**CPU Mode:**
```bash
docker compose --profile cpu up -d
```

**GPU Mode:**
```bash
docker compose --profile gpu up -d
```

**Stop the container:**
```bash
docker compose down
```

### Using Docker Directly

**CPU Mode:**
```bash
docker run -d \
  --name stem-separator \
  -p 8080:8080 \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:cpu
```

**GPU Mode:**
```bash
docker run -d \
  --name stem-separator \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:gpu
```

## Using the CLI in Docker

You can run the CLI directly without the web UI:

```bash
# Process a local file
docker run --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:cpu \
  python -m stem_separator /input/song.mp3 -o /output

# Process a YouTube URL
docker run --rm \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:cpu \
  python -m stem_separator "https://youtube.com/watch?v=VIDEO_ID" -o /output

# With specific options
docker run --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:cpu \
  python -m stem_separator /input/song.mp3 \
    -o /output \
    --model htdemucs_ft \
    --format mp3 \
    --stems vocals,drums
```

## Volume Mounts

| Mount Point | Purpose | Mode |
|-------------|---------|------|
| `/input` | Input audio files | Read-only |
| `/output` | Separated stem files | Read-write |
| `/cache` | Model cache (Demucs models ~1GB each) | Read-write |

### Important: Model Caching

The first time you run the container, Demucs will download the AI models (~1GB per model). To avoid re-downloading on every container restart, mount a persistent volume for `/cache`:

```bash
# Create a named volume
docker volume create stem-separator-cache

# Use it in your run command
-v stem-separator-cache:/cache
```

## Retrieving Separated Files

### Method 1: Web UI (Recommended)

1. Open http://localhost:8080 in your browser
2. Upload a file or paste a YouTube/Spotify URL
3. Wait for processing to complete
4. Click individual stem files to download, or use "Download All (ZIP)"

### Method 2: Mapped Volume

Files are saved to the `/output` directory in the container, which is mapped to `./output` on your host:

```bash
# After processing, files are in:
ls ./output/

# Each job creates a folder with the job ID:
# ./output/abc12345/
#   ├── vocals.wav
#   ├── drums.wav
#   ├── bass.wav
#   └── other.wav
```

### Method 3: Docker Copy

```bash
# Copy files from the container
docker cp stem-separator:/output/. ./my-stems/
```

### Method 4: API Endpoints

The web server provides REST API endpoints for downloading:

```bash
# Download a specific file
curl -O http://localhost:8080/api/download/{job_id}/{filename}

# Download all stems as ZIP
curl -O http://localhost:8080/api/download-all/{job_id}

# List all jobs
curl http://localhost:8080/api/jobs
```

## Web UI Features

The web UI at http://localhost:8080 provides:

- **File Upload**: Drag & drop or click to upload audio files (MP3, WAV, FLAC, etc.)
- **URL Processing**: Paste YouTube or Spotify URLs for direct processing
- **Playlist Support**: Process entire YouTube/Spotify playlists
- **Model Selection**: Choose between htdemucs, htdemucs_ft, or htdemucs_6s
- **Output Format**: WAV, MP3, FLAC, OGG, or AAC
- **Stem Selection**: All stems, vocals only, karaoke, drums only, etc.
- **Advanced Options**:
  - Force CPU mode
  - Audio normalization
  - Quality analysis
  - Low memory mode for long tracks
- **Real-time Progress**: WebSocket-based progress updates
- **Download Options**: Individual files or ZIP archive

## API Documentation

The container includes interactive API documentation:

- **Swagger UI**: http://localhost:8080/api/docs
- **ReDoc**: http://localhost:8080/api/redoc

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/input` | Input directory |
| `OUTPUT_DIR` | `/output` | Output directory |
| `MAX_UPLOAD_SIZE` | `524288000` | Max upload size (500MB) |
| `CUDA_VISIBLE_DEVICES` | (auto) | GPU devices to use |

### Custom Configuration File

Mount a config file to customize defaults:

```bash
docker run -d \
  -v ./my-config.yaml:/home/stemuser/.stem-separator.yaml:ro \
  ...
```

Example config:
```yaml
model: htdemucs_ft
format: mp3
normalize: true
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs stem-separator

# Ensure ports aren't in use
lsof -i :8080
```

### GPU not detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check container GPU access
docker exec stem-separator python -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory
- Use `--low-memory` option for long tracks
- Increase Docker memory limit in Docker Desktop settings
- Use CPU mode for very large files

### Model download fails
- Ensure the container has internet access
- Check if the cache volume is writable
- Models download to `/cache/demucs_cache/`

## Security Notes

- The container runs as non-root user `stemuser`
- Input directory is mounted read-only
- All user inputs are sanitized
- Subprocess calls use safe argument passing (no shell injection)
- Path traversal attacks are prevented in download endpoints

## Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 2GB | 4GB+ |
| Disk (cache) | 2GB | 5GB |
| CPU | 2 cores | 4+ cores |
| GPU (optional) | CUDA 12.1 compatible | RTX 3060+ |

### Processing Times (approximate)

| Mode | 4-minute song |
|------|---------------|
| GPU (RTX 3060) | ~30 seconds |
| CPU (8 cores) | ~3-4 minutes |

## Examples

### Process multiple files
```bash
# Place files in input folder
cp song1.mp3 song2.mp3 ./input/

# Run batch processing
docker run --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:cpu \
  python -m stem_separator /input --batch -o /output
```

### Extract vocals only (karaoke)
```bash
docker run --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:cpu \
  python -m stem_separator /input/song.mp3 -o /output --stems karaoke
```

### High-quality processing
```bash
docker run --rm \
  -v $(pwd)/input:/input:ro \
  -v $(pwd)/output:/output \
  -v stem-separator-cache:/cache \
  stem-separator:gpu \
  python -m stem_separator /input/song.mp3 -o /output \
    --model htdemucs_ft \
    --format flac \
    --normalize \
    --quality
```
