# Stem Separator Docker Image
# Supports both CPU and GPU (NVIDIA CUDA) modes
#
# Build for CPU:
#   docker build -t stem-separator:cpu --build-arg USE_GPU=false .
#
# Build for GPU (requires NVIDIA Container Toolkit):
#   docker build -t stem-separator:gpu --build-arg USE_GPU=true .

ARG USE_GPU=false

# =============================================================================
# Base stage for CPU
# =============================================================================
FROM python:3.11-slim-bookworm AS base-cpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =============================================================================
# Base stage for GPU (CUDA 12.1)
# =============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base-gpu

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# =============================================================================
# Select base based on build arg
# =============================================================================
FROM base-${USE_GPU:+gpu}${USE_GPU:-cpu} AS base
ARG USE_GPU

# =============================================================================
# Build stage - Install Python dependencies
# =============================================================================
FROM base AS builder

WORKDIR /app

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch based on GPU/CPU mode
ARG USE_GPU=false
RUN if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install web UI dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    aiofiles>=23.0.0 \
    websockets>=12.0

# Install optional dependencies for full functionality
RUN pip install --no-cache-dir \
    spotdl>=4.0.0 \
    mutagen>=1.45.0 || true

# =============================================================================
# Runtime stage
# =============================================================================
FROM base AS runtime

# Create non-root user for security
RUN groupadd -r stemuser && useradd -r -g stemuser -d /home/stemuser -m stemuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application code
COPY --chown=stemuser:stemuser stem_separator/ ./stem_separator/
COPY --chown=stemuser:stemuser stem_separator.py .
COPY --chown=stemuser:stemuser pyproject.toml .
COPY --chown=stemuser:stemuser requirements.txt .
COPY --chown=stemuser:stemuser README.md .

# Copy web UI
COPY --chown=stemuser:stemuser web/ ./web/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create directories for input/output/cache with proper permissions
RUN mkdir -p /input /output /cache /home/stemuser/.cache \
    && chown -R stemuser:stemuser /input /output /cache /home/stemuser

# Set cache directory for Demucs models
ENV TORCH_HOME=/cache
ENV HF_HOME=/cache

# Volume mount points
VOLUME ["/input", "/output", "/cache"]

# Expose web UI port
EXPOSE 8080

# Switch to non-root user
USER stemuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command: run web UI
CMD ["python", "-m", "web.server", "--host", "0.0.0.0", "--port", "8080"]

# Alternative: run CLI directly
# CMD ["python", "-m", "stem_separator"]
