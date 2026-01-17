#!/usr/bin/env python3
"""
Stem Separator Web Server

A FastAPI-based web interface for the Stem Separator CLI tool.
Supports file uploads, URL processing (YouTube/Spotify), and stem downloads.

Security Note: This module uses asyncio.create_subprocess_exec() which is the
safe alternative to shell execution - arguments are passed as a list, preventing
shell injection attacks.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stem_separator.config import MODELS, FORMATS, PRESETS
from stem_separator.utils import is_youtube_url, is_spotify_url

# Alias for compatibility
AVAILABLE_MODELS = MODELS
OUTPUT_FORMATS = FORMATS
STEM_PRESETS = PRESETS

# =============================================================================
# Configuration
# =============================================================================

# Directories
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/stem-separator-uploads"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/tmp/stem-separator-jobs"))

# Ensure directories exist
for d in [INPUT_DIR, OUTPUT_DIR, UPLOAD_DIR, JOBS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File size limits
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE", 500 * 1024 * 1024))  # 500MB default

# CORS configuration - defaults to same-origin only for security
# Set CORS_ORIGINS=* for development or specify comma-separated origins
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "").strip()
if CORS_ORIGINS == "*":
    ALLOWED_ORIGINS = ["*"]
elif CORS_ORIGINS:
    ALLOWED_ORIGINS = [origin.strip() for origin in CORS_ORIGINS.split(",")]
else:
    ALLOWED_ORIGINS = []  # Same-origin only (most secure default)

# Maximum jobs to return in list
MAX_JOBS_LIMIT = 100

# Job ID validation pattern (alphanumeric, 22 chars for secrets.token_urlsafe(16))
JOB_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{16,24}$")

# Job cleanup settings
JOB_EXPIRY_HOURS = int(os.environ.get("JOB_EXPIRY_HOURS", 24))  # Hours before jobs are cleaned up
MAX_JOBS_IN_MEMORY = int(os.environ.get("MAX_JOBS_IN_MEMORY", 1000))  # Maximum jobs to keep in memory

# Processing timeout (1 hour default)
PROCESSING_TIMEOUT = int(os.environ.get("PROCESSING_TIMEOUT", 3600))

# WebSocket connection limits
MAX_WEBSOCKET_CONNECTIONS_PER_JOB = 10

# Logger for this module
logger = logging.getLogger(__name__)

# Allowed audio extensions
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}

# =============================================================================
# Job Management
# =============================================================================

class JobStatus:
    """Job status constants."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def sanitize_error_message(error_msg: str, max_length: int = 500) -> str:
    """
    Sanitize error messages to prevent information disclosure.

    Args:
        error_msg: The raw error message.
        max_length: Maximum length of the returned message.

    Returns:
        Sanitized error message safe to return to clients.
    """
    if not error_msg:
        return "Unknown error"

    # Remove absolute paths (Unix and Windows style)
    sanitized = re.sub(r'/[a-zA-Z0-9/_.\-]+', '[PATH]', error_msg)
    sanitized = re.sub(r'[A-Z]:\\[a-zA-Z0-9\\_.\-]+', '[PATH]', sanitized)

    # Remove IP addresses
    sanitized = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', sanitized)

    # Remove potential credentials/tokens
    sanitized = re.sub(r'(password|token|key|secret|auth)[=:]\S+', r'\1=[REDACTED]', sanitized, flags=re.IGNORECASE)

    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


class JobManager:
    """Manages separation jobs and their status."""

    def __init__(self):
        self.jobs: dict = {}
        self.active_processes: dict = {}
        self.websockets: dict[str, list[WebSocket]] = {}
        self._last_cleanup = datetime.now()

    def create_job(self, source_type: str, source: str, options: dict) -> str:
        """Create a new job and return its ID."""
        # Use cryptographically secure ID (22 chars from 16 bytes)
        job_id = secrets.token_urlsafe(16)

        # Cleanup old jobs periodically
        self._maybe_cleanup()

        self.jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.PENDING,
            "source_type": source_type,
            "source": source,
            "options": options,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "progress": 0,
            "message": "Queued for processing",
            "output_files": [],
            "error": None,
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs):
        """Update job properties."""
        if job_id in self.jobs:
            self.jobs[job_id].update(kwargs)
            # Notify websocket subscribers
            asyncio.create_task(self._notify_subscribers(job_id))

    def list_jobs(self, limit: int = 50) -> list:
        """List recent jobs."""
        jobs = sorted(
            self.jobs.values(),
            key=lambda x: x["created_at"],
            reverse=True
        )
        return jobs[:limit]

    async def _notify_subscribers(self, job_id: str):
        """Notify WebSocket subscribers of job updates."""
        if job_id in self.websockets:
            job = self.jobs.get(job_id)
            if job:
                message = json.dumps({"type": "job_update", "job": job})
                dead_sockets = []
                for ws in self.websockets[job_id]:
                    try:
                        await ws.send_text(message)
                    except Exception:
                        dead_sockets.append(ws)
                # Remove dead sockets
                for ws in dead_sockets:
                    self.websockets[job_id].remove(ws)

    def subscribe(self, job_id: str, websocket: WebSocket):
        """Subscribe to job updates."""
        if job_id not in self.websockets:
            self.websockets[job_id] = []
        self.websockets[job_id].append(websocket)

    def unsubscribe(self, job_id: str, websocket: WebSocket):
        """Unsubscribe from job updates."""
        if job_id in self.websockets and websocket in self.websockets[job_id]:
            self.websockets[job_id].remove(websocket)

    def _maybe_cleanup(self):
        """Periodically clean up old jobs to prevent memory leaks."""
        now = datetime.now()

        # Only run cleanup every 10 minutes at most
        if (now - self._last_cleanup).total_seconds() < 600:
            return

        self._last_cleanup = now
        self._cleanup_old_jobs()

    def _cleanup_old_jobs(self):
        """Remove expired jobs and enforce memory limits."""
        now = datetime.now()
        expiry_threshold = now - timedelta(hours=JOB_EXPIRY_HOURS)

        # Find jobs to remove
        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            # Skip active jobs
            if job["status"] == JobStatus.PROCESSING:
                continue

            # Remove expired jobs
            try:
                created_at = datetime.fromisoformat(job["created_at"])
                if created_at < expiry_threshold:
                    jobs_to_remove.append(job_id)
            except (ValueError, KeyError):
                # Invalid date, mark for removal
                jobs_to_remove.append(job_id)

        # Remove expired jobs
        for job_id in jobs_to_remove:
            self._remove_job(job_id)

        # If still over limit, remove oldest completed jobs
        if len(self.jobs) > MAX_JOBS_IN_MEMORY:
            completed_jobs = [
                (job_id, job) for job_id, job in self.jobs.items()
                if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)
            ]
            # Sort by created_at, oldest first
            completed_jobs.sort(key=lambda x: x[1].get("created_at", ""))

            # Remove oldest jobs until under limit
            excess = len(self.jobs) - MAX_JOBS_IN_MEMORY
            for job_id, _ in completed_jobs[:excess]:
                self._remove_job(job_id)

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} expired jobs")

    def _remove_job(self, job_id: str):
        """Remove a job and its associated resources."""
        if job_id in self.jobs:
            del self.jobs[job_id]
        if job_id in self.websockets:
            del self.websockets[job_id]

    def get_websocket_count(self, job_id: str) -> int:
        """Get the number of WebSocket connections for a job."""
        return len(self.websockets.get(job_id, []))


# Global job manager
job_manager = JobManager()

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Stem Separator",
    description="AI-powered audio stem separation using Demucs",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware - configured via CORS_ORIGINS environment variable
if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

# =============================================================================
# Helper Functions
# =============================================================================

def validate_file_extension(filename: str) -> bool:
    """Check if file has an allowed extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_job_id(job_id: str) -> bool:
    """Validate job ID format to prevent path traversal."""
    return bool(JOB_ID_PATTERN.match(job_id))


def generate_file_id() -> str:
    """Generate a secure random file ID."""
    return secrets.token_hex(4)


def secure_filename(filename: str) -> str:
    """Sanitize filename for security."""
    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)
    # Replace spaces and special chars
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    filename = "".join(c if c in safe_chars else "_" for c in filename)
    # Ensure it has an extension
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        filename += ".mp3"
    return filename


def build_separation_command(input_path: str, output_dir: str, options: dict) -> list:
    """
    Build command arguments for stem separation.
    Returns a list of arguments for subprocess (safe, no shell injection).
    """
    cmd = [
        sys.executable, "-m", "stem_separator",
        input_path,
        "-o", output_dir,
    ]

    # Add validated options
    model = options.get("model", "htdemucs")
    if model in AVAILABLE_MODELS:
        cmd.extend(["--model", model])

    fmt = options.get("format", "wav")
    if fmt in OUTPUT_FORMATS:
        cmd.extend(["--format", fmt])

    stems = options.get("stems")
    if stems and isinstance(stems, str):
        # Validate stems value
        valid_stems = {"all", "vocals", "drums", "bass", "other", "guitar", "piano", "karaoke", "acapella", "instrumental"}
        stem_parts = [s.strip() for s in stems.split(",")]
        if all(s in valid_stems for s in stem_parts):
            cmd.extend(["--stems", stems])

    if options.get("cpu"):
        cmd.append("--cpu")

    if options.get("normalize"):
        cmd.append("--normalize")

    if options.get("quality"):
        cmd.append("--quality")

    if options.get("low_memory"):
        cmd.append("--low-memory")

    if options.get("playlist"):
        cmd.append("--playlist")

    return cmd


async def run_separation(job_id: str, input_path: str, options: dict):
    """Run the stem separation process using subprocess with list args (safe)."""
    job_manager.update_job(
        job_id,
        status=JobStatus.PROCESSING,
        started_at=datetime.now().isoformat(),
        message="Starting separation...",
        progress=5,
    )

    process = None
    try:
        output_dir = str(OUTPUT_DIR / job_id)
        cmd = build_separation_command(input_path, output_dir, options)

        job_manager.update_job(
            job_id,
            message="Processing audio...",
            progress=10,
        )

        # Run with asyncio.create_subprocess_exec (list args, no shell)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        job_manager.active_processes[job_id] = process

        # Monitor progress with timeout
        progress = 10
        start_time = datetime.now()
        while process.returncode is None:
            # Check for timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > PROCESSING_TIMEOUT:
                logger.warning(f"Job {job_id} exceeded timeout of {PROCESSING_TIMEOUT}s, terminating")
                process.terminate()
                await asyncio.sleep(1)
                if process.returncode is None:
                    process.kill()
                raise asyncio.TimeoutError(f"Processing timeout exceeded ({PROCESSING_TIMEOUT}s)")

            if progress < 90:
                progress += 5
                job_manager.update_job(job_id, progress=progress)
            await asyncio.sleep(2)
            # Check if process finished
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            # Find output files
            output_path = OUTPUT_DIR / job_id
            output_files = []

            if output_path.exists():
                for f in output_path.rglob("*"):
                    if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS | {".wav"}:
                        output_files.append({
                            "name": f.name,
                            "path": str(f.relative_to(OUTPUT_DIR)),
                            "size": f.stat().st_size,
                        })

            job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
                progress=100,
                message="Separation complete!",
                output_files=output_files,
            )
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now().isoformat(),
                message="Separation failed",
                error=sanitize_error_message(error_msg),
            )

    except asyncio.CancelledError:
        if process and process.returncode is None:
            process.terminate()
        job_manager.update_job(
            job_id,
            status=JobStatus.CANCELLED,
            completed_at=datetime.now().isoformat(),
            message="Job cancelled",
        )
    except asyncio.TimeoutError as e:
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            message="Processing timeout",
            error=sanitize_error_message(str(e)),
        )
    except Exception as e:
        logger.exception(f"Separation failed for job {job_id}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            message="Separation failed",
            error=sanitize_error_message(str(e)),
        )
    finally:
        if job_id in job_manager.active_processes:
            del job_manager.active_processes[job_id]


async def run_url_separation(job_id: str, url: str, options: dict):
    """Run separation for a URL source."""
    job_manager.update_job(
        job_id,
        status=JobStatus.PROCESSING,
        started_at=datetime.now().isoformat(),
        message="Downloading from URL...",
        progress=5,
    )

    process = None
    try:
        output_dir = str(OUTPUT_DIR / job_id)
        cmd = build_separation_command(url, output_dir, options)

        # Run with list args (safe, no shell)
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        job_manager.active_processes[job_id] = process

        # Monitor progress with timeout
        progress = 5
        start_time = datetime.now()
        while process.returncode is None:
            # Check for timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > PROCESSING_TIMEOUT:
                logger.warning(f"Job {job_id} exceeded timeout of {PROCESSING_TIMEOUT}s, terminating")
                process.terminate()
                await asyncio.sleep(1)
                if process.returncode is None:
                    process.kill()
                raise asyncio.TimeoutError(f"Processing timeout exceeded ({PROCESSING_TIMEOUT}s)")

            if progress < 90:
                progress += 3
                msg = "Processing..." if progress > 30 else "Downloading..."
                job_manager.update_job(job_id, progress=progress, message=msg)
            await asyncio.sleep(2)
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            output_path = OUTPUT_DIR / job_id
            output_files = []

            if output_path.exists():
                for f in output_path.rglob("*"):
                    if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS | {".wav"}:
                        output_files.append({
                            "name": f.name,
                            "path": str(f.relative_to(OUTPUT_DIR)),
                            "size": f.stat().st_size,
                        })

            job_manager.update_job(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=datetime.now().isoformat(),
                progress=100,
                message="Separation complete!",
                output_files=output_files,
            )
        else:
            error_msg = stderr.decode() if stderr else "Unknown error"
            job_manager.update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=datetime.now().isoformat(),
                message="Processing failed",
                error=sanitize_error_message(error_msg),
            )

    except asyncio.CancelledError:
        if process and process.returncode is None:
            process.terminate()
        job_manager.update_job(
            job_id,
            status=JobStatus.CANCELLED,
            completed_at=datetime.now().isoformat(),
            message="Job cancelled",
        )
    except asyncio.TimeoutError as e:
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            message="Processing timeout",
            error=sanitize_error_message(str(e)),
        )
    except Exception as e:
        logger.exception(f"URL separation failed for job {job_id}")
        job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            completed_at=datetime.now().isoformat(),
            message="Processing failed",
            error=sanitize_error_message(str(e)),
        )
    finally:
        if job_id in job_manager.active_processes:
            del job_manager.active_processes[job_id]


# =============================================================================
# API Routes
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/config")
async def get_config():
    """Get available configuration options."""
    return {
        "models": list(AVAILABLE_MODELS.keys()),
        "formats": list(OUTPUT_FORMATS.keys()),
        "presets": list(STEM_PRESETS.keys()),
        "max_upload_size": MAX_UPLOAD_SIZE,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    model: str = Form("htdemucs"),
    format: str = Form("wav"),
    stems: Optional[str] = Form(None),
    cpu: bool = Form(False),
    normalize: bool = Form(False),
    quality: bool = Form(False),
    low_memory: bool = Form(False),
):
    """Upload an audio file for processing."""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Save uploaded file with streaming to avoid loading entire file into memory
    safe_name = secure_filename(file.filename)
    file_id = generate_file_id()
    upload_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

    # Stream file and check size during upload
    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        async with aiofiles.open(upload_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE:
                    # Clean up partial file
                    await f.close()
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)}MB"
                    )
                await f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        upload_path.unlink(missing_ok=True)
        logger.exception("File upload failed")
        raise HTTPException(status_code=500, detail="File upload failed")

    # Create job
    options = {
        "model": model,
        "format": format,
        "stems": stems,
        "cpu": cpu,
        "normalize": normalize,
        "quality": quality,
        "low_memory": low_memory,
    }

    job_id = job_manager.create_job("upload", safe_name, options)

    # Start processing in background
    asyncio.create_task(run_separation(job_id, str(upload_path), options))

    return {"job_id": job_id, "message": "Upload successful, processing started"}


@app.post("/api/url")
async def process_url(
    url: str = Form(...),
    model: str = Form("htdemucs"),
    format: str = Form("wav"),
    stems: Optional[str] = Form(None),
    cpu: bool = Form(False),
    normalize: bool = Form(False),
    quality: bool = Form(False),
    low_memory: bool = Form(False),
    playlist: bool = Form(False),
):
    """Process a YouTube or Spotify URL."""
    # Validate URL
    url = url.strip()

    if not (is_youtube_url(url) or is_spotify_url(url)):
        raise HTTPException(
            status_code=400,
            detail="Invalid URL. Please provide a valid YouTube or Spotify URL."
        )

    # Create job
    options = {
        "model": model,
        "format": format,
        "stems": stems,
        "cpu": cpu,
        "normalize": normalize,
        "quality": quality,
        "low_memory": low_memory,
        "playlist": playlist,
    }

    source_type = "youtube" if is_youtube_url(url) else "spotify"
    job_id = job_manager.create_job(source_type, url, options)

    # Start processing
    asyncio.create_task(run_url_separation(job_id, url, options))

    return {"job_id": job_id, "message": "URL processing started"}


@app.get("/api/jobs")
async def list_jobs(limit: int = 50):
    """List recent jobs."""
    # Enforce maximum limit to prevent abuse
    safe_limit = min(max(1, limit), MAX_JOBS_LIMIT)
    return {"jobs": job_manager.list_jobs(safe_limit)}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and details."""
    if not validate_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    if not validate_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Job is not running")

    # Kill the process
    if job_id in job_manager.active_processes:
        process = job_manager.active_processes[job_id]
        process.terminate()
        await asyncio.sleep(0.5)
        if process.returncode is None:
            process.kill()

    job_manager.update_job(
        job_id,
        status=JobStatus.CANCELLED,
        completed_at=datetime.now().isoformat(),
        message="Job cancelled by user",
    )

    return {"message": "Job cancelled"}


@app.get("/api/download/{job_id}/{filename:path}")
async def download_file(job_id: str, filename: str):
    """Download a separated stem file."""
    # Security: validate job_id format
    if not validate_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    # Security: validate job exists
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Security: sanitize filename FIRST to prevent path traversal
    # Remove any path components and traversal sequences
    filename = os.path.basename(filename)
    if not filename or ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Build file path with sanitized filename
    file_path = OUTPUT_DIR / job_id / filename

    # Security: ensure path is within output directory (prevent traversal)
    try:
        file_path = file_path.resolve()
        output_resolved = OUTPUT_DIR.resolve()
        # Use is_relative_to for robust path checking (Python 3.9+)
        if not file_path.is_relative_to(output_resolved):
            raise HTTPException(status_code=403, detail="Access denied")
    except (ValueError, OSError):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )


@app.get("/api/download-all/{job_id}")
async def download_all(job_id: str):
    """Download all stems as a ZIP file."""
    # Security: validate job_id format
    if not validate_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    output_dir = OUTPUT_DIR / job_id
    if not output_dir.exists() or not output_dir.is_dir():
        raise HTTPException(status_code=404, detail="Output directory not found")

    # Create ZIP file
    zip_path = JOBS_DIR / f"{job_id}_stems.zip"
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", output_dir)

    return FileResponse(
        path=zip_path,
        filename=f"{job_id}_stems.zip",
        media_type="application/zip",
    )


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates."""
    # Validate job_id format before accepting connection
    if not validate_job_id(job_id):
        await websocket.close(code=4000, reason="Invalid job ID format")
        return

    # Check connection limit before accepting
    if job_manager.get_websocket_count(job_id) >= MAX_WEBSOCKET_CONNECTIONS_PER_JOB:
        await websocket.close(code=4003, reason="Too many connections for this job")
        return

    await websocket.accept()

    # Subscribe to job updates
    job_manager.subscribe(job_id, websocket)

    try:
        # Send current job status immediately
        job = job_manager.get_job(job_id)
        if job:
            await websocket.send_text(json.dumps({"type": "job_update", "job": job}))

        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))

    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, websocket)


# =============================================================================
# Static Files & Frontend
# =============================================================================

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main web UI."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        async with aiofiles.open(index_path, "r") as f:
            return await f.read()

    # Fallback minimal UI
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stem Separator</title>
        <meta charset="UTF-8">
    </head>
    <body>
        <h1>Stem Separator</h1>
        <p>Web UI files not found. Please ensure static files are installed.</p>
        <p>API documentation available at <a href="/api/docs">/api/docs</a></p>
    </body>
    </html>
    """


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the web server."""
    parser = argparse.ArgumentParser(description="Stem Separator Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
