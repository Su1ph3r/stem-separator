"""
Utility functions for Stem Separator.

Contains filename sanitization, validation, and helper functions.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Optional

from stem_separator.config import FORMATS, MODELS, PRESETS, FormatConfig
from stem_separator.logging_config import get_logger, print_error

# Security constants
SUBPROCESS_TIMEOUT = 300  # 5 minute timeout for subprocess calls
RESERVED_NAMES_WINDOWS = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})
SHELL_DANGEROUS_CHARS = frozenset("`$;|&<>(){}[]!\n\r")
ALLOWED_YOUTUBE_DOMAINS = frozenset({
    "www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com",
    "music.youtube.com", "www.youtube-nocookie.com",
})
ALLOWED_SPOTIFY_DOMAINS = frozenset({
    "open.spotify.com", "play.spotify.com", "spotify.com",
})

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """
    Remove problematic characters from filename with enhanced security.

    Args:
        name: Original filename.
        max_length: Maximum filename length (default 200 for path safety).

    Returns:
        Sanitized filename safe for all platforms.
    """
    # Unicode replacements for common problematic characters
    replacements = {
        "\uff02": '"',  # Full-width quotation mark
        "\uff07": "'",  # Full-width apostrophe
        "\uff0f": "-",  # Full-width solidus
        "\uff3c": "-",  # Full-width reverse solidus
        "\uff1a": "-",  # Full-width colon
        "\uff0a": "",  # Full-width asterisk
        "\uff1f": "",  # Full-width question mark
        "\uff1c": "",  # Full-width less-than
        "\uff1e": "",  # Full-width greater-than
        "\uff5c": "-",  # Full-width vertical line
        "\u201c": "",  # Left double quotation mark
        "\u201d": "",  # Right double quotation mark
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "/": "-",
        "\\": "-",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "-",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    # Remove remaining non-ASCII characters
    name = re.sub(r"[^\x00-\x7F]+", "", name)

    # Normalize whitespace
    name = re.sub(r"[-\s]+", " ", name).strip()
    name = re.sub(r"\s*-\s*", " - ", name)

    # Remove leading/trailing dots and spaces
    name = name.strip(". ")

    # Handle Windows reserved names
    if name:
        base_name = name.split(".")[0].upper()
        if base_name in RESERVED_NAMES_WINDOWS:
            name = f"file_{name}"

    # Limit length for path safety
    if len(name) > max_length:
        name = name[:max_length].rstrip(". ")

    return name if name and name.strip() else "audio"


def validate_path_safe(path: Path, base_dir: Optional[Path] = None) -> Path:
    """
    Validate that a path is safe and doesn't escape base directory.

    Args:
        path: Path to validate.
        base_dir: Optional base directory to constrain path within.

    Returns:
        Resolved absolute path.

    Raises:
        ValueError: If path is unsafe or escapes base directory.
    """
    resolved = path.resolve()

    # Check for path traversal if base_dir specified
    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Path '{path}' escapes base directory '{base_dir}'"
            )

    return resolved


def safe_subprocess_path(path: Path) -> str:
    """
    Prepare a path for safe use in subprocess calls.

    Args:
        path: Path to sanitize.

    Returns:
        Safe string representation of the path.

    Raises:
        ValueError: If path contains dangerous characters.
    """
    path_str = str(path.resolve())

    # Check for shell-dangerous characters
    if any(c in path_str for c in SHELL_DANGEROUS_CHARS):
        raise ValueError(f"Path contains unsafe characters: {path_str}")

    # Prevent flag injection (paths starting with -)
    if path_str.startswith("-"):
        raise ValueError(f"Path cannot start with hyphen: {path_str}")

    return path_str


def validate_youtube_url(url: str) -> bool:
    """
    Validate that a URL is a safe YouTube URL.

    Args:
        url: URL to validate.

    Returns:
        True if URL is a valid YouTube URL.
    """
    try:
        parsed = urllib.parse.urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Check domain
        if parsed.netloc not in ALLOWED_YOUTUBE_DOMAINS:
            return False

        # Check for dangerous characters that could be shell injection
        if any(c in url for c in SHELL_DANGEROUS_CHARS):
            return False

        return True
    except Exception:
        return False


def validate_spotify_url(url: str) -> bool:
    """
    Validate that a URL is a safe Spotify URL.

    Args:
        url: URL to validate.

    Returns:
        True if URL is a valid Spotify URL.
    """
    # Handle spotify: URI scheme
    if url.startswith("spotify:"):
        # Validate spotify URI format
        if any(c in url for c in SHELL_DANGEROUS_CHARS):
            return False
        return bool(re.match(r"^spotify:(track|album|playlist|artist):[a-zA-Z0-9]+$", url))

    try:
        parsed = urllib.parse.urlparse(url)

        # Check scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Check domain
        if parsed.netloc not in ALLOWED_SPOTIFY_DOMAINS:
            return False

        # Check for dangerous characters
        if any(c in url for c in SHELL_DANGEROUS_CHARS):
            return False

        return True
    except Exception:
        return False


def validate_model(model_name: str) -> str:
    """
    Validate model selection and return normalized name.

    Args:
        model_name: Model name to validate.

    Returns:
        Normalized model name.

    Raises:
        SystemExit: If model is invalid.
    """
    model_lower = model_name.lower()
    if model_lower not in MODELS:
        print_error(f"Invalid model '{model_name}'")
        print(f"Valid models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    return model_lower


def validate_format(format_name: str) -> str:
    """
    Validate output format and return normalized name.

    Args:
        format_name: Format name to validate.

    Returns:
        Normalized format name.

    Raises:
        SystemExit: If format is invalid.
    """
    format_lower = format_name.lower()
    if format_lower not in FORMATS:
        print_error(f"Invalid format '{format_name}'")
        print(f"Valid formats: {', '.join(FORMATS.keys())}")
        sys.exit(1)
    return format_lower


def parse_stem_selection(stems_arg: Optional[str], model_sources: list[str]) -> list[str]:
    """
    Parse --stems argument into list of stem names to export.

    Args:
        stems_arg: String like "vocals,drums" or preset name like "karaoke".
        model_sources: List of available stems from the model.

    Returns:
        List of stem names to export.

    Raises:
        SystemExit: If no valid stems specified.
    """
    if stems_arg is None:
        return list(model_sources)

    stems_lower = stems_arg.lower().strip()

    # Check if it's a preset
    if stems_lower in PRESETS:
        preset = PRESETS[stems_lower]

        if preset.include_all:
            return list(model_sources)
        elif preset.include:
            result = [s for s in model_sources if s in preset.include]
        elif preset.exclude:
            result = [s for s in model_sources if s not in preset.exclude]
        else:
            result = list(model_sources)

        if not result:
            print_error(f"Preset '{stems_lower}' results in no stems for this model")
            sys.exit(1)

        return result

    # Parse comma-separated list
    requested_stems = [s.strip().lower() for s in stems_arg.split(",") if s.strip()]

    if not requested_stems:
        print_error("No valid stems specified")
        sys.exit(1)

    # Validate all requested stems exist in model
    invalid_stems = [s for s in requested_stems if s not in model_sources]
    if invalid_stems:
        print_error(f"Invalid stem(s) for selected model: {', '.join(invalid_stems)}")
        print(f"Available stems: {', '.join(model_sources)}")
        sys.exit(1)

    return requested_stems


def check_dependencies(need_youtube: bool = False, need_spotify: bool = False) -> bool:
    """
    Check if required external dependencies are available.

    Args:
        need_youtube: Check for yt-dlp.
        need_spotify: Check for spotdl.

    Returns:
        True if all dependencies are available.

    Raises:
        SystemExit: If required dependency is missing.
    """
    logger = get_logger()

    # FFmpeg is always required
    if shutil.which("ffmpeg") is None:
        print_error("ffmpeg not found.")
        if sys.platform == "win32":
            print("  Install: winget install FFmpeg.FFmpeg")
        elif sys.platform == "darwin":
            print("  Install: brew install ffmpeg")
        else:
            print("  Install: sudo apt install ffmpeg  # or your package manager")
        sys.exit(1)

    if need_youtube and shutil.which("yt-dlp") is None:
        print_error("yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)

    if need_spotify and shutil.which("spotdl") is None:
        print_error("spotdl not found. Install with: pip install spotdl")
        sys.exit(1)

    logger.debug("All required dependencies found")
    return True


def is_youtube_url(input_str: str) -> bool:
    """
    Check if input is a YouTube URL.

    Note: This does basic domain checking. For security-critical operations,
    also use validate_youtube_url() which performs full validation.
    """
    input_lower = input_str.lower()
    return "youtube.com" in input_lower or "youtu.be" in input_lower


def is_youtube_playlist(input_str: str) -> bool:
    """Check if input is a YouTube playlist URL."""
    return is_youtube_url(input_str) and "list=" in input_str


def is_spotify_url(input_str: str) -> bool:
    """
    Check if input is a Spotify URL.

    Note: This does basic domain checking. For security-critical operations,
    also use validate_spotify_url() which performs full validation.
    """
    return "spotify.com" in input_str.lower() or "spotify:" in input_str.lower()


def is_spotify_playlist(input_str: str) -> bool:
    """Check if input is a Spotify playlist or album URL."""
    if not is_spotify_url(input_str):
        return False
    return "/playlist/" in input_str or "/album/" in input_str or "spotify:playlist:" in input_str


def convert_audio_format(
    input_wav: Path,
    output_path: Path,
    format_name: str,
    normalize: bool = False,
) -> Optional[Path]:
    """
    Convert WAV file to specified format using FFmpeg.

    Args:
        input_wav: Path to input WAV file.
        output_path: Base path for output (extension will be added).
        format_name: Format key from FORMATS dict.
        normalize: Whether to normalize audio levels.

    Returns:
        Path to converted file, or None on failure.
    """
    logger = get_logger()
    format_config = FORMATS[format_name]
    output_file = output_path.parent / f"{output_path.stem}{format_config.ext}"

    if format_name == "wav":
        if normalize:
            # Use FFmpeg to normalize even for WAV
            cmd = [
                "ffmpeg",
                "-i",
                str(input_wav),
                "-y",
                "-loglevel",
                "error",
                "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=11",
                str(output_file),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace",
                timeout=SUBPROCESS_TIMEOUT
            )
            if result.returncode != 0:
                logger.warning(f"Normalization failed, copying original: {result.stderr}")
                shutil.copy(input_wav, output_file)
        else:
            shutil.copy(input_wav, output_file)
        return output_file

    # Build FFmpeg command
    cmd = ["ffmpeg", "-i", str(input_wav), "-y", "-loglevel", "error"]

    # Add normalization filter if requested
    if normalize:
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])

    cmd.extend(["-acodec", format_config.codec])

    if format_config.bitrate:
        cmd.extend(["-b:a", format_config.bitrate])
    elif format_config.quality:
        cmd.extend(["-q:a", format_config.quality])

    cmd.append(str(output_file))

    result = subprocess.run(
        cmd, capture_output=True, text=True, errors="replace",
        timeout=SUBPROCESS_TIMEOUT
    )

    if result.returncode != 0:
        logger.warning(f"FFmpeg conversion failed for {output_file.name}: {result.stderr}")
        return None

    return output_file


def format_output_name(
    template: str,
    name: str,
    stem: str,
    model: str = "",
    format_name: str = "",
) -> str:
    """
    Format output filename using template.

    Args:
        template: Naming template with placeholders.
        name: Base name (track name).
        stem: Stem name.
        model: Model name.
        format_name: Output format.

    Returns:
        Formatted filename (without extension).
    """
    return template.format(
        name=sanitize_filename(name),
        stem=stem,
        model=model,
        format=format_name,
    )


def get_audio_duration(file_path: Path) -> Optional[float]:
    """
    Get duration of audio file in seconds.

    Args:
        file_path: Path to audio file.

    Returns:
        Duration in seconds, or None if unable to determine.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,  # 1 minute timeout for probe
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def estimate_memory_usage(duration_seconds: float, model_name: str) -> float:
    """
    Estimate memory usage for processing.

    Args:
        duration_seconds: Audio duration in seconds.
        model_name: Model name being used.

    Returns:
        Estimated memory usage in GB.
    """
    # Rough estimates based on model and duration
    # Base memory for model loading
    base_memory = 2.0  # GB

    # Memory per minute of audio (varies by model)
    memory_per_minute = {
        "htdemucs": 0.3,
        "htdemucs_ft": 0.35,
        "htdemucs_6s": 0.4,
    }

    minutes = duration_seconds / 60
    return base_memory + (memory_per_minute.get(model_name, 0.35) * minutes)


def collect_audio_files(path: Path, recursive: bool = False) -> list[Path]:
    """
    Collect audio files from a path.

    Args:
        path: File or directory path.
        recursive: Whether to search recursively in directories.

    Returns:
        List of audio file paths.
    """
    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus"}

    if path.is_file():
        if path.suffix.lower() in audio_extensions:
            return [path]
        return []

    if path.is_dir():
        if recursive:
            files = list(path.rglob("*"))
        else:
            files = list(path.iterdir())

        return sorted(
            [f for f in files if f.is_file() and f.suffix.lower() in audio_extensions]
        )

    return []
