"""
Configuration module for Stem Separator.

Contains model definitions, format configurations, presets, and user config handling.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a Demucs model."""

    sources: list[str]
    description: str


@dataclass
class FormatConfig:
    """Configuration for an audio output format."""

    ext: str
    description: str
    codec: Optional[str] = None
    bitrate: Optional[str] = None
    quality: Optional[str] = None
    requires_conversion: bool = True


@dataclass
class PresetConfig:
    """Configuration for a stem selection preset."""

    description: str
    include_all: bool = False
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class UserConfig:
    """User configuration loaded from config file."""

    model: str = "htdemucs"
    format: str = "wav"
    output_dir: str = "."
    stems: Optional[str] = None
    cpu: bool = False
    browser: Optional[str] = None
    verbose: bool = False
    quiet: bool = False
    normalize: bool = False
    naming_template: str = "{name}_{stem}"
    parallel_jobs: int = 1
    low_memory: bool = False


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS: dict[str, ModelConfig] = {
    "htdemucs": ModelConfig(
        sources=["drums", "bass", "other", "vocals"],
        description="Hybrid Transformer Demucs (default)",
    ),
    "htdemucs_ft": ModelConfig(
        sources=["drums", "bass", "other", "vocals"],
        description="Fine-tuned version (better quality)",
    ),
    "htdemucs_6s": ModelConfig(
        sources=["drums", "bass", "other", "vocals", "guitar", "piano"],
        description="6-stem model with guitar and piano",
    ),
}

# =============================================================================
# OUTPUT FORMAT CONFIGURATIONS
# =============================================================================

FORMATS: dict[str, FormatConfig] = {
    "wav": FormatConfig(
        ext=".wav",
        description="Lossless WAV (default)",
        requires_conversion=False,
    ),
    "mp3": FormatConfig(
        ext=".mp3",
        codec="libmp3lame",
        bitrate="320k",
        description="MP3 320kbps",
    ),
    "flac": FormatConfig(
        ext=".flac",
        codec="flac",
        description="Lossless FLAC",
    ),
    "ogg": FormatConfig(
        ext=".ogg",
        codec="libvorbis",
        quality="10",
        description="Ogg Vorbis (high quality)",
    ),
    "aac": FormatConfig(
        ext=".aac",
        codec="aac",
        bitrate="256k",
        description="AAC 256kbps",
    ),
}

# =============================================================================
# STEM SELECTION PRESETS
# =============================================================================

PRESETS: dict[str, PresetConfig] = {
    "all": PresetConfig(
        description="All stems (default)",
        include_all=True,
    ),
    "karaoke": PresetConfig(
        description="Everything except vocals",
        exclude=["vocals"],
    ),
    "acapella": PresetConfig(
        description="Vocals only",
        include=["vocals"],
    ),
    "instrumental": PresetConfig(
        description="Everything except vocals",
        exclude=["vocals"],
    ),
}

# =============================================================================
# CONFIGURATION FILE HANDLING
# =============================================================================

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".stem-separator.yaml",
    Path.home() / ".stem-separator.yml",
    Path.home() / ".config" / "stem-separator" / "config.yaml",
    Path.home() / ".config" / "stem-separator" / "config.yml",
]

if sys.platform == "win32":
    DEFAULT_CONFIG_PATHS.extend([
        Path(os.environ.get("APPDATA", "")) / "stem-separator" / "config.yaml",
        Path(os.environ.get("APPDATA", "")) / "stem-separator" / "config.yml",
    ])


def find_config_file() -> Optional[Path]:
    """Find the first existing configuration file."""
    for config_path in DEFAULT_CONFIG_PATHS:
        if config_path.exists():
            return config_path
    return None


def load_user_config(config_path: Optional[Path] = None) -> UserConfig:
    """
    Load user configuration from a YAML file.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        UserConfig with loaded settings or defaults.
    """
    if config_path is None:
        config_path = find_config_file()

    if config_path is None or not config_path.exists():
        return UserConfig()

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return UserConfig(
            model=data.get("model", "htdemucs"),
            format=data.get("format", "wav"),
            output_dir=data.get("output_dir", "."),
            stems=data.get("stems"),
            cpu=data.get("cpu", False),
            browser=data.get("browser"),
            verbose=data.get("verbose", False),
            quiet=data.get("quiet", False),
            normalize=data.get("normalize", False),
            naming_template=data.get("naming_template", "{name}_{stem}"),
            parallel_jobs=data.get("parallel_jobs", 1),
            low_memory=data.get("low_memory", False),
        )
    except (yaml.YAMLError, OSError):
        return UserConfig()


def save_user_config(config: UserConfig, config_path: Optional[Path] = None) -> Path:
    """
    Save user configuration to a YAML file.

    Args:
        config: UserConfig to save.
        config_path: Path to save to. Defaults to ~/.stem-separator.yaml.

    Returns:
        Path where config was saved.
    """
    if config_path is None:
        config_path = Path.home() / ".stem-separator.yaml"

    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "model": config.model,
        "format": config.format,
        "output_dir": config.output_dir,
        "stems": config.stems,
        "cpu": config.cpu,
        "browser": config.browser,
        "verbose": config.verbose,
        "quiet": config.quiet,
        "normalize": config.normalize,
        "naming_template": config.naming_template,
        "parallel_jobs": config.parallel_jobs,
        "low_memory": config.low_memory,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)

    return config_path


def generate_sample_config() -> str:
    """Generate a sample configuration file content."""
    return """# Stem Separator Configuration
# Place this file at ~/.stem-separator.yaml

# Default AI model: htdemucs, htdemucs_ft, htdemucs_6s
model: htdemucs

# Default output format: wav, mp3, flac, ogg, aac
format: wav

# Default output directory (use . for current directory)
output_dir: .

# Default stems to export: all, karaoke, acapella, instrumental, or comma-separated list
# stems: all

# Force CPU mode (skip GPU acceleration)
cpu: false

# Browser for YouTube cookies: chrome, firefox, edge, safari, opera, brave
# browser: chrome

# Verbose output (show detailed progress)
verbose: false

# Quiet mode (minimal output)
quiet: false

# Normalize audio levels in output stems
normalize: false

# Output naming template
# Available placeholders: {name}, {stem}, {model}, {format}
naming_template: "{name}_{stem}"

# Number of parallel jobs for batch processing
parallel_jobs: 1

# Low memory mode for processing very long tracks
low_memory: false
"""
