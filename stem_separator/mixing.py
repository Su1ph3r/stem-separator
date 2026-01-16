"""
Stem mixing module for Stem Separator.

Provides functionality to remix separated stems with volume control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from stem_separator.logging_config import (
    get_logger,
    print_error,
    print_info,
    print_status,
    print_success,
)


@dataclass
class MixComponent:
    """A single component in a mix."""

    stem_name: str
    volume: float = 1.0  # 0.0 to 2.0 (0 = mute, 1 = normal, 2 = double)


@dataclass
class MixResult:
    """Result of a mixing operation."""

    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    stems_used: list[str] = None

    def __post_init__(self):
        if self.stems_used is None:
            self.stems_used = []


def parse_remix_string(remix_str: str) -> list[MixComponent]:
    """
    Parse a remix specification string.

    Format: "stem1:volume,stem2:volume,..."
    Example: "vocals:0.5,drums:1.0,bass:0.8"

    Args:
        remix_str: Remix specification string.

    Returns:
        List of MixComponent objects.

    Raises:
        ValueError: If format is invalid.
    """
    components = []

    for part in remix_str.split(","):
        part = part.strip()
        if not part:
            continue

        if ":" in part:
            stem_name, volume_str = part.split(":", 1)
            try:
                volume = float(volume_str)
                if not 0.0 <= volume <= 2.0:
                    raise ValueError(f"Volume must be between 0.0 and 2.0, got {volume}")
            except ValueError as e:
                raise ValueError(f"Invalid volume for {stem_name}: {volume_str}") from e
        else:
            stem_name = part
            volume = 1.0

        components.append(MixComponent(stem_name=stem_name.lower(), volume=volume))

    if not components:
        raise ValueError("No valid mix components found")

    return components


def remix_stems(
    stems_dir: Path,
    output_path: Path,
    mix_components: list[MixComponent] | str,
    output_format: str = "wav",
    normalize_output: bool = True,
) -> MixResult:
    """
    Remix separated stems with volume control.

    Args:
        stems_dir: Directory containing stem files.
        output_path: Path for output mixed file.
        mix_components: List of MixComponent or remix string like "vocals:0.5,drums:1.0".
        output_format: Output format (wav, mp3, etc.).
        normalize_output: Whether to normalize the mixed output.

    Returns:
        MixResult with output path or error.
    """
    logger = get_logger()

    # Parse string format if needed
    if isinstance(mix_components, str):
        try:
            mix_components = parse_remix_string(mix_components)
        except ValueError as e:
            return MixResult(success=False, error=str(e))

    stems_dir = Path(stems_dir)
    output_path = Path(output_path)

    # Import required libraries
    try:
        import soundfile as sf
    except ImportError:
        return MixResult(success=False, error="soundfile not installed")

    # Find available stem files
    available_stems = {}
    for file in stems_dir.iterdir():
        if file.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".aac"}:
            # Extract stem name from filename (assumes format: name_stem.ext or stem.ext)
            stem_name = file.stem.lower()
            # Check common stem names
            for known_stem in ["vocals", "drums", "bass", "other", "guitar", "piano"]:
                if known_stem in stem_name:
                    available_stems[known_stem] = file
                    break
            else:
                # Use the full stem name
                available_stems[stem_name] = file

    logger.debug(f"Available stems: {list(available_stems.keys())}")

    # Validate requested stems exist
    missing_stems = []
    for component in mix_components:
        if component.stem_name not in available_stems:
            missing_stems.append(component.stem_name)

    if missing_stems:
        return MixResult(
            success=False,
            error=f"Stems not found: {', '.join(missing_stems)}. "
            f"Available: {', '.join(available_stems.keys())}",
        )

    print_status(f"Mixing {len(mix_components)} stems...")

    # Load and mix stems
    mixed_audio = None
    sample_rate = None
    stems_used = []

    for component in mix_components:
        stem_file = available_stems[component.stem_name]

        # Load audio
        audio, sr = sf.read(str(stem_file), dtype="float32")

        if sample_rate is None:
            sample_rate = sr
            mixed_audio = np.zeros_like(audio)
        elif sr != sample_rate:
            logger.warning(f"Sample rate mismatch for {component.stem_name}, resampling...")
            # Simple resampling (for production, use scipy.signal.resample)
            from scipy import signal

            num_samples = int(len(audio) * sample_rate / sr)
            audio = signal.resample(audio, num_samples)

        # Ensure same shape
        if len(audio) > len(mixed_audio):
            # Extend mixed audio
            new_mixed = np.zeros((len(audio), mixed_audio.shape[1] if mixed_audio.ndim > 1 else 1))
            new_mixed[: len(mixed_audio)] = mixed_audio
            mixed_audio = new_mixed
        elif len(audio) < len(mixed_audio):
            # Pad audio
            if audio.ndim == 1:
                audio = np.pad(audio, (0, len(mixed_audio) - len(audio)))
            else:
                audio = np.pad(audio, ((0, len(mixed_audio) - len(audio)), (0, 0)))

        # Apply volume and add to mix
        mixed_audio += audio * component.volume
        stems_used.append(f"{component.stem_name}:{component.volume}")

        logger.debug(f"Added {component.stem_name} at volume {component.volume}")

    if mixed_audio is None:
        return MixResult(success=False, error="No audio data loaded")

    # Normalize if requested
    if normalize_output:
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            # Normalize to -1dB headroom
            target_peak = 10 ** (-1 / 20)  # -1 dB
            mixed_audio = mixed_audio * (target_peak / max_val)
            logger.debug(f"Normalized output (peak was {max_val:.3f})")

    # Clip to prevent distortion
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as WAV first
    temp_wav = output_path.with_suffix(".wav")
    sf.write(str(temp_wav), mixed_audio, sample_rate)

    # Convert to target format if needed
    if output_format.lower() != "wav":
        from stem_separator.utils import convert_audio_format

        final_path = convert_audio_format(
            temp_wav,
            output_path.with_suffix(""),
            output_format,
        )

        if final_path:
            temp_wav.unlink(missing_ok=True)
            output_path = final_path
        else:
            output_path = temp_wav
            logger.warning("Format conversion failed, kept as WAV")
    else:
        # Rename temp to final
        final_path = output_path.with_suffix(".wav")
        if temp_wav != final_path:
            temp_wav.rename(final_path)
        output_path = final_path

    print_success(f"Mixed output saved: {output_path.name}")
    print_info(f"Components: {', '.join(stems_used)}")

    return MixResult(
        success=True,
        output_path=output_path,
        stems_used=stems_used,
    )


def create_karaoke(
    stems_dir: Path,
    output_path: Path,
    output_format: str = "wav",
) -> MixResult:
    """
    Create a karaoke version by mixing all stems except vocals.

    Args:
        stems_dir: Directory containing stem files.
        output_path: Path for output file.
        output_format: Output format.

    Returns:
        MixResult with output path or error.
    """
    # Find all non-vocal stems
    stems_dir = Path(stems_dir)
    components = []

    for file in stems_dir.iterdir():
        stem_name = file.stem.lower()
        if "vocal" not in stem_name:
            for known_stem in ["drums", "bass", "other", "guitar", "piano"]:
                if known_stem in stem_name:
                    components.append(MixComponent(stem_name=known_stem, volume=1.0))
                    break

    if not components:
        return MixResult(success=False, error="No non-vocal stems found")

    return remix_stems(stems_dir, output_path, components, output_format)


def create_acapella(
    stems_dir: Path,
    output_path: Path,
    output_format: str = "wav",
) -> MixResult:
    """
    Extract just the vocals.

    Args:
        stems_dir: Directory containing stem files.
        output_path: Path for output file.
        output_format: Output format.

    Returns:
        MixResult with output path or error.
    """
    return remix_stems(
        stems_dir,
        output_path,
        [MixComponent(stem_name="vocals", volume=1.0)],
        output_format,
    )
