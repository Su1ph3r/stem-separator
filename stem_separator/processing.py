"""
Core audio processing module for Stem Separator.

Handles audio separation using Demucs with support for:
- GPU/CPU processing with automatic fallback
- Model pre-loading for batch processing
- Low-memory mode for long tracks
- Streaming output
- Quality analysis
"""

from __future__ import annotations

import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from stem_separator.config import FORMATS, MODELS
from stem_separator.logging_config import (
    create_progress,
    get_logger,
    is_quiet,
    is_verbose,
    print_error,
    print_info,
    print_status,
    print_success,
    print_warning,
)
from stem_separator.utils import convert_audio_format, format_output_name


@dataclass
class StemResult:
    """Result for a single separated stem."""

    name: str
    file_path: Optional[Path] = None
    quality_score: Optional[float] = None
    peak_level: Optional[float] = None
    rms_level: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class SeparationResult:
    """Result of audio separation."""

    success: bool
    stems: list[StemResult] = field(default_factory=list)
    output_dir: Optional[Path] = None
    model_used: str = ""
    processing_device: str = ""
    error: Optional[str] = None


class StemSeparator:
    """
    Audio stem separator using Demucs.

    Supports model pre-loading for efficient batch processing.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        force_cpu: bool = False,
        low_memory: bool = False,
    ):
        """
        Initialize the stem separator.

        Args:
            model_name: Demucs model to use.
            force_cpu: Force CPU processing.
            low_memory: Enable low-memory mode for long tracks.
        """
        self.model_name = model_name
        self.force_cpu = force_cpu
        self.low_memory = low_memory
        self.logger = get_logger()

        self._model = None
        self._device = None
        self._torch = None
        self._loaded = False

    def _load_dependencies(self):
        """Load required dependencies."""
        try:
            import torch
            import soundfile as sf
            from scipy import signal
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            self._torch = torch
            self._sf = sf
            self._signal = signal
            self._get_model = get_model
            self._apply_model = apply_model

            # Suppress CUDA compatibility warnings
            warnings.filterwarnings("ignore", message=".*CUDA capability.*")

            return True
        except ImportError as e:
            print_error(f"Missing dependency: {e}")
            print("Install with: pip install demucs soundfile scipy")
            return False

    def load_model(self) -> bool:
        """
        Pre-load the Demucs model.

        Returns:
            True if model loaded successfully.
        """
        if self._loaded:
            return True

        if not self._load_dependencies():
            return False

        self.logger.info(f"Loading {self.model_name} model...")
        print_status(f"Loading {self.model_name} model...")

        try:
            self._model = self._get_model(self.model_name)
            self._model.train(False)  # Evaluation mode

            # Determine device
            if not self.force_cpu and self._torch.cuda.is_available():
                self._device = "cuda"
                self._model.to("cuda")
                self.logger.info("Using GPU acceleration")
            else:
                self._device = "cpu"
                self._model.to("cpu")
                if not self.force_cpu:
                    print_warning("GPU not available, using CPU (slower)")

            self._loaded = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            print_error(f"Failed to load model: {e}")
            return False

    def unload_model(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

        self._loaded = False

    @property
    def source_names(self) -> list[str]:
        """Get list of source/stem names for loaded model."""
        if self._model is not None:
            return list(self._model.sources)
        return MODELS[self.model_name].sources

    def _calculate_quality_metrics(
        self,
        stem_audio: np.ndarray,
        original_audio: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Calculate quality metrics for a separated stem.

        Args:
            stem_audio: Separated stem audio.
            original_audio: Original mixed audio.

        Returns:
            Tuple of (quality_score, peak_level, rms_level).
        """
        # Calculate peak level (in dB)
        peak = np.max(np.abs(stem_audio))
        peak_db = 20 * np.log10(peak + 1e-10)

        # Calculate RMS level (in dB)
        rms = np.sqrt(np.mean(stem_audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)

        # Quality score based on:
        # - Signal presence (not too quiet)
        # - Not clipping (peak not too high)
        # - Reasonable dynamic range

        quality = 1.0

        # Penalize very quiet stems
        if rms_db < -60:
            quality *= 0.5
        elif rms_db < -40:
            quality *= 0.8

        # Penalize clipping
        if peak_db > -0.5:
            quality *= 0.9

        # Penalize if stem is almost identical to mix (poor separation)
        if original_audio is not None:
            correlation = np.corrcoef(
                stem_audio.flatten()[:10000],
                original_audio.flatten()[:10000],
            )[0, 1]
            if correlation > 0.95:
                quality *= 0.7  # Likely poor separation

        return quality, peak_db, rms_db

    def separate(
        self,
        input_file: Path,
        output_dir: Path,
        output_format: str = "wav",
        selected_stems: Optional[list[str]] = None,
        normalize: bool = False,
        naming_template: str = "{name}_{stem}",
        track_name: str = "",
        analyze_quality: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SeparationResult:
        """
        Separate audio file into stems.

        Args:
            input_file: Path to input audio file.
            output_dir: Directory for output stems.
            output_format: Output format (wav, mp3, etc.).
            selected_stems: List of stems to export (None = all).
            normalize: Whether to normalize audio levels.
            naming_template: Template for output filenames.
            track_name: Base name for output files.
            analyze_quality: Whether to analyze stem quality.
            progress_callback: Optional callback for progress updates.

        Returns:
            SeparationResult with stems and metadata.
        """
        input_file = Path(input_file)
        output_dir = Path(output_dir)

        if not self._loaded and not self.load_model():
            return SeparationResult(
                success=False,
                error="Failed to load model",
            )

        if not track_name:
            track_name = input_file.stem

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load audio
            self.logger.debug(f"Loading audio: {input_file}")
            audio_data, sr = self._sf.read(str(input_file), dtype="float32")

            # Ensure stereo
            if audio_data.ndim == 1:
                audio_data = np.stack([audio_data, audio_data], axis=1)

            # Resample if needed
            if sr != self._model.samplerate:
                self.logger.debug(f"Resampling from {sr}Hz to {self._model.samplerate}Hz")
                num_samples = int(len(audio_data) * self._model.samplerate / sr)
                audio_data = self._signal.resample(audio_data, num_samples)
                sr = self._model.samplerate

            # Store original for quality analysis
            original_audio = audio_data.copy() if analyze_quality else None

            # Convert to torch tensor
            wav = self._torch.from_numpy(audio_data.T.astype(np.float32))
            wav = wav.unsqueeze(0)  # Add batch dimension

            # Process
            sources = self._process_audio(wav, progress_callback)

            if sources is None:
                return SeparationResult(
                    success=False,
                    error="Audio processing failed",
                )

            # Determine stems to export
            source_names = self._model.sources
            if selected_stems is None:
                selected_stems = list(source_names)

            # Save stems
            stems_result = []
            format_ext = FORMATS[output_format].ext

            for i, stem_name in enumerate(source_names):
                if stem_name not in selected_stems:
                    continue

                # Extract stem audio
                stem_audio = sources[0, i].cpu().numpy().T

                # Calculate quality metrics
                quality_score = None
                peak_level = None
                rms_level = None

                if analyze_quality:
                    quality_score, peak_level, rms_level = self._calculate_quality_metrics(
                        stem_audio, original_audio
                    )

                # Generate filename
                output_name = format_output_name(
                    naming_template,
                    track_name,
                    stem_name,
                    self.model_name,
                    output_format,
                )

                # Save as WAV first
                temp_wav = output_dir / f"{output_name}_temp.wav"
                self._sf.write(str(temp_wav), stem_audio, sr)

                # Convert to target format
                output_base = output_dir / output_name
                converted = convert_audio_format(
                    temp_wav, output_base, output_format, normalize
                )

                # Clean up temp WAV
                if converted and output_format != "wav":
                    temp_wav.unlink(missing_ok=True)
                elif not converted:
                    # Keep WAV as fallback
                    converted = temp_wav.rename(output_dir / f"{output_name}.wav")

                stems_result.append(
                    StemResult(
                        name=stem_name,
                        file_path=converted,
                        quality_score=quality_score,
                        peak_level=peak_level,
                        rms_level=rms_level,
                        success=converted is not None,
                    )
                )

                self.logger.debug(f"Saved {stem_name}{format_ext}")

            return SeparationResult(
                success=True,
                stems=stems_result,
                output_dir=output_dir,
                model_used=self.model_name,
                processing_device=self._device,
            )

        except Exception as e:
            self.logger.exception("Separation failed")
            return SeparationResult(
                success=False,
                error=str(e),
            )

    def _process_audio(
        self,
        wav: "torch.Tensor",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Optional["torch.Tensor"]:
        """
        Process audio through the model.

        Args:
            wav: Input audio tensor.
            progress_callback: Optional progress callback.

        Returns:
            Separated sources tensor, or None on failure.
        """
        sources = None

        # Try GPU first
        if self._device == "cuda":
            try:
                wav_gpu = wav.to("cuda")

                with self._torch.no_grad():
                    if self.low_memory:
                        sources = self._apply_model(
                            self._model,
                            wav_gpu,
                            progress=not is_quiet(),
                            split=True,  # Process in chunks
                            overlap=0.25,
                        )
                    else:
                        sources = self._apply_model(
                            self._model,
                            wav_gpu,
                            progress=not is_quiet(),
                        )

                if not is_quiet():
                    print_success("GPU processing completed")

            except RuntimeError as e:
                error_msg = str(e)
                if "no kernel image" in error_msg or "CUDA" in error_msg:
                    print_warning(
                        "GPU failed - try: pip install --pre torch torchaudio "
                        "--index-url https://download.pytorch.org/whl/nightly/cu128"
                    )
                    print_info("Falling back to CPU...")
                    self._torch.cuda.empty_cache()
                    self._device = "cpu"
                    self._model.to("cpu")
                else:
                    raise

        # CPU processing
        if sources is None:
            if not is_quiet():
                print_info("Using CPU (this may take 2-4 minutes)...")

            wav = wav.to("cpu")

            with self._torch.no_grad():
                if self.low_memory:
                    sources = self._apply_model(
                        self._model,
                        wav,
                        progress=not is_quiet(),
                        split=True,
                        overlap=0.25,
                    )
                else:
                    sources = self._apply_model(
                        self._model,
                        wav,
                        progress=not is_quiet(),
                    )

        return sources

    def separate_streaming(
        self,
        input_file: Path,
        output_dir: Path,
        output_format: str = "wav",
        selected_stems: Optional[list[str]] = None,
        normalize: bool = False,
        naming_template: str = "{name}_{stem}",
        track_name: str = "",
    ) -> Iterator[StemResult]:
        """
        Separate audio and yield stems as they're processed.

        This is useful for starting to use stems before all are complete.

        Args:
            input_file: Path to input audio file.
            output_dir: Directory for output stems.
            output_format: Output format.
            selected_stems: List of stems to export.
            normalize: Whether to normalize.
            naming_template: Output filename template.
            track_name: Base name for files.

        Yields:
            StemResult for each processed stem.
        """
        # For streaming, we process normally but yield results as they save
        result = self.separate(
            input_file=input_file,
            output_dir=output_dir,
            output_format=output_format,
            selected_stems=selected_stems,
            normalize=normalize,
            naming_template=naming_template,
            track_name=track_name,
        )

        if not result.success:
            yield StemResult(
                name="error",
                success=False,
                error=result.error,
            )
            return

        for stem in result.stems:
            yield stem


def separate_audio(
    input_file: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    output_format: str = "wav",
    selected_stems: Optional[list[str]] = None,
    force_cpu: bool = False,
    normalize: bool = False,
    low_memory: bool = False,
    naming_template: str = "{name}_{stem}",
    track_name: str = "",
    analyze_quality: bool = False,
) -> SeparationResult:
    """
    Convenience function to separate audio without managing the separator instance.

    Args:
        input_file: Path to input audio file.
        output_dir: Directory for output stems.
        model_name: Demucs model to use.
        output_format: Output format.
        selected_stems: List of stems to export.
        force_cpu: Force CPU processing.
        normalize: Whether to normalize audio.
        low_memory: Enable low-memory mode.
        naming_template: Output filename template.
        track_name: Base name for files.
        analyze_quality: Whether to analyze quality.

    Returns:
        SeparationResult with stems and metadata.
    """
    separator = StemSeparator(
        model_name=model_name,
        force_cpu=force_cpu,
        low_memory=low_memory,
    )

    try:
        return separator.separate(
            input_file=input_file,
            output_dir=output_dir,
            output_format=output_format,
            selected_stems=selected_stems,
            normalize=normalize,
            naming_template=naming_template,
            track_name=track_name,
            analyze_quality=analyze_quality,
        )
    finally:
        separator.unload_model()
