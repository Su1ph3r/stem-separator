"""
Stem Separator - Audio Stem Extraction Tool

Uses Demucs (by Facebook/Meta Research) for high-quality stem separation.
Supports YouTube URLs, Spotify tracks, and local audio files.
"""

__version__ = "2.0.0"
__author__ = "Stem Separator Contributors"

from stem_separator.config import MODELS, FORMATS, PRESETS
from stem_separator.processing import separate_audio, StemSeparator
from stem_separator.mixing import remix_stems

__all__ = [
    "MODELS",
    "FORMATS",
    "PRESETS",
    "separate_audio",
    "StemSeparator",
    "remix_stems",
    "__version__",
]
