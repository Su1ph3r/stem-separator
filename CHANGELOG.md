# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-01-16

### Added
- **Modular Package Architecture**: Refactored into installable Python package (`pip install -e .`)
- **Batch Processing**: Process entire directories with `--batch` flag
- **YouTube Playlist Support**: Download and process entire playlists with `--playlist`
- **Spotify Support**: Download and process Spotify tracks and playlists (requires spotdl)
- **Custom Stem Mixing**: Remix stems with volume control via `--remix "vocals:0.5,drums:1.0"`
- **Real-time Preview**: Preview separated stems with `--preview` (requires sounddevice)
- **Stem Quality Analysis**: Analyze separation quality with `--quality` flag
- **Progress Bars**: Visual progress feedback using rich library
- **Configuration File**: Save defaults in `~/.stem-separator.yaml`
- **Verbose/Quiet Modes**: `-v` for verbose, `-q` for quiet output
- **Output Naming Templates**: Customize output filenames with `--naming`
- **Resume Downloads**: Automatically resume interrupted playlist downloads
- **Parallel Processing**: Process multiple files with `-j N` flag
- **Model Pre-loading**: Efficient batch processing with shared model instance
- **Streaming Output**: Yields stems as they're processed
- **Low-memory Mode**: `--low-memory` flag for processing very long tracks
- **Metadata Preservation**: Copy ID3 tags from source to stems (requires mutagen)
- **Stem Normalization**: Normalize audio levels with `--normalize`
- **Dry Run Mode**: Preview operations with `--dry-run`
- **Cross-platform Install Scripts**: `install.sh` for Linux/macOS
- **PyPI Package Structure**: Full `pyproject.toml` with optional dependencies
- **Unit Tests**: Comprehensive test suite for core functionality
- **Type Hints**: Full type annotations throughout codebase
- **Logging Framework**: Configurable logging with rich formatting

### Changed
- Reorganized codebase into `stem_separator/` package
- Enhanced `sanitize_filename()` with Windows reserved name handling
- Improved error messages and user feedback
- Updated `install.bat` with optional dependency prompts

### Security
- Added URL validation for YouTube and Spotify URLs
- Added path traversal protection
- Added subprocess timeouts to prevent hangs
- Enhanced input sanitization

### Backward Compatibility
- Original `stem_separator.py` script still works as before
- Automatically uses new package if installed, falls back to legacy mode

## [1.0.0] - 2025-01-07

### Added
- Initial release
- Basic stem separation using Demucs
- YouTube URL support
- Multiple output formats (WAV, MP3, FLAC, OGG, AAC)
- Model selection (htdemucs, htdemucs_ft, htdemucs_6s)
- Stem presets (all, karaoke, acapella, instrumental)
- GPU acceleration with CUDA fallback
- Browser cookie support for YouTube authentication
