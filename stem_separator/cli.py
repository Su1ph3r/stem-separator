"""
Command-line interface for Stem Separator.

Provides the main entry point and argument parsing.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from stem_separator import __version__
from stem_separator.config import (
    FORMATS,
    MODELS,
    PRESETS,
    generate_sample_config,
    load_user_config,
)
from stem_separator.logging_config import (
    create_progress,
    is_quiet,
    print_error,
    print_info,
    print_status,
    print_success,
    print_warning,
    setup_logging,
)
from stem_separator.utils import (
    check_dependencies,
    is_spotify_playlist,
    is_spotify_url,
    is_youtube_playlist,
    is_youtube_url,
    parse_stem_selection,
    sanitize_filename,
    validate_format,
    validate_model,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Separate audio into stems using AI (Demucs)",
        epilog="""Examples:
  %(prog)s song.mp3
  %(prog)s song.mp3 --model htdemucs_6s --format mp3
  %(prog)s song.mp3 --stems karaoke --format flac
  %(prog)s "https://youtube.com/watch?v=..." --stems acapella
  %(prog)s ./music_folder --batch -o ./stems
  %(prog)s song.mp3 --remix "vocals:0.5,drums:1.0,bass:0.8"
  %(prog)s "https://open.spotify.com/track/..." --format mp3
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional arguments
    parser.add_argument(
        "input",
        nargs="?",
        help="Audio file, directory (with --batch), YouTube URL, or Spotify URL",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-o", "--output",
        default=".",
        help="Output directory (default: current)",
    )
    output_group.add_argument(
        "--format",
        default=None,
        help=f"Output format: {', '.join(FORMATS.keys())} (default: wav)",
    )
    output_group.add_argument(
        "--naming",
        dest="naming_template",
        default=None,
        help="Output naming template (default: {name}_{stem})",
    )

    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model",
        default=None,
        help=f"AI model: {', '.join(MODELS.keys())} (default: htdemucs)",
    )
    model_group.add_argument(
        "--stems",
        default=None,
        help="Stems to export: comma-separated (vocals,drums) or preset "
             "(all, karaoke, acapella, instrumental)",
    )

    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (skip GPU)",
    )
    proc_group.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize audio levels in output stems",
    )
    proc_group.add_argument(
        "--low-memory",
        action="store_true",
        help="Low memory mode for processing very long tracks",
    )
    proc_group.add_argument(
        "--quality",
        action="store_true",
        help="Analyze and report stem separation quality",
    )

    # Batch processing
    batch_group = parser.add_argument_group("Batch Processing")
    batch_group.add_argument(
        "--batch",
        action="store_true",
        help="Process directory of files or multiple inputs",
    )
    batch_group.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Search directories recursively (with --batch)",
    )
    batch_group.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel jobs for batch processing",
    )

    # Playlist options
    playlist_group = parser.add_argument_group("Playlist Options")
    playlist_group.add_argument(
        "--playlist",
        action="store_true",
        help="Process YouTube/Spotify playlist instead of single track",
    )

    # Source options
    source_group = parser.add_argument_group("Source Options")
    source_group.add_argument(
        "--browser",
        choices=["chrome", "firefox", "edge", "safari", "opera", "brave"],
        help="Use cookies from browser (helps with YouTube 403 errors)",
    )

    # Remix options
    remix_group = parser.add_argument_group("Remix Options")
    remix_group.add_argument(
        "--remix",
        metavar="SPEC",
        help="Remix stems with volume control (e.g., 'vocals:0.5,drums:1.0')",
    )

    # Preview options
    preview_group = parser.add_argument_group("Preview Options")
    preview_group.add_argument(
        "--preview",
        action="store_true",
        help="Preview stems interactively after processing",
    )

    # Metadata options
    meta_group = parser.add_argument_group("Metadata Options")
    meta_group.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't preserve metadata from input file",
    )

    # Output control
    verbosity_group = parser.add_argument_group("Output Control")
    verbosity_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show detailed progress)",
    )
    verbosity_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode (minimal output)",
    )
    verbosity_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually processing",
    )

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    util_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    util_group.add_argument(
        "--list-formats",
        action="store_true",
        help="List available output formats and exit",
    )
    util_group.add_argument(
        "--list-presets",
        action="store_true",
        help="List stem selection presets and exit",
    )
    util_group.add_argument(
        "--generate-config",
        action="store_true",
        help="Print sample configuration file and exit",
    )

    return parser


def handle_utility_commands(args) -> bool:
    """
    Handle utility commands that don't require processing.

    Returns:
        True if a utility command was handled.
    """
    if args.list_models:
        print("Available models:")
        for name, config in MODELS.items():
            print(f"  {name}: {config.description}")
            print(f"    Stems: {', '.join(config.sources)}")
        return True

    if args.list_formats:
        print("Available output formats:")
        for name, config in FORMATS.items():
            print(f"  {name}: {config.description}")
        return True

    if args.list_presets:
        print("Available stem presets:")
        for name, config in PRESETS.items():
            print(f"  {name}: {config.description}")
        return True

    if args.generate_config:
        print(generate_sample_config())
        return True

    return False


def process_single_file(
    input_path: Path,
    output_dir: Path,
    args,
    user_config,
    temp_dir: Path,
) -> bool:
    """
    Process a single local audio file.

    Returns:
        True if successful.
    """
    from stem_separator.metadata import copy_metadata, read_metadata
    from stem_separator.processing import separate_audio

    if not input_path.exists():
        print_error(f"File not found: {input_path}")
        return False

    display_name = sanitize_filename(input_path.stem)

    # Read metadata for preservation
    original_metadata = None
    if not args.no_metadata:
        original_metadata = read_metadata(input_path)

    # Convert to WAV if needed
    audio_file = temp_dir / "input.wav"
    if input_path.suffix.lower() == ".wav":
        shutil.copy(input_path, audio_file)
    else:
        if not is_quiet():
            print_status("Converting to WAV...")
        import subprocess
        subprocess.run(
            ["ffmpeg", "-i", str(input_path), "-y", "-loglevel", "error", str(audio_file)],
            check=True,
        )

    if not is_quiet():
        print_status(f"Processing: {display_name}")

    # Get settings
    model_name = args.model or user_config.model
    output_format = args.format or user_config.format
    naming_template = args.naming_template or user_config.naming_template
    normalize = args.normalize or user_config.normalize
    low_memory = args.low_memory or user_config.low_memory

    # Validate
    model_name = validate_model(model_name)
    output_format = validate_format(output_format)
    model_sources = MODELS[model_name].sources
    selected_stems = parse_stem_selection(args.stems or user_config.stems, model_sources)

    # Process
    stems_output = temp_dir / "stems"
    result = separate_audio(
        input_file=audio_file,
        output_dir=stems_output,
        model_name=model_name,
        output_format=output_format,
        selected_stems=selected_stems,
        force_cpu=args.cpu or user_config.cpu,
        normalize=normalize,
        low_memory=low_memory,
        naming_template=naming_template,
        track_name=display_name,
        analyze_quality=args.quality,
    )

    if not result.success:
        print_error(f"Processing failed: {result.error}")
        return False

    # Copy metadata to output files
    if original_metadata and not args.no_metadata:
        for stem in result.stems:
            if stem.file_path and stem.file_path.exists():
                copy_metadata(input_path, stem.file_path, stem.name)

    # Copy results to final output
    final_dir = output_dir / f"{display_name}_stems"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(stems_output, final_dir)

    # Show results
    print_success(f"Done! Stems saved to: {final_dir}")

    if args.quality:
        print_info("\nQuality Analysis:")
        for stem in result.stems:
            if stem.quality_score is not None:
                score_pct = int(stem.quality_score * 100)
                print_info(
                    f"  {stem.name}: {score_pct}% quality, "
                    f"peak: {stem.peak_level:.1f}dB, RMS: {stem.rms_level:.1f}dB"
                )

    print_info(f"\nCreated {len(result.stems)} stem(s) in {output_format.upper()} format:")
    for stem in result.stems:
        if stem.file_path:
            print_info(f"  - {stem.file_path.name}")

    return True


def process_youtube(
    url: str,
    output_dir: Path,
    args,
    user_config,
    temp_dir: Path,
) -> bool:
    """
    Process YouTube URL(s).

    Returns:
        True if successful.
    """
    from stem_separator.youtube import download_youtube, YouTubeDownloader

    is_playlist = args.playlist or is_youtube_playlist(url)

    downloader = YouTubeDownloader(
        temp_dir,
        browser=args.browser or user_config.browser,
    )

    if is_playlist:
        # Process playlist
        print_status("Processing YouTube playlist...")
        results = list(downloader.download_playlist(url))
        successful = 0

        for result in results:
            if result.success and result.file_path:
                # Process each downloaded file
                if process_single_file(
                    result.file_path,
                    output_dir,
                    args,
                    user_config,
                    temp_dir,
                ):
                    successful += 1
            else:
                print_warning(f"Failed to download: {result.title} - {result.error}")

        print_success(f"Processed {successful}/{len(results)} tracks from playlist")
        return successful > 0

    else:
        # Single video
        print_status("Downloading from YouTube...")
        result = downloader.download_single(url)

        if not result.success:
            print_error(f"Download failed: {result.error}")
            return False

        # Process the downloaded file
        return process_single_file(
            result.file_path,
            output_dir,
            args,
            user_config,
            temp_dir,
        )


def process_spotify(
    url: str,
    output_dir: Path,
    args,
    user_config,
    temp_dir: Path,
) -> bool:
    """
    Process Spotify URL(s).

    Returns:
        True if successful.
    """
    from stem_separator.spotify import SpotifyDownloader

    is_playlist = args.playlist or is_spotify_playlist(url)

    downloader = SpotifyDownloader(temp_dir)

    if is_playlist:
        # Process playlist/album
        print_status("Processing Spotify playlist/album...")
        results = list(downloader.download_playlist(url))
        successful = 0

        for result in results:
            if result.success and result.file_path:
                if process_single_file(
                    result.file_path,
                    output_dir,
                    args,
                    user_config,
                    temp_dir,
                ):
                    successful += 1
            else:
                print_warning(f"Failed: {result.artist} - {result.title} - {result.error}")

        print_success(f"Processed {successful}/{len(results)} tracks")
        return successful > 0

    else:
        # Single track
        print_status("Downloading from Spotify...")
        result = downloader.download_single(url)

        if not result.success:
            print_error(f"Download failed: {result.error}")
            return False

        return process_single_file(
            result.file_path,
            output_dir,
            args,
            user_config,
            temp_dir,
        )


def process_batch_mode(
    input_path: Path,
    output_dir: Path,
    args,
    user_config,
) -> bool:
    """
    Process batch of files.

    Returns:
        True if any successful.
    """
    from stem_separator.batch import BatchProcessor

    # Get settings
    model_name = validate_model(args.model or user_config.model)
    output_format = validate_format(args.format or user_config.format)
    model_sources = MODELS[model_name].sources
    selected_stems = parse_stem_selection(args.stems or user_config.stems, model_sources)

    processor = BatchProcessor(
        model_name=model_name,
        output_format=output_format,
        selected_stems=selected_stems,
        force_cpu=args.cpu or user_config.cpu,
        normalize=args.normalize or user_config.normalize,
        low_memory=args.low_memory or user_config.low_memory,
        naming_template=args.naming_template or user_config.naming_template,
        parallel_jobs=args.parallel,
        preserve_metadata=not args.no_metadata,
        analyze_quality=args.quality,
        dry_run=args.dry_run,
    )

    items = BatchProcessor.from_directory(input_path, output_dir, args.recursive)

    if not items:
        print_warning("No audio files found")
        return False

    print_status(f"Found {len(items)} audio files")

    result = processor.process_batch(items)

    print_success(
        f"\nBatch complete: {result.successful} successful, "
        f"{result.failed} failed, {result.skipped} skipped"
    )

    return result.successful > 0


def handle_remix(
    stems_dir: Path,
    output_path: Path,
    remix_spec: str,
    output_format: str,
) -> bool:
    """
    Handle remix command.

    Returns:
        True if successful.
    """
    from stem_separator.mixing import remix_stems

    result = remix_stems(
        stems_dir=stems_dir,
        output_path=output_path,
        mix_components=remix_spec,
        output_format=output_format,
    )

    if not result.success:
        print_error(f"Remix failed: {result.error}")
        return False

    print_success(f"Remix saved: {result.output_path}")
    return True


def handle_preview(stems_dir: Path):
    """Handle interactive preview."""
    from stem_separator.preview import preview_stems

    preview_stems(stems_dir, interactive=True)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle utility commands first
    if handle_utility_commands(args):
        return

    # Set up logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Load user config
    user_config = load_user_config()

    # Check for input
    if args.input is None:
        parser.print_help()
        sys.exit(1)

    # Determine input type
    input_str = args.input
    output_dir = Path(args.output or user_config.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check dependencies
    need_youtube = is_youtube_url(input_str)
    need_spotify = is_spotify_url(input_str)
    check_dependencies(need_youtube=need_youtube, need_spotify=need_spotify)

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="stems_"))

    try:
        # Batch processing mode
        if args.batch:
            input_path = Path(input_str)
            if not input_path.exists():
                print_error(f"Path not found: {input_path}")
                sys.exit(1)

            success = process_batch_mode(input_path, output_dir, args, user_config)

        # YouTube URL
        elif is_youtube_url(input_str):
            success = process_youtube(input_str, output_dir, args, user_config, temp_dir)

        # Spotify URL
        elif is_spotify_url(input_str):
            success = process_spotify(input_str, output_dir, args, user_config, temp_dir)

        # Local file
        else:
            input_path = Path(input_str).absolute()
            success = process_single_file(input_path, output_dir, args, user_config, temp_dir)

        # Handle remix if specified
        if success and args.remix:
            # Find the stems directory
            stems_dirs = list(output_dir.glob("*_stems"))
            if stems_dirs:
                latest_stems = max(stems_dirs, key=lambda p: p.stat().st_mtime)
                remix_output = output_dir / f"{latest_stems.stem}_remix"
                handle_remix(
                    latest_stems,
                    remix_output,
                    args.remix,
                    args.format or user_config.format,
                )

        # Handle preview if requested
        if success and args.preview:
            stems_dirs = list(output_dir.glob("*_stems"))
            if stems_dirs:
                latest_stems = max(stems_dirs, key=lambda p: p.stat().st_mtime)
                handle_preview(latest_stems)

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print_warning("\nInterrupted")
        sys.exit(130)

    except Exception as e:
        print_error(str(e))
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
