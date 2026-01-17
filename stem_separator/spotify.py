"""
Spotify download functionality for Stem Separator.

Uses spotdl for downloading tracks from Spotify.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from stem_separator.logging_config import (
    create_progress,
    get_logger,
    is_quiet,
    print_error,
    print_status,
    print_warning,
)
from stem_separator.utils import sanitize_filename, validate_spotify_url, SUBPROCESS_TIMEOUT


@dataclass
class SpotifyTrack:
    """Information about a Spotify track."""

    url: str
    title: str
    artist: str
    duration: Optional[float] = None
    index: int = 0
    total: int = 1


@dataclass
class SpotifyDownloadResult:
    """Result of a Spotify download."""

    success: bool
    file_path: Optional[Path] = None
    title: str = ""
    artist: str = ""
    error: Optional[str] = None


class SpotifyDownloader:
    """Handles Spotify downloads using spotdl."""

    def __init__(self, output_dir: Path):
        """
        Initialize Spotify downloader.

        Args:
            output_dir: Directory to save downloads.
        """
        self.output_dir = Path(output_dir)
        self.logger = get_logger()

    def _check_spotdl(self) -> bool:
        """Check if spotdl is installed."""
        try:
            result = subprocess.run(
                ["spotdl", "--version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_track_info(self, url: str) -> Optional[SpotifyTrack]:
        """
        Get track information from Spotify URL.

        Args:
            url: Spotify track URL.

        Returns:
            SpotifyTrack with info, or None on failure.
        """
        try:
            result = subprocess.run(
                ["spotdl", "url", url, "--print-meta"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=SUBPROCESS_TIMEOUT,
            )

            if result.returncode == 0 and result.stdout.strip():
                # Parse spotdl output (format varies)
                lines = result.stdout.strip().split("\n")
                title = "Unknown Track"
                artist = "Unknown Artist"

                for line in lines:
                    if ":" in line:
                        key, _, value = line.partition(":")
                        key = key.strip().lower()
                        value = value.strip()
                        if key == "name" or key == "title":
                            title = value
                        elif key == "artist" or key == "artists":
                            artist = value

                return SpotifyTrack(url=url, title=title, artist=artist)

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout fetching track info for: {url}")
        except subprocess.SubprocessError as e:
            self.logger.warning(f"Failed to get track info: {e}")

        return None

    def get_playlist_tracks(self, url: str) -> list[SpotifyTrack]:
        """
        Get all tracks from a Spotify playlist or album.

        Args:
            url: Spotify playlist/album URL.

        Returns:
            List of SpotifyTrack objects.
        """
        tracks = []

        try:
            # Use spotdl to get playlist info
            result = subprocess.run(
                ["spotdl", "url", url, "--output", "{artist} - {title}"],
                capture_output=True,
                text=True,
                errors="replace",
                cwd=str(self.output_dir),
                timeout=SUBPROCESS_TIMEOUT,
            )

            # Parse the output to get track list
            # spotdl prints track info during download preparation
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.split("\n")):
                    line = line.strip()
                    if line and " - " in line and not line.startswith("Found"):
                        parts = line.split(" - ", 1)
                        if len(parts) == 2:
                            artist, title = parts
                            tracks.append(
                                SpotifyTrack(
                                    url=url,  # Playlist URL
                                    title=title.strip(),
                                    artist=artist.strip(),
                                    index=i,
                                )
                            )

        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout fetching playlist tracks for: {url}")
        except subprocess.SubprocessError as e:
            self.logger.warning(f"Failed to get playlist tracks: {e}")

        # Update totals
        for track in tracks:
            track.total = len(tracks)

        return tracks

    def download_single(
        self,
        url: str,
        filename_prefix: Optional[str] = None,
    ) -> SpotifyDownloadResult:
        """
        Download a single track from Spotify.

        Args:
            url: Spotify track URL.
            filename_prefix: Optional prefix for output filename.

        Returns:
            SpotifyDownloadResult with status and file path.
        """
        # Security: Validate URL before use
        if not validate_spotify_url(url):
            return SpotifyDownloadResult(
                success=False,
                error=f"Invalid or unsafe Spotify URL: {url}",
            )

        if not self._check_spotdl():
            return SpotifyDownloadResult(
                success=False,
                error="spotdl not found. Install with: pip install spotdl",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get track info first
        track_info = self.get_track_info(url)
        title = track_info.title if track_info else "Unknown Track"
        artist = track_info.artist if track_info else "Unknown Artist"

        if not is_quiet():
            print_status(f"Downloading: {artist} - {title}")

        # Download using spotdl
        output_template = "{artist} - {title}"
        if filename_prefix:
            output_template = f"{filename_prefix}"

        try:
            result = subprocess.run(
                [
                    "spotdl",
                    "download",
                    url,
                    "--output",
                    output_template,
                    "--format",
                    "wav",  # Request WAV for best quality
                ],
                capture_output=True,
                text=True,
                errors="replace",
                cwd=str(self.output_dir),
                timeout=SUBPROCESS_TIMEOUT,
            )

            if result.returncode != 0:
                # spotdl might not support WAV, try mp3 and convert
                result = subprocess.run(
                    [
                        "spotdl",
                        "download",
                        url,
                        "--output",
                        output_template,
                        "--format",
                        "mp3",
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    cwd=str(self.output_dir),
                    timeout=SUBPROCESS_TIMEOUT,
                )

                if result.returncode == 0:
                    # Find the downloaded file and convert to WAV
                    for file in self.output_dir.iterdir():
                        if file.suffix == ".mp3" and file.stem.startswith(
                            filename_prefix or artist
                        ):
                            wav_file = file.with_suffix(".wav")
                            try:
                                convert_result = subprocess.run(
                                    [
                                        "ffmpeg",
                                        "-i",
                                        str(file),
                                        "-y",
                                        "-loglevel",
                                        "error",
                                        str(wav_file),
                                    ],
                                    capture_output=True,
                                    timeout=SUBPROCESS_TIMEOUT,
                                )
                            except subprocess.TimeoutExpired:
                                file.unlink(missing_ok=True)
                                return SpotifyDownloadResult(
                                    success=False,
                                    title=title,
                                    artist=artist,
                                    error="Audio conversion timeout exceeded",
                                )
                            if convert_result.returncode == 0:
                                file.unlink()  # Remove mp3
                                return SpotifyDownloadResult(
                                    success=True,
                                    file_path=wav_file,
                                    title=title,
                                    artist=artist,
                                )

                return SpotifyDownloadResult(
                    success=False,
                    title=title,
                    artist=artist,
                    error=f"Download failed: {result.stderr}",
                )

            # Find the downloaded WAV file
            safe_name = sanitize_filename(f"{artist} - {title}")
            for file in self.output_dir.iterdir():
                if file.suffix == ".wav":
                    return SpotifyDownloadResult(
                        success=True,
                        file_path=file,
                        title=title,
                        artist=artist,
                    )

            return SpotifyDownloadResult(
                success=False,
                title=title,
                artist=artist,
                error="Downloaded file not found",
            )

        except subprocess.TimeoutExpired:
            return SpotifyDownloadResult(
                success=False,
                title=title,
                artist=artist,
                error=f"Download timeout exceeded ({SUBPROCESS_TIMEOUT}s)",
            )
        except subprocess.SubprocessError as e:
            return SpotifyDownloadResult(
                success=False,
                title=title,
                artist=artist,
                error=str(e),
            )

    def download_playlist(
        self,
        url: str,
    ) -> Iterator[SpotifyDownloadResult]:
        """
        Download all tracks from a Spotify playlist or album.

        Args:
            url: Spotify playlist/album URL.

        Yields:
            SpotifyDownloadResult for each track.
        """
        # Security: Validate URL before use
        if not validate_spotify_url(url):
            yield SpotifyDownloadResult(
                success=False,
                error=f"Invalid or unsafe Spotify URL: {url}",
            )
            return

        if not self._check_spotdl():
            yield SpotifyDownloadResult(
                success=False,
                error="spotdl not found. Install with: pip install spotdl",
            )
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        print_status("Fetching playlist information...")

        try:
            # Download entire playlist at once for efficiency
            result = subprocess.run(
                [
                    "spotdl",
                    "download",
                    url,
                    "--output",
                    "{list-position:03d}_{artist} - {title}",
                    "--format",
                    "mp3",  # More reliable than WAV
                ],
                capture_output=True,
                text=True,
                errors="replace",
                cwd=str(self.output_dir),
                timeout=SUBPROCESS_TIMEOUT,
            )

            # Find all downloaded files
            mp3_files = sorted(self.output_dir.glob("*.mp3"))

            if not mp3_files:
                yield SpotifyDownloadResult(
                    success=False,
                    error="No tracks downloaded from playlist",
                )
                return

            print_status(f"Downloaded {len(mp3_files)} tracks, converting to WAV...")

            with create_progress("Converting") as progress:
                task = progress.add_task("Convert", total=len(mp3_files))

                for mp3_file in mp3_files:
                    wav_file = mp3_file.with_suffix(".wav")

                    # Extract title and artist from filename
                    name_parts = mp3_file.stem.split("_", 1)
                    if len(name_parts) == 2:
                        name = name_parts[1]
                    else:
                        name = mp3_file.stem

                    if " - " in name:
                        artist, title = name.split(" - ", 1)
                    else:
                        artist = "Unknown"
                        title = name

                    # Convert to WAV
                    try:
                        convert_result = subprocess.run(
                            [
                                "ffmpeg",
                                "-i",
                                str(mp3_file),
                                "-y",
                                "-loglevel",
                                "error",
                                str(wav_file),
                            ],
                            capture_output=True,
                            timeout=SUBPROCESS_TIMEOUT,
                        )
                    except subprocess.TimeoutExpired:
                        yield SpotifyDownloadResult(
                            success=False,
                            title=title,
                            artist=artist,
                            error="WAV conversion timeout exceeded",
                        )
                        progress.advance(task)
                        continue

                    if convert_result.returncode == 0 and wav_file.exists():
                        mp3_file.unlink()
                        yield SpotifyDownloadResult(
                            success=True,
                            file_path=wav_file,
                            title=title.strip(),
                            artist=artist.strip(),
                        )
                    else:
                        yield SpotifyDownloadResult(
                            success=False,
                            title=title,
                            artist=artist,
                            error="WAV conversion failed",
                        )

                    progress.advance(task)

        except subprocess.TimeoutExpired:
            yield SpotifyDownloadResult(
                success=False,
                error=f"Playlist download timeout exceeded ({SUBPROCESS_TIMEOUT}s)",
            )
        except subprocess.SubprocessError as e:
            yield SpotifyDownloadResult(success=False, error=str(e))


def download_spotify(
    url: str,
    output_dir: Path,
    is_playlist: bool = False,
) -> Iterator[SpotifyDownloadResult] | SpotifyDownloadResult:
    """
    Download audio from Spotify.

    Args:
        url: Spotify URL (track, album, or playlist).
        output_dir: Directory to save downloads.
        is_playlist: Whether to treat as playlist/album.

    Returns:
        SpotifyDownloadResult for single track, or Iterator for playlist.
    """
    downloader = SpotifyDownloader(output_dir)

    if is_playlist or "/playlist/" in url or "/album/" in url:
        return downloader.download_playlist(url)
    else:
        return downloader.download_single(url)
