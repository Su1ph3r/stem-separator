"""
YouTube download functionality for Stem Separator.

Supports single videos and playlists with resume capability.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from stem_separator.logging_config import (
    create_progress,
    get_logger,
    is_quiet,
    print_error,
    print_info,
    print_status,
    print_warning,
)
from stem_separator.utils import sanitize_filename, validate_youtube_url, SUBPROCESS_TIMEOUT


@dataclass
class YouTubeTrack:
    """Information about a YouTube track."""

    url: str
    title: str
    duration: Optional[float] = None
    index: int = 0
    total: int = 1


@dataclass
class DownloadResult:
    """Result of a YouTube download."""

    success: bool
    file_path: Optional[Path] = None
    title: str = ""
    error: Optional[str] = None


class YouTubeDownloader:
    """Handles YouTube downloads with progress and resume support."""

    def __init__(
        self,
        output_dir: Path,
        browser: Optional[str] = None,
        resume: bool = True,
    ):
        """
        Initialize YouTube downloader.

        Args:
            output_dir: Directory to save downloads.
            browser: Browser to use for cookies (for auth).
            resume: Whether to resume partial downloads.
        """
        self.output_dir = Path(output_dir)
        self.browser = browser
        self.resume = resume
        self.logger = get_logger()
        self._download_archive = self.output_dir / ".download_archive"

    def _get_base_cmd(self) -> list[str]:
        """Get base yt-dlp command with common options."""
        cmd = ["yt-dlp"]
        if self.browser:
            # Security note: browser cookies are used for authentication
            self.logger.debug(f"Using cookies from browser: {self.browser}")
            cmd.extend(["--cookies-from-browser", self.browser])
        return cmd

    def get_video_info(self, url: str) -> Optional[dict]:
        """
        Get video information without downloading.

        Args:
            url: YouTube URL.

        Returns:
            Video info dict or None on failure.
        """
        cmd = self._get_base_cmd() + [
            "-j",  # JSON output
            "--no-playlist",
            url,
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace"
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            self.logger.debug(f"Failed to get video info: {e}")
        return None

    def get_playlist_info(self, url: str) -> list[YouTubeTrack]:
        """
        Get information about all videos in a playlist.

        Args:
            url: YouTube playlist URL.

        Returns:
            List of YouTubeTrack objects.
        """
        # Security: Validate URL before use
        if not validate_youtube_url(url):
            self.logger.warning(f"Invalid YouTube playlist URL: {url}")
            return []

        cmd = self._get_base_cmd() + [
            "-j",
            "--flat-playlist",
            url,
        ]

        tracks = []
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, errors="replace",
                timeout=SUBPROCESS_TIMEOUT
            )
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split("\n")):
                    if line:
                        try:
                            info = json.loads(line)
                            tracks.append(
                                YouTubeTrack(
                                    url=f"https://youtube.com/watch?v={info.get('id', '')}",
                                    title=info.get("title", f"Track {i + 1}"),
                                    duration=info.get("duration"),
                                    index=i,
                                )
                            )
                        except json.JSONDecodeError:
                            continue

            # Update total count
            for track in tracks:
                track.total = len(tracks)

        except subprocess.SubprocessError as e:
            self.logger.error(f"Failed to get playlist info: {e}")

        return tracks

    def download_single(
        self,
        url: str,
        filename_prefix: str = "input",
    ) -> DownloadResult:
        """
        Download a single video as audio.

        Args:
            url: YouTube URL.
            filename_prefix: Prefix for output filename.

        Returns:
            DownloadResult with status and file path.
        """
        # Security: Validate URL before use
        if not validate_youtube_url(url):
            return DownloadResult(
                success=False,
                error=f"Invalid or unsafe YouTube URL: {url}",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"{filename_prefix}.wav"

        # Get title first
        title = "audio"
        title_cmd = self._get_base_cmd() + ["--get-title", "--no-playlist", url]
        try:
            title_result = subprocess.run(
                title_cmd, capture_output=True, text=True, errors="replace",
                timeout=60  # 1 minute timeout for title fetch
            )
        except subprocess.TimeoutExpired:
            title_result = type('obj', (object,), {'returncode': 1, 'stdout': ''})()
        if title_result.returncode == 0:
            title = title_result.stdout.strip()

        if not is_quiet():
            print_status(f"Downloading: {title[:60]}...")

        # Download with progress
        temp_template = str(self.output_dir / "temp_download.%(ext)s")
        cmd = self._get_base_cmd() + [
            "-x",  # Extract audio
            "--audio-quality",
            "0",  # Best quality
            "-o",
            temp_template,
            "--no-playlist",
            "--progress",
            "--newline",
        ]

        if self.resume:
            cmd.append("--continue")

        cmd.append(url)

        # Run download
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, errors="replace")

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            if "403" in error_msg:
                return DownloadResult(
                    success=False,
                    title=title,
                    error="YouTube blocked the download (403 Forbidden). Try --browser edge",
                )
            return DownloadResult(success=False, title=title, error=error_msg)

        # Find and convert downloaded file
        for file in self.output_dir.iterdir():
            if file.name.startswith("temp_download"):
                self.logger.debug(f"Converting {file} to WAV...")
                convert_result = subprocess.run(
                    ["ffmpeg", "-i", str(file), "-y", "-loglevel", "error", str(output_file)],
                    capture_output=True,
                    text=True,
                    errors="replace",
                )
                file.unlink()  # Remove temp file

                if convert_result.returncode == 0 and output_file.exists():
                    return DownloadResult(
                        success=True,
                        file_path=output_file,
                        title=sanitize_filename(title),
                    )
                else:
                    return DownloadResult(
                        success=False,
                        title=title,
                        error=f"Conversion failed: {convert_result.stderr}",
                    )

        return DownloadResult(success=False, title=title, error="Downloaded file not found")

    def download_playlist(
        self,
        url: str,
        skip_existing: bool = True,
    ) -> Iterator[DownloadResult]:
        """
        Download all videos from a playlist.

        Args:
            url: YouTube playlist URL.
            skip_existing: Skip videos already in download archive.

        Yields:
            DownloadResult for each video.
        """
        tracks = self.get_playlist_info(url)

        if not tracks:
            yield DownloadResult(
                success=False, error="No tracks found in playlist or failed to fetch playlist"
            )
            return

        print_status(f"Found {len(tracks)} tracks in playlist")

        # Load download archive for resuming
        downloaded = set()
        if skip_existing and self._download_archive.exists():
            downloaded = set(self._download_archive.read_text().strip().split("\n"))

        with create_progress("Downloading playlist") as progress:
            task = progress.add_task("Downloads", total=len(tracks))

            for track in tracks:
                # Check if already downloaded
                track_id = track.url.split("v=")[-1] if "v=" in track.url else track.url
                if track_id in downloaded:
                    self.logger.info(f"Skipping already downloaded: {track.title}")
                    progress.advance(task)
                    continue

                # Create unique filename for this track
                safe_title = sanitize_filename(track.title)[:50]
                prefix = f"{track.index + 1:03d}_{safe_title}"

                result = self.download_single(track.url, prefix)
                result.title = track.title  # Use original title

                if result.success:
                    # Record successful download
                    with open(self._download_archive, "a", encoding="utf-8") as f:
                        f.write(f"{track_id}\n")

                progress.advance(task)
                yield result


def download_youtube(
    url: str,
    output_dir: Path,
    browser: Optional[str] = None,
    is_playlist: bool = False,
) -> Iterator[DownloadResult] | DownloadResult:
    """
    Download audio from YouTube.

    Args:
        url: YouTube URL (video or playlist).
        output_dir: Directory to save downloads.
        browser: Browser for cookies.
        is_playlist: Whether to treat as playlist.

    Returns:
        DownloadResult for single video, or Iterator for playlist.
    """
    downloader = YouTubeDownloader(output_dir, browser)

    if is_playlist or "list=" in url:
        return downloader.download_playlist(url)
    else:
        return downloader.download_single(url)
