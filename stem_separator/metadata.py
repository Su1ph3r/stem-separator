"""
Metadata handling module for Stem Separator.

Handles reading and writing audio file metadata (ID3 tags, etc.).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from stem_separator.logging_config import get_logger


@dataclass
class AudioMetadata:
    """Audio file metadata."""

    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    year: Optional[str] = None
    genre: Optional[str] = None
    track_number: Optional[str] = None
    comment: Optional[str] = None
    album_artist: Optional[str] = None
    composer: Optional[str] = None
    cover_art: Optional[bytes] = None
    extra: dict[str, str] = field(default_factory=dict)

    def to_ffmpeg_args(self, stem_name: str = "") -> list[str]:
        """
        Convert metadata to FFmpeg metadata arguments.

        Args:
            stem_name: Stem name to append to title.

        Returns:
            List of FFmpeg arguments for metadata.
        """
        args = []

        title = self.title
        if title and stem_name:
            title = f"{title} ({stem_name})"

        if title:
            args.extend(["-metadata", f"title={title}"])
        if self.artist:
            args.extend(["-metadata", f"artist={self.artist}"])
        if self.album:
            args.extend(["-metadata", f"album={self.album}"])
        if self.year:
            args.extend(["-metadata", f"date={self.year}"])
        if self.genre:
            args.extend(["-metadata", f"genre={self.genre}"])
        if self.track_number:
            args.extend(["-metadata", f"track={self.track_number}"])
        if self.album_artist:
            args.extend(["-metadata", f"album_artist={self.album_artist}"])
        if self.composer:
            args.extend(["-metadata", f"composer={self.composer}"])

        # Add comment indicating this is a separated stem
        comment = f"Separated stem: {stem_name}" if stem_name else "Created by Stem Separator"
        if self.comment:
            comment = f"{self.comment} | {comment}"
        args.extend(["-metadata", f"comment={comment}"])

        # Add extra metadata
        for key, value in self.extra.items():
            args.extend(["-metadata", f"{key}={value}"])

        return args


def read_metadata(file_path: Path) -> Optional[AudioMetadata]:
    """
    Read metadata from an audio file.

    Args:
        file_path: Path to audio file.

    Returns:
        AudioMetadata object or None if failed.
    """
    logger = get_logger()

    try:
        # Try using mutagen first (more reliable)
        try:
            from mutagen import File as MutagenFile
            from mutagen.easyid3 import EasyID3
            from mutagen.id3 import ID3
            from mutagen.flac import FLAC
            from mutagen.oggvorbis import OggVorbis
            from mutagen.mp4 import MP4

            audio = MutagenFile(str(file_path), easy=True)

            if audio is None:
                raise ImportError("Mutagen could not read file")

            def get_tag(tag_names: list[str]) -> Optional[str]:
                for tag in tag_names:
                    if tag in audio:
                        val = audio[tag]
                        if isinstance(val, list) and val:
                            return str(val[0])
                        return str(val) if val else None
                return None

            metadata = AudioMetadata(
                title=get_tag(["title"]),
                artist=get_tag(["artist", "albumartist"]),
                album=get_tag(["album"]),
                year=get_tag(["date", "year"]),
                genre=get_tag(["genre"]),
                track_number=get_tag(["tracknumber"]),
                album_artist=get_tag(["albumartist"]),
                composer=get_tag(["composer"]),
            )

            # Try to get cover art
            try:
                raw_audio = MutagenFile(str(file_path))
                if hasattr(raw_audio, "pictures") and raw_audio.pictures:
                    metadata.cover_art = raw_audio.pictures[0].data
                elif hasattr(raw_audio, "tags"):
                    tags = raw_audio.tags
                    if tags:
                        # ID3 cover art
                        for key in tags.keys():
                            if key.startswith("APIC"):
                                metadata.cover_art = tags[key].data
                                break
            except Exception:
                pass

            return metadata

        except ImportError:
            logger.debug("Mutagen not available, falling back to ffprobe")

        # Fallback to ffprobe
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(file_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return None

        import json

        data = json.loads(result.stdout)
        tags = data.get("format", {}).get("tags", {})

        # ffprobe returns lowercase tag names
        return AudioMetadata(
            title=tags.get("title") or tags.get("TITLE"),
            artist=tags.get("artist") or tags.get("ARTIST"),
            album=tags.get("album") or tags.get("ALBUM"),
            year=tags.get("date") or tags.get("DATE") or tags.get("year"),
            genre=tags.get("genre") or tags.get("GENRE"),
            track_number=tags.get("track") or tags.get("TRACK"),
            album_artist=tags.get("album_artist") or tags.get("ALBUMARTIST"),
            composer=tags.get("composer") or tags.get("COMPOSER"),
            comment=tags.get("comment") or tags.get("COMMENT"),
        )

    except Exception as e:
        logger.debug(f"Failed to read metadata: {e}")
        return None


def write_metadata(
    file_path: Path,
    metadata: AudioMetadata,
    stem_name: str = "",
) -> bool:
    """
    Write metadata to an audio file.

    Args:
        file_path: Path to audio file.
        metadata: Metadata to write.
        stem_name: Stem name to append to title.

    Returns:
        True if successful.
    """
    logger = get_logger()

    try:
        # Try using mutagen first
        try:
            from mutagen import File as MutagenFile
            from mutagen.easyid3 import EasyID3
            from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC, TCON, TRCK, COMM

            audio = MutagenFile(str(file_path), easy=True)

            if audio is None:
                raise ImportError("Mutagen could not open file")

            title = metadata.title
            if title and stem_name:
                title = f"{title} ({stem_name})"

            if title:
                audio["title"] = title
            if metadata.artist:
                audio["artist"] = metadata.artist
            if metadata.album:
                audio["album"] = metadata.album
            if metadata.year:
                audio["date"] = metadata.year
            if metadata.genre:
                audio["genre"] = metadata.genre
            if metadata.track_number:
                audio["tracknumber"] = metadata.track_number
            if metadata.album_artist:
                audio["albumartist"] = metadata.album_artist
            if metadata.composer:
                audio["composer"] = metadata.composer

            audio.save()
            return True

        except ImportError:
            logger.debug("Mutagen not available, falling back to ffmpeg")

        # Fallback to ffmpeg (requires re-encoding or copying)
        temp_path = file_path.with_suffix(f".temp{file_path.suffix}")

        cmd = [
            "ffmpeg",
            "-i",
            str(file_path),
            "-y",
            "-loglevel",
            "error",
            "-c",
            "copy",  # Copy without re-encoding
        ]
        cmd.extend(metadata.to_ffmpeg_args(stem_name))
        cmd.append(str(temp_path))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            temp_path.replace(file_path)
            return True
        else:
            temp_path.unlink(missing_ok=True)
            logger.warning(f"Failed to write metadata: {result.stderr}")
            return False

    except Exception as e:
        logger.debug(f"Failed to write metadata: {e}")
        return False


def copy_metadata(
    source_file: Path,
    dest_file: Path,
    stem_name: str = "",
) -> bool:
    """
    Copy metadata from source to destination file.

    Args:
        source_file: Source file to read metadata from.
        dest_file: Destination file to write metadata to.
        stem_name: Stem name to append to title.

    Returns:
        True if successful.
    """
    metadata = read_metadata(source_file)
    if metadata:
        return write_metadata(dest_file, metadata, stem_name)
    return False


def extract_cover_art(file_path: Path, output_path: Path) -> bool:
    """
    Extract cover art from an audio file.

    Args:
        file_path: Path to audio file.
        output_path: Path to save cover art image.

    Returns:
        True if successful.
    """
    logger = get_logger()

    try:
        # Try using ffmpeg
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(file_path),
                "-an",  # No audio
                "-vcodec",
                "copy",
                "-y",
                "-loglevel",
                "error",
                str(output_path),
            ],
            capture_output=True,
        )

        if result.returncode == 0 and output_path.exists():
            return True

        # Try using mutagen
        metadata = read_metadata(file_path)
        if metadata and metadata.cover_art:
            output_path.write_bytes(metadata.cover_art)
            return True

    except Exception as e:
        logger.debug(f"Failed to extract cover art: {e}")

    return False


def embed_cover_art(audio_path: Path, image_path: Path) -> bool:
    """
    Embed cover art into an audio file.

    Args:
        audio_path: Path to audio file.
        image_path: Path to cover art image.

    Returns:
        True if successful.
    """
    logger = get_logger()

    if not image_path.exists():
        return False

    try:
        temp_path = audio_path.with_suffix(f".temp{audio_path.suffix}")

        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(audio_path),
                "-i",
                str(image_path),
                "-map",
                "0:a",
                "-map",
                "1:v",
                "-c",
                "copy",
                "-disposition:v:0",
                "attached_pic",
                "-y",
                "-loglevel",
                "error",
                str(temp_path),
            ],
            capture_output=True,
        )

        if result.returncode == 0:
            temp_path.replace(audio_path)
            return True
        else:
            temp_path.unlink(missing_ok=True)

    except Exception as e:
        logger.debug(f"Failed to embed cover art: {e}")

    return False
