"""Tests for utility functions."""

import pytest
from pathlib import Path

from stem_separator.utils import (
    sanitize_filename,
    is_youtube_url,
    is_youtube_playlist,
    is_spotify_url,
    is_spotify_playlist,
    validate_model,
    validate_format,
    parse_stem_selection,
    format_output_name,
    collect_audio_files,
)
from stem_separator.config import MODELS


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_filename(self):
        """Test basic filename passes through."""
        assert sanitize_filename("my song") == "my song"

    def test_removes_special_characters(self):
        """Test removal of special characters."""
        assert sanitize_filename("song:with*special?chars") == "song-withspecialchars"

    def test_handles_unicode(self):
        """Test handling of unicode characters."""
        result = sanitize_filename("日本語タイトル")
        assert result == "audio"  # Non-ASCII removed, falls back to 'audio'

    def test_handles_full_width_characters(self):
        """Test handling of full-width characters."""
        result = sanitize_filename("Song／Title")
        assert "/" not in result and "\\" not in result

    def test_normalizes_whitespace(self):
        """Test whitespace normalization."""
        assert sanitize_filename("  too   many   spaces  ") == "too many spaces"

    def test_handles_curly_quotes(self):
        """Test handling of curly quotes."""
        result = sanitize_filename("Song "Title"")
        assert """ not in result and """ not in result

    def test_empty_string_returns_audio(self):
        """Test empty string returns 'audio'."""
        assert sanitize_filename("") == "audio"

    def test_only_special_chars_returns_audio(self):
        """Test string with only special chars returns 'audio'."""
        assert sanitize_filename("***???") == "audio"


class TestUrlDetection:
    """Tests for URL detection functions."""

    def test_youtube_url_detection(self):
        """Test YouTube URL detection."""
        assert is_youtube_url("https://www.youtube.com/watch?v=abc123")
        assert is_youtube_url("https://youtu.be/abc123")
        assert is_youtube_url("http://youtube.com/watch?v=abc123")
        assert not is_youtube_url("https://vimeo.com/123")
        assert not is_youtube_url("song.mp3")

    def test_youtube_playlist_detection(self):
        """Test YouTube playlist detection."""
        assert is_youtube_playlist("https://youtube.com/playlist?list=PLabc123")
        assert is_youtube_playlist("https://youtube.com/watch?v=abc&list=PLabc123")
        assert not is_youtube_playlist("https://youtube.com/watch?v=abc123")

    def test_spotify_url_detection(self):
        """Test Spotify URL detection."""
        assert is_spotify_url("https://open.spotify.com/track/abc123")
        assert is_spotify_url("spotify:track:abc123")
        assert not is_spotify_url("https://youtube.com/watch?v=abc")
        assert not is_spotify_url("song.mp3")

    def test_spotify_playlist_detection(self):
        """Test Spotify playlist detection."""
        assert is_spotify_playlist("https://open.spotify.com/playlist/abc123")
        assert is_spotify_playlist("https://open.spotify.com/album/abc123")
        assert not is_spotify_playlist("https://open.spotify.com/track/abc123")


class TestValidation:
    """Tests for validation functions."""

    def test_validate_model_valid(self):
        """Test valid model names."""
        assert validate_model("htdemucs") == "htdemucs"
        assert validate_model("HTDEMUCS") == "htdemucs"
        assert validate_model("htdemucs_ft") == "htdemucs_ft"
        assert validate_model("htdemucs_6s") == "htdemucs_6s"

    def test_validate_model_invalid(self):
        """Test invalid model names exit."""
        with pytest.raises(SystemExit):
            validate_model("invalid_model")

    def test_validate_format_valid(self):
        """Test valid format names."""
        assert validate_format("wav") == "wav"
        assert validate_format("MP3") == "mp3"
        assert validate_format("flac") == "flac"
        assert validate_format("ogg") == "ogg"
        assert validate_format("aac") == "aac"

    def test_validate_format_invalid(self):
        """Test invalid format names exit."""
        with pytest.raises(SystemExit):
            validate_format("invalid_format")


class TestStemSelection:
    """Tests for stem selection parsing."""

    def test_none_returns_all_stems(self):
        """Test None returns all model stems."""
        sources = MODELS["htdemucs"].sources
        result = parse_stem_selection(None, sources)
        assert result == sources

    def test_preset_all(self):
        """Test 'all' preset."""
        sources = MODELS["htdemucs"].sources
        result = parse_stem_selection("all", sources)
        assert result == sources

    def test_preset_karaoke(self):
        """Test 'karaoke' preset excludes vocals."""
        sources = MODELS["htdemucs"].sources
        result = parse_stem_selection("karaoke", sources)
        assert "vocals" not in result
        assert "drums" in result
        assert "bass" in result

    def test_preset_acapella(self):
        """Test 'acapella' preset includes only vocals."""
        sources = MODELS["htdemucs"].sources
        result = parse_stem_selection("acapella", sources)
        assert result == ["vocals"]

    def test_comma_separated(self):
        """Test comma-separated stem list."""
        sources = MODELS["htdemucs"].sources
        result = parse_stem_selection("vocals,drums", sources)
        assert result == ["vocals", "drums"]

    def test_invalid_stem_exits(self):
        """Test invalid stem name exits."""
        sources = MODELS["htdemucs"].sources
        with pytest.raises(SystemExit):
            parse_stem_selection("invalid_stem", sources)

    def test_case_insensitive(self):
        """Test case insensitivity."""
        sources = MODELS["htdemucs"].sources
        result = parse_stem_selection("VOCALS,DRUMS", sources)
        assert result == ["vocals", "drums"]


class TestNamingTemplate:
    """Tests for output naming templates."""

    def test_basic_template(self):
        """Test basic naming template."""
        result = format_output_name("{name}_{stem}", "My Song", "vocals")
        assert result == "My Song_vocals"

    def test_template_with_model(self):
        """Test template with model name."""
        result = format_output_name(
            "{name}_{stem}_{model}",
            "My Song",
            "vocals",
            model="htdemucs",
        )
        assert result == "My Song_vocals_htdemucs"

    def test_template_sanitizes_name(self):
        """Test that name is sanitized."""
        result = format_output_name("{name}_{stem}", "My:Song*With?Chars", "vocals")
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result


class TestCollectAudioFiles:
    """Tests for audio file collection."""

    def test_single_file(self, tmp_path):
        """Test collecting a single file."""
        audio_file = tmp_path / "song.mp3"
        audio_file.touch()
        result = collect_audio_files(audio_file)
        assert result == [audio_file]

    def test_directory(self, tmp_path):
        """Test collecting from directory."""
        (tmp_path / "song1.mp3").touch()
        (tmp_path / "song2.wav").touch()
        (tmp_path / "not_audio.txt").touch()
        result = collect_audio_files(tmp_path)
        assert len(result) == 2
        assert all(f.suffix in {".mp3", ".wav"} for f in result)

    def test_recursive(self, tmp_path):
        """Test recursive collection."""
        (tmp_path / "song1.mp3").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "song2.mp3").touch()

        result = collect_audio_files(tmp_path, recursive=False)
        assert len(result) == 1

        result = collect_audio_files(tmp_path, recursive=True)
        assert len(result) == 2

    def test_non_audio_file(self, tmp_path):
        """Test non-audio file returns empty list."""
        text_file = tmp_path / "readme.txt"
        text_file.touch()
        result = collect_audio_files(text_file)
        assert result == []
