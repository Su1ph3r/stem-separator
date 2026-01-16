"""Tests for mixing module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from stem_separator.mixing import (
    MixComponent,
    MixResult,
    parse_remix_string,
    remix_stems,
    create_karaoke,
    create_acapella,
)


class TestMixComponent:
    """Tests for MixComponent dataclass."""

    def test_default_volume(self):
        """Test default volume is 1.0."""
        component = MixComponent(stem_name="vocals")
        assert component.volume == 1.0

    def test_custom_volume(self):
        """Test custom volume setting."""
        component = MixComponent(stem_name="vocals", volume=0.5)
        assert component.volume == 0.5


class TestParseRemixString:
    """Tests for remix string parsing."""

    def test_simple_stem_list(self):
        """Test parsing simple stem list."""
        result = parse_remix_string("vocals,drums,bass")
        assert len(result) == 3
        assert all(c.volume == 1.0 for c in result)
        assert result[0].stem_name == "vocals"
        assert result[1].stem_name == "drums"
        assert result[2].stem_name == "bass"

    def test_stems_with_volumes(self):
        """Test parsing stems with volumes."""
        result = parse_remix_string("vocals:0.5,drums:1.0,bass:0.8")
        assert len(result) == 3
        assert result[0].stem_name == "vocals"
        assert result[0].volume == 0.5
        assert result[1].stem_name == "drums"
        assert result[1].volume == 1.0
        assert result[2].stem_name == "bass"
        assert result[2].volume == 0.8

    def test_mixed_format(self):
        """Test parsing mixed format (some with volume, some without)."""
        result = parse_remix_string("vocals:0.5,drums,bass:1.5")
        assert result[0].volume == 0.5
        assert result[1].volume == 1.0  # Default
        assert result[2].volume == 1.5

    def test_case_insensitive(self):
        """Test case insensitivity."""
        result = parse_remix_string("VOCALS:0.5,Drums:1.0")
        assert result[0].stem_name == "vocals"
        assert result[1].stem_name == "drums"

    def test_handles_whitespace(self):
        """Test handling of whitespace."""
        result = parse_remix_string("  vocals : 0.5 , drums : 1.0  ")
        assert len(result) == 2
        assert result[0].stem_name == "vocals"

    def test_invalid_volume_raises_error(self):
        """Test invalid volume raises ValueError."""
        with pytest.raises(ValueError):
            parse_remix_string("vocals:invalid")

    def test_volume_out_of_range_raises_error(self):
        """Test volume out of range raises ValueError."""
        with pytest.raises(ValueError):
            parse_remix_string("vocals:3.0")  # > 2.0

        with pytest.raises(ValueError):
            parse_remix_string("vocals:-0.5")  # < 0.0

    def test_empty_string_raises_error(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError):
            parse_remix_string("")

    def test_only_commas_raises_error(self):
        """Test string with only commas raises ValueError."""
        with pytest.raises(ValueError):
            parse_remix_string(",,,")


class TestMixResult:
    """Tests for MixResult dataclass."""

    def test_success_result(self):
        """Test successful mix result."""
        result = MixResult(
            success=True,
            output_path=Path("/output/mix.wav"),
            stems_used=["vocals:0.5", "drums:1.0"],
        )
        assert result.success
        assert result.output_path is not None
        assert len(result.stems_used) == 2

    def test_failure_result(self):
        """Test failed mix result."""
        result = MixResult(
            success=False,
            error="Stems not found",
        )
        assert not result.success
        assert result.error == "Stems not found"

    def test_default_stems_used(self):
        """Test default stems_used is empty list."""
        result = MixResult(success=True)
        assert result.stems_used == []


class TestRemixStemsIntegration:
    """Integration tests for remix_stems function."""

    @pytest.fixture
    def stems_dir(self, tmp_path):
        """Create a temporary directory with mock stem files."""
        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not available")

        # Create mock stem files (1 second of audio)
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)

        # Generate test audio (stereo sine waves)
        t = np.linspace(0, duration, samples)

        # Vocals: 440 Hz
        vocals = np.stack([
            np.sin(2 * np.pi * 440 * t) * 0.5,
            np.sin(2 * np.pi * 440 * t) * 0.5,
        ], axis=1)
        sf.write(str(tmp_path / "test_vocals.wav"), vocals, sample_rate)

        # Drums: 100 Hz
        drums = np.stack([
            np.sin(2 * np.pi * 100 * t) * 0.5,
            np.sin(2 * np.pi * 100 * t) * 0.5,
        ], axis=1)
        sf.write(str(tmp_path / "test_drums.wav"), drums, sample_rate)

        # Bass: 80 Hz
        bass = np.stack([
            np.sin(2 * np.pi * 80 * t) * 0.5,
            np.sin(2 * np.pi * 80 * t) * 0.5,
        ], axis=1)
        sf.write(str(tmp_path / "test_bass.wav"), bass, sample_rate)

        return tmp_path

    def test_remix_all_stems(self, stems_dir, tmp_path):
        """Test remixing all stems."""
        output_path = tmp_path / "output" / "mix.wav"

        result = remix_stems(
            stems_dir=stems_dir,
            output_path=output_path,
            mix_components="vocals:1.0,drums:1.0,bass:1.0",
        )

        assert result.success
        assert result.output_path.exists()
        assert len(result.stems_used) == 3

    def test_remix_with_different_volumes(self, stems_dir, tmp_path):
        """Test remixing with different volumes."""
        output_path = tmp_path / "output" / "mix.wav"

        result = remix_stems(
            stems_dir=stems_dir,
            output_path=output_path,
            mix_components="vocals:0.5,drums:1.5",
        )

        assert result.success
        assert "vocals:0.5" in result.stems_used
        assert "drums:1.5" in result.stems_used

    def test_remix_missing_stem_fails(self, stems_dir, tmp_path):
        """Test remixing with missing stem fails."""
        output_path = tmp_path / "output" / "mix.wav"

        result = remix_stems(
            stems_dir=stems_dir,
            output_path=output_path,
            mix_components="vocals:1.0,nonexistent:1.0",
        )

        assert not result.success
        assert "not found" in result.error.lower()

    def test_remix_with_normalization(self, stems_dir, tmp_path):
        """Test remixing with normalization."""
        output_path = tmp_path / "output" / "mix.wav"

        result = remix_stems(
            stems_dir=stems_dir,
            output_path=output_path,
            mix_components="vocals:2.0,drums:2.0",  # High volumes
            normalize_output=True,
        )

        assert result.success

        # Check output doesn't clip
        try:
            import soundfile as sf
            audio, _ = sf.read(str(result.output_path))
            assert np.max(np.abs(audio)) <= 1.0
        except ImportError:
            pass
