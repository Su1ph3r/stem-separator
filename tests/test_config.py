"""Tests for configuration module."""

import pytest
from pathlib import Path
import tempfile

from stem_separator.config import (
    MODELS,
    FORMATS,
    PRESETS,
    UserConfig,
    load_user_config,
    save_user_config,
    generate_sample_config,
    ModelConfig,
    FormatConfig,
    PresetConfig,
)


class TestModelConfigs:
    """Tests for model configurations."""

    def test_htdemucs_exists(self):
        """Test htdemucs model is defined."""
        assert "htdemucs" in MODELS
        assert isinstance(MODELS["htdemucs"], ModelConfig)

    def test_htdemucs_has_four_stems(self):
        """Test htdemucs has 4 stems."""
        assert len(MODELS["htdemucs"].sources) == 4
        assert "vocals" in MODELS["htdemucs"].sources
        assert "drums" in MODELS["htdemucs"].sources
        assert "bass" in MODELS["htdemucs"].sources
        assert "other" in MODELS["htdemucs"].sources

    def test_htdemucs_6s_has_six_stems(self):
        """Test htdemucs_6s has 6 stems."""
        assert "htdemucs_6s" in MODELS
        assert len(MODELS["htdemucs_6s"].sources) == 6
        assert "guitar" in MODELS["htdemucs_6s"].sources
        assert "piano" in MODELS["htdemucs_6s"].sources

    def test_all_models_have_descriptions(self):
        """Test all models have descriptions."""
        for name, config in MODELS.items():
            assert config.description, f"Model {name} missing description"


class TestFormatConfigs:
    """Tests for format configurations."""

    def test_wav_format_exists(self):
        """Test WAV format is defined."""
        assert "wav" in FORMATS
        assert FORMATS["wav"].ext == ".wav"
        assert not FORMATS["wav"].requires_conversion

    def test_mp3_format_exists(self):
        """Test MP3 format is defined."""
        assert "mp3" in FORMATS
        assert FORMATS["mp3"].ext == ".mp3"
        assert FORMATS["mp3"].codec == "libmp3lame"
        assert FORMATS["mp3"].bitrate == "320k"

    def test_flac_format_exists(self):
        """Test FLAC format is defined."""
        assert "flac" in FORMATS
        assert FORMATS["flac"].ext == ".flac"
        assert FORMATS["flac"].codec == "flac"

    def test_all_formats_have_extensions(self):
        """Test all formats have extensions."""
        for name, config in FORMATS.items():
            assert config.ext.startswith("."), f"Format {name} extension doesn't start with '.'"


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_all_preset_exists(self):
        """Test 'all' preset is defined."""
        assert "all" in PRESETS
        assert PRESETS["all"].include_all

    def test_karaoke_preset_excludes_vocals(self):
        """Test 'karaoke' preset excludes vocals."""
        assert "karaoke" in PRESETS
        assert "vocals" in PRESETS["karaoke"].exclude

    def test_acapella_preset_includes_only_vocals(self):
        """Test 'acapella' preset includes only vocals."""
        assert "acapella" in PRESETS
        assert PRESETS["acapella"].include == ["vocals"]

    def test_instrumental_preset_same_as_karaoke(self):
        """Test 'instrumental' is same as 'karaoke'."""
        assert "instrumental" in PRESETS
        assert PRESETS["instrumental"].exclude == PRESETS["karaoke"].exclude


class TestUserConfig:
    """Tests for user configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = UserConfig()
        assert config.model == "htdemucs"
        assert config.format == "wav"
        assert config.output_dir == "."
        assert config.stems is None
        assert config.cpu is False
        assert config.verbose is False
        assert config.quiet is False
        assert config.normalize is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = UserConfig(
            model="htdemucs_6s",
            format="mp3",
            normalize=True,
            parallel_jobs=4,
        )
        assert config.model == "htdemucs_6s"
        assert config.format == "mp3"
        assert config.normalize is True
        assert config.parallel_jobs == 4


class TestConfigFilePersistence:
    """Tests for config file loading and saving."""

    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "test_config.yaml"

            # Save config
            original = UserConfig(
                model="htdemucs_ft",
                format="flac",
                normalize=True,
                naming_template="{stem}_{name}",
            )
            saved_path = save_user_config(original, config_path)
            assert saved_path == config_path
            assert config_path.exists()

            # Load config
            loaded = load_user_config(config_path)
            assert loaded.model == original.model
            assert loaded.format == original.format
            assert loaded.normalize == original.normalize
            assert loaded.naming_template == original.naming_template

    def test_load_nonexistent_returns_defaults(self):
        """Test loading nonexistent file returns defaults."""
        config = load_user_config(Path("/nonexistent/path/config.yaml"))
        assert config.model == "htdemucs"
        assert config.format == "wav"

    def test_load_invalid_yaml_returns_defaults(self):
        """Test loading invalid YAML returns defaults."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid.yaml"
            config_path.write_text("{{invalid yaml")

            config = load_user_config(config_path)
            assert config.model == "htdemucs"


class TestSampleConfig:
    """Tests for sample config generation."""

    def test_generate_sample_config(self):
        """Test sample config generation."""
        sample = generate_sample_config()
        assert "model:" in sample
        assert "format:" in sample
        assert "htdemucs" in sample
        assert "wav" in sample

    def test_sample_config_is_valid_yaml(self):
        """Test sample config is valid YAML."""
        import yaml

        sample = generate_sample_config()
        parsed = yaml.safe_load(sample)
        assert isinstance(parsed, dict)
        assert "model" in parsed
