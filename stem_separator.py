#!/usr/bin/env python3
"""
Stem Separator - Audio Stem Extraction Tool
Uses Demucs (by Facebook/Meta Research) for high-quality stem separation
Supports YouTube URLs, playlists, and local audio files

Features:
- Batch processing (multiple files/URLs)
- YouTube playlist support
- Configuration file support
- BPM & key detection
- Audio normalization
- DAW project export (Audacity)
- API server mode
"""

import os
import sys
import argparse
import subprocess
import shutil
import tempfile
import re
import json
import glob as glob_module
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Supported Demucs models
MODELS = {
    'htdemucs': {
        'sources': ['drums', 'bass', 'other', 'vocals'],
        'description': 'Hybrid Transformer Demucs (default)'
    },
    'htdemucs_ft': {
        'sources': ['drums', 'bass', 'other', 'vocals'],
        'description': 'Fine-tuned version (better quality)'
    },
    'htdemucs_6s': {
        'sources': ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
        'description': '6-stem model with guitar and piano'
    }
}

# Output format configurations
FORMATS = {
    'wav': {
        'ext': '.wav',
        'description': 'Lossless WAV (default)',
        'requires_conversion': False
    },
    'mp3': {
        'ext': '.mp3',
        'codec': 'libmp3lame',
        'bitrate': '320k',
        'description': 'MP3 320kbps'
    },
    'flac': {
        'ext': '.flac',
        'codec': 'flac',
        'description': 'Lossless FLAC'
    },
    'ogg': {
        'ext': '.ogg',
        'codec': 'libvorbis',
        'quality': '10',
        'description': 'Ogg Vorbis (high quality)'
    },
    'aac': {
        'ext': '.aac',
        'codec': 'aac',
        'bitrate': '256k',
        'description': 'AAC 256kbps'
    }
}

# Stem selection presets
PRESETS = {
    'all': {
        'description': 'All stems (default)',
        'include_all': True
    },
    'karaoke': {
        'description': 'Everything except vocals',
        'exclude': ['vocals']
    },
    'acapella': {
        'description': 'Vocals only',
        'include': ['vocals']
    },
    'instrumental': {
        'description': 'Everything except vocals',
        'exclude': ['vocals']
    }
}

# Default configuration file locations
CONFIG_LOCATIONS = [
    '.stem_separator.yaml',
    '.stem_separator.yml',
    os.path.expanduser('~/.stem_separator.yaml'),
    os.path.expanduser('~/.stem_separator.yml'),
    os.path.expanduser('~/.config/stem_separator/config.yaml'),
]

# =============================================================================
# CONFIGURATION FILE SUPPORT
# =============================================================================

@dataclass
class Config:
    """Configuration settings with defaults"""
    model: str = 'htdemucs'
    format: str = 'wav'
    output: str = '.'
    stems: Optional[str] = None
    cpu: bool = False
    browser: Optional[str] = None
    normalize: bool = False
    normalize_level: float = -14.0
    analyze: bool = False
    export_daw: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(path, 'r') as f:
                data = yaml.safe_load(f) or {}
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except ImportError:
            print("Warning: pyyaml not installed. Config file support disabled.")
            return cls()
        except Exception as e:
            print(f"Warning: Could not load config from {path}: {e}")
            return cls()

    @classmethod
    def find_and_load(cls) -> 'Config':
        """Find and load configuration from standard locations"""
        for path in CONFIG_LOCATIONS:
            if os.path.exists(path):
                print(f"Loading config from: {path}")
                return cls.from_yaml(path)
        return cls()

    def save(self, path: str):
        """Save configuration to YAML file"""
        try:
            import yaml
            with open(path, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
            print(f"Config saved to: {path}")
        except ImportError:
            print("Error: pyyaml not installed. Cannot save config.")


# =============================================================================
# PROGRESS BAR UTILITIES
# =============================================================================

class ProgressBar:
    """Wrapper for tqdm progress bar with fallback"""

    def __init__(self, total: int, desc: str = "", unit: str = "it"):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.pbar = None

        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=total, desc=desc, unit=unit, ncols=80)
        except ImportError:
            pass

    def update(self, n: int = 1):
        if self.pbar:
            self.pbar.update(n)
        else:
            self.current += n
            pct = (self.current / self.total) * 100 if self.total > 0 else 0
            print(f"\r{self.desc}: {pct:.1f}% ({self.current}/{self.total})", end="", flush=True)

    def set_description(self, desc: str):
        self.desc = desc
        if self.pbar:
            self.pbar.set_description(desc)

    def close(self):
        if self.pbar:
            self.pbar.close()
        else:
            print()  # New line after progress


def print_stage(stage: int, total: int, message: str):
    """Print a processing stage message"""
    print(f"\n[{stage}/{total}] {message}")


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model(model_name):
    """Validate model selection and return normalized name"""
    model_lower = model_name.lower()
    if model_lower not in MODELS:
        print(f"Error: Invalid model '{model_name}'")
        print(f"Valid models: {', '.join(MODELS.keys())}")
        sys.exit(1)
    return model_lower


def validate_format(format_name):
    """Validate output format and return normalized name"""
    format_lower = format_name.lower()
    if format_lower not in FORMATS:
        print(f"Error: Invalid format '{format_name}'")
        print(f"Valid formats: {', '.join(FORMATS.keys())}")
        sys.exit(1)
    return format_lower


def parse_stem_selection(stems_arg, model_sources):
    """
    Parse --stems argument into list of stem names to export.
    """
    if stems_arg is None:
        return list(model_sources)

    stems_lower = stems_arg.lower().strip()

    if stems_lower in PRESETS:
        preset = PRESETS[stems_lower]
        if preset.get('include_all'):
            return list(model_sources)
        elif 'include' in preset:
            result = [s for s in model_sources if s in preset['include']]
        elif 'exclude' in preset:
            result = [s for s in model_sources if s not in preset['exclude']]
        else:
            result = list(model_sources)

        if not result:
            print(f"Error: Preset '{stems_lower}' results in no stems for this model")
            sys.exit(1)
        return result

    requested_stems = [s.strip().lower() for s in stems_arg.split(',') if s.strip()]
    if not requested_stems:
        print("Error: No valid stems specified")
        sys.exit(1)

    invalid_stems = [s for s in requested_stems if s not in model_sources]
    if invalid_stems:
        print(f"Error: Invalid stem(s) for selected model: {', '.join(invalid_stems)}")
        print(f"Available stems: {', '.join(model_sources)}")
        sys.exit(1)

    return requested_stems


# =============================================================================
# FORMAT CONVERSION
# =============================================================================

def convert_audio_format(input_wav, output_path, format_name):
    """Convert WAV file to specified format using FFmpeg."""
    format_config = FORMATS[format_name]
    output_file = output_path + format_config['ext']

    if format_name == 'wav':
        shutil.copy(input_wav, output_file)
        return output_file

    cmd = ['ffmpeg', '-i', input_wav, '-y', '-loglevel', 'error']
    cmd.extend(['-acodec', format_config['codec']])

    if 'bitrate' in format_config:
        cmd.extend(['-b:a', format_config['bitrate']])
    elif 'quality' in format_config:
        cmd.extend(['-q:a', format_config['quality']])

    cmd.append(output_file)

    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')

    if result.returncode != 0:
        print(f"  Warning: FFmpeg conversion failed for {os.path.basename(output_file)}")
        return None

    return output_file


# =============================================================================
# AUDIO ANALYSIS (BPM & KEY DETECTION)
# =============================================================================

def analyze_audio(audio_file: str) -> Dict[str, Any]:
    """
    Analyze audio file for BPM and musical key.

    Returns dict with 'bpm', 'key', 'duration', and 'sample_rate'
    """
    results = {
        'bpm': None,
        'key': None,
        'duration': None,
        'sample_rate': None
    }

    try:
        import librosa
        import numpy as np

        print("  Analyzing audio...")

        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        results['sample_rate'] = sr
        results['duration'] = len(y) / sr

        # BPM detection
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Handle both old and new librosa versions
        if hasattr(tempo, '__len__'):
            results['bpm'] = round(float(tempo[0]), 1)
        else:
            results['bpm'] = round(float(tempo), 1)

        # Key detection using chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)

        # Map to key names
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        key_idx = np.argmax(chroma_avg)

        # Simple major/minor detection
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Rotate profiles and correlate
        major_corr = np.correlate(chroma_avg, np.roll(major_profile, key_idx))[0]
        minor_corr = np.correlate(chroma_avg, np.roll(minor_profile, key_idx))[0]

        mode = 'major' if major_corr > minor_corr else 'minor'
        results['key'] = f"{key_names[key_idx]} {mode}"

    except ImportError:
        print("  Warning: librosa not installed. Audio analysis disabled.")
        print("  Install with: pip install librosa")
    except Exception as e:
        print(f"  Warning: Audio analysis failed: {e}")

    return results


def print_analysis(analysis: Dict[str, Any], name: str = ""):
    """Print audio analysis results"""
    if not any(analysis.values()):
        return

    title = f"[Analysis] {name}" if name else "[Analysis]"
    print(f"\n{title}")

    if analysis['duration']:
        mins = int(analysis['duration'] // 60)
        secs = int(analysis['duration'] % 60)
        print(f"  Duration: {mins}:{secs:02d}")

    if analysis['bpm']:
        print(f"  BPM: {analysis['bpm']}")

    if analysis['key']:
        print(f"  Key: {analysis['key']}")


# =============================================================================
# AUDIO NORMALIZATION
# =============================================================================

def normalize_audio(audio_file: str, target_lufs: float = -14.0) -> bool:
    """
    Normalize audio to target loudness (LUFS).

    Args:
        audio_file: Path to audio file (will be modified in-place)
        target_lufs: Target loudness in LUFS (default: -14.0)

    Returns:
        True if successful, False otherwise
    """
    try:
        import soundfile as sf
        import pyloudnorm as pyln
        import numpy as np

        # Read audio
        data, rate = sf.read(audio_file)

        # Measure loudness
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)

        if np.isinf(loudness):
            print(f"    Warning: Could not measure loudness for {os.path.basename(audio_file)}")
            return False

        # Normalize
        normalized = pyln.normalize.loudness(data, loudness, target_lufs)

        # Prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 1.0:
            normalized = normalized / peak * 0.99

        # Write back
        sf.write(audio_file, normalized, rate)
        return True

    except ImportError:
        print("  Warning: pyloudnorm not installed. Normalization disabled.")
        print("  Install with: pip install pyloudnorm")
        return False
    except Exception as e:
        print(f"  Warning: Normalization failed: {e}")
        return False


# =============================================================================
# DAW PROJECT EXPORT
# =============================================================================

def create_audacity_project(stems_dir: str, output_path: str, sample_rate: int = 44100) -> Optional[str]:
    """
    Create an Audacity project file (.aup3) for the separated stems.

    Note: Creates an Audacity label file that can be imported, since .aup3
    is a SQLite database format that's complex to generate.
    Also creates a project structure file for reference.
    """
    try:
        project_name = os.path.basename(output_path).replace('.aup3', '')
        project_dir = os.path.dirname(output_path) or '.'

        # Get all audio files in stems directory
        audio_files = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac']:
            audio_files.extend(glob_module.glob(os.path.join(stems_dir, f'*{ext}')))

        if not audio_files:
            print("  Warning: No audio files found for DAW export")
            return None

        # Create project info JSON (can be used by scripts to import into DAW)
        project_info = {
            'project_name': project_name,
            'created': datetime.now().isoformat(),
            'sample_rate': sample_rate,
            'tracks': []
        }

        for i, audio_file in enumerate(sorted(audio_files)):
            stem_name = os.path.splitext(os.path.basename(audio_file))[0]
            project_info['tracks'].append({
                'name': stem_name,
                'file': os.path.abspath(audio_file),
                'position': 0,
                'color': get_stem_color(stem_name)
            })

        # Write project info
        info_path = os.path.join(project_dir, f"{project_name}_project.json")
        with open(info_path, 'w') as f:
            json.dump(project_info, f, indent=2)

        # Create Audacity macro/import script
        lof_path = os.path.join(project_dir, f"{project_name}.lof")
        with open(lof_path, 'w') as f:
            f.write("# Audacity List of Files (LOF)\n")
            f.write("# Open this file in Audacity to import all stems as separate tracks\n")
            for audio_file in sorted(audio_files):
                f.write(f'file "{os.path.abspath(audio_file)}" offset 0\n')

        print(f"    Created: {os.path.basename(info_path)}")
        print(f"    Created: {os.path.basename(lof_path)}")
        print(f"    Tip: Open the .lof file in Audacity to import all stems")

        return info_path

    except Exception as e:
        print(f"  Warning: DAW export failed: {e}")
        return None


def get_stem_color(stem_name: str) -> str:
    """Get a color for a stem track (for DAW visualization)"""
    colors = {
        'vocals': '#FF6B6B',
        'drums': '#4ECDC4',
        'bass': '#45B7D1',
        'other': '#96CEB4',
        'guitar': '#FFEAA7',
        'piano': '#DDA0DD'
    }
    return colors.get(stem_name.lower(), '#808080')


# =============================================================================
# UTILITIES
# =============================================================================

def sanitize_filename(name):
    """Remove problematic characters from filename"""
    replacements = {
        '＂': '"', '＇': "'", '／': '-', '＼': '-',
        '：': '-', '＊': '', '？': '', '＜': '', '＞': '', '｜': '-',
        '"': '', '"': '', ''': "'", ''': "'",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r'[^\x00-\x7F]+', '', name)
    name = re.sub(r'[-\s]+', ' ', name).strip()
    name = re.sub(r'\s*-\s*', ' - ', name)
    return name if name else 'audio'


def is_youtube_url(input_str):
    return 'youtube.com' in input_str.lower() or 'youtu.be' in input_str.lower()


def is_youtube_playlist(input_str):
    """Check if URL is a YouTube playlist"""
    return is_youtube_url(input_str) and ('list=' in input_str or '/playlist' in input_str)


def get_playlist_videos(url: str, browser: Optional[str] = None) -> List[Dict[str, str]]:
    """Get list of videos from a YouTube playlist"""
    cmd = ['yt-dlp', '--flat-playlist', '-J', url]
    if browser:
        cmd.extend(['--cookies-from-browser', browser])

    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')

    if result.returncode != 0:
        print(f"Error getting playlist info: {result.stderr}")
        return []

    try:
        data = json.loads(result.stdout)
        entries = data.get('entries', [])
        videos = []
        for entry in entries:
            if entry:
                videos.append({
                    'id': entry.get('id', ''),
                    'title': entry.get('title', 'Unknown'),
                    'url': f"https://youtube.com/watch?v={entry.get('id', '')}"
                })
        return videos
    except json.JSONDecodeError:
        print("Error parsing playlist data")
        return []


def download_youtube(url, output_dir, browser=None):
    """Download audio from YouTube using yt-dlp with progress display"""
    print("  Downloading from YouTube...")

    output_file = os.path.join(output_dir, 'input.wav')

    title_cmd = ['yt-dlp', '--get-title', '--no-playlist', url]
    if browser:
        title_cmd.extend(['--cookies-from-browser', browser])
    title_result = subprocess.run(title_cmd, capture_output=True, text=True, errors='replace')
    original_title = title_result.stdout.strip() if title_result.returncode == 0 else "audio"

    temp_file = os.path.join(output_dir, 'temp_download.%(ext)s')

    cmd = ['yt-dlp', '-x', '--audio-quality', '0', '-o', temp_file, '--no-playlist',
           '--progress', '--newline']
    if browser:
        cmd.extend(['--cookies-from-browser', browser])
    cmd.append(url)

    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, errors='replace')

    if result.returncode != 0:
        if '403' in (result.stderr or ''):
            print("\nError: YouTube blocked the download (403 Forbidden)")
            print("\nTry: python stem_separator.py URL --browser edge")
            return None, None
        print(f"Download error: {result.stderr}")
        return None, None

    for file in os.listdir(output_dir):
        if file.startswith('temp_download'):
            filepath = os.path.join(output_dir, file)
            print("  Converting to WAV...")
            subprocess.run(['ffmpeg', '-i', filepath, '-y', '-loglevel', 'error', output_file])
            os.remove(filepath)
            if os.path.exists(output_file):
                return output_file, sanitize_filename(original_title)

    print("Error: Could not find downloaded file")
    return None, None


def expand_input_list(inputs: List[str]) -> List[str]:
    """
    Expand input list to handle batch files and glob patterns.

    Supports:
    - Regular files
    - Glob patterns (*.mp3)
    - Batch files (@batch.txt or --batch-file)
    - YouTube URLs
    """
    expanded = []

    for input_item in inputs:
        # Batch file (starts with @)
        if input_item.startswith('@'):
            batch_file = input_item[1:]
            if os.path.exists(batch_file):
                with open(batch_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            expanded.append(line)
            else:
                print(f"Warning: Batch file not found: {batch_file}")

        # YouTube URL
        elif is_youtube_url(input_item):
            expanded.append(input_item)

        # Glob pattern or regular file
        else:
            matches = glob_module.glob(input_item)
            if matches:
                expanded.extend(sorted(matches))
            elif os.path.exists(input_item):
                expanded.append(input_item)
            else:
                print(f"Warning: No matches for: {input_item}")

    return expanded


# =============================================================================
# CORE SEPARATION
# =============================================================================

def separate_audio(input_file, output_dir, model_name='htdemucs', output_format='wav',
                   selected_stems=None, force_cpu=False, normalize=False,
                   normalize_level=-14.0, analyze=False, export_daw=None):
    """
    Separate audio using Demucs.

    Returns:
        Tuple of (stems_dir, analysis_results)
    """
    analysis_results = {}

    try:
        import torch
        import numpy as np
        import soundfile as sf
        from scipy import signal
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install demucs soundfile scipy")
        return None, None

    import warnings
    warnings.filterwarnings('ignore', message='.*CUDA capability.*')

    # Load model
    print(f"  Loading {model_name} model...")
    model = get_model(model_name)
    model.train(False)
    source_names = model.sources

    if selected_stems is None:
        selected_stems = list(source_names)

    # Load audio
    print("  Loading audio file...")
    audio_data, sr = sf.read(input_file, dtype='float32')

    # Analyze if requested
    if analyze:
        analysis_results = analyze_audio(input_file)

    # Ensure stereo
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=1)

    # Resample if needed
    if sr != model.samplerate:
        print(f"  Resampling from {sr} Hz to {model.samplerate} Hz...")
        num_samples = int(len(audio_data) * model.samplerate / sr)
        audio_data = signal.resample(audio_data, num_samples)
        sr = model.samplerate

    wav = torch.from_numpy(audio_data.T.astype(np.float32))
    wav = wav.unsqueeze(0)

    # Try GPU first
    sources = None
    if not force_cpu and torch.cuda.is_available():
        try:
            print("  Trying GPU...")
            model.to('cuda')
            wav_gpu = wav.to('cuda')

            with torch.no_grad():
                sources = apply_model(model, wav_gpu, progress=True)

            print("  GPU acceleration worked!")

        except RuntimeError as e:
            if 'no kernel image' in str(e) or 'CUDA' in str(e):
                print("  GPU failed - falling back to CPU...")
                torch.cuda.empty_cache()
                model.to('cpu')
                sources = None
            else:
                raise

    # CPU fallback
    if sources is None:
        print("  Using CPU (this takes 2-4 minutes)...")
        model.to('cpu')
        wav = wav.to('cpu')

        with torch.no_grad():
            sources = apply_model(model, wav, progress=True)

    # Save stems
    format_ext = FORMATS[output_format]['ext']
    print(f"  Saving stems as {output_format.upper()}...")
    stems_dir = os.path.join(output_dir, 'stems')
    os.makedirs(stems_dir, exist_ok=True)

    pbar = ProgressBar(len(selected_stems), desc="  Saving stems", unit="stem")

    for i, name in enumerate(source_names):
        if name not in selected_stems:
            continue

        stem = sources[0, i].cpu().numpy().T
        temp_wav = os.path.join(stems_dir, f'{name}_temp.wav')
        sf.write(temp_wav, stem, sr)

        # Normalize if requested
        if normalize:
            normalize_audio(temp_wav, normalize_level)

        output_base = os.path.join(stems_dir, name)
        converted = convert_audio_format(temp_wav, output_base, output_format)

        if converted:
            if output_format != 'wav' and os.path.exists(temp_wav):
                os.remove(temp_wav)
        else:
            fallback = os.path.join(stems_dir, f'{name}.wav')
            os.rename(temp_wav, fallback)

        pbar.update(1)

    pbar.close()

    # DAW export
    if export_daw:
        print(f"  Creating DAW project ({export_daw})...")
        create_audacity_project(stems_dir, os.path.join(output_dir, f"stems_{export_daw}"), sr)

    return stems_dir, analysis_results


def process_single_item(input_item: str, output_dir: str, config: Config,
                        item_num: int = 1, total_items: int = 1) -> bool:
    """
    Process a single input item (file or URL).

    Returns True on success, False on failure.
    """
    print(f"\n{'='*60}")
    print(f"Processing [{item_num}/{total_items}]: {input_item[:60]}...")
    print('='*60)

    model_name = validate_model(config.model)
    output_format = validate_format(config.format)
    model_sources = MODELS[model_name]['sources']
    selected_stems = parse_stem_selection(config.stems, model_sources)

    temp_dir = tempfile.mkdtemp(prefix='stems_')

    try:
        # Get audio file
        if is_youtube_url(input_item):
            print_stage(1, 2, "Downloading from YouTube")
            audio_file, display_name = download_youtube(input_item, temp_dir, config.browser)
            if audio_file is None:
                return False
        else:
            if not os.path.exists(input_item):
                print(f"Error: File not found: {input_item}")
                return False
            original_path = os.path.abspath(input_item)
            display_name = sanitize_filename(Path(original_path).stem)
            audio_file = os.path.join(temp_dir, 'input.wav')

            if original_path.lower().endswith('.wav'):
                shutil.copy(original_path, audio_file)
            else:
                print_stage(1, 2, "Converting to WAV")
                subprocess.run(['ffmpeg', '-i', original_path, '-y', '-loglevel', 'error', audio_file])

        print(f"\nFile: {display_name}")

        # Separate
        print_stage(2, 2, f"Separating with {model_name}")
        stems_dir, analysis = separate_audio(
            audio_file,
            temp_dir,
            model_name=model_name,
            output_format=output_format,
            selected_stems=selected_stems,
            force_cpu=config.cpu,
            normalize=config.normalize,
            normalize_level=config.normalize_level,
            analyze=config.analyze,
            export_daw=config.export_daw
        )

        if stems_dir is None:
            return False

        # Print analysis if available
        if analysis:
            print_analysis(analysis, display_name)

        # Check results
        stem_files = [f for f in os.listdir(stems_dir) if not f.startswith('.')]
        if not stem_files:
            print("Error: No stems created")
            return False

        # Output stems to folder
        final_dir = os.path.join(output_dir, f"{display_name}_stems")
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(stems_dir, final_dir)

        # Also copy DAW project files if created
        for f in os.listdir(temp_dir):
            if f.endswith('.json') or f.endswith('.lof'):
                shutil.copy(os.path.join(temp_dir, f), final_dir)

        print(f"\nDone! Stems saved to: {final_dir}")
        print(f"Created {len(stem_files)} stem(s) in {output_format.upper()} format")

        return True

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Separate audio into stems using AI (Demucs)',
        epilog='''Examples:
  %(prog)s song.mp3
  %(prog)s song.mp3 --model htdemucs_6s --format mp3
  %(prog)s song.mp3 --stems karaoke --format flac
  %(prog)s "https://youtube.com/watch?v=..." --stems acapella
  %(prog)s *.mp3 --format mp3                     # Batch: all MP3s
  %(prog)s @songs.txt --analyze                   # Batch: from file
  %(prog)s "https://youtube.com/playlist?list=..." # YouTube playlist
  %(prog)s song.mp3 --normalize --analyze         # With analysis
  %(prog)s song.mp3 --export-daw audacity         # DAW export
  %(prog)s --server                               # Start API server''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', nargs='*', help='Audio file(s), YouTube URL(s), or @batch_file.txt')
    parser.add_argument('-o', '--output', default='.', help='Output directory')

    # Model and format
    parser.add_argument('--model', default='htdemucs',
                       help=f"AI model: {', '.join(MODELS.keys())} (default: htdemucs)")
    parser.add_argument('--format', default='wav',
                       help=f"Output format: {', '.join(FORMATS.keys())} (default: wav)")
    parser.add_argument('--stems', default=None,
                       help='Stems to export: comma-separated or preset (all, karaoke, acapella, instrumental)')

    # Processing options
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (skip GPU)')
    parser.add_argument('--browser', choices=['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave'],
                       help='Use cookies from browser (helps with YouTube 403 errors)')

    # New features
    parser.add_argument('--normalize', action='store_true', help='Normalize audio loudness')
    parser.add_argument('--normalize-level', type=float, default=-14.0,
                       help='Target loudness in LUFS (default: -14.0)')
    parser.add_argument('--analyze', action='store_true', help='Analyze BPM and musical key')
    parser.add_argument('--export-daw', choices=['audacity'],
                       help='Export DAW project file')

    # Configuration
    parser.add_argument('--save-config', metavar='FILE', help='Save current settings to config file')
    parser.add_argument('--config', metavar='FILE', help='Load settings from config file')

    # Server mode
    parser.add_argument('--server', action='store_true', help='Start API server mode')
    parser.add_argument('--host', default='127.0.0.1', help='API server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='API server port (default: 8000)')

    args = parser.parse_args()

    # Server mode
    if args.server:
        run_api_server(args.host, args.port)
        return

    # Load config from file if specified, otherwise find default
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.find_and_load()

    # Override config with CLI arguments
    if args.model != 'htdemucs':
        config.model = args.model
    if args.format != 'wav':
        config.format = args.format
    if args.output != '.':
        config.output = args.output
    if args.stems:
        config.stems = args.stems
    if args.cpu:
        config.cpu = True
    if args.browser:
        config.browser = args.browser
    if args.normalize:
        config.normalize = True
    if args.normalize_level != -14.0:
        config.normalize_level = args.normalize_level
    if args.analyze:
        config.analyze = True
    if args.export_daw:
        config.export_daw = args.export_daw

    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        if not args.input:
            return

    # Check for input
    if not args.input:
        parser.print_help()
        print("\nError: No input specified")
        sys.exit(1)

    # Check dependencies
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg not found.")
        print("  Install: winget install FFmpeg.FFmpeg")
        sys.exit(1)

    # Expand input list (batch files, globs, playlists)
    inputs = []
    for inp in args.input:
        if is_youtube_playlist(inp):
            print(f"Fetching playlist: {inp}")
            videos = get_playlist_videos(inp, config.browser)
            if videos:
                print(f"Found {len(videos)} videos in playlist")
                inputs.extend([v['url'] for v in videos])
            else:
                print("Warning: Could not fetch playlist videos")
        else:
            inputs.append(inp)

    inputs = expand_input_list(inputs)

    if not inputs:
        print("Error: No valid inputs found")
        sys.exit(1)

    # Check yt-dlp for YouTube URLs
    youtube_inputs = [i for i in inputs if is_youtube_url(i)]
    if youtube_inputs and shutil.which('yt-dlp') is None:
        print("Error: yt-dlp not found. Install with: pip install yt-dlp")
        sys.exit(1)

    output_dir = os.path.abspath(config.output)
    os.makedirs(output_dir, exist_ok=True)

    # Process all inputs
    total = len(inputs)
    success = 0
    failed = []

    print(f"\nProcessing {total} item(s)...")

    for i, input_item in enumerate(inputs, 1):
        try:
            if process_single_item(input_item, output_dir, config, i, total):
                success += 1
            else:
                failed.append(input_item)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError processing {input_item}: {e}")
            failed.append(input_item)

    # Summary
    if total > 1:
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {success}/{total} successful")
        if failed:
            print(f"\nFailed items:")
            for item in failed:
                print(f"  - {item[:60]}")
        print('='*60)


# =============================================================================
# API SERVER MODE
# =============================================================================

def run_api_server(host: str = '127.0.0.1', port: int = 8000):
    """Start the FastAPI server for API mode"""
    try:
        from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        import uvicorn
        import uuid
    except ImportError:
        print("Error: FastAPI and uvicorn not installed.")
        print("Install with: pip install fastapi uvicorn python-multipart")
        sys.exit(1)

    app = FastAPI(
        title="Stem Separator API",
        description="AI-powered audio stem separation using Demucs",
        version="2.0.0"
    )

    # Job storage
    jobs: Dict[str, Dict[str, Any]] = {}
    jobs_dir = tempfile.mkdtemp(prefix='stem_api_')

    @app.get("/")
    async def root():
        return {
            "name": "Stem Separator API",
            "version": "2.0.0",
            "endpoints": {
                "POST /separate": "Submit audio for separation",
                "GET /jobs/{job_id}": "Check job status",
                "GET /jobs/{job_id}/download/{stem}": "Download a stem",
                "GET /models": "List available models",
                "GET /formats": "List available formats"
            }
        }

    @app.get("/models")
    async def list_models():
        return MODELS

    @app.get("/formats")
    async def list_formats():
        return FORMATS

    @app.post("/separate")
    async def separate(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        model: str = Form("htdemucs"),
        format: str = Form("wav"),
        stems: str = Form("all"),
        normalize: bool = Form(False)
    ):
        """Submit an audio file for stem separation"""
        job_id = str(uuid.uuid4())[:8]
        job_dir = os.path.join(jobs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        # Save uploaded file
        input_path = os.path.join(job_dir, file.filename)
        with open(input_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        jobs[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "model": model,
            "format": format,
            "stems": [],
            "error": None
        }

        # Process in background
        def process_job():
            try:
                jobs[job_id]["status"] = "processing"

                config = Config(
                    model=model,
                    format=format,
                    stems=stems if stems != "all" else None,
                    normalize=normalize,
                    output=job_dir
                )

                if process_single_item(input_path, job_dir, config):
                    # Find output stems
                    stems_dir = None
                    for d in os.listdir(job_dir):
                        if d.endswith('_stems'):
                            stems_dir = os.path.join(job_dir, d)
                            break

                    if stems_dir:
                        jobs[job_id]["stems"] = [
                            f for f in os.listdir(stems_dir)
                            if not f.startswith('.') and not f.endswith('.json') and not f.endswith('.lof')
                        ]
                        jobs[job_id]["stems_dir"] = stems_dir

                    jobs[job_id]["status"] = "completed"
                else:
                    jobs[job_id]["status"] = "failed"
                    jobs[job_id]["error"] = "Processing failed"

            except Exception as e:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)

        background_tasks.add_task(process_job)

        return {"job_id": job_id, "status": "queued"}

    @app.get("/jobs/{job_id}")
    async def get_job_status(job_id: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        return jobs[job_id]

    @app.get("/jobs/{job_id}/download/{stem}")
    async def download_stem(job_id: str, stem: str):
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")

        job = jobs[job_id]
        if job["status"] != "completed":
            raise HTTPException(status_code=400, detail="Job not completed")

        stems_dir = job.get("stems_dir")
        if not stems_dir:
            raise HTTPException(status_code=404, detail="Stems not found")

        stem_path = os.path.join(stems_dir, stem)
        if not os.path.exists(stem_path):
            raise HTTPException(status_code=404, detail=f"Stem '{stem}' not found")

        return FileResponse(stem_path, filename=stem)

    print(f"\nStarting Stem Separator API server...")
    print(f"  URL: http://{host}:{port}")
    print(f"  Docs: http://{host}:{port}/docs")
    print(f"\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
