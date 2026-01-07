#!/usr/bin/env python3
"""
Stem Separator - Audio Stem Extraction Tool
Uses Demucs (by Facebook/Meta Research) for high-quality stem separation
Supports YouTube URLs and local audio files
"""

import os
import sys
import argparse
import subprocess
import shutil
import tempfile
import re
from pathlib import Path

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

    Args:
        stems_arg: String like "vocals,drums" or preset name like "karaoke"
        model_sources: List of available stems from the model

    Returns:
        List of stem names to export
    """
    if stems_arg is None:
        return list(model_sources)

    stems_lower = stems_arg.lower().strip()

    # Check if it's a preset
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

    # Parse comma-separated list
    requested_stems = [s.strip().lower() for s in stems_arg.split(',') if s.strip()]

    if not requested_stems:
        print("Error: No valid stems specified")
        sys.exit(1)

    # Validate all requested stems exist in model
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
    """
    Convert WAV file to specified format using FFmpeg.

    Args:
        input_wav: Path to input WAV file
        output_path: Base path for output (extension will be added)
        format_name: Format key from FORMATS dict

    Returns:
        Path to converted file, or None on failure
    """
    format_config = FORMATS[format_name]
    output_file = output_path + format_config['ext']

    if format_name == 'wav':
        shutil.copy(input_wav, output_file)
        return output_file

    # Build FFmpeg command
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


def download_youtube(url, output_dir, browser=None):
    """Download audio from YouTube using yt-dlp with progress display"""
    print("\n[1/2] Downloading from YouTube...")

    output_file = os.path.join(output_dir, 'input.wav')

    # Get video title first (silent)
    title_cmd = ['yt-dlp', '--get-title', '--no-playlist', url]
    if browser:
        title_cmd.extend(['--cookies-from-browser', browser])
    title_result = subprocess.run(title_cmd, capture_output=True, text=True, errors='replace')
    original_title = title_result.stdout.strip() if title_result.returncode == 0 else "audio"

    temp_file = os.path.join(output_dir, 'temp_download.%(ext)s')

    # Download with progress bar visible
    cmd = ['yt-dlp', '-x', '--audio-quality', '0', '-o', temp_file, '--no-playlist',
           '--progress', '--newline']
    if browser:
        cmd.extend(['--cookies-from-browser', browser])
    cmd.append(url)

    # Show progress by not capturing stdout, but capture stderr for errors
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, errors='replace')

    if result.returncode != 0:
        if '403' in (result.stderr or ''):
            print("\nError: YouTube blocked the download (403 Forbidden)")
            print("\nTry: python stem_separator.py URL --browser edge")
            sys.exit(1)
        print(f"Download error: {result.stderr}")
        sys.exit(1)

    for file in os.listdir(output_dir):
        if file.startswith('temp_download'):
            filepath = os.path.join(output_dir, file)
            print("  Converting to WAV...")
            subprocess.run(['ffmpeg', '-i', filepath, '-y', '-loglevel', 'error', output_file])
            os.remove(filepath)
            if os.path.exists(output_file):
                return output_file, sanitize_filename(original_title)

    print("Error: Could not find downloaded file")
    sys.exit(1)


def separate_audio(input_file, output_dir, model_name='htdemucs', output_format='wav',
                   selected_stems=None, force_cpu=False):
    """
    Separate audio using Demucs.

    Args:
        input_file: Path to input audio file
        output_dir: Directory for output
        model_name: Demucs model to use (from MODELS dict)
        output_format: Output format (from FORMATS dict)
        selected_stems: List of stem names to export (None = all)
        force_cpu: Force CPU processing
    """
    print(f"\n[2/2] Separating audio with {model_name}...")

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
        sys.exit(1)

    # Suppress the CUDA compatibility warning
    import warnings
    warnings.filterwarnings('ignore', message='.*CUDA capability.*')

    # Load model
    print(f"  Loading {model_name} model...")
    model = get_model(model_name)
    model.train(False)  # Set to evaluation mode
    source_names = model.sources

    # Determine which stems to export
    if selected_stems is None:
        selected_stems = list(source_names)
    
    # Load audio using soundfile first (before device selection)
    print("  Loading audio file...")
    audio_data, sr = sf.read(input_file, dtype='float32')
    
    # Ensure stereo
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=1)
    
    # Resample if needed (demucs expects 44100 Hz)
    if sr != model.samplerate:
        print(f"  Resampling from {sr} Hz to {model.samplerate} Hz...")
        num_samples = int(len(audio_data) * model.samplerate / sr)
        audio_data = signal.resample(audio_data, num_samples)
        sr = model.samplerate
    
    # Convert to torch tensor [channels, samples]
    wav = torch.from_numpy(audio_data.T.astype(np.float32))
    wav = wav.unsqueeze(0)  # Add batch dimension
    
    # Try GPU first, fall back to CPU if it fails
    sources = None
    if not force_cpu and torch.cuda.is_available():
        try:
            print("  Trying GPU...")
            model.to('cuda')
            wav_gpu = wav.to('cuda')
            
            # Try running the model - this will fail if kernels don't exist
            with torch.no_grad():
                sources = apply_model(model, wav_gpu, progress=True)
            
            print("  GPU acceleration worked!")
            
        except RuntimeError as e:
            if 'no kernel image' in str(e) or 'CUDA' in str(e):
                print("  GPU failed - try: pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
                print("  Using CPU for now...")
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

    for i, name in enumerate(source_names):
        # Skip stems not in selection
        if name not in selected_stems:
            continue

        stem = sources[0, i].cpu().numpy().T  # [channels, samples] -> [samples, channels]

        # Save as WAV first
        temp_wav = os.path.join(stems_dir, f'{name}_temp.wav')
        sf.write(temp_wav, stem, sr)

        # Convert to target format
        output_base = os.path.join(stems_dir, name)
        converted = convert_audio_format(temp_wav, output_base, output_format)

        if converted:
            print(f"    Saved {name}{format_ext}")
            # Clean up temp WAV if we converted to different format
            if output_format != 'wav' and os.path.exists(temp_wav):
                os.remove(temp_wav)
        else:
            # Keep WAV as fallback
            fallback = os.path.join(stems_dir, f'{name}.wav')
            os.rename(temp_wav, fallback)
            print(f"    Saved {name}.wav (fallback)")

    return stems_dir


def main():
    parser = argparse.ArgumentParser(
        description='Separate audio into stems using AI (Demucs)',
        epilog='Examples:\n'
               '  %(prog)s song.mp3\n'
               '  %(prog)s song.mp3 --model htdemucs_6s --format mp3\n'
               '  %(prog)s song.mp3 --stems karaoke --format flac\n'
               '  %(prog)s "https://youtube.com/watch?v=..." --stems acapella',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', help='Audio file or YouTube URL')
    parser.add_argument('-o', '--output', default='.', help='Output directory')

    # New feature arguments
    parser.add_argument('--model', default='htdemucs',
                       help=f"AI model: {', '.join(MODELS.keys())} (default: htdemucs)")
    parser.add_argument('--format', default='wav',
                       help=f"Output format: {', '.join(FORMATS.keys())} (default: wav)")
    parser.add_argument('--stems', default=None,
                       help='Stems to export: comma-separated (vocals,drums) or preset '
                            '(all, karaoke, acapella, instrumental)')

    # Existing arguments
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (skip GPU)')
    parser.add_argument('--browser', choices=['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave'],
                       help='Use cookies from browser (helps with YouTube 403 errors)')

    args = parser.parse_args()

    # Validate new arguments
    model_name = validate_model(args.model)
    output_format = validate_format(args.format)
    model_sources = MODELS[model_name]['sources']
    selected_stems = parse_stem_selection(args.stems, model_sources)
    
    # Check dependencies
    if is_youtube_url(args.input):
        if shutil.which('yt-dlp') is None:
            print("Error: yt-dlp not found. Install with: pip install yt-dlp")
            sys.exit(1)
    
    if shutil.which('ffmpeg') is None:
        print("Error: ffmpeg not found.")
        print("  Install: winget install FFmpeg.FFmpeg")
        sys.exit(1)
    
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix='stems_')
    
    try:
        # Get audio file
        if is_youtube_url(args.input):
            audio_file, display_name = download_youtube(args.input, temp_dir, args.browser)
        else:
            if not os.path.exists(args.input):
                print(f"Error: File not found: {args.input}")
                sys.exit(1)
            original_path = os.path.abspath(args.input)
            display_name = sanitize_filename(Path(original_path).stem)
            audio_file = os.path.join(temp_dir, 'input.wav')
            
            if original_path.lower().endswith('.wav'):
                shutil.copy(original_path, audio_file)
            else:
                print("\n[1/2] Converting to WAV...")
                subprocess.run(['ffmpeg', '-i', original_path, '-y', '-loglevel', 'error', audio_file])
        
        print(f"\nProcessing: {display_name}")

        # Separate
        stems_dir = separate_audio(
            audio_file,
            temp_dir,
            model_name=model_name,
            output_format=output_format,
            selected_stems=selected_stems,
            force_cpu=args.cpu
        )

        # Check results
        format_ext = FORMATS[output_format]['ext']
        stem_files = [f for f in os.listdir(stems_dir) if not f.startswith('.')]
        if not stem_files:
            print("Error: No stems created")
            sys.exit(1)

        # Output stems to folder
        final_dir = os.path.join(output_dir, f"{display_name}_stems")
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(stems_dir, final_dir)

        print(f"\nDone! Stems saved to: {final_dir}")

        print(f"\nCreated {len(stem_files)} stem(s) in {output_format.upper()} format:")
        for f in sorted(stem_files):
            print(f"  - {f}")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
