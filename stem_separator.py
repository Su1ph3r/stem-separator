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
    """Download audio from YouTube using yt-dlp"""
    print("\n[1/2] Downloading from YouTube...")
    
    output_file = os.path.join(output_dir, 'input.wav')
    
    title_cmd = ['yt-dlp', '--get-title', '--no-playlist', url]
    if browser:
        title_cmd.extend(['--cookies-from-browser', browser])
    title_result = subprocess.run(title_cmd, capture_output=True, text=True, errors='replace')
    original_title = title_result.stdout.strip() if title_result.returncode == 0 else "audio"
    
    temp_file = os.path.join(output_dir, 'temp_download.%(ext)s')
    
    cmd = ['yt-dlp', '-x', '--audio-quality', '0', '-o', temp_file, '--no-playlist']
    if browser:
        cmd.extend(['--cookies-from-browser', browser])
    cmd.append(url)
    
    result = subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    
    if result.returncode != 0:
        if '403' in result.stderr:
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


def separate_audio(input_file, output_dir, force_cpu=False):
    """Separate audio using Demucs as a library"""
    print("\n[2/2] Separating audio (this takes a few minutes)...")
    
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
    print("  Loading AI model...")
    model = get_model('htdemucs')
    model.eval()
    source_names = model.sources  # ['drums', 'bass', 'other', 'vocals']
    
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
    print("  Saving stems...")
    stems_dir = os.path.join(output_dir, 'stems')
    os.makedirs(stems_dir, exist_ok=True)
    
    for i, name in enumerate(source_names):
        stem = sources[0, i].cpu().numpy().T  # [channels, samples] -> [samples, channels]
        stem_path = os.path.join(stems_dir, f'{name}.wav')
        sf.write(stem_path, stem, sr)
        print(f"    Saved {name}.wav")
    
    return stems_dir


def main():
    parser = argparse.ArgumentParser(
        description='Separate audio into stems (vocals, drums, bass, other)',
        epilog='Examples:\n'
               '  %(prog)s song.mp3\n'
               '  %(prog)s "https://youtube.com/watch?v=..."\n'
               '  %(prog)s song.mp3 -o C:\\Music\\Stems',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', help='Audio file or YouTube URL')
    parser.add_argument('-o', '--output', default='.', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (skip GPU)')
    parser.add_argument('--browser', choices=['chrome', 'firefox', 'edge', 'safari', 'opera', 'brave'],
                       help='Use cookies from browser (helps with YouTube 403 errors)')
    
    args = parser.parse_args()
    
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
        stems_dir = separate_audio(audio_file, temp_dir, force_cpu=args.cpu)
        
        # Check results
        wav_files = [f for f in os.listdir(stems_dir) if f.endswith('.wav')]
        if not wav_files:
            print("Error: No stems created")
            sys.exit(1)
        
        # Output stems to folder
        final_dir = os.path.join(output_dir, f"{display_name}_stems")
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(stems_dir, final_dir)
        
        print(f"\nDone! Stems saved to: {final_dir}")
        
        print(f"\nCreated {len(wav_files)} stems:")
        for f in sorted(wav_files):
            print(f"  - {f}")
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
