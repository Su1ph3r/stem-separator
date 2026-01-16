"""
Real-time audio preview module for Stem Separator.

Provides functionality to preview separated stems before saving.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from stem_separator.logging_config import get_logger, print_error, print_info, print_status


@dataclass
class PreviewState:
    """Current state of audio preview."""

    is_playing: bool = False
    current_stem: str = ""
    position: float = 0.0
    duration: float = 0.0
    volume: float = 1.0


class AudioPreview:
    """
    Audio preview player for stem files.

    Uses sounddevice for cross-platform audio playback.
    """

    def __init__(self):
        """Initialize the audio preview system."""
        self.logger = get_logger()
        self._state = PreviewState()
        self._audio_data = None
        self._sample_rate = None
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._sd = None
        self._available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if audio preview is available."""
        try:
            import sounddevice as sd

            self._sd = sd
            return True
        except ImportError:
            self.logger.warning("sounddevice not available, preview disabled")
            return False

    @property
    def available(self) -> bool:
        """Check if preview functionality is available."""
        return self._available

    @property
    def state(self) -> PreviewState:
        """Get current preview state."""
        return self._state

    def load(self, file_path: Path, stem_name: str = "") -> bool:
        """
        Load an audio file for preview.

        Args:
            file_path: Path to audio file.
            stem_name: Name of the stem (for display).

        Returns:
            True if loaded successfully.
        """
        if not self._available:
            print_error("Preview not available. Install sounddevice: pip install sounddevice")
            return False

        try:
            import soundfile as sf

            self._audio_data, self._sample_rate = sf.read(str(file_path), dtype="float32")
            self._state.current_stem = stem_name or file_path.stem
            self._state.duration = len(self._audio_data) / self._sample_rate
            self._state.position = 0.0
            return True

        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            return False

    def play(
        self,
        start_position: float = 0.0,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> bool:
        """
        Start playback.

        Args:
            start_position: Start position in seconds.
            on_complete: Callback when playback completes.

        Returns:
            True if playback started.
        """
        if not self._available or self._audio_data is None:
            return False

        self.stop()
        self._stop_flag.clear()
        self._state.position = start_position
        self._state.is_playing = True

        def playback_worker():
            try:
                start_sample = int(start_position * self._sample_rate)
                audio_chunk = self._audio_data[start_sample:]

                # Apply volume
                if self._state.volume != 1.0:
                    audio_chunk = audio_chunk * self._state.volume

                self._sd.play(audio_chunk, self._sample_rate)

                # Update position while playing
                while self._sd.get_stream().active and not self._stop_flag.is_set():
                    elapsed = self._sd.get_stream().time
                    self._state.position = start_position + elapsed
                    time.sleep(0.1)

                self._state.is_playing = False

                if on_complete and not self._stop_flag.is_set():
                    on_complete()

            except Exception as e:
                self.logger.error(f"Playback error: {e}")
                self._state.is_playing = False

        self._playback_thread = threading.Thread(target=playback_worker, daemon=True)
        self._playback_thread.start()
        return True

    def pause(self):
        """Pause playback."""
        if self._available and self._state.is_playing:
            self._sd.stop()
            self._state.is_playing = False

    def stop(self):
        """Stop playback."""
        if not self._available:
            return

        self._stop_flag.set()
        if self._sd:
            self._sd.stop()
        self._state.is_playing = False
        self._state.position = 0.0

        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)

    def seek(self, position: float):
        """
        Seek to position.

        Args:
            position: Position in seconds.
        """
        was_playing = self._state.is_playing
        self.stop()
        self._state.position = max(0.0, min(position, self._state.duration))

        if was_playing:
            self.play(self._state.position)

    def set_volume(self, volume: float):
        """
        Set playback volume.

        Args:
            volume: Volume level (0.0 to 2.0).
        """
        self._state.volume = max(0.0, min(2.0, volume))

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        self._audio_data = None


class InteractivePreview:
    """
    Interactive preview mode for auditioning stems.

    Provides a simple text-based interface for previewing stems.
    """

    def __init__(self, stems_dir: Path):
        """
        Initialize interactive preview.

        Args:
            stems_dir: Directory containing stem files.
        """
        self.stems_dir = Path(stems_dir)
        self.preview = AudioPreview()
        self.logger = get_logger()
        self._stems: dict[str, Path] = {}
        self._load_stems()

    def _load_stems(self):
        """Load available stems from directory."""
        for file in self.stems_dir.iterdir():
            if file.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
                # Extract stem name
                stem_name = file.stem.lower()
                for known in ["vocals", "drums", "bass", "other", "guitar", "piano"]:
                    if known in stem_name:
                        self._stems[known] = file
                        break

    @property
    def available_stems(self) -> list[str]:
        """Get list of available stem names."""
        return list(self._stems.keys())

    def preview_stem(self, stem_name: str) -> bool:
        """
        Preview a specific stem.

        Args:
            stem_name: Name of stem to preview.

        Returns:
            True if preview started.
        """
        stem_name = stem_name.lower()
        if stem_name not in self._stems:
            print_error(f"Stem not found: {stem_name}")
            print_info(f"Available stems: {', '.join(self._stems.keys())}")
            return False

        if self.preview.load(self._stems[stem_name], stem_name):
            print_status(f"Playing: {stem_name}")
            return self.preview.play()

        return False

    def run_interactive(self):
        """Run interactive preview mode."""
        if not self.preview.available:
            print_error("Preview not available. Install sounddevice: pip install sounddevice")
            return

        if not self._stems:
            print_error("No stems found in directory")
            return

        print_status("Interactive Preview Mode")
        print_info(f"Available stems: {', '.join(self._stems.keys())}")
        print_info("Commands: play <stem>, stop, volume <0-100>, quit")
        print()

        try:
            while True:
                try:
                    cmd = input("> ").strip().lower()
                except EOFError:
                    break

                if not cmd:
                    continue

                parts = cmd.split()
                command = parts[0]

                if command in ("quit", "exit", "q"):
                    break
                elif command == "play" and len(parts) > 1:
                    self.preview_stem(parts[1])
                elif command == "stop":
                    self.preview.stop()
                    print_info("Stopped")
                elif command == "pause":
                    self.preview.pause()
                    print_info("Paused")
                elif command == "volume" and len(parts) > 1:
                    try:
                        vol = float(parts[1]) / 100.0
                        self.preview.set_volume(vol)
                        print_info(f"Volume: {int(vol * 100)}%")
                    except ValueError:
                        print_error("Invalid volume")
                elif command == "list":
                    print_info(f"Stems: {', '.join(self._stems.keys())}")
                elif command == "status":
                    state = self.preview.state
                    print_info(
                        f"Playing: {state.is_playing}, "
                        f"Stem: {state.current_stem}, "
                        f"Position: {state.position:.1f}s / {state.duration:.1f}s"
                    )
                else:
                    print_info("Commands: play <stem>, stop, pause, volume <0-100>, list, status, quit")

        finally:
            self.preview.cleanup()
            print_info("Preview ended")


def preview_stems(stems_dir: Path, interactive: bool = True):
    """
    Preview separated stems.

    Args:
        stems_dir: Directory containing stem files.
        interactive: Whether to run in interactive mode.
    """
    preview = InteractivePreview(stems_dir)

    if interactive:
        preview.run_interactive()
    else:
        # Just list available stems
        stems = preview.available_stems
        if stems:
            print_info(f"Available stems for preview: {', '.join(stems)}")
        else:
            print_error("No stems found")
