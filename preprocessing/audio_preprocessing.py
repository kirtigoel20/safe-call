"""
Audio Preprocessing Module for SAFE CALL.

Provides universal audio ingestion, normalization, and validation.
Supports .wav, .flac, .mp3, .m4a, .aac, .ogg, .wma, and other standard formats.
Standardizes all signals to 16,000 Hz single-channel mono float32.
"""

import os
import subprocess
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import librosa

try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    # Ensure ffmpeg directory is in PATH for any downstream tools
    ffmpeg_dir = str(Path(FFMPEG_BIN).parent)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception:
    FFMPEG_BIN = "ffmpeg"


def _load_via_ffmpeg(file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Decodes audio using bundled FFmpeg.
    Handles .m4a, .mp3, .aac, .mp4, and compressed formats seamlessly.
    """
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-threads", "0",
        "-i", str(file_path),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to decode audio file {file_path}: {exc.stderr.decode(errors='replace')}"
        ) from exc

    raw_bytes = proc.stdout
    if len(raw_bytes) == 0:
        raise ValueError("Audio file is empty.")

    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, target_sr


def load_audio(file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Robust audio loader supporting all common formats (.wav, .flac, .mp3, .m4a, .aac, etc.).

    Parameters:
        file_path: Path to the audio file.
        target_sr: Desired sample rate (default: 16000 Hz).

    Returns:
        (audio_array, sample_rate): 1D float32 numpy array and sample rate.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 1. Try fast soundfile load first (ideal for .wav, .flac, .ogg)
    audio = None
    sample_rate = target_sr
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
    except Exception:
        # 2. Seamless fallback to FFmpeg for .m4a, .mp3, .aac, .mp4, etc.
        audio, sample_rate = _load_via_ffmpeg(path, target_sr=target_sr)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    return audio, sample_rate


def preprocess_audio(file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio to standard 16 kHz mono float32.

    Parameters:
        file_path: Path to audio file.
        target_sr: Target sample rate (16000 Hz).

    Returns:
        (audio, sample_rate)
    """
    audio, original_sr = load_audio(file_path, target_sr=target_sr)

    # Resample if sample rate doesn't match target
    if original_sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=target_sr
        )

    return audio.astype(np.float32), target_sr
