"""
Unit tests for Speech-to-Text Module (Phase 3).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_to_text.transcriber import (
    SpeechTranscriber,
    get_transcriber,
    transcribe_audio,
)

TEMP_DIR = PROJECT_ROOT / "tests" / "temp_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def transcriber():
    return get_transcriber("tiny")


@pytest.fixture
def synthetic_speech_wav():
    """Generates a 1-second 16 kHz sine audio file for testing pipeline flow."""
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    path = TEMP_DIR / "test_synth_speech.wav"
    sf.write(path, audio, sr)
    return path


def test_transcriber_initialization(transcriber):
    assert transcriber is not None
    assert transcriber.model is not None
    assert transcriber.model_name == "tiny"
    assert transcriber.device in ("cpu", "cuda")


def test_transcribe_structure(transcriber, synthetic_speech_wav):
    result = transcriber.transcribe(synthetic_speech_wav)

    assert isinstance(result, dict)
    required_keys = [
        "text",
        "language",
        "segments",
        "audio_duration_seconds",
        "model_name",
        "device",
    ]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in transcription output"

    assert isinstance(result["text"], str)
    assert isinstance(result["language"], str)
    assert isinstance(result["segments"], list)
    assert result["audio_duration_seconds"] > 0.5
    assert "whisper" in result["model_name"]


def test_transcribe_numpy_array(transcriber):
    sr = 16000
    audio_arr = (0.2 * np.random.randn(sr * 1)).astype(np.float32)
    result = transcriber.transcribe(audio_arr)

    assert isinstance(result, dict)
    assert "text" in result
    assert result["audio_duration_seconds"] == 1.0


def test_convenience_function(synthetic_speech_wav):
    result = transcribe_audio(synthetic_speech_wav, model_name="tiny")
    assert isinstance(result, dict)
    assert "text" in result


def test_file_not_found(transcriber):
    missing_path = TEMP_DIR / "non_existent_speech_file.wav"
    with pytest.raises(FileNotFoundError):
        transcriber.transcribe(missing_path)


def test_audio_too_short(transcriber):
    short_audio = np.array([0.1] * 10, dtype=np.float32)
    with pytest.raises(ValueError, match="too short"):
        transcriber.transcribe(short_audio)


def test_empty_audio_rejection(transcriber):
    empty_path = TEMP_DIR / "empty_speech_test.wav"
    sf.write(empty_path, np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="empty"):
        transcriber.transcribe(empty_path)

