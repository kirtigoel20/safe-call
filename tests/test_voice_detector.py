"""
Unit tests for Voice Deepfake Detection Module (Phase 2).
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_detection.voice_detector import (
    VoiceDeepfakeDetector,
    get_voice_detector,
    predict_voice,
    LABEL_HUMAN,
    LABEL_SPOOF,
    DEFAULT_EER_THRESHOLD,
)

TEMP_DIR = PROJECT_ROOT / "tests" / "temp_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def detector():
    return get_voice_detector()


@pytest.fixture
def synthetic_wav():
    """Generates a valid 1-second 16 kHz sine wave audio file."""
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = TEMP_DIR / "test_synth_audio.wav"
    sf.write(path, audio, sr)
    return path


def test_detector_initialization(detector):
    assert detector is not None
    assert detector.model is not None
    assert abs(detector.threshold - DEFAULT_EER_THRESHOLD) < 1e-4


def test_predict_structure(detector, synthetic_wav):
    result = detector.predict(synthetic_wav)
    
    assert isinstance(result, dict)
    required_keys = [
        "prediction",
        "raw_label",
        "confidence",
        "spoof_probability",
        "bonafide_probability",
        "audio_duration_seconds",
        "sample_rate",
        "threshold_used",
        "feature_dim",
    ]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in prediction output"

    assert result["prediction"] in (LABEL_HUMAN, LABEL_SPOOF)
    assert result["raw_label"] in (0, 1)
    assert 0.0 <= result["confidence"] <= 1.0
    assert 0.0 <= result["spoof_probability"] <= 1.0
    assert 0.0 <= result["bonafide_probability"] <= 1.0
    assert abs(result["spoof_probability"] + result["bonafide_probability"] - 1.0) < 1e-2
    assert result["feature_dim"] == 40
    assert result["sample_rate"] == 16000
    assert result["audio_duration_seconds"] > 0.5


def test_predict_with_numpy_array(detector):
    sr = 16000
    audio_arr = (0.3 * np.random.randn(sr * 2)).astype(np.float32)
    result = detector.predict(audio_arr)

    assert result["prediction"] in (LABEL_HUMAN, LABEL_SPOOF)
    assert result["audio_duration_seconds"] == 2.0


def test_convenience_function(synthetic_wav):
    result = predict_voice(synthetic_wav)
    assert result["prediction"] in (LABEL_HUMAN, LABEL_SPOOF)


def test_file_not_found(detector):
    missing_path = TEMP_DIR / "definitely_non_existent_file.wav"
    with pytest.raises(FileNotFoundError):
        detector.predict(missing_path)


def test_audio_too_short(detector):
    # Audio with only 10 samples (~0.0006s)
    short_audio = np.array([0.1] * 10, dtype=np.float32)
    with pytest.raises(ValueError, match="too short"):
        detector.predict(short_audio)


def test_empty_audio_rejection(detector):
    empty_path = TEMP_DIR / "empty_test.wav"
    sf.write(empty_path, np.array([], dtype=np.float32), 16000)
    with pytest.raises(ValueError, match="empty"):
        detector.predict(empty_path)


def test_custom_threshold_override(detector, synthetic_wav):
    # With threshold 0.0, anything should be classified as spoof
    res_spoof = detector.predict(synthetic_wav, threshold=0.0)
    assert res_spoof["prediction"] == LABEL_SPOOF
    assert res_spoof["raw_label"] == 1

    # With threshold 1.0, anything should be classified as human
    res_human = detector.predict(synthetic_wav, threshold=1.0)
    assert res_human["prediction"] == LABEL_HUMAN
    assert res_human["raw_label"] == 0

