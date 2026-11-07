
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_preprocessing import (
    load_audio,
    preprocess_audio,
)

# Temporary test audio files will be stored here
TEST_DIR = PROJECT_ROOT / "tests" / "temp_audio"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def create_test_audio(filename, sample_rate, audio):
    path = TEST_DIR / filename
    sf.write(path, audio, sample_rate)
    return path


def main():
    # Test 1: Stereo audio should become mono
    sr = 22050
    duration = 2
    t = np.arange(sr * duration) / sr

    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack((left, right))

    stereo_path = create_test_audio(
        "stereo_test.wav", sr, stereo
    )

    mono_audio, original_sr = load_audio(stereo_path)

    assert mono_audio.ndim == 1
    assert original_sr == sr
    print("PASS: Stereo audio converted to mono.")

    # Test 2: Audio should be resampled to 16 kHz
    processed_audio, processed_sr = preprocess_audio(
        stereo_path, target_sr=16000
    )

    assert processed_sr == 16000
    assert processed_audio.ndim == 1
    assert len(processed_audio) == 32000

    print("PASS: Audio resampled to 16000 Hz.")
    print("Processed shape:", processed_audio.shape)
    print("Processed duration:",
          round(len(processed_audio) / processed_sr, 2),
          "seconds")

    # Test 3: Empty audio should be rejected
    empty_path = TEST_DIR / "empty_test.wav"
    sf.write(empty_path, np.array([], dtype=np.float32), 16000)

    try:
        load_audio(empty_path)
    except ValueError as error:
        assert "empty" in str(error).lower()
        print("PASS: Empty audio correctly rejected.")
    else:
        raise AssertionError("Empty audio was not rejected.")

    # Test 4: Non-finite audio should be rejected
    invalid_path = TEST_DIR / "invalid_test.wav"

    # Write a valid file first, then replace its samples
    sf.write(invalid_path, np.zeros(1600, dtype=np.float32), 16000)

    # SoundFile cannot reliably store NaN/Inf in every format.
    # Test the validation directly by mocking sf.read.
    import unittest.mock as mock

    with mock.patch(
        "preprocessing.audio_preprocessing.sf.read",
        return_value=(
            np.array([0.1, np.nan, 0.2], dtype=np.float32),
            16000,
        ),
    ):
        try:
            load_audio(invalid_path)
        except ValueError as error:
            assert "invalid values" in str(error).lower()
            print("PASS: NaN/Inf audio correctly rejected.")
        else:
            raise AssertionError("Invalid audio was not rejected.")

    print("\nAll preprocessing checks passed!")


if __name__ == "__main__":
    main()
