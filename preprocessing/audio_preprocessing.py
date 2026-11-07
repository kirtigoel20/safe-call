
import numpy as np
import soundfile as sf
import librosa


def load_audio(file_path):
    audio, sample_rate = sf.read(
        file_path,
        dtype="float32"
    )

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid values.")

    return audio, sample_rate


def preprocess_audio(file_path, target_sr=16000):
    audio, original_sr = load_audio(file_path)

    if original_sr != target_sr:
        audio = librosa.resample(
            audio,
            orig_sr=original_sr,
            target_sr=target_sr
        )

    return audio, target_sr