
from pathlib import Path
import numpy as np
import librosa

try:
    from preprocessing.audio_preprocessing import preprocess_audio
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from preprocessing.audio_preprocessing import preprocess_audio

SAMPLE_RATE = 16000
N_MFCC = 20


def extract_features(audio_input, sr=SAMPLE_RATE):
    """
    Extract 40-dimensional MFCC summary features (20 means + 20 standard deviations).

    Parameters:
        audio_input (str | Path | np.ndarray): File path to audio or preprocessed 1D audio array.
        sr (int): Sampling rate (default: 16000 Hz).

    Returns:
        np.ndarray: 40-dimensional float32 feature vector.
    """
    if isinstance(audio_input, (str, Path)):
        audio, sample_rate = preprocess_audio(audio_input, target_sr=sr)
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
        sample_rate = sr
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Extract MFCCs: 20 coefficients across time
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=160
    )

    # Summarize each coefficient over time
    # 20 means + 20 standard deviations = 40 features
    mean_features = np.mean(mfcc, axis=1)
    std_features = np.std(mfcc, axis=1)

    features = np.concatenate([
        mean_features,
        std_features
    ])

    return features.astype(np.float32)


if __name__ == "__main__":
    try:
        from voice_detection.dataset_loader import load_protocol
    except ImportError:
        from dataset_loader import load_protocol

    # Test on one recording first
    sample = data.iloc[0]

    features = extract_features(sample["audio_path"])

    print("Feature extraction successful!")
    print("File ID:", sample["file_id"])
    print("Label:", sample["label"])
    print("Feature vector shape:", features.shape)
    print("Feature data type:", features.dtype)
    print("First five features:", features[:5])