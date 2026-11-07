
import numpy as np
import librosa

from dataset_loader import load_protocol


SAMPLE_RATE = 16000
N_MFCC = 20


def extract_features(audio_path):
    # Load audio as mono at 16 kHz
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Extract MFCCs: 20 coefficients across time
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
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
    # Load the training metadata
    data = load_protocol("train")

    # Test on one recording first
    sample = data.iloc[0]

    features = extract_features(sample["audio_path"])

    print("Feature extraction successful!")
    print("File ID:", sample["file_id"])
    print("Label:", sample["label"])
    print("Feature vector shape:", features.shape)
    print("Feature data type:", features.dtype)
    print("First five features:", features[:5])