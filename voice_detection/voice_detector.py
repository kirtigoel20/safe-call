"""
Voice Deepfake Detection Module for SAFE CALL.

Implements the complete inference pipeline:
1. Audio validation & preprocessing (16 kHz mono float32)
2. 40-dimensional MFCC acoustic feature extraction (mean + std)
3. Biological vocal naturalness verification (autocorrelation pitch jitter & energy dynamics)
4. Model inference with calibrated Equal Error Rate (EER) thresholding
5. Robust human-readable and structured risk outputs
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
from numpy.lib.stride_tricks import as_strided
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_preprocessing import preprocess_audio
from voice_detection.feature_extraction import extract_features, SAMPLE_RATE

# Default project paths
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "safecall_logistic_regression.joblib"

# Equal Error Rate (EER) threshold derived from ASVspoof 2019 dev evaluation (EER: 17.35%)
DEFAULT_EER_THRESHOLD = 0.6107

# User-facing classification labels
LABEL_HUMAN = "Likely Human-generated"
LABEL_SPOOF = "Likely AI-generated/Cloned"


def _compute_vocal_naturalness(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Computes biological human vocal cord naturalness in real time (<10ms).
    Real human voices exhibit pitch micro-jitter (natural vocal fold muscle tremor)
    and dynamic energy contrast between voiced phonation and unvoiced stops/breathing.
    Synthetic TTS and voice clones have unnaturally rigid pitch and uniform phonation.
    """
    if len(audio) < sr * 0.2:
        return 0.0

    # 1. Pitch Autocorrelation at 4 kHz
    audio_4k = audio[::4]
    sr_4k = sr // 4
    min_lag = int(sr_4k / 400)  # 400 Hz max pitch
    max_lag = int(sr_4k / 70)   # 70 Hz min pitch
    frame_len_pitch = int(sr_4k * 0.04)  # 40 ms
    hop_len_pitch = int(sr_4k * 0.02)    # 20 ms

    num_p_frames = (len(audio_4k) - frame_len_pitch) // hop_len_pitch
    pitches = []
    if num_p_frames >= 5:
        for i in range(num_p_frames):
            frame = audio_4k[i * hop_len_pitch : i * hop_len_pitch + frame_len_pitch]
            energy = np.sum(frame ** 2)
            if energy < 1e-4:
                continue
            corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
            search_region = corr[min_lag:max_lag]
            if len(search_region) == 0:
                continue
            peak_lag = min_lag + int(np.argmax(search_region))
            peak_val = corr[peak_lag] / (corr[0] + 1e-6)
            if peak_val > 0.35:
                pitches.append(sr_4k / peak_lag)

    if len(pitches) < 5:
        return 0.0

    jitter = float(np.mean(np.abs(np.diff(pitches)))) if len(pitches) > 1 else 0.0
    pitch_std = float(np.std(pitches))

    # 2. Energy and Voicing Dynamics
    frame_len_dyn = int(sr * 0.025)
    hop_len_dyn = int(sr * 0.010)
    num_d_frames = 1 + (len(audio) - frame_len_dyn) // hop_len_dyn
    frames = as_strided(
        audio,
        shape=(num_d_frames, frame_len_dyn),
        strides=(audio.strides[0] * hop_len_dyn, audio.strides[0]),
    )
    energies = np.sqrt(np.mean(frames ** 2, axis=1))
    max_energy = np.max(energies)
    if max_energy < 1e-6:
        return 0.0

    voiced_ratio = float(np.mean(energies > (0.1 * max_energy)))
    energy_contrast = float(np.std(energies) / (np.mean(energies) + 1e-6))

    # 3. Composite Biological Naturalness Score [0.0 to 1.0]
    j_score = float(np.clip((jitter - 11.0) / 5.0, 0.0, 1.0))
    std_score = float(np.clip((pitch_std - 38.0) / 12.0, 0.0, 1.0))
    ec_score = float(np.clip((energy_contrast - 1.05) / 0.30, 0.0, 1.0))
    vr_score = float(np.clip((0.55 - voiced_ratio) / 0.15, 0.0, 1.0))

    naturalness = 0.35 * j_score + 0.35 * std_score + 0.15 * ec_score + 0.15 * vr_score
    return float(np.clip(naturalness, 0.0, 1.0))


class VoiceDeepfakeDetector:
    """
    End-to-end Voice Deepfake Detector for SAFE CALL.
    Loads trained acoustic classifier and provides calibrated inference on audio files or signals.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        threshold: float = DEFAULT_EER_THRESHOLD,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.threshold = float(threshold)
        self.model = self._load_model()

    def _load_model(self):
        """Loads and verifies the trained scikit-learn model pipeline."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Voice detection model not found at: {self.model_path}. "
                "Ensure trained weights exist in the models/ directory."
            )
        try:
            model = joblib.load(self.model_path)
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load voice detection model from {self.model_path}: {exc}"
            ) from exc

    def predict(
        self,
        audio_input: Union[str, Path, np.ndarray],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Perform voice deepfake inference on an audio file or numpy array.

        Parameters:
            audio_input: File path (str/Path) or 1D float32 numpy audio array.
            threshold: Optional custom decision threshold for spoof detection.

        Returns:
            Dict containing:
                - prediction: "Likely Human-generated" | "Likely AI-generated/Cloned"
                - raw_label: 0 (bonafide) | 1 (spoof)
                - confidence: Float (0.0 to 1.0) probability of the predicted class
                - spoof_probability: Float (0.0 to 1.0)
                - bonafide_probability: Float (0.0 to 1.0)
                - naturalness_score: Float (0.0 to 1.0) biological vocal cord naturalness
                - audio_duration_seconds: Duration in seconds
                - sample_rate: Effective sample rate (16000 Hz)
                - threshold_used: The decision threshold applied
        """
        applied_threshold = float(threshold if threshold is not None else self.threshold)

        # 1. Preprocess audio & extract duration
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            audio, sr = preprocess_audio(audio_path, target_sr=SAMPLE_RATE)
        elif isinstance(audio_input, np.ndarray):
            if audio_input.ndim != 1 or len(audio_input) == 0:
                raise ValueError("Audio array must be non-empty 1D array.")
            audio = audio_input.astype(np.float32)
            sr = SAMPLE_RATE
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        duration = round(len(audio) / sr, 3)
        if duration < 0.2:
            raise ValueError(
                f"Audio recording too short ({duration:.2f}s). "
                "At least 0.2 seconds required for reliable feature extraction."
            )

        # 2. Extract 40-dimensional MFCC features (20 mean + 20 std)
        features = extract_features(audio, sr=sr)
        features_2d = features.reshape(1, -1)

        # 3. Raw acoustic model inference
        raw_probabilities = self.model.predict_proba(features_2d)[0]
        raw_spoof_prob = float(raw_probabilities[1])

        # 4. Biological Vocal Cord Naturalness Verification
        # Mitigates phone mic / room reverberation false positives on real human speech
        naturalness = _compute_vocal_naturalness(audio, sr=sr)
        calibrated_spoof = float(np.clip(raw_spoof_prob * (1.0 - 0.85 * naturalness), 0.0, 1.0))
        calibrated_bonafide = float(1.0 - calibrated_spoof)

        # 5. Decision using calibrated threshold
        is_spoof = calibrated_spoof >= applied_threshold

        if is_spoof:
            prediction = LABEL_SPOOF
            raw_label = 1
            confidence = calibrated_spoof
        else:
            prediction = LABEL_HUMAN
            raw_label = 0
            confidence = calibrated_bonafide

        return {
            "prediction": prediction,
            "raw_label": raw_label,
            "confidence": round(confidence, 4),
            "spoof_probability": round(calibrated_spoof, 4),
            "bonafide_probability": round(calibrated_bonafide, 4),
            "naturalness_score": round(naturalness, 4),
            "audio_duration_seconds": duration,
            "sample_rate": sr,
            "threshold_used": round(applied_threshold, 4),
            "feature_dim": features.shape[0],
        }


# Global singleton instance for efficient reuse across API calls
_DETECTOR_INSTANCE: Optional[VoiceDeepfakeDetector] = None


def get_voice_detector() -> VoiceDeepfakeDetector:
    """Retrieve or initialize the singleton VoiceDeepfakeDetector instance."""
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        _DETECTOR_INSTANCE = VoiceDeepfakeDetector()
    return _DETECTOR_INSTANCE


def predict_voice(
    audio_path: Union[str, Path],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience function to predict voice deepfake for an audio file."""
    detector = get_voice_detector()
    return detector.predict(audio_path, threshold=threshold)


if __name__ == "__main__":
    detector = get_voice_detector()
    print("VoiceDeepfakeDetector initialized successfully.")
    print("Model loaded from:", detector.model_path)
    print("Calibrated decision threshold (EER):", detector.threshold)
