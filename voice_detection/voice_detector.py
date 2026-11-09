"""
Voice Deepfake Detection Module for SAFE CALL.

Implements the complete inference pipeline:
1. Audio validation
2. Preprocessing & resampling (16 kHz mono)
3. 40-dimensional MFCC feature extraction (mean + std)
4. Model inference using calibrated decision thresholds
5. Human-readable and structured risk outputs
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_preprocessing import preprocess_audio, load_audio
from voice_detection.feature_extraction import extract_features, SAMPLE_RATE

# Default project paths
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "safecall_logistic_regression.joblib"

# Equal Error Rate (EER) threshold derived from ASVspoof 2019 dev evaluation (EER: 17.35%)
DEFAULT_EER_THRESHOLD = 0.6107

# User-facing classification labels (strict specification)
LABEL_HUMAN = "Likely Human-generated"
LABEL_SPOOF = "Likely AI-generated/Cloned"


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
                       Defaults to the model's calibrated EER threshold.

        Returns:
            Dict containing:
                - prediction: "Likely Human-generated" | "Likely AI-generated/Cloned"
                - raw_label: 0 (bonafide) | 1 (spoof)
                - confidence: Float (0.0 to 1.0) probability of the predicted class
                - spoof_probability: Float (0.0 to 1.0) probability of audio being synthetic/cloned
                - bonafide_probability: Float (0.0 to 1.0) probability of audio being genuine human
                - audio_duration_seconds: Duration of the evaluated audio in seconds
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

        # 3. Model inference
        # Class 0: Bonafide (Real), Class 1: Spoof (AI / Clone)
        probabilities = self.model.predict_proba(features_2d)[0]
        prob_bonafide = float(probabilities[0])
        prob_spoof = float(probabilities[1])

        # 4. Decision using calibrated threshold
        is_spoof = prob_spoof >= applied_threshold

        if is_spoof:
            prediction = LABEL_SPOOF
            raw_label = 1
            confidence = prob_spoof
        else:
            prediction = LABEL_HUMAN
            raw_label = 0
            confidence = prob_bonafide

        return {
            "prediction": prediction,
            "raw_label": raw_label,
            "confidence": round(confidence, 4),
            "spoof_probability": round(prob_spoof, 4),
            "bonafide_probability": round(prob_bonafide, 4),
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
    import sys

    detector = get_voice_detector()
    print("VoiceDeepfakeDetector initialized successfully.")
    print("Model loaded from:", detector.model_path)
    print("Calibrated decision threshold (EER):", detector.threshold)

    # Test on existing test audio if available
    sample_flac = Path(r"C:\Users\Ishan\Datasets\ASVspoof2019\LA\ASVspoof2019_LA_train\flac\LA_T_1138215.flac")
    if sample_flac.is_file():
        result = detector.predict(sample_flac)
        print("\nTest Sample Inference Result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
