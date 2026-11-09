"""
Speech-to-Text Transcription Module for SAFE CALL.

Uses OpenAI Whisper to perform robust automatic speech recognition (ASR)
on caller audio, extracting clean transcripts, detected language, and timestamped
segments for downstream scam intent analysis.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import numpy as np
import torch
import whisper

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.audio_preprocessing import preprocess_audio

DEFAULT_MODEL_NAME = "base"
TARGET_SAMPLE_RATE = 16000


class SpeechTranscriber:
    """
    Automatic Speech Recognition (ASR) pipeline using OpenAI Whisper.
    Converts audio into text, detects language, and provides timestamped segments.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        """Loads the Whisper model onto the configured device."""
        try:
            return whisper.load_model(self.model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Whisper model '{self.model_name}' on {self.device}: {exc}"
            ) from exc

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file or 1D float32 numpy array into text.

        Parameters:
            audio_input: Path to audio file or 1D float32 numpy array (16 kHz).
            language: Optional language code (e.g., 'en'). If None, automatically detected.

        Returns:
            Dict containing:
                - text: Transcribed text string
                - language: Detected/specified language code (e.g., 'en')
                - segments: List of segment dicts with start, end, text, and confidence
                - audio_duration_seconds: Total audio duration in seconds
                - model_name: Name of model used
                - device: Hardware device used ('cpu' or 'cuda')
        """
        # 1. Load and preprocess audio to 16 kHz mono float32
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio, sr = preprocess_audio(audio_path, target_sr=TARGET_SAMPLE_RATE)
        elif isinstance(audio_input, np.ndarray):
            if audio_input.ndim != 1 or len(audio_input) == 0:
                raise ValueError("Audio array must be a non-empty 1D array.")
            audio = audio_input.astype(np.float32)
            sr = TARGET_SAMPLE_RATE
        else:
            raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

        duration = round(len(audio) / sr, 3)
        if duration < 0.2:
            raise ValueError(
                f"Audio recording too short ({duration:.2f}s). "
                "At least 0.2 seconds required for transcription."
            )

        # 2. Whisper transcription
        use_fp16 = self.device == "cuda"
        transcribe_kwargs: Dict[str, Any] = {"fp16": use_fp16}
        if language is not None:
            transcribe_kwargs["language"] = language

        result = self.model.transcribe(audio, **transcribe_kwargs)

        # 3. Format segments with confidence estimation
        formatted_segments: List[Dict[str, Any]] = []
        for seg in result.get("segments", []):
            avg_logprob = seg.get("avg_logprob", 0.0)
            confidence = float(np.clip(np.exp(avg_logprob), 0.0, 1.0))
            formatted_segments.append({
                "id": seg.get("id"),
                "start": round(float(seg.get("start", 0.0)), 2),
                "end": round(float(seg.get("end", 0.0)), 2),
                "text": seg.get("text", "").strip(),
                "confidence": round(confidence, 4),
                "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0)), 4),
            })

        cleaned_text = result.get("text", "").strip()

        return {
            "text": cleaned_text,
            "language": result.get("language", "unknown"),
            "segments": formatted_segments,
            "audio_duration_seconds": duration,
            "model_name": f"whisper-{self.model_name}",
            "device": self.device,
        }


# Global singleton instance for efficient reuse across API calls
_TRANSCRIBER_INSTANCE: Optional[SpeechTranscriber] = None


def get_transcriber(model_name: str = DEFAULT_MODEL_NAME) -> SpeechTranscriber:
    """Retrieve or initialize the singleton SpeechTranscriber instance."""
    global _TRANSCRIBER_INSTANCE
    if _TRANSCRIBER_INSTANCE is None or _TRANSCRIBER_INSTANCE.model_name != model_name:
        _TRANSCRIBER_INSTANCE = SpeechTranscriber(model_name=model_name)
    return _TRANSCRIBER_INSTANCE


def transcribe_audio(
    audio_path: Union[str, Path, np.ndarray],
    model_name: str = DEFAULT_MODEL_NAME,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function to transcribe an audio file or signal."""
    transcriber = get_transcriber(model_name=model_name)
    return transcriber.transcribe(audio_path, language=language)


if __name__ == "__main__":
    transcriber = get_transcriber("base")
    print("SpeechTranscriber initialized successfully.")
    print("Device:", transcriber.device)
    print("Model:", transcriber.model_name)

