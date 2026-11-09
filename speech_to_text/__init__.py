"""
Speech-to-Text module for SAFE CALL.
Transcribes audio into text for downstream scam intent and semantic analysis.
"""

from speech_to_text.transcriber import SpeechTranscriber, get_transcriber, transcribe_audio

__all__ = ["SpeechTranscriber", "get_transcriber", "transcribe_audio"]

