"""
Multi-Modal Risk Assessment & Fusion Engine for SAFE CALL.

Fuses acoustic voice deepfake signals and semantic scam intent analysis
into a unified risk score (0.0 to 1.0) and categorical risk levels.

Prevents false alarms on compressed / microphone audio by cross-validating
acoustic findings against conversational intent.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from voice_detection.voice_detector import get_voice_detector, VoiceDeepfakeDetector
from speech_to_text.transcriber import get_transcriber, SpeechTranscriber
from scam_detection.scam_detector import get_scam_detector, ScamDetector

DEFAULT_WEIGHT_VOICE = 0.40
DEFAULT_WEIGHT_SCAM = 0.60


class MultiModalRiskAssessor:
    """
    Fuses acoustic deepfake detection and conversational scam intent.
    Computes unified threat assessment for real-time call protection.
    """

    def __init__(
        self,
        weight_voice: float = DEFAULT_WEIGHT_VOICE,
        weight_scam: float = DEFAULT_WEIGHT_SCAM,
        voice_detector: Optional[VoiceDeepfakeDetector] = None,
        transcriber: Optional[SpeechTranscriber] = None,
        scam_detector: Optional[ScamDetector] = None,
    ):
        total_w = weight_voice + weight_scam
        self.w_voice = weight_voice / total_w
        self.w_scam = weight_scam / total_w

        self.voice_detector = voice_detector or get_voice_detector()
        self.transcriber = transcriber or get_transcriber()
        self.scam_detector = scam_detector or get_scam_detector()

    def fuse(
        self,
        voice_result: Dict[str, Any],
        scam_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fuse individual module results into an overall risk profile.

        Parameters:
            voice_result: Dictionary output from VoiceDeepfakeDetector.predict()
            scam_result: Dictionary output from ScamDetector.analyze()

        Returns:
            Dict containing unified risk score, risk level, threat category, and recommendations.
        """
        p_voice = float(voice_result.get("spoof_probability", 0.0))
        p_scam = float(scam_result.get("scam_probability", 0.0))
        is_voice_spoof = bool(voice_result.get("raw_label", 0) == 1)
        is_scam_text = bool(scam_result.get("is_scam", False))

        # Base weighted fusion
        base_score = (self.w_voice * p_voice) + (self.w_scam * p_scam)

        # Smart Multi-Modal Contextual Multipliers:
        # 1. AI Clone + Scam Language = CRITICAL EMERGENCY
        if is_voice_spoof and is_scam_text:
            synergy_boost = 0.25
            primary_threat = "AI_CLONE_FRAUD"

        # 2. Human Voice + Scam Language = CONFIRMED HUMAN FRAUD
        elif not is_voice_spoof and is_scam_text:
            synergy_boost = 0.15
            primary_threat = "HUMAN_SCAM_CALL"

        # 3. Synthetic voice detected BUT conversation is completely normal:
        #    Mitigates false alarms from lossy phone codecs (.m4a), room echo, or robocalls.
        #    Caps composite threat at MODERATE rather than creating a false emergency.
        elif is_voice_spoof and not is_scam_text:
            synergy_boost = -0.15
            primary_threat = "SUSPICIOUS_SYNTHETIC_VOICE"

        # 4. Human voice + Normal talk = SAFE BENIGN CALL
        else:
            synergy_boost = 0.0
            primary_threat = "BENIGN_CALL"

        composite_risk = float(np.clip(base_score + synergy_boost, 0.0, 1.0))

        # If conversation is completely free of scam language, cap risk to MODERATE
        if not is_scam_text:
            composite_risk = min(composite_risk, 0.35)

        # Categorical Risk Tiers
        if composite_risk >= 0.75:
            risk_level = "CRITICAL"
            requires_action = True
        elif composite_risk >= 0.50:
            risk_level = "HIGH"
            requires_action = True
        elif composite_risk >= 0.20:
            risk_level = "MODERATE"
            requires_action = False
        else:
            risk_level = "LOW"
            requires_action = False

        return {
            "composite_risk_score": round(composite_risk, 4),
            "risk_level": risk_level,
            "primary_threat": primary_threat,
            "requires_immediate_action": requires_action,
            "voice_contribution": round(p_voice, 4),
            "scam_contribution": round(p_scam, 4),
            "voice_is_spoof": is_voice_spoof,
            "scam_is_detected": is_scam_text,
        }

    def analyze_call(
        self,
        audio_input: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Execute end-to-end multi-modal pipeline on an incoming audio recording:
        1. Voice deepfake detection
        2. Speech-to-text transcription
        3. Scam language analysis
        4. Multi-modal fusion
        """
        # 1. Voice Deepfake Detection
        voice_result = self.voice_detector.predict(audio_input)

        # 2. Speech-to-Text Transcription
        transcription_result = self.transcriber.transcribe(audio_input)
        transcript_text = transcription_result.get("text", "")

        # 3. Scam Intent Analysis
        scam_result = self.scam_detector.analyze(transcript_text)

        # 4. Multi-modal Fusion
        fusion_result = self.fuse(voice_result, scam_result)

        return {
            "fusion": fusion_result,
            "voice_detection": voice_result,
            "transcription": transcription_result,
            "scam_detection": scam_result,
        }


_RISK_ASSESSOR_INSTANCE: Optional[MultiModalRiskAssessor] = None


def get_risk_assessor() -> MultiModalRiskAssessor:
    """Retrieve or initialize singleton MultiModalRiskAssessor."""
    global _RISK_ASSESSOR_INSTANCE
    if _RISK_ASSESSOR_INSTANCE is None:
        _RISK_ASSESSOR_INSTANCE = MultiModalRiskAssessor()
    return _RISK_ASSESSOR_INSTANCE


def assess_call_risk(
    audio_path: Union[str, Path, np.ndarray],
) -> Dict[str, Any]:
    """Convenience function to run end-to-end multi-modal call risk assessment."""
    assessor = get_risk_assessor()
    return assessor.analyze_call(audio_path)
