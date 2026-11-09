"""
SAFE CALL: End-to-End Unified Call Analysis Pipeline.

Executes the complete multi-stage defense architecture:
1. Audio Preprocessing (16 kHz mono standardization)
2. Voice Deepfake Detection (ASVspoof-trained acoustic classifier)
3. Speech-to-Text Transcription (OpenAI Whisper)
4. Scam Intent & Coercion Analysis (NLP Multi-Pattern Categorizer)
5. Multi-Modal Risk Fusion (Composite Threat Scoring)
6. Explainable Safety Guidance & Actionable Countermeasures
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.risk_assessor import get_risk_assessor
from explainability.guidance_generator import get_guidance_generator


class SafeCallPipeline:
    """
    Unified end-to-end pipeline orchestrator for SAFE CALL.
    """

    def __init__(self):
        self.assessor = get_risk_assessor()
        self.guidance_gen = get_guidance_generator()

    def process_call(
        self,
        audio_input: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Processes an incoming audio recording or signal through the entire pipeline.

        Returns a comprehensive dictionary with all module outputs and explainability.
        """
        # 1. Multi-Modal Analysis (Voice + Transcription + Scam Intent + Fusion)
        analysis = self.assessor.analyze_call(audio_input)

        # 2. Explainability & Guidance Generation
        guidance = self.guidance_gen.generate_guidance(analysis)

        return {
            "status": "success",
            "guidance": guidance,
            "fusion": analysis["fusion"],
            "voice_detection": analysis["voice_detection"],
            "transcription": analysis["transcription"],
            "scam_detection": analysis["scam_detection"],
        }


_PIPELINE_INSTANCE = None


def get_pipeline() -> SafeCallPipeline:
    """Retrieve or initialize the singleton SafeCallPipeline."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = SafeCallPipeline()
    return _PIPELINE_INSTANCE


def run_safecall(
    audio_path: Union[str, Path, np.ndarray],
) -> Dict[str, Any]:
    """Convenience function to run the full SAFE CALL pipeline on an audio input."""
    pipeline = get_pipeline()
    return pipeline.process_call(audio_path)


if __name__ == "__main__":
    print("==================================================")
    print("     SAFE CALL: END-TO-END PIPELINE SYSTEM        ")
    print("==================================================")

    pipeline = get_pipeline()
    print("All modules loaded and ready.")

    # Test with synthetic speech wav if present
    test_audio = PROJECT_ROOT / "tests" / "temp_audio" / "test_synth_speech.wav"
    if test_audio.is_file():
        print(f"\nRunning test pipeline on: {test_audio.name}...")
        result = pipeline.process_call(test_audio)
        print("\n--- SAFETY GUIDANCE ---")
        print(f"Title:        {result['guidance']['title']}")
        print(f"Risk Level:   {result['guidance']['risk_level']}")
        print(f"Risk Score:   {result['guidance']['composite_risk_score']}")
        print(f"Reason:       {result['guidance']['summary_reason']}")
        print(f"Voice Status: {result['voice_detection']['prediction']}")
        print(f"Transcript:   \"{result['transcription']['text']}\"")
        print(f"Action Items: {result['guidance']['action_items']}")

