"""
Unit & Integration Tests for SAFE CALL End-to-End Pipeline & Explainability.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import SafeCallPipeline, get_pipeline, run_safecall
from explainability.guidance_generator import generate_call_guidance


TEMP_DIR = PROJECT_ROOT / "tests" / "temp_audio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def pipeline():
    return get_pipeline()


@pytest.fixture
def sample_wav():
    sr = 16000
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = TEMP_DIR / "test_pipeline_audio.wav"
    sf.write(path, audio, sr)
    return path


def test_pipeline_initialization(pipeline):
    assert pipeline is not None
    assert pipeline.assessor is not None
    assert pipeline.guidance_gen is not None


def test_pipeline_process_call_structure(pipeline, sample_wav):
    result = pipeline.process_call(sample_wav)

    assert result["status"] == "success"
    assert "guidance" in result
    assert "fusion" in result
    assert "voice_detection" in result
    assert "transcription" in result
    assert "scam_detection" in result

    # Check guidance details
    guidance = result["guidance"]
    assert "title" in guidance
    assert "alert_type" in guidance
    assert "risk_level" in guidance
    assert "action_items" in guidance
    assert isinstance(guidance["action_items"], list)
    assert len(guidance["action_items"]) > 0

    # Check fusion details
    fusion = result["fusion"]
    assert 0.0 <= fusion["composite_risk_score"] <= 1.0
    assert fusion["risk_level"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")
    assert fusion["primary_threat"] in (
        "AI_CLONE_FRAUD",
        "HUMAN_SCAM_CALL",
        "SUSPICIOUS_SYNTHETIC_VOICE",
        "BENIGN_CALL",
    )


def test_convenience_function(sample_wav):
    result = run_safecall(sample_wav)
    assert result["status"] == "success"


def test_guidance_critical_ai_fraud():
    mock_analysis = {
        "fusion": {
            "risk_level": "CRITICAL",
            "primary_threat": "AI_CLONE_FRAUD",
            "composite_risk_score": 0.95,
            "voice_is_spoof": True,
            "scam_is_detected": True,
        },
        "voice_detection": {
            "prediction": "Likely AI-generated/Cloned",
            "spoof_probability": 0.92,
        },
        "scam_detection": {
            "is_scam": True,
            "scam_probability": 0.98,
            "matched_categories": ["financial_credential", "urgency"],
            "red_flags": [{"category": "financial_credential", "matched_phrase": "share OTP"}],
        },
    }
    guidance = generate_call_guidance(mock_analysis)

    assert guidance["alert_type"] == "CRITICAL"
    assert "AI Voice Clone" in guidance["title"]
    assert any("HANG UP" in act for act in guidance["action_items"])
    assert any("OTP" in act for act in guidance["action_items"])


def test_guidance_benign_call():
    mock_analysis = {
        "fusion": {
            "risk_level": "LOW",
            "primary_threat": "BENIGN_CALL",
            "composite_risk_score": 0.05,
            "voice_is_spoof": False,
            "scam_is_detected": False,
        },
        "voice_detection": {
            "prediction": "Likely Human-generated",
            "spoof_probability": 0.05,
        },
        "scam_detection": {
            "is_scam": False,
            "scam_probability": 0.0,
            "matched_categories": [],
            "red_flags": [],
        },
    }
    guidance = generate_call_guidance(mock_analysis)

    assert guidance["alert_type"] == "SAFE"
    assert "NORMAL" in guidance["title"].upper()
