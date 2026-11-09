"""
Unit tests for Scam Detection Module (Phase 4).
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scam_detection.scam_detector import (
    ScamDetector,
    get_scam_detector,
    detect_scam,
)


@pytest.fixture(scope="module")
def detector():
    return get_scam_detector()


def test_detector_initialization(detector):
    assert detector is not None
    assert 0.0 < detector.threshold < 1.0


def test_empty_and_whitespace_text(detector):
    res_empty = detector.analyze("")
    assert res_empty["is_scam"] is False
    assert res_empty["scam_probability"] == 0.0
    assert res_empty["risk_level"] == "LOW"
    assert res_empty["matched_categories"] == []

    res_spaces = detector.analyze("     ")
    assert res_spaces["is_scam"] is False


def test_benign_conversation(detector):
    text = "Hey John, are we still meeting for lunch at the cafeteria at 1 PM?"
    res = detector.analyze(text)

    assert res["is_scam"] is False
    assert res["scam_probability"] < 0.2
    assert res["risk_level"] == "LOW"
    assert len(res["red_flags"]) == 0


def test_high_risk_banking_scam(detector):
    text = (
        "I am calling from your bank manager office. Your account is blocked. "
        "Please provide your one-time password OTP immediately to restore access."
    )
    res = detector.analyze(text)

    assert res["is_scam"] is True
    assert res["scam_probability"] >= 0.5
    assert res["risk_level"] in ("HIGH", "CRITICAL")
    assert "financial_credential" in res["matched_categories"]
    assert "urgency" in res["matched_categories"] or "impersonation" in res["matched_categories"]
    assert len(res["red_flags"]) >= 2


def test_threat_and_police_impersonation(detector):
    text = "This is crime branch police department. We have an arrest warrant for illegal transactions."
    res = detector.analyze(text)

    assert "impersonation" in res["matched_categories"]
    assert "threats_legal" in res["matched_categories"]
    assert res["scam_probability"] > 0.3


def test_convenience_function():
    res = detect_scam("Send me the OTP right now.")
    assert isinstance(res, dict)
    assert res["is_scam"] is True


def test_custom_threshold_override(detector):
    text = "Congratulations, you won a reward."
    # At very low threshold, should trigger is_scam
    res_low = detector.analyze(text, threshold=0.05)
    assert res_low["is_scam"] is True

    # At very high threshold, should not trigger is_scam
    res_high = detector.analyze(text, threshold=0.99)
    assert res_high["is_scam"] is False

