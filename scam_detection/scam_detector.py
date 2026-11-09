"""
Scam Detection & Intent Analysis Module for SAFE CALL.

Analyzes transcribed speech text to detect social engineering indicators:
- Credential & financial theft (OTP, CVV, bank account)
- False authority impersonation (Police, Bank manager, Tax, Support)
- Artificial urgency and pressure
- Intimidation & arrest threats
- Fraudulent prizes & refunds

Generates:
- Scam probability score (0.0 to 1.0)
- Categorical risk tier (LOW, MODERATE, HIGH, CRITICAL)
- Structured red flag evidence for explainability
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scam_detection.scam_patterns import SCAM_CATEGORIES, COMPILED_PATTERNS

DEFAULT_SCAM_THRESHOLD = 0.45


class ScamDetector:
    """
    NLP Intent and Scam Risk Analyzer for SAFE CALL.
    Identifies psychological coercion and fraud patterns in text transcripts.
    """

    def __init__(self, threshold: float = DEFAULT_SCAM_THRESHOLD):
        self.threshold = float(threshold)

    def analyze(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a transcribed text string for scam and social engineering indicators.

        Parameters:
            text: Input transcript string.
            threshold: Optional custom decision threshold (defaults to 0.45).

        Returns:
            Dict containing:
                - is_scam: Boolean flag
                - scam_probability: Float (0.0 to 1.0)
                - risk_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'
                - matched_categories: List of detected category names
                - red_flags: List of matched evidence dictionaries
                - red_flag_count: Total distinct pattern matches
                - threshold_used: Float threshold applied
                - summary: Human-readable summary of detected threats
        """
        applied_threshold = float(threshold if threshold is not None else self.threshold)

        if not text or not text.strip():
            return {
                "is_scam": False,
                "scam_probability": 0.0,
                "risk_level": "LOW",
                "matched_categories": [],
                "red_flags": [],
                "red_flag_count": 0,
                "threshold_used": applied_threshold,
                "summary": "No speech content provided for scam analysis.",
            }

        cleaned_text = text.strip()
        matched_categories: List[str] = []
        red_flags: List[Dict[str, Any]] = []
        accumulated_weight = 0.0

        for category, patterns in COMPILED_PATTERNS.items():
            cat_meta = SCAM_CATEGORIES[category]
            category_hits = []

            for pattern in patterns:
                for match in pattern.finditer(cleaned_text):
                    category_hits.append(match.group(0))

            if category_hits:
                # Deduplicate matching phrases while preserving order
                unique_hits = list(dict.fromkeys(category_hits))
                matched_categories.append(category)
                accumulated_weight += cat_meta["weight"]

                for hit in unique_hits:
                    red_flags.append({
                        "category": category,
                        "matched_phrase": hit,
                        "severity": cat_meta["severity"],
                        "description": cat_meta["description"],
                    })

        # Multi-factor compound boost:
        # Scammers frequently combine Impersonation + Urgency + Financial / Threat.
        # If multiple high-risk vectors occur together, boost the probability.
        combo_boost = 0.0
        if "financial_credential" in matched_categories and "urgency" in matched_categories:
            combo_boost += 0.25
        if "impersonation" in matched_categories and ("threats_legal" in matched_categories or "urgency" in matched_categories):
            combo_boost += 0.20
        if "financial_credential" in matched_categories and "impersonation" in matched_categories:
            combo_boost += 0.30

        raw_score = accumulated_weight + combo_boost
        scam_probability = float(np.clip(raw_score, 0.0, 1.0))
        is_scam = scam_probability >= applied_threshold

        # Determine risk level
        if scam_probability >= 0.75:
            risk_level = "CRITICAL"
        elif scam_probability >= 0.50:
            risk_level = "HIGH"
        elif scam_probability >= 0.25:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Generate human-readable summary
        if not red_flags:
            summary = "No obvious scam patterns or social engineering indicators detected."
        else:
            cat_names = [c.replace("_", " ").title() for c in matched_categories]
            summary = (
                f"Suspicious activity detected! Found {len(red_flags)} indicator(s) "
                f"across categories: {', '.join(cat_names)}."
            )

        return {
            "is_scam": is_scam,
            "scam_probability": round(scam_probability, 4),
            "risk_level": risk_level,
            "matched_categories": matched_categories,
            "red_flags": red_flags,
            "red_flag_count": len(red_flags),
            "threshold_used": round(applied_threshold, 4),
            "summary": summary,
        }


# Global singleton instance
_SCAM_DETECTOR_INSTANCE: Optional[ScamDetector] = None


def get_scam_detector() -> ScamDetector:
    """Retrieve or initialize the singleton ScamDetector instance."""
    global _SCAM_DETECTOR_INSTANCE
    if _SCAM_DETECTOR_INSTANCE is None:
        _SCAM_DETECTOR_INSTANCE = ScamDetector()
    return _SCAM_DETECTOR_INSTANCE


def detect_scam(
    text: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience function to analyze a transcript for scam indicators."""
    detector = get_scam_detector()
    return detector.analyze(text, threshold=threshold)


if __name__ == "__main__":
    detector = get_scam_detector()

    test_samples = [
        "Hello mom, I will be home around 7 PM for dinner tonight.",
        "This is SBI bank manager. Your account is blocked immediately. Share your OTP and debit card CVV to unblock right now or police will take legal action.",
        "Congratulations! You won the international lottery prize of 10 lakh rupees. Click link to claim your reward.",
    ]

    print("==================================================")
    print("      SAFE CALL - SCAM DETECTOR SAMPLE TEST       ")
    print("==================================================")
    for sample in test_samples:
        res = detector.analyze(sample)
        print(f"\nText: \"{sample}\"")
        print(f"  Risk Level:       {res['risk_level']}")
        print(f"  Scam Probability: {res['scam_probability']}")
        print(f"  Is Scam:          {res['is_scam']}")
        print(f"  Categories:       {res['matched_categories']}")
        print(f"  Summary:          {res['summary']}")

