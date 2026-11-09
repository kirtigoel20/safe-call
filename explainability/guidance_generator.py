"""
Explainable Safety Guidance & Recommendation Engine for SAFE CALL.

Translates complex acoustic deepfake metrics and NLP scam detections
into human-understandable explanations and actionable safety instructions.
"""

from typing import Dict, Any, List, Optional


class GuidanceGenerator:
    """
    Generates explainable rationale and proactive safety countermeasures
    for call recipients based on multi-modal threat assessments.
    """

    def generate_guidance(self, call_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explainable guidance from a full call analysis dictionary.

        Parameters:
            call_analysis: Output dictionary containing 'fusion', 'voice_detection',
                           'scam_detection', and optionally 'transcription'.

        Returns:
            Dict containing:
                - title: High-level alert banner title
                - alert_type: 'SAFE' | 'WARNING' | 'DANGER' | 'CRITICAL'
                - summary_reason: Concise 1-sentence explanation of why the call was flagged
                - acoustic_explanation: Explanation of voice clone assessment
                - semantic_explanation: Explanation of conversation content assessment
                - action_items: Bulleted list of immediate concrete safety actions
                - hotline_info: Official reporting details (e.g., Cyber Crime Helpline 1930)
        """
        fusion = call_analysis.get("fusion", {})
        voice = call_analysis.get("voice_detection", {})
        scam = call_analysis.get("scam_detection", {})

        risk_level = fusion.get("risk_level", "LOW")
        primary_threat = fusion.get("primary_threat", "BENIGN_CALL")
        is_spoof = fusion.get("voice_is_spoof", False)
        is_scam = fusion.get("scam_is_detected", False)
        spoof_prob = float(voice.get("spoof_probability", 0.0))
        scam_prob = float(scam.get("scam_probability", 0.0))
        red_flags = scam.get("red_flags", [])

        action_items: List[str] = []

        # 1. Acoustic Explanation
        if is_spoof:
            acoustic_exp = (
                f"Acoustic analysis detected synthetic audio artifacts and vocoder anomalies "
                f"with {spoof_prob * 100:.1f}% confidence. The voice exhibits signatures of an AI-generated or cloned voice."
            )
        else:
            acoustic_exp = (
                f"Acoustic features align with natural human speech patterns "
                f"({(1.0 - spoof_prob) * 100:.1f}% authentic human confidence)."
            )

        # 2. Semantic Explanation
        if is_scam:
            categories_str = ", ".join(
                [c.replace("_", " ").title() for c in scam.get("matched_categories", [])]
            )
            semantic_exp = (
                f"Transcript analysis identified high-risk social engineering techniques ({categories_str}) "
                f"with {len(red_flags)} detected red flag phrase(s)."
            )
        else:
            semantic_exp = "No deceptive language, financial solicitations, or urgency triggers detected in conversation."

        # 3. Decision Matrix for Guidance & Countermeasures
        if primary_threat == "AI_CLONE_FRAUD":
            title = "CRITICAL ALERT: Suspected AI Voice Clone Scam"
            alert_type = "CRITICAL"
            summary_reason = (
                "Both synthetic voice synthesis AND high-pressure scam language were detected. "
                "This indicates an active AI-cloned impersonation attack."
            )
            action_items.extend([
                "DO NOT share any OTP, PIN, password, or financial information.",
                "DO NOT transfer money, purchase gift cards, or approve payment requests.",
                "HANG UP IMMEDIATELY. If the caller claims to be a relative or colleague, call them back directly on their trusted saved phone number.",
                "Ask a personal question only the real person would know (establish a 'Safe Word').",
            ])

        elif primary_threat == "HUMAN_SCAM_CALL":
            title = "HIGH ALERT: Suspected Fraud / Scam Call"
            alert_type = "DANGER"
            summary_reason = (
                "The caller's voice appears human, but the conversation contains confirmed fraud patterns "
                "(such as credential harvesting, false authority, or urgency)."
            )
            action_items.extend([
                "Legitimate organizations (Banks, Police, Government) will NEVER demand OTPs or passwords over the phone.",
                "Do not install remote assistance applications (AnyDesk, TeamViewer).",
                "Hang up and independently verify by contacting the official organization via their verified website.",
            ])

        elif primary_threat == "SUSPICIOUS_SYNTHETIC_VOICE":
            title = "CAUTION: Synthetic / Automated Voice Detected"
            alert_type = "WARNING"
            summary_reason = (
                "The audio exhibits synthetic or robocall acoustic characteristics, though no overt scam demands have occurred yet."
            )
            action_items.extend([
                "Remain vigilant and avoid confirming personal or sensitive identity details.",
                "Be wary if the caller transitions into urgent requests or financial topics.",
            ])

        else:  # BENIGN_CALL
            title = "NORMAL: Call Assessed as Safe"
            alert_type = "SAFE"
            summary_reason = "Audio characteristics and conversation content are consistent with normal genuine communication."
            action_items.append("No suspicious indicators detected. Continue regular conversation safely.")

        return {
            "title": title,
            "alert_type": alert_type,
            "risk_level": risk_level,
            "composite_risk_score": fusion.get("composite_risk_score", 0.0),
            "summary_reason": summary_reason,
            "acoustic_explanation": acoustic_exp,
            "semantic_explanation": semantic_exp,
            "detected_red_flags": red_flags,
            "action_items": action_items,
            "hotline_info": {
                "india_cyber_crime": "National Cyber Crime Reporting Portal (1930 / cybercrime.gov.in)",
                "financial_fraud": "Contact your bank's official fraud helpline immediately to freeze accounts.",
            },
        }


_GUIDANCE_INSTANCE: Optional[GuidanceGenerator] = None


def get_guidance_generator() -> GuidanceGenerator:
    """Retrieve or initialize singleton GuidanceGenerator."""
    global _GUIDANCE_INSTANCE
    if _GUIDANCE_INSTANCE is None:
        _GUIDANCE_INSTANCE = GuidanceGenerator()
    return _GUIDANCE_INSTANCE


def generate_call_guidance(call_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to generate user-facing explainable guidance."""
    generator = get_guidance_generator()
    return generator.generate_guidance(call_analysis)

