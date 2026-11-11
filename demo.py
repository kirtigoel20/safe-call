"""
SAFE CALL: Simple, Accurate, and Interactive Demonstration CLI.

Supports direct command-line arguments:
    python demo.py "path/to/recording.m4a"
Or interactive menu mode:
    python demo.py
"""

import sys
import os
from pathlib import Path
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import get_pipeline
from scam_detection.scam_detector import get_scam_detector


def print_banner():
    print("=" * 72)
    print("   🛡️   SAFE CALL: AI Voice Clone & Scam Detection System   🛡️")
    print("        Fast, Accurate Call Protection & Guidance              ")
    print("=" * 72)


def display_simple_card(result: dict):
    """
    Displays a clean, simple, highly readable report for teachers & users.
    """
    guidance = result.get("guidance", {})
    fusion = result.get("fusion", {})
    voice = result.get("voice_detection", {})
    trans = result.get("transcription", {})
    scam = result.get("scam_detection", {})

    risk_level = guidance.get("risk_level", "LOW")
    title = guidance.get("title", "ANALYSIS COMPLETE")

    # Visual risk banner
    print("\n" + "=" * 72)
    print(f"  🚨 {title.upper()}")
    print("=" * 72)

    # 1. Voice Clone Check
    is_spoof = voice.get("raw_label", 0) == 1
    spoof_pct = voice.get("spoof_probability", 0.0) * 100
    human_pct = voice.get("bonafide_probability", 0.0) * 100

    print("\n[1] 🎙️ VOICE DEEPFAKE STATUS")
    if is_spoof:
        print(f"    • Verdict:       ⚠️ LIKELY AI-GENERATED / CLONED VOICE")
        print(f"    • AI Confidence: {spoof_pct:.1f}%")
        print(f"    • Finding:       Synthetic acoustic vocoder artifacts detected.")
    else:
        print(f"    • Verdict:       ✅ GENUINE HUMAN VOICE")
        print(f"    • Authenticity:  {human_pct:.1f}%")
        print(f"    • Finding:       Natural acoustic vocal resonances detected.")

    # 2. Transcription
    print("\n[2] 📝 SPEECH TRANSCRIPTION")
    transcript = trans.get("text") or "(No clear speech detected / quiet audio)"
    print(f"    \"{transcript}\"")
    print(f"    • Language:      {trans.get('language', 'en').upper()}  |  Duration: {voice.get('audio_duration_seconds', 0)}s")

    # 3. Scam Red Flags
    print("\n[3] 🚩 SCAM INTENT ANALYSIS")
    is_scam = scam.get("is_scam", False)
    red_flags = scam.get("red_flags", [])

    if is_scam and red_flags:
        print(f"    • Status:        ⚠️ SUSPICIOUS LANGUAGE DETECTED")
        print(f"    • Red Flags Caught ({len(red_flags)}):")
        for rf in red_flags:
            cat_name = rf['category'].replace('_', ' ').title()
            print(f"      - [{rf['severity']}] {cat_name}: \"{rf['matched_phrase']}\"")
    else:
        print(f"    • Status:        ✅ No deceptive keywords or pressure tactics found.")

    # 4. Final Verdict & Actionable Guidance
    print("\n[4] 🛡️ FINAL THREAT LEVEL & RECOMMENDED ACTION")
    print(f"    • Threat Level:  {risk_level} (Score: {fusion.get('composite_risk_score', 0.0)}/1.0)")
    print(f"    • Primary Threat:{fusion.get('primary_threat', 'BENIGN_CALL')}")
    print(f"\n    👉 WHAT TO DO:")
    for i, action in enumerate(guidance.get("action_items", []), 1):
        print(f"       {i}. {action}")

    hotlines = guidance.get("hotline_info", {})
    if hotlines and risk_level in ("HIGH", "CRITICAL"):
        print(f"\n    📞 OFFICIAL HELPLINE:")
        for k, v in hotlines.items():
            print(f"       • {v}")

    print("=" * 72 + "\n")


def analyze_file(audio_path_str: str):
    cleaned_path = audio_path_str.strip().strip('"').strip("'")
    path = Path(cleaned_path)
    if not path.is_file():
        print(f"\n❌ Error: File not found: '{cleaned_path}'")
        print("Tip: Make sure the file path is correct (e.g. C:\\path\\to\\audio.m4a or audio.wav).")
        return

    print(f"\n🔄 Processing '{path.name}' through 6-stage SAFE CALL pipeline...")
    pipeline = get_pipeline()
    result = pipeline.process_call(path)
    display_simple_card(result)


def run_interactive():
    print_banner()

    # Check if user supplied file path directly as CLI argument:
    # e.g., python demo.py "sample.m4a"
    if len(sys.argv) > 1:
        analyze_file(sys.argv[1])
        return

    print("\nChoose an option:")
    print("  1. Analyze an Audio File (.m4a, .wav, .mp3, .aac, .flac)")
    print("  2. Test Scam Text Detector (Type or paste conversation)")
    print("  3. View Evaluation Metrics (ASVspoof 2019 Benchmark)")
    print("  4. Exit")

    choice = input("\nEnter choice (1-4) [default 1]: ").strip() or "1"

    if choice == "1":
        path_input = input("\nEnter or drag-and-drop audio file path: ").strip()
        if not path_input:
            # Fall back to preset test audio if empty
            path_input = str(PROJECT_ROOT / "tests" / "temp_audio" / "test_synth_speech.wav")
            print(f"Using default test audio: {path_input}")
        analyze_file(path_input)

    elif choice == "2":
        detector = get_scam_detector()
        print("\nEnter or paste dialogue (or press Enter for bank scam example):")
        text = input("> ").strip()
        if not text:
            text = (
                "Hello, I am calling from State Bank fraud division. "
                "Your account is blocked right now. Share your OTP and card CVV immediately or police will take legal action."
            )
            print(f"Testing default: \"{text}\"")

        res = detector.analyze(text)
        print("\n" + "=" * 60)
        print(f"  SCAM ANALYSIS RESULT: {res['risk_level']} (Score: {res['scam_probability'] * 100:.1f}%)")
        print("=" * 60)
        print(f"Is Scam:          {res['is_scam']}")
        print(f"Matched Threat:   {res['matched_categories']}")
        print("Red Flags Caught:")
        for rf in res.get("red_flags", []):
            print(f"  - [{rf['severity']}] \"{rf['matched_phrase']}\"")
        print(f"Summary:          {res['summary']}")
        print("=" * 60 + "\n")

    elif choice == "3":
        metrics_file = PROJECT_ROOT / "data" / "eval_results" / "voice_eval_metrics.json"
        if metrics_file.is_file():
            import json
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            print("\n" + "=" * 60)
            print("   ASVSPOOF 2019 LOGICAL ACCESS EVALUATION")
            print("=" * 60)
            print(f"Recordings Evaluated: {metrics.get('total_recordings'):,}")
            print(f"ROC-AUC:              {metrics.get('roc_auc') * 100:.2f}%")
            print(f"Equal Error Rate:     {metrics.get('equal_error_rate') * 100:.2f}%")
            print(f"Calibrated Threshold: {metrics.get('calibrated_threshold')}")
            cal = metrics.get("metrics_at_calibrated_threshold", {})
            print(f"Balanced Accuracy:    {cal.get('balanced_accuracy') * 100:.2f}%")
            print(f"Spoof Precision:      {cal.get('spoof', {}).get('precision') * 100:.2f}%")
            print(f"Spoof Recall:         {cal.get('spoof', {}).get('recall') * 100:.2f}%")
            print("=" * 60 + "\n")
        else:
            print("Metrics file not found.")

    else:
        print("Exiting.")


if __name__ == "__main__":
    run_interactive()
