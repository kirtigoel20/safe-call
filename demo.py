"""
Interactive Demonstration CLI for SAFE CALL.

Allows testing preset scenarios or custom audio files with colored terminal output,
demonstrating the complete 6-stage AI defense pipeline to evaluators and teachers.
"""

import sys
import time
from pathlib import Path
import soundfile as sf
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import get_pipeline
from scam_detection.scam_detector import get_scam_detector


def print_banner():
    print("=" * 70)
    print("   🛡️   SAFE CALL: AI Voice Clone & Scam Detection System   🛡️   ")
    print("      Multi-Modal Real-Time Call Protection & Guidance          ")
    print("=" * 70)


def create_demo_audio(filename: str, duration: float = 2.0, freq: float = 440.0) -> Path:
    """Creates a temporary audio file for testing if none exists."""
    demo_dir = PROJECT_ROOT / "tests" / "temp_audio"
    demo_dir.mkdir(parents=True, exist_ok=True)
    path = demo_dir / filename
    if not path.is_file():
        sr = 16000
        t = np.arange(int(sr * duration)) / sr
        audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(path, audio, sr)
    return path


def display_result(result: dict):
    guidance = result.get("guidance", {})
    fusion = result.get("fusion", {})
    voice = result.get("voice_detection", {})
    trans = result.get("transcription", {})
    scam = result.get("scam_detection", {})

    print("\n" + "#" * 70)
    print(f"  {guidance.get('title', 'ANALYSIS REPORT')}")
    print("#" * 70)

    print(f"\n[OVERALL THREAT ASSESSMENT]")
    print(f"  Risk Level:          {guidance.get('risk_level')}")
    print(f"  Composite Risk Score:{fusion.get('composite_risk_score')} / 1.00")
    print(f"  Threat Category:     {fusion.get('primary_threat')}")
    print(f"  Summary:             {guidance.get('summary_reason')}")

    print(f"\n[STAGE 1 & 2: ACOUSTIC VOICE DETECTION]")
    print(f"  Prediction:          {voice.get('prediction')}")
    print(f"  Spoof Probability:   {voice.get('spoof_probability') * 100:.2f}%")
    print(f"  Human Probability:   {voice.get('bonafide_probability') * 100:.2f}%")
    print(f"  EER Threshold Used:  {voice.get('threshold_used')}")
    print(f"  Acoustic Details:    {guidance.get('acoustic_explanation')}")

    print(f"\n[STAGE 3: SPEECH-TO-TEXT TRANSCRIPTION]")
    transcript = trans.get("text") or "(No spoken speech or quiet audio)"
    print(f"  Transcript:          \"{transcript}\"")
    print(f"  Detected Language:   {trans.get('language')}")
    print(f"  Audio Duration:      {voice.get('audio_duration_seconds')}s")

    print(f"\n[STAGE 4: SCAM INTENT ANALYSIS]")
    print(f"  Is Scam Intent:      {scam.get('is_scam')}")
    print(f"  Scam Probability:    {scam.get('scam_probability') * 100:.2f}%")
    print(f"  Matched Categories:  {scam.get('matched_categories')}")
    red_flags = scam.get("red_flags", [])
    if red_flags:
        print("  Red Flags Caught:")
        for rf in red_flags:
            print(f"    - [{rf['severity']}] {rf['category'].upper()}: \"{rf['matched_phrase']}\"")
    else:
        print("  Red Flags Caught:    None")

    print(f"\n[STAGE 5 & 6: RECOMMENDED SAFETY ACTIONS]")
    for i, action in enumerate(guidance.get("action_items", []), 1):
        print(f"  {i}. {action}")

    hotlines = guidance.get("hotline_info", {})
    if hotlines:
        print(f"\n[OFFICIAL HELPLINE]")
        for k, v in hotlines.items():
            print(f"  • {v}")
    print("#" * 70 + "\n")


def run_interactive():
    print_banner()
    pipeline = get_pipeline()

    print("\nSelect a Demonstration Scenario:")
    print("  1. Run Complete Pipeline on Preset Sample Audio")
    print("  2. Test Live Scam Intent Analyzer with Custom Text")
    print("  3. Run Complete Pipeline on Custom Audio File (.wav / .flac)")
    print("  4. View Model Evaluation Metrics & Benchmark Summary")
    print("  5. Exit")

    choice = input("\nEnter choice (1-5) [default 1]: ").strip() or "1"

    if choice == "1":
        sample_path = create_demo_audio("demo_sample.wav")
        print(f"\nAnalyzing audio file: {sample_path}...")
        res = pipeline.process_call(sample_path)
        display_result(res)

    elif choice == "2":
        scam_detector = get_scam_detector()
        print("\nEnter or paste caller dialogue to test (or press Enter for default banking scam):")
        text = input("> ").strip()
        if not text:
            text = (
                "Hello, I am calling from State Bank fraud prevention. "
                "Your account is blocked right now due to suspicious activity. "
                "Share your one-time password OTP immediately to restore access."
            )
            print(f"Testing default text: \"{text}\"")
        res = scam_detector.analyze(text)
        print("\n--- SCAM DETECTION RESULTS ---")
        print(f"Is Scam:          {res['is_scam']}")
        print(f"Risk Level:       {res['risk_level']}")
        print(f"Scam Probability: {res['scam_probability'] * 100:.2f}%")
        print(f"Categories:       {res['matched_categories']}")
        print("Red Flags Caught:")
        for rf in res.get("red_flags", []):
            print(f"  - [{rf['severity']}] \"{rf['matched_phrase']}\" ({rf['description']})")
        print(f"Summary:          {res['summary']}")

    elif choice == "3":
        path_str = input("Enter path to audio file: ").strip().strip('"')
        path = Path(path_str)
        if not path.is_file():
            print(f"Error: File '{path}' does not exist.")
            return
        print(f"\nProcessing call: {path.name}...")
        res = pipeline.process_call(path)
        display_result(res)

    elif choice == "4":
        metrics_file = PROJECT_ROOT / "data" / "eval_results" / "voice_eval_metrics.json"
        if metrics_file.is_file():
            import json
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            print("\n================ ASVSPOOF 2019 EVALUATION ================")
            print(f"Dataset:            {metrics.get('dataset')}")
            print(f"Recordings Tested:  {metrics.get('total_recordings'):,}")
            print(f"ROC-AUC:            {metrics.get('roc_auc') * 100:.2f}%")
            print(f"Equal Error Rate:   {metrics.get('equal_error_rate') * 100:.2f}%")
            print(f"EER Threshold:      {metrics.get('calibrated_threshold')}")
            cal = metrics.get("metrics_at_calibrated_threshold", {})
            print(f"Balanced Accuracy:  {cal.get('balanced_accuracy') * 100:.2f}%")
            print(f"Spoof Precision:    {cal.get('spoof', {}).get('precision') * 100:.2f}%")
            print(f"Spoof Recall:       {cal.get('spoof', {}).get('recall') * 100:.2f}%")
            print("==========================================================\n")
        else:
            print("Evaluation metrics file not found.")

    else:
        print("Exiting.")


if __name__ == "__main__":
    run_interactive()

