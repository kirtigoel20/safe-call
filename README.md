# SAFE CALL: AI Voice Clone Detection and Intelligent Scam Response Guidance

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2Bcu128-red.svg)](https://pytorch.org/)
[![OpenAI Whisper](https://img.shields.io/badge/ASR-OpenAI%20Whisper-orange.svg)](https://github.com/openai/whisper)
[![Tests](https://img.shields.io/badge/Tests-27%2F27%20Passing-brightgreen.svg)]()

---

## 📌 Project Overview

**SAFE CALL** is a multi-modal artificial intelligence system designed to detect AI-generated voice clones, identify social engineering fraud during voice-based communication, and provide explainable real-time safety recommendations to protect users from high-tech telephone scams.

The system fuses **acoustic deepfake detection** with **automatic speech recognition (ASR)** and **NLP intent analysis** to compute unified threat levels and generate immediate, actionable countermeasures.

---

## ⚙️ Implemented Architecture & Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Incoming Call Audio                              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 1: Audio Preprocessing (`preprocessing/audio_preprocessing.py`)      │
│  - Stereo-to-mono downmixing                                                │
│  - Standardized 16,000 Hz resampling                                        │
│  - Signal validation (empty audio, non-finite values, duration bounds)      │
└──────────────────────┬───────────────────────────────┬──────────────────────┘
                       │                               │
                       ▼                               ▼
┌─────────────────────────────────────────┐  ┌────────────────────────────────┐
│ Phase 2: Voice Deepfake Detection       │  │ Phase 3: Speech-to-Text (ASR)  │
│ - 40-dim MFCC Feature Extractor         │  │ - OpenAI Whisper (Base / Tiny) │
│ - ASVspoof 2019-Trained Classifier      │  │ - Automatic CUDA Acceleration  │
│ - Calibrated EER Decision Threshold     │  │ - Timestamped Segments         │
│ (`voice_detection/voice_detector.py`)   │  │ (`speech_to_text/transcriber.py`)│
└──────────────────────┬──────────────────┘  └────────────────┬───────────────┘
                       │                                      │
                       │                                      ▼
                       │                     ┌────────────────────────────────┐
                       │                     │ Phase 4: Scam Intent Analysis  │
                       │                     │ - Multi-Vector Social Eng. NLP │
                       │                     │ - OTP, Bank & Police Imperson. │
                       │                     │ - Compound Synergy Multipliers │
                       │                     │ (`scam_detection/`)            │
                       │                     └────────────────┬───────────────┘
                       │                                      │
                       └──────────────────────┬───────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 5: Multi-Modal Risk Fusion (`fusion/risk_assessor.py`)               │
│  - Threat Quadrant Mapping (AI_CLONE_FRAUD, HUMAN_SCAM, BENIGN)            │
│  - Composite Threat Score (0.0 to 1.0) & Risk Tiers (LOW, HIGH, CRITICAL)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Phase 6: Explainable Safety Guidance (`explainability/guidance_generator.py`)
│  - Plain-English rationale for detected acoustic & transcript anomalies    │
│  - Immediate action items (hang up, verify via safe word, never share OTP)  │
│  - Official reporting integration (National Cyber Crime Helpline 1930)      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Evaluation Results

Evaluated on the official **ASVspoof 2019 Logical Access (LA)** Development Split (**24,844 unseen recordings**):

| Metric | Score | Explanation |
| :--- | :--- | :--- |
| **ROC-AUC** | **91.07%** (`0.9107`) | Strong overall discriminative power on unseen deepfakes |
| **Equal Error Rate (EER)** | **17.35%** (`0.1735`) | Standard biometric anti-spoofing performance metric |
| **Calibrated Threshold** | **`0.6107`** | Optimal operating point balancing false alarms and misses |
| **Balanced Accuracy** | **82.65%** | Equal fairness across both bonafide and spoof distributions |
| **Spoof Precision** | **97.66%** | High certainty when alerting user of an AI voice clone |

Evaluation visual artifacts:
* ROC Curve: `data/eval_results/voice_roc_curve.png`
* Confusion Matrix: `data/eval_results/voice_confusion_matrix.png`
* Metrics JSON: `data/eval_results/voice_eval_metrics.json`

---

## 🛠️ Repository Structure

```
Safecall/
├── preprocessing/
│   └── audio_preprocessing.py        # 16 kHz mono resampling & audio validation
├── voice_detection/
│   ├── dataset_loader.py             # ASVspoof 2019 protocol loader
│   ├── feature_extraction.py         # 40-dim MFCC feature extraction (mean + std)
│   ├── extract_dataset_features.py   # Batch feature extraction for train/dev sets
│   ├── train_model.py                # Balanced Logistic Regression training
│   ├── evaluate_model.py             # EER calculation & ROC/confusion matrix plots
│   └── voice_detector.py             # Production inference engine with EER thresholding
├── speech_to_text/
│   └── transcriber.py                # Whisper ASR transcription & confidence scoring
├── scam_detection/
│   ├── scam_patterns.py              # Social engineering taxonomies & compiled regexes
│   └── scam_detector.py              # Intent analyzer with compound risk multipliers
├── fusion/
│   └── risk_assessor.py              # Acoustic + NLP multi-modal risk fusion
├── explainability/
│   └── guidance_generator.py         # Plain-English alert generator & countermeasures
├── tests/
│   ├── test_voice_detector.py        # 8 tests for acoustic inference engine
│   ├── test_speech_to_text.py        # 7 tests for ASR transcriber
│   ├── test_scam_detector.py         # 7 tests for scam intent analysis
│   └── test_pipeline.py              # 5 tests for end-to-end integration
├── models/
│   └── safecall_logistic_regression.joblib  # Trained acoustic model weights
├── pipeline.py                       # Unified pipeline orchestrator
├── demo.py                           # Interactive CLI presentation demo
├── web_app.py                        # Interactive Web Browser Dashboard
├── VIVA_GUIDE.md                     # Academic Defense & Viva Preparation Guide
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Launch Interactive Web Browser Dashboard (Recommended for Demos)
```powershell
python web_app.py
```
Open **`http://localhost:8000`** in Google Chrome or Microsoft Edge to drag-and-drop audio or record live speech directly from your microphone.

### 2. Run Interactive Demo CLI
```powershell
python demo.py
# Or analyze an audio file directly:
python demo.py "path/to/recording.m4a"
```

### 3. Run Complete Automated Test Suite (27 Tests)
```powershell
.venv\Scripts\python.exe -m pytest tests/test_voice_detector.py tests/test_speech_to_text.py tests/test_scam_detector.py tests/test_pipeline.py -v
```

### 4. Run Pipeline in Python Code
```python
from pipeline import run_safecall

# Analyze an incoming audio recording
result = run_safecall("call_recording.wav")

print("Risk Level:  ", result["guidance"]["risk_level"])
print("Threat:      ", result["fusion"]["primary_threat"])
print("Transcript:  ", result["transcription"]["text"])
print("Action Items:", result["guidance"]["action_items"])
```
