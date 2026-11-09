"""
Evaluation & Visualization Suite for Voice Deepfake Detection.

Computes:
- Accuracy, Balanced Accuracy
- Precision, Recall, F1-Score (Bonafide & Spoof)
- ROC-AUC Score
- Equal Error Rate (EER) and optimal operating threshold
- Normalized & Raw Confusion Matrices
- ROC Curve visualization saved to disk
"""

import json
from pathlib import Path
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_PATH = PROJECT_ROOT / "models" / "safecall_logistic_regression.joblib"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_eer(y_true, y_scores):
    """Computes Equal Error Rate (EER) and the corresponding threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    # Locate threshold closest to EER
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    optimal_threshold = float(thresholds[idx])
    return float(eer), optimal_threshold, fpr, tpr


def evaluate_voice_model():
    print("==================================================")
    print("   SAFE CALL - VOICE DETECTION EVALUATION SUITE   ")
    print("==================================================")

    dev_path = FEATURE_DIR / "dev_features.npz"
    if not dev_path.is_file():
        raise FileNotFoundError(f"Development features not found at {dev_path}")

    print(f"Loading development features from: {dev_path}...")
    with np.load(dev_path, allow_pickle=False) as dev_data:
        X_dev = dev_data["X"]
        y_dev = dev_data["y"]

    print(f"Loading trained model from: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # 1. Predictions & Probabilities
    y_probs = model.predict_proba(X_dev)[:, 1]
    y_preds_default = model.predict(X_dev)

    # 2. Calculate EER and calibrated threshold
    eer, eer_threshold, fpr, tpr = compute_eer(y_dev, y_probs)
    y_preds_calibrated = (y_probs >= eer_threshold).astype(int)

    # 3. Calculate metrics
    roc_auc = float(roc_auc_score(y_dev, y_probs))
    acc_default = float(accuracy_score(y_dev, y_preds_default))
    bacc_default = float(balanced_accuracy_score(y_dev, y_preds_default))
    acc_calibrated = float(accuracy_score(y_dev, y_preds_calibrated))
    bacc_calibrated = float(balanced_accuracy_score(y_dev, y_preds_calibrated))

    cm_default = confusion_matrix(y_dev, y_preds_default, labels=[0, 1])
    cm_calibrated = confusion_matrix(y_dev, y_preds_calibrated, labels=[0, 1])

    report_default = classification_report(
        y_dev, y_preds_default, target_names=["Bonafide", "Spoof"], output_dict=True
    )
    report_calibrated = classification_report(
        y_dev, y_preds_calibrated, target_names=["Bonafide", "Spoof"], output_dict=True
    )

    print("\n--- RESULTS AT DEFAULT THRESHOLD (0.50) ---")
    print(f"Accuracy:          {acc_default:.4f}")
    print(f"Balanced Accuracy: {bacc_default:.4f}")
    print(f"Bonafide Precision: {report_default['Bonafide']['precision']:.4f} | Recall: {report_default['Bonafide']['recall']:.4f} | F1: {report_default['Bonafide']['f1-score']:.4f}")
    print(f"Spoof Precision:    {report_default['Spoof']['precision']:.4f} | Recall: {report_default['Spoof']['recall']:.4f} | F1: {report_default['Spoof']['f1-score']:.4f}")

    print("\n--- RESULTS AT CALIBRATED EER THRESHOLD ---")
    print(f"Calibrated Threshold: {eer_threshold:.4f}")
    print(f"Equal Error Rate (EER): {eer * 100:.2f}%")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print(f"Accuracy:              {acc_calibrated:.4f}")
    print(f"Balanced Accuracy:     {bacc_calibrated:.4f}")
    print(f"Bonafide Precision:    {report_calibrated['Bonafide']['precision']:.4f} | Recall: {report_calibrated['Bonafide']['recall']:.4f} | F1: {report_calibrated['Bonafide']['f1-score']:.4f}")
    print(f"Spoof Precision:       {report_calibrated['Spoof']['precision']:.4f} | Recall: {report_calibrated['Spoof']['recall']:.4f} | F1: {report_calibrated['Spoof']['f1-score']:.4f}")

    # 4. Generate & Save Visualizations
    # Figure 1: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cm, title in zip(
        axes,
        [cm_default, cm_calibrated],
        ["Confusion Matrix (Default Thresh 0.50)", f"Confusion Matrix (Calibrated EER Thresh {eer_threshold:.3f})"],
    ):
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(title, fontsize=12, pad=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Bonafide", "Spoof"], fontsize=10)
        ax.set_yticklabels(["Bonafide", "Spoof"], fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
        ax.set_ylabel("True Label", fontsize=11, labelpad=8)

        # Text annotations
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10,
                    fontweight="bold",
                )

    plt.tight_layout()
    cm_plot_path = RESULTS_DIR / "voice_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_plot_path}")

    # Figure 2: ROC Curve with EER Point
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"Logistic Regression ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.scatter([eer], [1.0 - eer], color="#dc2626", s=80, zorder=5, label=f"EER = {eer * 100:.2f}% (Thresh = {eer_threshold:.3f})")
    plt.title("SAFE CALL — Voice Deepfake Detection ROC Curve", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (Bonafide misclassified as Spoof)", fontsize=10)
    plt.ylabel("True Positive Rate (Spoof correctly detected)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    roc_plot_path = RESULTS_DIR / "voice_roc_curve.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve plot to: {roc_plot_path}")

    # 5. Save structured metrics JSON
    summary = {
        "dataset": "ASVspoof 2019 LA Development Set",
        "total_recordings": int(len(y_dev)),
        "bonafide_count": int(np.sum(y_dev == 0)),
        "spoof_count": int(np.sum(y_dev == 1)),
        "roc_auc": round(roc_auc, 4),
        "equal_error_rate": round(eer, 4),
        "calibrated_threshold": round(eer_threshold, 4),
        "default_threshold": 0.5,
        "metrics_at_calibrated_threshold": {
            "accuracy": round(acc_calibrated, 4),
            "balanced_accuracy": round(bacc_calibrated, 4),
            "bonafide": {
                "precision": round(report_calibrated["Bonafide"]["precision"], 4),
                "recall": round(report_calibrated["Bonafide"]["recall"], 4),
                "f1_score": round(report_calibrated["Bonafide"]["f1-score"], 4),
            },
            "spoof": {
                "precision": round(report_calibrated["Spoof"]["precision"], 4),
                "recall": round(report_calibrated["Spoof"]["recall"], 4),
                "f1_score": round(report_calibrated["Spoof"]["f1-score"], 4),
            },
            "confusion_matrix": {
                "true_bonafide": int(cm_calibrated[0, 0]),
                "false_spoof": int(cm_calibrated[0, 1]),
                "false_bonafide": int(cm_calibrated[1, 0]),
                "true_spoof": int(cm_calibrated[1, 1]),
            },
        },
    }

    metrics_json_path = RESULTS_DIR / "voice_eval_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary JSON to: {metrics_json_path}")
    print("==================================================")
    return summary


if __name__ == "__main__":
    evaluate_voice_model()

