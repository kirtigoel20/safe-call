
from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_features(split):
    path = FEATURE_DIR / f"{split}_features.npz"

    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        X = data["X"]
        y = data["y"]

    return X, y


if __name__ == "__main__":
    print("Loading training features...")
    X_train, y_train = load_features("train")

    print("Loading development features...")
    X_dev, y_dev = load_features("dev")

    print("\nTraining feature shape:", X_train.shape)
    print("Development feature shape:", X_dev.shape)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])

    print("\nTraining Logistic Regression...")
    model.fit(X_train, y_train)
    print("Training complete!")

    predictions = model.predict(X_dev)
    probabilities = model.predict_proba(X_dev)[:, 1]

    print("\n========== DEVELOPMENT RESULTS ==========")
    print(f"Accuracy:          {accuracy_score(y_dev, predictions):.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy_score(y_dev, predictions):.4f}"
    )
    print(f"ROC-AUC:           {roc_auc_score(y_dev, probabilities):.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_dev,
        predictions,
        labels=[0, 1],
        target_names=["Bonafide", "Spoof"],
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_dev, predictions, labels=[0, 1]))

    model_path = MODEL_DIR / "safecall_logistic_regression.joblib"
    joblib.dump(model, model_path)

    print("\nModel saved to:", model_path.resolve())
