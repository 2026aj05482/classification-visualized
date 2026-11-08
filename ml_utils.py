"""Shared input contract and inference-only evaluation for the five classifiers."""

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
TARGET = "diagnosis"
FEATURES = [
    f"{prefix}_{measurement}"
    for prefix in ("mean", "error", "worst")
    for measurement in (
        "radius", "texture", "perimeter", "area", "smoothness", "compactness",
        "concavity", "concave_points", "symmetry", "fractal_dimension",
    )
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
MAX_ROWS = 10_000
MAX_BYTES = 5 * 1024 * 1024


def validate_data(frame):
    """Return features in training order and a binary target (malignant = 1)."""
    if frame.empty:
        raise ValueError("The CSV must contain at least one test row.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Upload at most {MAX_ROWS:,} test rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    required = set(FEATURES + [TARGET])
    missing = sorted(required - set(frame.columns))
    extra = sorted(set(frame.columns) - required)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected columns (remove IDs/index columns): " + ", ".join(extra))
    try:
        features = frame[FEATURES].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All 30 input features must be numeric.") from exc
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Input features must not contain missing or infinite values.")
    labels = frame[TARGET].astype("string").str.strip().str.upper()
    mapping = {"0": 0, "1": 1, "0.0": 0, "1.0": 1, "B": 0, "M": 1}
    if labels.isna().any() or not labels.isin(mapping).all():
        raise ValueError("diagnosis must contain only 0/1 (0 = benign, 1 = malignant), or B/M.")
    return features, labels.map(mapping).astype(int)


def read_test_csv(contents):
    """Read CSV bytes without silently allowing pandas to rename duplicate headers."""
    if len(contents) > MAX_BYTES:
        raise ValueError("CSV must be no larger than 5 MB.")
    try:
        decoded = contents.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)))
        if len(header) != len(set(header)):
            raise ValueError("Duplicate column names are not allowed.")
        frame = pd.read_csv(io.StringIO(decoded))
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration, csv.Error) as exc:
        raise ValueError("Upload a non-empty, UTF-8, comma-separated CSV file.") from exc
    features, target = validate_data(frame)
    return frame, features, target


def evaluate_model(model, features, target):
    """Use fitted artifacts only. AUC uses scores; class metrics use predict()."""
    predictions = model.predict(features)
    if hasattr(model, "predict_proba"):
        positive_index = list(model.classes_).index(1)
        scores = model.predict_proba(features)[:, positive_index]
    else:
        scores = model.decision_function(features)
    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(roc_auc_score(target, scores)) if target.nunique() == 2 else float("nan"),
        "Precision": float(precision_score(target, predictions, pos_label=1, zero_division=0)),
        "Recall": float(recall_score(target, predictions, pos_label=1, zero_division=0)),
        "F1": float(f1_score(target, predictions, pos_label=1, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    return {
        "metrics": metrics,
        "predictions": predictions,
        "scores": scores,
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]),
        "classification_report": classification_report(
            target, predictions, labels=[0, 1],
            target_names=["Benign (0)", "Malignant (1)"], output_dict=True, zero_division=0,
        ),
    }

