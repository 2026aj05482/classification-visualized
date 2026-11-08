"""Reproduce all artifacts offline from scikit-learn's bundled UCI WDBC data.

Run explicitly with `python train_models.py`. Streamlit never calls this module.
"""

import hashlib
import importlib.metadata
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import FEATURES, MODEL_FILES, ROOT, TARGET, evaluate_model

SEED = 42


def build_models():
    estimators = {
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=2, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="uniform"),
        "Naive Bayes": GaussianNB(var_smoothing=1e-9),
        "Support Vector Machine": SVC(C=1.0, kernel="rbf", gamma="scale", probability=False, random_state=SEED),
    }
    return {
        name: Pipeline([
            ("scaler", "passthrough" if name == "Decision Tree" else StandardScaler()),
            ("classifier", estimator),
        ])
        for name, estimator in estimators.items()
    }


def main():
    for directory in ("model", "data", "reports"):
        (ROOT / directory).mkdir(exist_ok=True)
    dataset = load_breast_cancer()
    # Native feature order is mean, standard error, worst; ten measurements each.
    frame = pd.DataFrame(dataset.data, columns=FEATURES)
    # sklearn labels malignant=0, benign=1; invert to make malignant positive.
    frame[TARGET] = 1 - dataset.target
    frame.to_csv(ROOT / "data" / "breast_cancer_wisconsin.csv", index=False)
    train, test = train_test_split(frame, test_size=0.2, stratify=frame[TARGET], random_state=SEED)
    test.to_csv(ROOT / "test_data.csv", index=False)
    # Evaluate exactly the serialized CSV that is supplied to app users.
    test = pd.read_csv(ROOT / "test_data.csv")
    summary = []
    details = {}
    artifacts = {}
    for name, pipeline in build_models().items():
        pipeline.fit(train[FEATURES], train[TARGET])
        path = ROOT / "model" / MODEL_FILES[name]
        joblib.dump(pipeline, path, compress=3)
        loaded = joblib.load(path)
        result = evaluate_model(loaded, test[FEATURES], test[TARGET])
        summary.append({"ML Model Name": name, **result["metrics"]})
        details[name] = {
            "confusion_matrix": result["confusion_matrix"].tolist(),
            "classification_report": result["classification_report"],
        }
        artifacts[name] = {"file": MODEL_FILES[name], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    pd.DataFrame(summary).to_csv(ROOT / "reports" / "metrics.csv", index=False, float_format="%.10f")
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    metadata = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic)",
        "source": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        "loader": "sklearn.datasets.load_breast_cancer",
        "seed": SEED, "features": FEATURES, "target": TARGET,
        "labels": {"0": "benign", "1": "malignant"},
        "instances": len(frame), "train_instances": len(train), "test_instances": len(test),
        "class_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().items()},
        "train_class_counts": {str(k): int(v) for k, v in train[TARGET].value_counts().items()},
        "test_class_counts": {str(k): int(v) for k, v in test[TARGET].value_counts().items()},
        "train_indices": train.index.tolist(),
        "test_indices": train_test_split(frame.index, test_size=0.2, stratify=frame[TARGET], random_state=SEED)[1].tolist(),
        "test_csv_sha256": hashlib.sha256((ROOT / "test_data.csv").read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ("scikit-learn", "numpy", "pandas", "scipy", "joblib")},
        "artifacts": artifacts,
        "hyperparameters_selected": "Fixed before evaluation; no test-set tuning or model retraining in the app.",
    }
    (ROOT / "model" / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Saved five fitted pipelines. Train: {len(train)}; held-out test: {len(test)}.")


if __name__ == "__main__":
    main()

