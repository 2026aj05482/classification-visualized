"""Streamlit evaluation dashboard: load saved pipelines, never train on uploads."""

import hashlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from ml_utils import FEATURES, METRICS, MODEL_FILES, ROOT, TARGET, evaluate_model, read_test_csv

st.set_page_config(page_title="Classification Visualized", page_icon="🔬", layout="wide")


@st.cache_resource
def load_models():
    """Only repository-owned artifacts are deserialized; uploads are CSV only."""
    metadata = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    models = {}
    for name, filename in MODEL_FILES.items():
        path = ROOT / "model" / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["artifacts"][name]["sha256"]:
            raise ValueError(f"Saved artifact checksum mismatch: {filename}")
        models[name] = joblib.load(path)
    return models


@st.cache_data(show_spinner=False)
def evaluate_csv(contents):
    frame, features, target = read_test_csv(contents)
    results = {name: evaluate_model(model, features, target) for name, model in load_models().items()}
    comparison = pd.DataFrame({name: value["metrics"] for name, value in results.items()}).T
    comparison.index.name = "ML Model Name"
    return frame, target, results, comparison


st.title("Classification Visualized")
st.markdown("### One dataset. Five classifiers. A transparent comparison.")
st.write("Explore how five saved models classify UCI breast cancer cell measurements as benign or malignant.")
st.caption("569 original instances · 30 numeric features · 455 training rows · 114 held-out test rows")

with st.sidebar:
    st.header("Evaluation controls")
    source = st.radio("Test data source", ["Bundled test data", "Upload test CSV"])
    uploaded = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])
    selected = st.selectbox("Model", list(MODEL_FILES))
    st.download_button("Download test_data.csv", (ROOT / "test_data.csv").read_bytes(),
                       "test_data.csv", "text/csv", width="stretch")
    st.caption("Use the download as your CSV template. Labels: 0/B = benign, 1/M = malignant.")
    st.divider()
    st.markdown("[Dataset on UCI](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)")
    st.markdown("[GitHub source](https://github.com/2026aj05482/classification-visualized)")

if source == "Upload test CSV" and uploaded is None:
    st.info("Upload a CSV containing the 30 input features and diagnosis column to begin.")
    st.stop()

contents = uploaded.getvalue() if source == "Upload test CSV" else (ROOT / "test_data.csv").read_bytes()
try:
    frame, target, results, comparison = evaluate_csv(contents)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except (FileNotFoundError, KeyError, ImportError) as exc:
    st.error("The saved model artifacts could not be loaded. Check model/ and the pinned requirements.txt.")
    st.stop()

st.info(f"Evaluating {len(frame):,} rows from {'uploaded CSV' if source == 'Upload test CSV' else 'the bundled held-out test set'}. "
        "Models and scalers are already fitted; this app performs inference only.")
if target.nunique() < 2:
    st.warning("This file contains one target class. ROC AUC is undefined (shown as N/A); upload both classes for AUC.")
    st.caption("Zero-division precision/recall/F1 and degenerate MCC are reported as 0 by scikit-learn.")

st.subheader("All five models on the same test data")
st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
st.caption("Precision, recall and F1 use malignant (1) as the positive class. AUC uses continuous scores. "
           "SVM uses its decision function; other models use malignant-class probabilities.")
st.download_button("Download comparison metrics", comparison.to_csv().encode("utf-8"),
                   "model_comparison.csv", "text/csv")

st.subheader(selected)
result = results[selected]
for column, name in zip(st.columns(6), METRICS):
    value = result["metrics"][name]
    column.metric(name, "N/A" if np.isnan(value) else f"{value:.4f}")

diagnostics, predictions_tab, data_tab = st.tabs(["Model diagnostics", "Predictions", "Data & method"])
with diagnostics:
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(result["confusion_matrix"], display_labels=["Benign (0)", "Malignant (1)"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**ROC comparison**")
        if target.nunique() == 2:
            fig, ax = plt.subplots(figsize=(5, 4))
            for name, evaluated in results.items():
                RocCurveDisplay.from_predictions(target, evaluated["scores"], name=name, ax=ax)
            ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
            ax.legend(fontsize=7, loc="lower right")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("ROC curves require both benign and malignant labels.")
    st.markdown("**Classification report**")
    report = {name: values for name, values in result["classification_report"].items() if isinstance(values, dict)}
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch")
    st.caption("Overall accuracy is shown in the metric cards above; support is the number of actual rows in each class.")

with predictions_tab:
    predictions = frame.copy()
    predictions["actual_label"] = target.map({0: "Benign", 1: "Malignant"})
    predictions["predicted_diagnosis"] = result["predictions"]
    predictions["predicted_label"] = pd.Series(result["predictions"], index=frame.index).map({0: "Benign", 1: "Malignant"})
    score_name = "decision_score" if selected == "Support Vector Machine" else "malignant_probability"
    predictions[score_name] = result["scores"]
    st.dataframe(predictions, width="stretch")
    st.caption("SVM scores are signed distances to the decision boundary, not probabilities.")
    st.download_button("Download selected-model predictions", predictions.to_csv(index=False).encode("utf-8"),
                       "predictions.csv", "text/csv")

with data_tab:
    st.write(f"Rows: {len(frame):,} | Benign: {(target == 0).sum():,} | Malignant: {(target == 1).sum():,}")
    st.dataframe(frame.head(20), width="stretch")
    st.markdown("**CSV input contract**")
    st.write("Provide all 30 finite numeric features below plus diagnosis, with no extra ID/index columns. "
             "Column order may vary. Missing values, invalid labels and duplicate headers are rejected. "
             "Upload limit: 5 MB and 10,000 rows.")
    st.code(", ".join(FEATURES + [TARGET]), language=None)
    st.markdown("**Training and evaluation**")
    st.write("A stratified 80/20 split with random_state=42 is shared by every model. StandardScaler is fitted "
             "on training data inside each pipeline except the Decision Tree, which uses raw features. "
             "Hyperparameters were fixed before evaluation. No fitting or parameter selection occurs in the app.")
    st.caption("Results describe this dataset and split; they do not establish performance on other populations.")
