"""Integration checks for saved artifacts, input validation and Streamlit flows."""

import hashlib
import io
import json
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from ml_utils import FEATURES, MODEL_FILES, ROOT, TARGET, evaluate_model, read_test_csv, validate_data


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contents = (ROOT / "test_data.csv").read_bytes()
        cls.frame, cls.features, cls.target = read_test_csv(cls.contents)
        cls.metadata = json.loads((ROOT / "model" / "metadata.json").read_text())

    def test_shared_split_is_disjoint_and_stratified(self):
        meta = self.metadata
        train, test = set(meta["train_indices"]), set(meta["test_indices"])
        self.assertFalse(train & test)
        self.assertEqual(len(train | test), 569)
        self.assertEqual(len(self.frame), 114)
        self.assertEqual(self.target.value_counts().to_dict(), {0: 72, 1: 42})
        self.assertEqual(hashlib.sha256(self.contents).hexdigest(), meta["test_csv_sha256"])
        full = pd.read_csv(ROOT / "data" / "breast_cancer_wisconsin.csv")
        expected = full.iloc[meta["test_indices"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(self.frame, expected, check_exact=False)

    def test_saved_pipelines_reproduce_report_without_fitting(self):
        expected = pd.read_csv(ROOT / "reports" / "metrics.csv").set_index("ML Model Name")
        full = pd.read_csv(ROOT / "data" / "breast_cancer_wisconsin.csv")
        training_means = full.iloc[self.metadata["train_indices"]][FEATURES].mean().to_numpy()
        with patch.object(Pipeline, "fit", side_effect=AssertionError("Inference must not fit")):
            for name, filename in MODEL_FILES.items():
                with self.subTest(model=name):
                    path = ROOT / "model" / filename
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.metadata["artifacts"][name]["sha256"])
                    model = joblib.load(path)
                    result = evaluate_model(model, self.features, self.target)
                    np.testing.assert_allclose(list(result["metrics"].values()), expected.loc[name].to_numpy(), atol=1e-9)
                    self.assertEqual(result["confusion_matrix"].sum(), len(self.frame))
                    if name != "Decision Tree":
                        self.assertEqual(model.named_steps["scaler"].n_samples_seen_, 455)
                        np.testing.assert_allclose(model.named_steps["scaler"].mean_, training_means)

    def test_reordered_columns_and_letter_labels(self):
        frame = self.frame[self.frame.columns[::-1]].copy()
        frame[TARGET] = frame[TARGET].map({0: " B ", 1: "M"})
        features, target = validate_data(frame)
        pd.testing.assert_frame_equal(features, self.features)
        pd.testing.assert_series_equal(target, self.target)

    def test_invalid_uploads_are_rejected(self):
        invalid = {
            "missing feature": self.frame.drop(columns=FEATURES[0]),
            "missing target": self.frame.drop(columns=TARGET),
            "extra id": self.frame.assign(id=1),
            "missing value": self.frame.assign(mean_radius=np.nan),
            "infinite": self.frame.assign(mean_radius=np.inf),
            "non numeric": self.frame.assign(mean_radius="oops"),
            "unknown target": self.frame.assign(diagnosis=2),
            "missing target value": self.frame.assign(diagnosis=np.nan),
            "empty rows": self.frame.iloc[:0],
        }
        for name, frame in invalid.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                read_test_csv(frame.to_csv(index=False).encode())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_test_csv(b"mean_radius,mean_radius,diagnosis\n1,2,0\n")
        with self.assertRaises(ValueError):
            read_test_csv(b"")
        with self.assertRaises(ValueError):
            read_test_csv(b"x" * (5 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            validate_data(pd.concat([self.frame] * 88, ignore_index=True))

    def test_single_class_auc_is_undefined(self):
        features, target = validate_data(self.frame[self.frame[TARGET] == 0])
        result = evaluate_model(joblib.load(ROOT / "model" / "knn.joblib"), features, target)
        self.assertTrue(np.isnan(result["metrics"]["AUC"]))
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))

    def test_streamlit_all_five_models_and_upload_flow(self):
        with patch.object(Pipeline, "fit", side_effect=AssertionError("App must never fit")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value), 5)
            for name in MODEL_FILES:
                app.selectbox[0].select(name).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.metric), 6)
            app.radio[0].set_value("Upload test CSV").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Upload a CSV" in item.value for item in app.info))
            with patch("streamlit.file_uploader", return_value=io.BytesIO(self.contents)):
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.dataframe[0].value), 5)
                self.assertTrue(any("uploaded CSV" in item.value for item in app.info))
            with patch("streamlit.file_uploader", return_value=io.BytesIO(b"wrong\n1\n")):
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Missing columns" in item.value for item in app.error))
            single_class = self.frame[self.frame[TARGET] == 0].to_csv(index=False).encode()
            with patch("streamlit.file_uploader", return_value=io.BytesIO(single_class)):
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any("one target class" in item.value for item in app.warning))
                self.assertEqual(app.metric[1].value, "N/A")


if __name__ == "__main__":
    unittest.main()
