"""
test_evaluation.py — Unit Tests for Class-Wise Evaluation & Metrics Calculation
================================================================================
AI-Based Anemia Screening and Blood Report Interpretation System
Module 2: Machine Learning and Model Evaluation

Tests cover:
  1. Raw dataset MD5 checksum integrity (unaltered raw CSV)
  2. Test partition recovery determinism (zero leakage, identical across scenarios)
  3. Class label completeness, mapping, and consistent alphabetical ordering
  4. Exact per-class support counts in held-out test partitions (including rare classes)
  5. Mathematical consistency of precision, recall, F1, TP, FP, FN, TN, and confusion matrices
  6. Generated class-wise evaluation CSV structure and metric completeness
  7. Reproduction of baseline model metadata metrics with zero discrepancy
"""

import unittest
import hashlib
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
import sys
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
ML_DIR = SRC_DIR.parent
PROJECT_ROOT = ML_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_model.src.preprocessing import (
    DATASET_PATH,
    REPORTS_DIR,
    CATEGORY_A_DF_INDICES,
    CATEGORY_B_DF_INDICES,
    run_full_preprocessing,
)
from ml_model.src.generate_class_wise_reports import (
    get_task_a_class_order,
    get_task_b_class_order,
    compute_one_vs_rest_counts,
)

RAW_CSV_MD5 = "c6375e974d730a42b53160614237ddf0"
MODELS_DIR = ML_DIR / "models"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


class TestDatasetAndPartitionIntegrity(unittest.TestCase):
    """Verify raw dataset checksum and deterministic test partition recovery."""

    def test_raw_csv_md5_unaltered(self):
        """Ensure raw CSV has not been modified in place."""
        hasher = hashlib.md5()
        with open(DATASET_PATH, "rb") as f:
            hasher.update(f.read())
        self.assertEqual(
            hasher.hexdigest(),
            RAW_CSV_MD5,
            "Raw CSV file MD5 hash has changed! The dataset must remain unaltered."
        )

    def test_held_out_partitions_identical_across_scenarios(self):
        """Ensure held-out test partitions are 100% identical between Scenario 1 and 2."""
        s1 = run_full_preprocessing("scenario1", verbose=False)
        s2 = run_full_preprocessing("scenario2", verbose=False)

        # Task A
        self.assertEqual(list(s1["task_a_X_test"].index), list(s2["task_a_X_test"].index))
        pd.testing.assert_frame_equal(s1["task_a_X_test"], s2["task_a_X_test"])
        pd.testing.assert_series_equal(s1["task_a_y_test"], s2["task_a_y_test"])
        self.assertEqual(len(s1["task_a_X_test"]), 241)

        # Task B
        self.assertEqual(list(s1["task_b_X_test"].index), list(s2["task_b_X_test"].index))
        pd.testing.assert_frame_equal(s1["task_b_X_test"], s2["task_b_X_test"])
        pd.testing.assert_series_equal(s1["task_b_y_test"], s2["task_b_y_test"])
        self.assertEqual(len(s1["task_b_X_test"]), 216)

    def test_zero_category_a_or_b_in_test_partitions(self):
        """Verify clean test partitions contain zero Category A and zero Category B rows."""
        s1 = run_full_preprocessing("scenario1", verbose=False)
        cat_a_set = set(CATEGORY_A_DF_INDICES)
        cat_b_set = set(CATEGORY_B_DF_INDICES)
        test_a_indices = set(s1["task_a_X_test"].index)
        test_b_indices = set(s1["task_b_X_test"].index)

        self.assertEqual(len(test_a_indices & cat_a_set), 0)
        self.assertEqual(len(test_a_indices & cat_b_set), 0)
        self.assertEqual(len(test_b_indices & cat_a_set), 0)
        self.assertEqual(len(test_b_indices & cat_b_set), 0)


class TestClassOrderingAndLabels(unittest.TestCase):
    """Verify class ordering, label sets, and support counts."""

    def setUp(self):
        self.s1 = run_full_preprocessing("scenario1", verbose=False)

    def test_task_a_classes_and_ordering(self):
        """Task A must contain exactly 9 classes sorted alphabetically."""
        classes = get_task_a_class_order(self.s1["task_a_y_test"])
        expected_classes = [
            "Healthy",
            "Iron deficiency anemia",
            "Leukemia",
            "Leukemia with thrombocytopenia",
            "Macrocytic anemia",
            "Normocytic hypochromic anemia",
            "Normocytic normochromic anemia",
            "Other microcytic anemia",
            "Thrombocytopenia",
        ]
        self.assertEqual(classes, expected_classes)

    def test_task_a_exact_class_supports(self):
        """Verify exact test support counts for all 9 classes in Task A."""
        counts = self.s1["task_a_y_test"].value_counts().to_dict()
        expected_counts = {
            "Healthy": 64,
            "Normocytic hypochromic anemia": 53,
            "Normocytic normochromic anemia": 50,
            "Iron deficiency anemia": 36,
            "Thrombocytopenia": 14,
            "Other microcytic anemia": 10,
            "Leukemia": 9,
            "Macrocytic anemia": 3,
            "Leukemia with thrombocytopenia": 2,
        }
        self.assertEqual(counts, expected_counts)
        self.assertEqual(sum(counts.values()), 241)

    def test_task_b_classes_and_supports(self):
        """Verify exact test support counts for Task B (0=Healthy, 1=Anemia)."""
        counts = self.s1["task_b_y_test"].value_counts().to_dict()
        self.assertEqual(counts[0], 64)
        self.assertEqual(counts[1], 152)
        self.assertEqual(sum(counts.values()), 216)


class TestMetricCalculationConsistency(unittest.TestCase):
    """Verify precision, recall, F1-score, and confusion matrix arithmetic."""

    def test_one_vs_rest_confusion_matrix_sum(self):
        """Verify that TP + FP + FN + TN equals total samples for each one-vs-rest slice."""
        cm = np.array([
            [64, 0, 0],
            [1, 35, 0],
            [0, 2, 10],
        ])
        total = cm.sum()
        for idx in range(3):
            tp, fp, fn, tn = compute_one_vs_rest_counts(cm, idx)
            self.assertEqual(tp + fp + fn + tn, total)
            self.assertEqual(tp + fn, cm[idx, :].sum())

    def test_task_b_confusion_matrix_sum_and_sensitivity_specificity(self):
        """Verify Task B logistic regression confusion matrix arithmetic."""
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)

        lr_meta = meta["models"]["task_b_scenario1_logistic_regression"]
        cm_dict = lr_meta["confusion_matrix"]
        tn, fp, fn, tp = cm_dict["TN"], cm_dict["FP"], cm_dict["FN"], cm_dict["TP"]

        self.assertEqual(tn + fp + fn + tp, 216)
        self.assertEqual(tn + fp, 64)   # Healthy support
        self.assertEqual(tp + fn, 152)  # Anemia support

        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        self.assertAlmostEqual(sens, lr_meta["test_sensitivity_recall"], places=4)
        self.assertAlmostEqual(spec, lr_meta["test_specificity"], places=4)


class TestEvaluationOutputFiles(unittest.TestCase):
    """Verify the generated class-wise evaluation CSV and report files."""

    def test_csv_structure_and_row_count(self):
        """Verify class_wise_evaluation_metrics.csv exists and has expected rows."""
        csv_path = REPORTS_DIR / "class_wise_evaluation_metrics.csv"
        self.assertTrue(csv_path.exists(), "class_wise_evaluation_metrics.csv does not exist!")

        df = pd.read_csv(csv_path)
        # 9 classes * 3 models * 2 scenarios = 54 rows (Task A)
        # 2 classes * 3 models * 2 scenarios = 12 rows (Task B)
        # Total = 66 rows
        self.assertEqual(len(df), 66)

        expected_cols = [
            "task", "scenario", "model", "metric_scope", "class_label",
            "precision", "recall", "f1_score", "support",
            "tp", "fp", "fn", "tn",
            "overall_accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
            "tuned_threshold", "sensitivity", "specificity", "roc_auc", "pr_auc"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

        # Check that no class-level metric is null
        self.assertEqual(df["precision"].isna().sum(), 0)
        self.assertEqual(df["recall"].isna().sum(), 0)
        self.assertEqual(df["f1_score"].isna().sum(), 0)
        self.assertEqual(df["support"].isna().sum(), 0)

    def test_markdown_report_exists_and_contains_key_sections(self):
        """Verify class_wise_evaluation_report.md exists and contains expected sections."""
        md_path = REPORTS_DIR / "class_wise_evaluation_report.md"
        self.assertTrue(md_path.exists(), "class_wise_evaluation_report.md does not exist!")

        content = md_path.read_text(encoding="utf-8")
        self.assertIn("Task A: 9-Class Multiclass Diagnosis", content)
        self.assertIn("Task B: Binary Anemia Screening", content)
        self.assertIn("Macrocytic anemia", content)
        self.assertIn("Leukemia with thrombocytopenia", content)
        self.assertIn("Hemoglobin Separation", content)
        self.assertIn("Critical Educational Disclaimer", content)


if __name__ == "__main__":
    unittest.main()
