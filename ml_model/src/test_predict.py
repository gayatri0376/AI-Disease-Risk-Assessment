"""
test_predict.py — Unit Tests for ML Inference Module
=====================================================
AI-Based Anemia Screening and Blood Report Interpretation System
Module 2: Machine Learning and Model Evaluation

Tests cover:
  1. Valid input inference for Task A (multiclass) and Task B (binary screening)
  2. Missing required CBC features detection
  3. Non-numeric and None values rejection
  4. NaN, +inf, and -inf values rejection
  5. Negative counts and concentration values rejection
  6. Impossible percentages (>100%) and differential sum violations rejection
  7. Extreme non-physiological values rejection
  8. Configurable model selection across Logistic Regression, Decision Tree, and Random Forest
  9. Scenario selection across Scenario 1 and Scenario 2
  10. Threshold handling: metadata lookup and custom threshold overrides for Task B
  11. Task A threshold isolation (ensures thresholds are never applied to Task A)
  12. Non-CBC metadata handling (ensures age, sex, patient_id are kept outside feature vector)
  13. Configuration error handling (invalid task, scenario, model name)
  14. Mandatory research disclaimers and uncalibrated score notices presence
"""

import math
import unittest
from pathlib import Path

from ml_model.src.predict import (
    predict,
    predict_task_a,
    predict_task_b,
    validate_cbc_input,
    load_model_pipeline,
    get_tuned_threshold,
    InputValidationError,
    ModelArtifactError,
    FEATURE_COLUMNS,
    TASK_B_CLASS_LABELS,
)


class TestInferenceValidInputs(unittest.TestCase):
    """Test valid CBC profile inputs across Task A and Task B."""

    def setUp(self):
        self.healthy_patient = {
            "WBC": 5.2,
            "LYMp": 19.7,
            "NEUTp": 72.4,
            "LYMn": 1.0,
            "NEUTn": 3.8,
            "RBC": 4.85,
            "HGB": 13.2,
            "HCT": 41.0,
            "MCV": 84.7,
            "MCH": 27.2,
            "MCHC": 32.1,
            "PLT": 181.0,
            "PDW": 10.0,
            "PCT": 0.15,
        }
        self.anemia_patient = {
            "WBC": 7.2,
            "LYMp": 30.7,
            "NEUTp": 60.7,
            "LYMn": 2.2,
            "NEUTn": 4.4,
            "RBC": 3.97,
            "HGB": 9.0,
            "HCT": 30.5,
            "MCV": 77.0,
            "MCH": 22.6,
            "MCHC": 29.5,
            "PLT": 148.0,
            "PDW": 14.3,
            "PCT": 0.14,
        }

    def test_task_a_prediction_structure_and_labels(self):
        """Task A returns expected dictionary structure with one of 9 dataset labels."""
        res = predict(self.anemia_patient, task="task_a", model_name="random_forest", scenario="scenario2")

        self.assertEqual(res["task"], "task_a")
        self.assertEqual(res["scenario"], "scenario2")
        self.assertEqual(res["model_name"], "random_forest")
        self.assertIn("predicted_label", res)
        self.assertIsNone(res["decision_threshold"])
        self.assertIsNone(res["predicted_code"])

        # Check class scores map
        self.assertIn("class_scores", res)
        self.assertEqual(len(res["class_scores"]), 9)
        self.assertIn("Iron deficiency anemia", res["class_scores"])
        self.assertIn("Healthy", res["class_scores"])

        # Check disclaimer and score interpretation note
        self.assertIn("RESEARCH AND EDUCATIONAL USE ONLY", res["disclaimer"])
        self.assertIn("uncalibrated model outputs", res["score_interpretation"])

    def test_task_b_prediction_structure_and_threshold(self):
        """Task B returns expected binary prediction structure with metadata threshold."""
        res = predict(self.anemia_patient, task="task_b", model_name="random_forest", scenario="scenario2")

        self.assertEqual(res["task"], "task_b")
        self.assertEqual(res["scenario"], "scenario2")
        self.assertEqual(res["model_name"], "random_forest")
        self.assertIn(res["predicted_label"], [TASK_B_CLASS_LABELS[0], TASK_B_CLASS_LABELS[1]])
        self.assertIn(res["predicted_code"], [0, 1])
        self.assertEqual(res["predicted_code"], 1)  # Patient has HGB=9.0, should screen positive

        # Tuned threshold for scenario2 RF is 0.370
        self.assertAlmostEqual(res["decision_threshold"], 0.370, places=2)

        # Check class scores map
        self.assertEqual(len(res["class_scores"]), 2)
        self.assertIn("Healthy (0)", res["class_scores"])
        self.assertIn("Anemia Subtypes (1)", res["class_scores"])

    def test_task_b_healthy_screens_negative(self):
        """Healthy patient with HGB=13.2 screens negative (Class 0, Healthy) in Task B."""
        res = predict_task_b(self.healthy_patient, model_name="random_forest", scenario="scenario2")
        self.assertEqual(res["predicted_code"], 0)
        self.assertEqual(res["predicted_label"], "Healthy")

    def test_convenience_wrappers(self):
        """predict_task_a and predict_task_b produce results identical to generic predict()."""
        res_a1 = predict(self.healthy_patient, task="task_a")
        res_a2 = predict_task_a(self.healthy_patient)
        self.assertEqual(res_a1["predicted_label"], res_a2["predicted_label"])

        res_b1 = predict(self.healthy_patient, task="task_b")
        res_b2 = predict_task_b(self.healthy_patient)
        self.assertEqual(res_b1["predicted_code"], res_b2["predicted_code"])


class TestInputValidationAndRejection(unittest.TestCase):
    """Test rejection of missing, non-numeric, NaN, infinite, or out-of-range features."""

    def setUp(self):
        self.valid_patient = {
            "WBC": 6.5,
            "LYMp": 28.0,
            "NEUTp": 65.0,
            "LYMn": 1.8,
            "NEUTn": 4.2,
            "RBC": 4.5,
            "HGB": 13.5,
            "HCT": 40.0,
            "MCV": 88.0,
            "MCH": 29.0,
            "MCHC": 33.0,
            "PLT": 250.0,
            "PDW": 12.0,
            "PCT": 0.20,
        }

    def test_missing_features_raises_error(self):
        """Missing one or more required features must raise InputValidationError."""
        # Drop HGB
        incomplete = self.valid_patient.copy()
        del incomplete["HGB"]
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(incomplete)
        self.assertIn("Missing required CBC feature(s)", str(ctx.exception))
        self.assertIn("HGB", str(ctx.exception))

    def test_empty_input_raises_error(self):
        """Empty input dictionary raises InputValidationError."""
        with self.assertRaises(InputValidationError):
            validate_cbc_input({})

    def test_non_dict_input_raises_error(self):
        """Non-dictionary input raises InputValidationError."""
        with self.assertRaises(InputValidationError):
            validate_cbc_input([1.0, 2.0, 3.0])  # type: ignore

    def test_non_numeric_values_raise_error(self):
        """Non-numeric strings and None raise InputValidationError."""
        bad_val = self.valid_patient.copy()
        bad_val["HGB"] = "elevated"
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_val)
        self.assertIn("must be numeric", str(ctx.exception))

        bad_none = self.valid_patient.copy()
        bad_none["WBC"] = None
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_none)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_nan_and_infinity_raise_error(self):
        """NaN and positive/negative infinity values raise InputValidationError."""
        bad_nan = self.valid_patient.copy()
        bad_nan["HGB"] = float("nan")
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_nan)
        self.assertIn("cannot be NaN", str(ctx.exception))

        bad_inf = self.valid_patient.copy()
        bad_inf["RBC"] = float("inf")
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_inf)
        self.assertIn("cannot be infinite", str(ctx.exception))

    def test_negative_values_raise_error(self):
        """Negative counts and concentrations raise InputValidationError."""
        # Negative HGB (Category A row 245 had -10.0)
        neg_hgb = self.valid_patient.copy()
        neg_hgb["HGB"] = -10.0
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(neg_hgb)
        self.assertIn("below minimum physiological boundary", str(ctx.exception))

        # Negative MCV (Category A row 344 had -79.3)
        neg_mcv = self.valid_patient.copy()
        neg_mcv["MCV"] = -79.3
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(neg_mcv)
        self.assertIn("below minimum physiological boundary", str(ctx.exception))

    def test_impossible_percentages_raise_error(self):
        """Percentages exceeding 100% raise InputValidationError."""
        # HCT > 100% (Category A row 4 had 316.0%)
        bad_hct = self.valid_patient.copy()
        bad_hct["HCT"] = 316.0
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_hct)
        self.assertIn("exceeds maximum physiological boundary", str(ctx.exception))

        # NEUTp > 100% (Category A row 216 had 5317.0%)
        bad_neut = self.valid_patient.copy()
        bad_neut["NEUTp"] = 5317.0
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_neut)
        self.assertIn("exceeds maximum physiological boundary", str(ctx.exception))

    def test_differential_sum_exceeding_100_percent_raises_error(self):
        """NEUTp + LYMp summing to > 100% raises InputValidationError."""
        bad_diff = self.valid_patient.copy()
        bad_diff["NEUTp"] = 75.0
        bad_diff["LYMp"] = 35.0  # Sum = 110%
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_diff)
        self.assertIn("sum to 110.0%, which exceeds 100%", str(ctx.exception))

    def test_extreme_physiological_values_raise_error(self):
        """Physiologically impossible cell counts and indices raise InputValidationError."""
        # MCH > 150 pg (Category A row 465 had 3117.0 pg)
        bad_mch = self.valid_patient.copy()
        bad_mch["MCH"] = 3117.0
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_mch)
        self.assertIn("MCH", str(ctx.exception))

        # RBC > 15 x10^6/uL (Category B row 273 had 90.8)
        bad_rbc = self.valid_patient.copy()
        bad_rbc["RBC"] = 90.8
        with self.assertRaises(InputValidationError) as ctx:
            validate_cbc_input(bad_rbc)
        self.assertIn("RBC", str(ctx.exception))


class TestModelSelectionAndConfiguration(unittest.TestCase):
    """Test model selection, scenario switching, and threshold override handling."""

    def setUp(self):
        self.patient = {
            "WBC": 6.0,
            "LYMp": 25.0,
            "NEUTp": 68.0,
            "LYMn": 1.5,
            "NEUTn": 4.1,
            "RBC": 4.2,
            "HGB": 10.5,
            "HCT": 33.0,
            "MCV": 78.5,
            "MCH": 25.0,
            "MCHC": 31.8,
            "PLT": 210.0,
            "PDW": 11.5,
            "PCT": 0.18,
        }

    def test_all_models_run_task_a(self):
        """All 3 models run successfully for Task A under both scenarios."""
        models = ["random_forest", "logistic_regression", "decision_tree"]
        scenarios = ["scenario1", "scenario2"]

        for sc in scenarios:
            for m in models:
                res = predict(self.patient, task="task_a", model_name=m, scenario=sc)
                self.assertEqual(res["task"], "task_a")
                self.assertEqual(res["scenario"], sc)
                self.assertEqual(res["model_name"], m)
                self.assertIsInstance(res["predicted_label"], str)
                self.assertEqual(len(res["class_scores"]), 9)

    def test_all_models_run_task_b(self):
        """All 3 models run successfully for Task B under both scenarios."""
        models = ["random_forest", "logistic_regression", "decision_tree"]
        scenarios = ["scenario1", "scenario2"]

        for sc in scenarios:
            for m in models:
                res = predict(self.patient, task="task_b", model_name=m, scenario=sc)
                self.assertEqual(res["task"], "task_b")
                self.assertEqual(res["scenario"], sc)
                self.assertEqual(res["model_name"], m)
                self.assertIn(res["predicted_code"], [0, 1])
                self.assertIsNotNone(res["decision_threshold"])

    def test_task_b_tuned_threshold_lookup(self):
        """Task B properly retrieves tuned threshold from metadata for each model."""
        # Scenario 2 thresholds
        th_rf = get_tuned_threshold("task_b", "scenario2", "random_forest")
        th_lr = get_tuned_threshold("task_b", "scenario2", "logistic_regression")
        th_dt = get_tuned_threshold("task_b", "scenario2", "decision_tree")

        self.assertAlmostEqual(th_rf, 0.370, places=2)
        self.assertAlmostEqual(th_lr, 0.340, places=2)
        self.assertAlmostEqual(th_dt, 0.100, places=2)

    def test_custom_threshold_override(self):
        """Custom threshold parameter overrides metadata default in Task B."""
        # Use borderline patient with HGB=12.95 where Random Forest score is ~0.4804
        borderline_patient = self.patient.copy()
        borderline_patient["HGB"] = 12.95
        borderline_patient["MCV"] = 86.0
        borderline_patient["MCH"] = 29.0
        borderline_patient["MCHC"] = 33.0
        borderline_patient["RBC"] = 4.5
        borderline_patient["HCT"] = 39.0

        # With default tuned threshold (0.370), score 0.4804 >= 0.370 -> Anemia (1)
        res_default = predict(borderline_patient, task="task_b", model_name="random_forest")
        self.assertEqual(res_default["predicted_code"], 1)
        self.assertAlmostEqual(res_default["decision_threshold"], 0.370, places=2)

        # Higher threshold (0.60) forces prediction to Healthy (0) since 0.4804 < 0.60
        res_high = predict(borderline_patient, task="task_b", threshold=0.60, model_name="random_forest")
        self.assertEqual(res_high["predicted_code"], 0)
        self.assertEqual(res_high["decision_threshold"], 0.60)

        # Lower threshold (0.30) forces prediction to Anemia (1) since 0.4804 >= 0.30
        res_low = predict(borderline_patient, task="task_b", threshold=0.30, model_name="random_forest")
        self.assertEqual(res_low["predicted_code"], 1)
        self.assertEqual(res_low["decision_threshold"], 0.30)

    def test_task_a_never_applies_threshold(self):
        """Passing threshold to Task A does not set a decision threshold."""
        res = predict(self.patient, task="task_a", threshold=0.35)
        self.assertIsNone(res["decision_threshold"])

    def test_invalid_task_or_scenario_raises_value_error(self):
        """Invalid task, scenario, or model name raises ValueError."""
        with self.assertRaises(ValueError):
            predict(self.patient, task="task_c")

        with self.assertRaises(ValueError):
            predict(self.patient, scenario="scenario3")

        with self.assertRaises(ValueError):
            predict(self.patient, model_name="neural_network")

    def test_extra_metadata_like_age_and_sex_are_excluded(self):
        """Non-CBC metadata fields are tolerated but excluded from ML feature vector."""
        payload_with_meta = self.patient.copy()
        payload_with_meta["age"] = 42
        payload_with_meta["sex"] = "Female"
        payload_with_meta["patient_id"] = "P-10023"

        res = predict(payload_with_meta, task="task_a")
        # Ensure inference succeeded
        self.assertIn("predicted_label", res)
        # Ensure validated features contain exactly the 14 CBC features
        self.assertEqual(len(res["validated_features"]), 14)
        self.assertEqual(set(res["validated_features"].keys()), set(FEATURE_COLUMNS))
        self.assertNotIn("age", res["validated_features"])
        self.assertNotIn("sex", res["validated_features"])


if __name__ == "__main__":
    unittest.main()
