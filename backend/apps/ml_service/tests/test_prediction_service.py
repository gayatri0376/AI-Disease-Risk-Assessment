"""
test_prediction_service.py — Unit Tests for ML Prediction Service
==================================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Phase 6.1 Validation

TEST COVERAGE:
--------------
  1. Both prediction tasks: Task B (screening) and Task A (multiclass diagnosis).
  2. Correct feature ordering handling (permuted dictionary keys).
  3. Missing feature rejection.
  4. Extra / unexpected feature rejection.
  5. Invalid numeric values, non-numeric strings, NaN, +/- infinity rejection.
  6. Physical sanity bounds enforcement (negative values, percentages > 100%).
  7. Invalid scenario and model selection error handling.
  8. Consistent repeated predictions (deterministic output).
  9. Correct mapping of predicted classes and 9 probability columns.
 10. Correct application of audited binary thresholds for Task B.
 11. Graceful model-file loading errors when model file is missing or invalid.
 12. Evaluation on healthy vs. anemic sample profiles.
"""

import math
import unittest
from pathlib import Path

# Add project root to sys.path
import sys
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.apps.ml_service.validators import (
    validate_cbc_input,
    ValidationError,
    EXPECTED_FEATURES,
    PHYSICAL_SANITY_BOUNDS,
)
from backend.apps.ml_service.predictor import (
    predict_anemia_screening,
    predict_diagnosis,
    load_pipeline,
    get_model_metadata,
    CLINICAL_DISCLAIMER,
)


# Canonical realistic test blood profiles derived from dataset test records
HEALTHY_CBC_SAMPLE = {
    "WBC": 7.5,
    "LYMp": 32.0,
    "NEUTp": 60.0,
    "LYMn": 2.4,
    "NEUTn": 4.5,
    "RBC": 4.8,
    "HGB": 14.5,
    "HCT": 43.0,
    "MCV": 89.0,
    "MCH": 30.0,
    "MCHC": 33.7,
    "PLT": 250.0,
    "PDW": 12.0,
    "PCT": 0.25,
}

ANEMIA_CBC_SAMPLE = {
    "WBC": 6.8,
    "LYMp": 28.0,
    "NEUTp": 65.0,
    "LYMn": 1.9,
    "NEUTn": 4.4,
    "RBC": 3.8,
    "HGB": 8.5,
    "HCT": 28.0,
    "MCV": 73.0,
    "MCH": 22.0,
    "MCHC": 30.0,
    "PLT": 310.0,
    "PDW": 13.5,
    "PCT": 0.28,
}


class TestInputValidation(unittest.TestCase):
    """Test comprehensive input validation logic."""

    def test_valid_input_passes_and_orders_columns(self):
        """Valid dictionary with scrambled key order must produce 1-row DataFrame with exact EXPECTED_FEATURES order."""
        # Reverse key order
        scrambled = {k: HEALTHY_CBC_SAMPLE[k] for k in reversed(list(HEALTHY_CBC_SAMPLE.keys()))}
        df = validate_cbc_input(scrambled)
        self.assertEqual(len(df), 1)
        self.assertEqual(list(df.columns), EXPECTED_FEATURES)
        self.assertAlmostEqual(df.loc[0, "HGB"], 14.5)

    def test_missing_features_rejected(self):
        """Dropping any feature must raise ValidationError specifying the missing field."""
        incomplete = dict(HEALTHY_CBC_SAMPLE)
        del incomplete["HGB"]
        with self.assertRaises(ValidationError) as ctx:
            validate_cbc_input(incomplete)
        self.assertIn("Missing required CBC features", str(ctx.exception))
        self.assertIn("HGB", str(ctx.exception))

    def test_unexpected_extra_features_rejected(self):
        """Extra or target columns must be strictly rejected."""
        extra = dict(HEALTHY_CBC_SAMPLE)
        extra["Diagnosis"] = "Healthy"
        with self.assertRaises(ValidationError) as ctx:
            validate_cbc_input(extra)
        self.assertIn("Unexpected extra features detected", str(ctx.exception))
        self.assertIn("Diagnosis", str(ctx.exception))

    def test_nan_and_infinite_rejected(self):
        """NaN, +inf, -inf must raise ValidationError."""
        for bad_val in [float("nan"), float("inf"), float("-inf"), None]:
            bad_data = dict(HEALTHY_CBC_SAMPLE)
            bad_data["HGB"] = bad_val
            with self.assertRaises(ValidationError):
                validate_cbc_input(bad_data)

    def test_non_numeric_strings_and_booleans_rejected(self):
        """Non-numeric string values and booleans must be rejected."""
        for bad_val in ["low", "twelve", True, False, [12.0]]:
            bad_data = dict(HEALTHY_CBC_SAMPLE)
            bad_data["RBC"] = bad_val
            with self.assertRaises(ValidationError):
                validate_cbc_input(bad_data)

    def test_physically_impossible_negative_rejected(self):
        """Negative values for non-negative features like HGB or RBC must be rejected."""
        bad_data = dict(HEALTHY_CBC_SAMPLE)
        bad_data["HGB"] = -10.0
        with self.assertRaises(ValidationError) as ctx:
            validate_cbc_input(bad_data)
        self.assertIn("Physically invalid value for 'HGB'", str(ctx.exception))

    def test_physically_impossible_percentage_rejected(self):
        """Percentages exceeding 100% must be rejected."""
        bad_data = dict(HEALTHY_CBC_SAMPLE)
        bad_data["HCT"] = 316.0
        with self.assertRaises(ValidationError) as ctx:
            validate_cbc_input(bad_data)
        self.assertIn("Physically invalid value for 'HCT'", str(ctx.exception))

    def test_non_dict_input_rejected(self):
        """Passing a list or string instead of a dictionary must raise ValidationError."""
        with self.assertRaises(ValidationError):
            validate_cbc_input([1.0, 2.0, 3.0])


class TestPredictionServiceTaskB(unittest.TestCase):
    """Test Task B binary anemia screening function."""

    def test_predict_anemia_screening_healthy(self):
        """Healthy sample profile should predict class 0 with high healthy probability."""
        res = predict_anemia_screening(HEALTHY_CBC_SAMPLE, scenario="scenario_2", model_name="random_forest")
        self.assertEqual(res["task"], "Task B (Binary Anemia Screening)")
        self.assertEqual(res["predicted_class"], 0)
        self.assertIn("Healthy", res["screening_result"])
        self.assertEqual(res["scenario"], "scenario2")
        self.assertEqual(res["model_used"], "random_forest")
        self.assertGreater(res["probability_healthy"], 0.5)
        self.assertIn(CLINICAL_DISCLAIMER, res["clinical_disclaimer"])

    def test_predict_anemia_screening_anemia(self):
        """Anemia sample profile should predict class 1 with high anemia probability."""
        res = predict_anemia_screening(ANEMIA_CBC_SAMPLE, scenario="scenario_2", model_name="random_forest")
        self.assertEqual(res["predicted_class"], 1)
        self.assertIn("Anemia", res["screening_result"])
        self.assertGreater(res["probability_anemia"], 0.5)

    def test_threshold_application(self):
        """Verify that the exact threshold from metadata is recorded and applied for all 6 Task B combinations."""
        # Expected authoritative thresholds established in Phase 5 benchmark artifacts
        expected_thresholds = {
            ("scenario1", "logistic_regression"): 0.39,
            ("scenario1", "decision_tree"): 0.10,
            ("scenario1", "random_forest"): 0.41,
            ("scenario2", "logistic_regression"): 0.34,
            ("scenario2", "decision_tree"): 0.10,
            ("scenario2", "random_forest"): 0.37,
        }

        meta = get_model_metadata()

        for (scenario, model_name), expected_th in expected_thresholds.items():
            model_key = f"task_b_{scenario}_{model_name}"
            # Verify threshold stored in metadata matches authoritative value
            meta_th = meta["models"][model_key]["tuned_threshold"]
            self.assertAlmostEqual(
                meta_th,
                expected_th,
                places=3,
                msg=f"Metadata threshold mismatch for {model_key}",
            )

            # Verify prediction service retrieves and applies this exact threshold
            res = predict_anemia_screening(
                HEALTHY_CBC_SAMPLE,
                scenario=scenario,
                model_name=model_name,
            )
            self.assertAlmostEqual(
                res["applied_threshold"],
                expected_th,
                places=3,
                msg=f"Applied threshold mismatch in prediction result for {model_key}",
            )

    def test_threshold_decision_boundary(self):
        """Verify that probability >= threshold classifies as 1 (Anemia) and < threshold classifies as 0 (Healthy)."""
        # Test with Logistic Regression on Scenario 1 (threshold 0.39) and Scenario 2 (threshold 0.34)
        for sc in ["scenario_1", "scenario_2"]:
            res = predict_anemia_screening(HEALTHY_CBC_SAMPLE, scenario=sc, model_name="logistic_regression")
            prob = res["probability_anemia"]
            thresh = res["applied_threshold"]
            if prob >= thresh:
                self.assertEqual(res["predicted_class"], 1)
            else:
                self.assertEqual(res["predicted_class"], 0)


class TestPredictionServiceTaskA(unittest.TestCase):
    """Test Task A multiclass diagnosis classification function."""

    def test_predict_diagnosis_healthy(self):
        """Healthy blood profile should predict 'Healthy' diagnosis with 9 class probabilities."""
        res = predict_diagnosis(HEALTHY_CBC_SAMPLE, scenario="scenario_2", model_name="random_forest")
        self.assertEqual(res["task"], "Task A (Multiclass Diagnosis Classification)")
        self.assertEqual(res["predicted_diagnosis"], "Healthy")
        self.assertIsNotNone(res["class_probabilities"])
        self.assertEqual(len(res["class_probabilities"]), 9)
        self.assertAlmostEqual(sum(res["class_probabilities"].values()), 1.0, places=1)
        self.assertIn(CLINICAL_DISCLAIMER, res["clinical_disclaimer"])

    def test_predict_diagnosis_iron_deficiency(self):
        """Iron deficiency sample profile should predict an anemia diagnosis."""
        res = predict_diagnosis(ANEMIA_CBC_SAMPLE, scenario="scenario_2", model_name="random_forest")
        self.assertIn("anemia", res["predicted_diagnosis"].lower())
        self.assertEqual(len(res["class_probabilities"]), 9)

    def test_all_three_models_task_a(self):
        """All three candidate models (logistic_regression, decision_tree, random_forest) run cleanly."""
        for m in ["logistic_regression", "decision_tree", "random_forest"]:
            res = predict_diagnosis(HEALTHY_CBC_SAMPLE, scenario="scenario_2", model_name=m)
            self.assertIn(res["predicted_diagnosis"], [
                "Healthy", "Iron deficiency anemia", "Leukemia", "Leukemia with thrombocytopenia",
                "Macrocytic anemia", "Normocytic hypochromic anemia", "Normocytic normochromic anemia",
                "Other microcytic anemia", "Thrombocytopenia"
            ])


class TestDeterministicAndEdgeCases(unittest.TestCase):
    """Test repeatability, invalid configurations, and missing files."""

    def test_deterministic_predictions(self):
        """Calling prediction functions repeatedly on same input must produce identical results."""
        res1 = predict_anemia_screening(HEALTHY_CBC_SAMPLE)
        res2 = predict_anemia_screening(HEALTHY_CBC_SAMPLE)
        self.assertEqual(res1["predicted_class"], res2["predicted_class"])
        self.assertEqual(res1["probability_anemia"], res2["probability_anemia"])

        res_a1 = predict_diagnosis(HEALTHY_CBC_SAMPLE)
        res_a2 = predict_diagnosis(HEALTHY_CBC_SAMPLE)
        self.assertEqual(res_a1["predicted_diagnosis"], res_a2["predicted_diagnosis"])
        self.assertEqual(res_a1["class_probabilities"], res_a2["class_probabilities"])

    def test_invalid_scenario_raises_value_error(self):
        """Non-existent scenario name raises ValueError."""
        with self.assertRaises(ValueError):
            predict_anemia_screening(HEALTHY_CBC_SAMPLE, scenario="scenario_99")

    def test_invalid_model_raises_value_error(self):
        """Non-existent model name raises ValueError."""
        with self.assertRaises(ValueError):
            predict_diagnosis(HEALTHY_CBC_SAMPLE, model_name="deep_neural_net")

    def test_load_pipeline_graceful_missing_file_error(self):
        """load_pipeline raises FileNotFoundError if model file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_pipeline("task_c", "scenario1", "non_existent_model")


if __name__ == "__main__":
    unittest.main()
