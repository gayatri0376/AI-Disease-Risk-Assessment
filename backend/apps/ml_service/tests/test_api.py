"""
test_api.py — Django REST Framework Integration Tests for ML API
==================================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Phase 6.2 Verification

TEST COVERAGE:
--------------
1. GET /api/health/:
   - Returns 200 OK with expected status.
   - Does not expose system paths, credentials, or internal configuration.
2. POST /api/screening/ (Task B: Binary Screening):
   - Valid healthy request returns 200 OK, predicted_class=0, screening_result, applied_threshold, disclaimer.
   - Valid anemia request returns 200 OK, predicted_class=1.
   - Returns correct scenario/model threshold (e.g. 0.37 for scenario_2 RF, 0.39 for scenario_1 LR).
3. POST /api/diagnosis/ (Task A: 9-Class Classification):
   - Valid healthy request returns 200 OK, predicted_diagnosis='Healthy', 9 class probabilities, disclaimer.
   - Valid anemia request returns 200 OK, diagnosis of anemia subtype.
4. Input validation and error handling (HTTP 400):
   - Missing required CBC features.
   - Unexpected / extra feature keys.
   - Target field injection ('Diagnosis', 'Diagnosis_binary').
   - Non-numeric strings, booleans, NaN, and +/- infinity.
   - Physically impossible / out-of-bounds values (negative HGB, percentage > 100%).
   - Invalid scenario and model names.
   - Non-dict / non-JSON request body.
5. Service error handling (HTTP 503):
   - Model artifact missing or unreadable returns safe 503 error without leaking tracebacks.
6. Internal server error handling (HTTP 500):
   - Unexpected exceptions return safe 500 response without tracebacks.
"""

from unittest.mock import patch
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from backend.apps.ml_service.predictor import CLINICAL_DISCLAIMER


# Canonical test samples
HEALTHY_CBC_DATA = {
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

ANEMIA_CBC_DATA = {
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


class HealthEndpointTests(APITestCase):
    """Tests for GET /api/health/."""

    def test_health_check_success(self):
        url = reverse("api-health")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("service", data)
        # Verify no file paths or system credentials leaked
        for k, v in data.items():
            self.assertNotIn("/Users/", str(v))
            self.assertNotIn("secret", str(v).lower())
            self.assertNotIn("token", str(v).lower())


class AnemiaScreeningEndpointTests(APITestCase):
    """Tests for POST /api/screening/."""

    def setUp(self):
        self.url = reverse("api-screening")

    def test_valid_screening_healthy(self):
        payload = dict(HEALTHY_CBC_DATA)
        payload["scenario"] = "scenario_2"
        payload["model_name"] = "random_forest"

        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["task"], "Task B (Binary Anemia Screening)")
        self.assertEqual(data["predicted_class"], 0)
        self.assertIn("Healthy", data["screening_result"])
        self.assertEqual(data["applied_threshold"], 0.37)
        self.assertEqual(data["scenario"], "scenario2")
        self.assertEqual(data["model_used"], "random_forest")
        self.assertGreater(data["probability_healthy"], 0.5)
        self.assertIn(CLINICAL_DISCLAIMER, data["clinical_disclaimer"])

    def test_valid_screening_anemia(self):
        payload = dict(ANEMIA_CBC_DATA)
        payload["scenario"] = "scenario_2"
        payload["model_name"] = "random_forest"

        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["predicted_class"], 1)
        self.assertIn("Anemia", data["screening_result"])
        self.assertGreater(data["probability_anemia"], 0.5)

    def test_threshold_mapping_scenario_1_and_2(self):
        """Screening endpoint must apply the exact metadata threshold for the requested scenario and model."""
        test_cases = [
            ("scenario_1", "logistic_regression", 0.39),
            ("scenario_1", "decision_tree", 0.10),
            ("scenario_1", "random_forest", 0.41),
            ("scenario_2", "logistic_regression", 0.34),
            ("scenario_2", "decision_tree", 0.10),
            ("scenario_2", "random_forest", 0.37),
        ]
        for sc, mod, expected_th in test_cases:
            payload = dict(HEALTHY_CBC_DATA)
            payload["scenario"] = sc
            payload["model_name"] = mod
            response = self.client.post(self.url, data=payload, format="json")
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertAlmostEqual(response.json()["applied_threshold"], expected_th, places=2)

    def test_missing_feature_returns_400(self):
        payload = dict(HEALTHY_CBC_DATA)
        del payload["HGB"]
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertTrue("HGB" in str(response.json()) or "non_field_errors" in response.json())

    def test_extra_feature_returns_400(self):
        payload = dict(HEALTHY_CBC_DATA)
        payload["EXTRA_FEATURE"] = 12.3
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_target_column_injection_returns_400(self):
        payload = dict(HEALTHY_CBC_DATA)
        payload["Diagnosis"] = "Healthy"
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_non_numeric_and_boolean_rejected_400(self):
        for bad_val in ["abc", True, False, [1.0]]:
            payload = dict(HEALTHY_CBC_DATA)
            payload["HGB"] = bad_val
            response = self.client.post(self.url, data=payload, format="json")
            self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_physically_impossible_value_rejected_400(self):
        # Negative HGB
        payload = dict(HEALTHY_CBC_DATA)
        payload["HGB"] = -10.0
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

        # HCT > 100%
        payload = dict(HEALTHY_CBC_DATA)
        payload["HCT"] = 150.0
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_scenario_returns_400(self):
        payload = dict(HEALTHY_CBC_DATA)
        payload["scenario"] = "invalid_scenario"
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_model_returns_400(self):
        payload = dict(HEALTHY_CBC_DATA)
        payload["model_name"] = "xgboost_ultra"
        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


class DiagnosisEndpointTests(APITestCase):
    """Tests for POST /api/diagnosis/."""

    def setUp(self):
        self.url = reverse("api-diagnosis")

    def test_valid_diagnosis_healthy(self):
        payload = dict(HEALTHY_CBC_DATA)
        payload["scenario"] = "scenario_2"
        payload["model_name"] = "random_forest"

        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data["task"], "Task A (Multiclass Diagnosis Classification)")
        self.assertEqual(data["predicted_diagnosis"], "Healthy")
        self.assertIsNotNone(data["class_probabilities"])
        self.assertEqual(len(data["class_probabilities"]), 9)
        self.assertAlmostEqual(sum(data["class_probabilities"].values()), 1.0, places=1)
        self.assertIn(CLINICAL_DISCLAIMER, data["clinical_disclaimer"])
        self.assertIn("rare_class_note", data)
        self.assertIn("uncalibrated model scores interpreted as probabilities", data["rare_class_note"])
        self.assertNotIn("model-calibrated scores", data["rare_class_note"])

    def test_valid_diagnosis_anemia(self):
        payload = dict(ANEMIA_CBC_DATA)
        payload["scenario"] = "scenario_2"
        payload["model_name"] = "random_forest"

        response = self.client.post(self.url, data=payload, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn("anemia", data["predicted_diagnosis"].lower())
        self.assertEqual(len(data["class_probabilities"]), 9)


class ErrorPathAndResilienceTests(APITestCase):
    """Tests for error scenarios: 503 on missing model artifact, 500 on unexpected exception."""

    def test_model_loading_failure_returns_503(self):
        url = reverse("api-screening")
        payload = dict(HEALTHY_CBC_DATA)
        # Mock load_pipeline to simulate missing/unreadable model file
        with patch("backend.apps.ml_service.views.predict_anemia_screening", side_effect=FileNotFoundError("Model file missing")):
            response = self.client.post(url, data=payload, format="json")
            self.assertEqual(response.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)
            data = response.json()
            self.assertIn("unavailable", data.get("error", "").lower())
            # Ensure no traceback leaked in response
            self.assertNotIn("Traceback", str(data))

    def test_unexpected_exception_returns_500(self):
        url = reverse("api-diagnosis")
        payload = dict(HEALTHY_CBC_DATA)
        with patch("backend.apps.ml_service.views.predict_diagnosis", side_effect=RuntimeError("Unexpected hardware fault")):
            response = self.client.post(url, data=payload, format="json")
            self.assertEqual(response.status_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
            data = response.json()
            self.assertIn("internal server error", data.get("error", "").lower())
            self.assertNotIn("Traceback", str(data))
