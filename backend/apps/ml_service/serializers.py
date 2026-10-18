"""
serializers.py — Request and Response Serializers for ML Prediction Endpoints
=============================================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Educational Use Only (Not Clinically Validated)

PURPOSE:
--------
Serializers for validating incoming CBC request payloads and serializing
prediction outputs for Task B (binary screening) and Task A (multiclass classification).
Reuses existing validation rules from backend.apps.ml_service.validators.
"""

from rest_framework import serializers
from typing import Dict, Any

from backend.apps.ml_service.validators import (
    validate_cbc_input,
    ValidationError as MLValidationError,
    EXPECTED_FEATURES,
    PHYSICAL_SANITY_BOUNDS,
)
from backend.apps.ml_service.predictor import (
    VALID_SCENARIOS,
    VALID_MODELS,
)


class CBCPredictionRequestSerializer(serializers.Serializer):
    """
    Serializer for validating patient CBC blood report input.

    Accepts:
      - 14 CBC features as top-level fields (or inside a 'cbc_values' dict if provided).
      - Optional 'scenario': 'scenario_1', 'scenario_2' (default: 'scenario_2').
      - Optional 'model_name': 'random_forest' (default), 'decision_tree', 'logistic_regression'.

    Rejects:
      - Missing CBC features
      - Unexpected / extra features (including target columns like 'Diagnosis')
      - Non-numeric, boolean, NaN, or infinite values
      - Out-of-bounds / physically impossible values
    """

    scenario = serializers.ChoiceField(
        choices=sorted(list(VALID_SCENARIOS.keys())),
        default="scenario_2",
        required=False,
        help_text="Preprocessing scenario: 'scenario_1' (primary) or 'scenario_2' (conservative, default).",
    )
    model_name = serializers.ChoiceField(
        choices=sorted(list(VALID_MODELS)),
        default="random_forest",
        required=False,
        help_text="Model architecture: 'random_forest' (default), 'decision_tree', or 'logistic_regression'.",
    )

    def to_internal_value(self, data):
        """
        Validate incoming data against CBC feature requirements and configuration.
        """
        if not isinstance(data, dict):
            raise serializers.ValidationError({
                "non_field_errors": ["Request body must be a JSON object."]
            })

        # Separate scenario and model_name from CBC features
        scenario = data.get("scenario", "scenario_2")
        model_name = data.get("model_name", "random_forest")

        # Validate scenario choice
        if scenario not in VALID_SCENARIOS:
            raise serializers.ValidationError({
                "scenario": [
                    f"Invalid scenario '{scenario}'. Allowed choices: {sorted(list(VALID_SCENARIOS.keys()))}."
                ]
            })

        # Validate model_name choice
        if model_name not in VALID_MODELS:
            raise serializers.ValidationError({
                "model_name": [
                    f"Invalid model_name '{model_name}'. Allowed choices: {sorted(list(VALID_MODELS))}."
                ]
            })

        # Check for explicit target column leakage
        forbidden_targets = {"Diagnosis", "Diagnosis_binary", "target", "label"}
        detected_targets = forbidden_targets.intersection(set(data.keys()))
        if detected_targets:
            raise serializers.ValidationError({
                "non_field_errors": [
                    f"Target column(s) {sorted(list(detected_targets))} detected in input. Target fields cannot be passed to prediction endpoints."
                ]
            })

        # Extract CBC features: support flat payload or nested 'cbc_values'
        if "cbc_values" in data and isinstance(data["cbc_values"], dict):
            cbc_dict = data["cbc_values"]
            # Check for extra keys in top level
            top_level_extras = set(data.keys()) - {"scenario", "model_name", "cbc_values"}
            if top_level_extras:
                raise serializers.ValidationError({
                    "non_field_errors": [
                        f"Unexpected extra fields in request: {sorted(list(top_level_extras))}."
                    ]
                })
        else:
            # Flat payload
            cbc_dict = {k: v for k, v in data.items() if k not in ("scenario", "model_name")}

        # Perform strict ML service validation
        try:
            # validate_cbc_input raises ValidationError with descriptive message
            validate_cbc_input(cbc_dict)
        except MLValidationError as e:
            err_msg = str(e)
            # Try to associate error with specific field if possible
            for feat in EXPECTED_FEATURES:
                if f"'{feat}'" in err_msg or f"for '{feat}'" in err_msg or f"Missing required CBC features" in err_msg and feat in err_msg:
                    raise serializers.ValidationError({feat: [err_msg]})
            # Otherwise return as non_field_errors
            raise serializers.ValidationError({"non_field_errors": [err_msg]})
        except Exception as e:
            raise serializers.ValidationError({"non_field_errors": [f"Invalid input: {str(e)}"]})

        return {
            "cbc_values": cbc_dict,
            "scenario": scenario,
            "model_name": model_name,
        }
