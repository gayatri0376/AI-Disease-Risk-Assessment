"""
predict.py — ML Inference Module
=================================
AI-Based Anemia Screening and Blood Report Interpretation System
Module 2: Machine Learning and Model Evaluation

PURPOSE:
--------
Provides standalone, robust, and tested inference capabilities for:
  • Task A: 9-class multiclass diagnosis prediction.
  • Task B: Binary anemia screening (0 = Healthy, 1 = Anemia Subtypes).

KEY DESIGN PRINCIPLES:
----------------------
1. Decoupled & Configurable:
   - Supports Logistic Regression, Decision Tree, and Random Forest models.
   - Defaults to Scenario 2 provisional models (clean training data).
2. Strict Physiological Input Validation:
   - Verifies all 14 required CBC features are present and numeric.
   - Rejects NaN, infinity, negative counts, impossible percentages (>100%),
     and clearly impossible physiological values using documented rules.
   - Distinguishes basic input validation from clinical diagnosis.
3. Preserves Pipeline Integrity:
   - Passes unscaled, raw inputs in the exact column order expected by the pipeline.
   - Feature scaling and imputation remain encapsulated inside the trained pipeline.
4. Threshold-Aware:
   - Dynamically loads training-tuned decision thresholds for Task B from metadata.
   - Thresholds are never applied to Task A.
5. Educational & Research Compliance:
   - Explicitly designates scores as uncalibrated model outputs (not clinical risk).
   - Embeds mandatory clinical disclaimers on every prediction.
   - Excludes age and sex from the feature vector.
"""

from __future__ import annotations

import json
import joblib
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

# =============================================================================
# Path & Directory Setup
# =============================================================================
THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
ML_DIR = SRC_DIR.parent
PROJECT_ROOT = ML_DIR.parent

MODELS_DIR = ML_DIR / "models"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Exact 14 CBC feature columns in order
FEATURE_COLUMNS: list[str] = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC",
    "PLT", "PDW", "PCT",
]

# Task & Model Constants
VALID_TASKS = {"task_a", "task_b"}
VALID_SCENARIOS = {"scenario1", "scenario2"}
VALID_MODELS = {"logistic_regression", "decision_tree", "random_forest"}

DEFAULT_TASK = "task_a"
DEFAULT_SCENARIO = "scenario2"
DEFAULT_MODEL = "random_forest"

TASK_B_CLASS_LABELS = {
    0: "Healthy",
    1: "Anemia Subtypes",
}

# =============================================================================
# Documented Physiological Plausibility Limits
# =============================================================================
# These limits represent broad biological and physical plausibility boundaries
# to catch corrupted inputs, unit mismatches, or data entry errors.
# They are NOT clinical reference ranges for normal health.
DEFAULT_PHYSIOLOGICAL_LIMITS: dict[str, dict[str, Any]] = {
    "WBC": {
        "min": 0.1,
        "max": 500.0,
        "unit": "x10^3/uL",
        "description": "White blood cell count (non-zero living count up to severe leukemoid reaction)",
    },
    "LYMp": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "description": "Lymphocyte percentage (physical percentage bound)",
    },
    "NEUTp": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "description": "Neutrophil percentage (physical percentage bound)",
    },
    "LYMn": {
        "min": 0.0,
        "max": 100.0,
        "unit": "x10^3/uL",
        "description": "Absolute lymphocyte count (non-negative count)",
    },
    "NEUTn": {
        "min": 0.0,
        "max": 100.0,
        "unit": "x10^3/uL",
        "description": "Absolute neutrophil count (non-negative count)",
    },
    "RBC": {
        "min": 0.5,
        "max": 15.0,
        "unit": "x10^6/uL",
        "description": "Red blood cell count (extreme anemia to absolute polycythemia ceiling)",
    },
    "HGB": {
        "min": 0.5,
        "max": 30.0,
        "unit": "g/dL",
        "description": "Hemoglobin concentration (extreme anemia to polycythemic ceiling)",
    },
    "HCT": {
        "min": 1.0,
        "max": 100.0,
        "unit": "%",
        "description": "Hematocrit percentage (physical percentage bound)",
    },
    "MCV": {
        "min": 10.0,
        "max": 250.0,
        "unit": "fL",
        "description": "Mean corpuscular volume (extreme microcytosis to megaloblastic ceiling)",
    },
    "MCH": {
        "min": 5.0,
        "max": 150.0,
        "unit": "pg",
        "description": "Mean corpuscular hemoglobin (severe hypochromia to physiological ceiling)",
    },
    "MCHC": {
        "min": 10.0,
        "max": 60.0,
        "unit": "g/dL",
        "description": "Mean corpuscular hemoglobin concentration (hemoglobin solubility limit ~50-60 g/dL)",
    },
    "PLT": {
        "min": 1.0,
        "max": 3000.0,
        "unit": "x10^3/uL",
        "description": "Platelet count (extreme thrombocytopenia to extreme thrombocythemia)",
    },
    "PDW": {
        "min": 1.0,
        "max": 60.0,
        "unit": "fL",
        "description": "Platelet distribution width",
    },
    "PCT": {
        "min": 0.0,
        "max": 10.0,
        "unit": "%",
        "description": "Plateletcrit percentage (platelet volume fraction; normal 0.1-0.5%, ceiling 10%)",
    },
}

MANDATORY_DISCLAIMER = (
    "RESEARCH AND EDUCATIONAL USE ONLY: This output is generated by a provisional machine learning "
    "baseline and does NOT constitute a medical diagnosis, clinical finding, or treatment plan. "
    "The models were trained on a single curated dataset lacking patient demographic variables "
    "(age, biological sex, pregnancy status) and have not undergone multi-center external clinical validation. "
    "All complete blood count (CBC) values must be evaluated in clinical context by a licensed healthcare professional."
)

SCORE_INTERPRETATION_NOTE = (
    "Scores represent raw uncalibrated model outputs (decision tree leaf empirical proportions, "
    "random forest ensemble vote fractions, or logistic regression sigmoid outputs). "
    "They do NOT represent calibrated posterior probabilities or clinical risk percentages."
)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================
class InferenceError(Exception):
    """Base exception for all inference and prediction errors."""
    pass


class InputValidationError(InferenceError, ValueError):
    """Raised when an input payload fails feature presence, datatype, or physiological checks."""
    pass


class ModelArtifactError(InferenceError, FileNotFoundError):
    """Raised when a requested model pipeline file or metadata artifact cannot be found."""
    pass


# =============================================================================
# Model Caching & Metadata Loading
# =============================================================================
_PIPELINE_CACHE: dict[str, Any] = {}
_METADATA_CACHE: dict[str, Any] | None = None


def load_metadata(path: Path = METADATA_PATH) -> dict[str, Any]:
    """Load model metadata JSON with in-memory caching."""
    global _METADATA_CACHE
    if _METADATA_CACHE is not None:
        return _METADATA_CACHE

    if not path.exists():
        raise ModelArtifactError(f"Model metadata file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        _METADATA_CACHE = json.load(f)
    return _METADATA_CACHE


def get_model_pipeline_path(task: str, scenario: str, model_name: str) -> Path:
    """Generate expected path for a saved model pipeline."""
    return MODELS_DIR / f"{task}_{scenario}_{model_name}.joblib"


def load_model_pipeline(
    task: str = DEFAULT_TASK,
    scenario: str = DEFAULT_SCENARIO,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> Any:
    """
    Load a serialized scikit-learn Pipeline from disk.

    Parameters
    ----------
    task : str
        'task_a' or 'task_b'.
    scenario : str
        'scenario1' or 'scenario2'.
    model_name : str
        'random_forest', 'logistic_regression', or 'decision_tree'.
    use_cache : bool
        If True, caches loaded pipelines in memory for fast repeated calls.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The loaded pipeline object.
    """
    task = _normalize_task(task)
    scenario = _normalize_scenario(scenario)
    model_name = _normalize_model_name(model_name)

    cache_key = f"{task}_{scenario}_{model_name}"
    if use_cache and cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    model_path = get_model_pipeline_path(task, scenario, model_name)
    if not model_path.exists():
        raise ModelArtifactError(f"Saved pipeline artifact not found at: {model_path}")

    pipeline = joblib.load(model_path)
    if use_cache:
        _PIPELINE_CACHE[cache_key] = pipeline

    return pipeline


def get_tuned_threshold(
    task: str,
    scenario: str,
    model_name: str,
    metadata_path: Path = METADATA_PATH,
) -> float | None:
    """
    Retrieve the training-tuned decision threshold for Task B from model metadata.

    Returns None for Task A (thresholds do not apply to multiclass prediction).
    """
    task = _normalize_task(task)
    if task != "task_b":
        return None

    scenario = _normalize_scenario(scenario)
    model_name = _normalize_model_name(model_name)
    model_key = f"{task}_{scenario}_{model_name}"

    try:
        metadata = load_metadata(metadata_path)
        if "models" in metadata and model_key in metadata["models"]:
            m_info = metadata["models"][model_key]
            if "tuned_threshold" in m_info:
                return float(m_info["tuned_threshold"])
    except Exception:
        pass

    # Fallback default if metadata cannot be read
    return 0.50


# =============================================================================
# Input Validation Logic
# =============================================================================
def validate_cbc_input(
    features: dict[str, Any],
    custom_limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Validate that an input dictionary contains all 14 required CBC features
    with finite, physically plausible numeric values.

    Parameters
    ----------
    features : dict
        Mapping of feature names to values.
    custom_limits : dict, optional
        Custom physiological boundary definitions overriding defaults.

    Returns
    -------
    dict[str, float]
        Dictionary of validated float features.

    Raises
    ------
    InputValidationError
        If any feature is missing, non-numeric, NaN, infinite, or out of bounds.
    """
    if not isinstance(features, dict):
        raise InputValidationError(f"Input features must be a dictionary, got {type(features).__name__}")

    limits = custom_limits if custom_limits is not None else DEFAULT_PHYSIOLOGICAL_LIMITS
    errors: list[str] = []
    validated: dict[str, float] = {}

    # 1. Check for missing required features
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise InputValidationError(f"Missing required CBC feature(s): {missing}. Exactly 14 features are required.")

    # 2. Check each required feature
    for col in FEATURE_COLUMNS:
        raw_val = features[col]

        # Check for None / non-numeric
        if raw_val is None:
            errors.append(f"Feature '{col}' cannot be None.")
            continue

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            errors.append(f"Feature '{col}' must be numeric, got {raw_val!r} of type {type(raw_val).__name__}.")
            continue

        # Check for NaN / Infinity
        if math.isnan(val):
            errors.append(f"Feature '{col}' cannot be NaN.")
            continue
        if math.isinf(val):
            errors.append(f"Feature '{col}' cannot be infinite.")
            continue

        # Check physiological limits
        if col in limits:
            col_limits = limits[col]
            min_limit = col_limits.get("min")
            max_limit = col_limits.get("max")
            unit = col_limits.get("unit", "")

            if min_limit is not None and val < min_limit:
                errors.append(
                    f"Feature '{col}' value {val} is below minimum physiological boundary of {min_limit} {unit}."
                )
            elif max_limit is not None and val > max_limit:
                errors.append(
                    f"Feature '{col}' value {val} exceeds maximum physiological boundary of {max_limit} {unit}."
                )

        validated[col] = val

    # 3. Check differential percentage consistency
    # NEUTp + LYMp should not physically exceed 100% in a blood smear
    if "NEUTp" in validated and "LYMp" in validated:
        diff_sum = validated["NEUTp"] + validated["LYMp"]
        if diff_sum > 100.0:
            errors.append(
                f"Differential percentages NEUTp ({validated['NEUTp']}%) + LYMp ({validated['LYMp']}%) "
                f"sum to {diff_sum:.1f}%, which exceeds 100%."
            )

    if errors:
        raise InputValidationError(
            f"Input validation failed with {len(errors)} error(s):\n  - " + "\n  - ".join(errors)
        )

    return validated


# =============================================================================
# Core Prediction Interface
# =============================================================================
def predict(
    features: dict[str, Any],
    task: str = DEFAULT_TASK,
    model_name: str = DEFAULT_MODEL,
    scenario: str = DEFAULT_SCENARIO,
    threshold: float | None = None,
    custom_limits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Execute standalone ML inference for an individual patient CBC profile.

    Parameters
    ----------
    features : dict[str, Any]
        Dictionary containing all 14 required CBC parameters.
        Extraneous keys (e.g., 'age', 'sex', 'patient_id') are ignored and not
        passed to the machine learning feature vector.
    task : str
        'task_a' (9-class multiclass diagnosis) or
        'task_b' (binary screening: 0=Healthy, 1=Anemia).
    model_name : str
        'random_forest' (default), 'logistic_regression', or 'decision_tree'.
    scenario : str
        'scenario2' (default clean training) or 'scenario1'.
    threshold : float, optional
        Custom decision threshold for Task B. If None, the optimal
        cross-validation threshold from model metadata is used.
        Ignored for Task A.
    custom_limits : dict, optional
        Custom boundary limits to override physiological defaults.

    Returns
    -------
    dict[str, Any]
        Structured, documented prediction payload.
    """
    # 1. Normalize configuration parameters
    norm_task = _normalize_task(task)
    norm_scenario = _normalize_scenario(scenario)
    norm_model = _normalize_model_name(model_name)

    # 2. Validate input features
    validated_features = validate_cbc_input(features, custom_limits=custom_limits)

    # 3. Construct input DataFrame in exact feature order without manual scaling
    input_row = [validated_features[col] for col in FEATURE_COLUMNS]
    df_input = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)

    # 4. Load pipeline
    pipeline = load_model_pipeline(norm_task, norm_scenario, norm_model)

    # 5. Execute inference
    raw_pred = pipeline.predict(df_input)[0]

    # Calculate class scores if estimator supports predict_proba
    class_scores: dict[str, float] = {}
    has_proba = hasattr(pipeline, "predict_proba")
    proba_arr = pipeline.predict_proba(df_input)[0] if has_proba else None

    # 6. Task-specific formatting and decision logic
    effective_threshold: float | None = None
    predicted_code: int | None = None

    if norm_task == "task_a":
        # Multiclass diagnosis
        predicted_label = str(raw_pred)
        effective_threshold = None

        if proba_arr is not None:
            # Map probabilities to classes
            classes = pipeline.classes_
            class_scores = {
                str(cls_name): round(float(prob), 4)
                for cls_name, prob in zip(classes, proba_arr)
            }

    else:
        # Task B: Binary Anemia Screening
        # Determine threshold
        if threshold is not None:
            effective_threshold = float(threshold)
        else:
            meta_thresh = get_tuned_threshold(norm_task, norm_scenario, norm_model)
            effective_threshold = float(meta_thresh) if meta_thresh is not None else 0.50

        if proba_arr is not None:
            # Class 1 is Anemia Subtypes
            score_anemia = float(proba_arr[1])
            predicted_code = 1 if score_anemia >= effective_threshold else 0
            predicted_label = TASK_B_CLASS_LABELS[predicted_code]

            class_scores = {
                "Healthy (0)": round(float(proba_arr[0]), 4),
                "Anemia Subtypes (1)": round(float(score_anemia), 4),
            }
        else:
            predicted_code = int(raw_pred)
            predicted_label = TASK_B_CLASS_LABELS[predicted_code]

    # 7. Build structured result payload
    return {
        "task": norm_task,
        "task_description": "9-Class Multiclass Diagnosis" if norm_task == "task_a" else "Binary Anemia Screening",
        "scenario": norm_scenario,
        "model_name": norm_model,
        "model_artifact": f"{norm_task}_{norm_scenario}_{norm_model}.joblib",
        "predicted_label": predicted_label,
        "predicted_code": predicted_code,
        "decision_threshold": round(effective_threshold, 3) if effective_threshold is not None else None,
        "class_scores": class_scores,
        "score_interpretation": SCORE_INTERPRETATION_NOTE,
        "validated_features": validated_features,
        "validation_status": {
            "is_valid": True,
            "feature_count": len(validated_features),
            "feature_order": FEATURE_COLUMNS,
        },
        "disclaimer": MANDATORY_DISCLAIMER,
    }


def predict_task_a(
    features: dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    scenario: str = DEFAULT_SCENARIO,
) -> dict[str, Any]:
    """Convenience wrapper for Task A multiclass diagnosis."""
    return predict(features, task="task_a", model_name=model_name, scenario=scenario)


def predict_task_b(
    features: dict[str, Any],
    model_name: str = DEFAULT_MODEL,
    scenario: str = DEFAULT_SCENARIO,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for Task B binary screening."""
    return predict(features, task="task_b", model_name=model_name, scenario=scenario, threshold=threshold)


# =============================================================================
# Helper String Normalizers
# =============================================================================
def _normalize_task(task: str) -> str:
    t = str(task).strip().lower()
    if t in {"task_a", "a", "multiclass"}:
        return "task_a"
    if t in {"task_b", "b", "binary", "screening"}:
        return "task_b"
    raise ValueError(f"Invalid task '{task}'. Expected one of: {sorted(VALID_TASKS)}")


def _normalize_scenario(scenario: str) -> str:
    s = str(scenario).strip().lower()
    if s in {"scenario2", "s2", "clean"}:
        return "scenario2"
    if s in {"scenario1", "s1", "conservative"}:
        return "scenario1"
    raise ValueError(f"Invalid scenario '{scenario}'. Expected one of: {sorted(VALID_SCENARIOS)}")


def _normalize_model_name(model_name: str) -> str:
    m = str(model_name).strip().lower()
    if m in {"random_forest", "rf", "forest"}:
        return "random_forest"
    if m in {"logistic_regression", "lr", "logistic"}:
        return "logistic_regression"
    if m in {"decision_tree", "dt", "tree"}:
        return "decision_tree"
    raise ValueError(f"Invalid model_name '{model_name}'. Expected one of: {sorted(VALID_MODELS)}")


# =============================================================================
# CLI Demo & Execution
# =============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("AI-Based Anemia Screening — Inference Module Demo")
    print("=" * 75)

    sample_patient = {
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

    print("\n[Demo 1] Running Task A (Multiclass Diagnosis)...")
    res_a = predict(sample_patient, task="task_a", model_name="random_forest")
    print(f"  Model: {res_a['model_name']} ({res_a['scenario']})")
    print(f"  Predicted Label: {res_a['predicted_label']}")
    print(f"  Top Class Scores:")
    sorted_scores = sorted(res_a["class_scores"].items(), key=lambda x: x[1], reverse=True)
    for c, s in sorted_scores[:3]:
        print(f"    - {c}: {s:.4f}")

    print("\n[Demo 2] Running Task B (Binary Anemia Screening)...")
    res_b = predict(sample_patient, task="task_b", model_name="random_forest")
    print(f"  Model: {res_b['model_name']} ({res_b['scenario']})")
    print(f"  Tuned Threshold: {res_b['decision_threshold']}")
    print(f"  Predicted Label: {res_b['predicted_label']} (Code: {res_b['predicted_code']})")
    print(f"  Class Scores: {res_b['class_scores']}")
    print("\nDone.")
