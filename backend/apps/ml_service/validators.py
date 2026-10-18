"""
validators.py — Input Validation for CBC Blood Test Features
============================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Educational Use Only (Not Clinically Validated)

PURPOSE:
--------
Provides rigorous, independent input validation for patient Complete Blood Count (CBC)
records prior to machine learning inference.

Guarantees:
  1. Exactly the 14 expected CBC features are present (no missing, no extra keys).
  2. All values are convertible to finite floats (no NaN, null, +/-inf, strings).
  3. No target columns or surrogate identifiers are accepted.
  4. Physical/biological sanity bounds are enforced using conservative limits
     documented in Phase 3 exploratory data analysis to reject clearly corrupt records.
  5. Does NOT claim that values within range represent clinically normal physiology.
"""

import math
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd


# Exact 14 CBC features in standardized order
EXPECTED_FEATURES: List[str] = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC",
    "PLT", "PDW", "PCT",
]

# Physical and biological sanity bounds
# Format: feature_name -> (min_valid, max_valid, unit, clinical_rationale)
# Note: These are conservative sanity limits to detect physical impossibilities
# and gross measurement/unit errors (e.g. negative HGB, percentage > 100%).
# They do NOT define diagnostic reference intervals.
PHYSICAL_SANITY_BOUNDS: Dict[str, Tuple[float, float, str, str]] = {
    "WBC": (
        0.1, 500.0, "x10^3/uL",
        "White blood cell count. Values below 0.1 or above 500 indicate profound instrument error."
    ),
    "LYMp": (
        0.0, 100.0, "%",
        "Lymphocyte differential percentage. Cannot be negative or exceed 100%."
    ),
    "NEUTp": (
        0.0, 100.0, "%",
        "Neutrophil differential percentage. Cannot be negative or exceed 100%."
    ),
    "LYMn": (
        0.0, 100.0, "x10^3/uL",
        "Absolute lymphocyte count. Cannot be negative."
    ),
    "NEUTn": (
        0.0, 100.0, "x10^3/uL",
        "Absolute neutrophil count. Cannot be negative."
    ),
    "RBC": (
        0.5, 15.0, "x10^6/uL",
        "Red blood cell count. Physiological maximum rarely exceeds 10; values >15 are unit errors."
    ),
    "HGB": (
        0.5, 30.0, "g/dL",
        "Hemoglobin concentration. Cannot be negative; values >30 are physically impossible."
    ),
    "HCT": (
        1.0, 100.0, "%",
        "Hematocrit percentage. Cannot be negative or exceed 100%."
    ),
    "MCV": (
        10.0, 250.0, "fL",
        "Mean corpuscular volume. Erythrocyte volume cannot be negative; clinical macrocytosis <250."
    ),
    "MCH": (
        5.0, 150.0, "pg",
        "Mean corpuscular hemoglobin. Physiological upper clinical limit is ~45-50 pg."
    ),
    "MCHC": (
        10.0, 60.0, "g/dL",
        "Mean corpuscular hemoglobin concentration. Hyperchromia >40 is rare; >50 is physiologically implausible."
    ),
    "PLT": (
        1.0, 3000.0, "x10^3/uL",
        "Platelet count. Cannot be negative; extreme thrombocytosis rarely exceeds 2000."
    ),
    "PDW": (
        1.0, 60.0, "fL",
        "Platelet distribution width. Width cannot be negative; clinical PDW >40 indicates instrument artifact."
    ),
    "PCT": (
        0.0, 10.0, "%",
        "Plateletcrit percentage. Plateletcrit >5% is extreme; typical normal is 0.1-0.5%, ceiling 10%."
    ),
}


class ValidationError(ValueError):
    """Custom exception raised when blood report data fails input validation."""
    pass


def validate_cbc_input(cbc_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate and format input dictionary of CBC measurements into a single-row DataFrame.

    Parameters
    ----------
    cbc_data : dict
        Mapping of feature name to numeric measurement.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with exact columns and ordering matching EXPECTED_FEATURES.

    Raises
    ------
    ValidationError
        If missing features, extra features, non-numeric values, NaN/infinity,
        or physically impossible values are detected.
    """
    if not isinstance(cbc_data, dict):
        raise ValidationError(f"Expected input to be a dictionary, got {type(cbc_data).__name__}.")

    input_keys = set(cbc_data.keys())
    expected_set = set(EXPECTED_FEATURES)

    # 1. Check for missing features
    missing = expected_set - input_keys
    if missing:
        raise ValidationError(
            f"Missing required CBC features: {sorted(list(missing))}. "
            f"Exactly 14 features required: {EXPECTED_FEATURES}."
        )

    # 2. Check for unexpected / extra features
    extras = input_keys - expected_set
    if extras:
        raise ValidationError(
            f"Unexpected extra features detected: {sorted(list(extras))}. "
            f"Input must contain only the 14 expected CBC features."
        )

    # 3. Numeric conversion and finiteness check
    cleaned_values: Dict[str, float] = {}
    for feat in EXPECTED_FEATURES:
        val = cbc_data[feat]

        if val is None:
            raise ValidationError(f"Feature '{feat}' cannot be None / null.")

        # Reject booleans explicitly (since in Python bool is a subclass of int)
        if isinstance(val, bool):
            raise ValidationError(f"Feature '{feat}' cannot be boolean ({val}).")

        # Reject non-scalar types (list, tuple, dict, etc.)
        if isinstance(val, (list, tuple, dict, set)):
            raise ValidationError(f"Feature '{feat}' cannot be a container type ({type(val).__name__}).")

        # For strings, attempt float conversion
        if isinstance(val, str):
            try:
                float_val = float(val)
            except (ValueError, TypeError):
                raise ValidationError(f"Feature '{feat}' must be numeric, got non-numeric string '{val}'.")
        else:
            try:
                float_val = float(val)
            except (ValueError, TypeError):
                raise ValidationError(f"Feature '{feat}' must be numeric, got {type(val).__name__} ('{val}').")

        if math.isnan(float_val):
            raise ValidationError(f"Feature '{feat}' has NaN (missing) value. NaN is rejected.")

        if math.isinf(float_val):
            raise ValidationError(f"Feature '{feat}' has infinite value ({float_val}). Infinite values are rejected.")

        # 4. Check physical/biological sanity limits
        min_bound, max_bound, unit, rationale = PHYSICAL_SANITY_BOUNDS[feat]
        if float_val < min_bound or float_val > max_bound:
            raise ValidationError(
                f"Physically invalid value for '{feat}': {float_val} {unit}. "
                f"Permissible range is [{min_bound}, {max_bound}] {unit}. "
                f"Rationale: {rationale}"
            )

        cleaned_values[feat] = float_val

    # Construct single-row DataFrame with strict column order
    df_row = pd.DataFrame([cleaned_values], columns=EXPECTED_FEATURES)
    return df_row
