"""
preprocessing.py — Phase 4: Data Preprocessing Pipeline
=========================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Educational Use Only

PURPOSE
-------
Provides reusable, well-documented functions for loading, cleaning,
and preparing the anemia dataset for machine learning.

Supports TWO preprocessing scenarios (approved in Phase 3 review):
  Scenario 1 (conservative): Remove duplicates + exclude 8 Category A rows only.
  Scenario 2 (aggressive):   Same as Scenario 1, plus exclude 21 Category B rows.

Does NOT train any model. Does NOT touch the original CSV.

HOW TO RUN (from project root):
    source .venv/bin/activate
    python ml_model/src/preprocessing.py

OUTPUTS
-------
  ml_model/reports/cleaning_log.csv         — every excluded record, with reason
  ml_model/reports/preprocessing_summary.txt — row counts and class distributions

IMPORTANT NOTES ON ROW NUMBERING
---------------------------------
  Original DataFrame index: 0-indexed (row 0 = first data row in CSV)
  CSV row number          : 1-indexed (row 1 = header, row 2 = first data row)
  Relationship            : csv_row = df_index + 2
  We use ORIGINAL df_index (from the full unmodified DataFrame) throughout,
  so indices never shift as rows are removed.
"""

# =============================================================================
# SECTION 0: Imports and Path Setup
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer   # kept for extensibility; no NaN currently

# ---------------------------------------------------------------------------
# Project-root-safe paths
# ---------------------------------------------------------------------------
THIS_FILE    = Path(__file__).resolve()
SRC_DIR      = THIS_FILE.parent          # ml_model/src/
ML_DIR       = SRC_DIR.parent            # ml_model/
PROJECT_ROOT = ML_DIR.parent             # AI-Anemia-Screening/

DATASET_PATH = ML_DIR / "dataset" / "anemia_types.csv"
REPORTS_DIR  = ML_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 1: Constants and Approved Exclusion Lists
# =============================================================================

# -- Column names --
TARGET_COLUMN   = "Diagnosis"
FEATURE_COLUMNS = [
    "WBC", "LYMp", "NEUTp", "LYMn", "NEUTn",
    "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC",
    "PLT", "PDW", "PCT",
]

# -- Splitting configuration --
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# -- Task B label mapping --
# Five anemia subtypes → label 1
# Healthy              → label 0
# Three non-anemia blood disorders → EXCLUDED from Task B entirely
ANEMIA_CLASSES = [
    "Iron deficiency anemia",
    "Normocytic hypochromic anemia",
    "Normocytic normochromic anemia",
    "Other microcytic anemia",
    "Macrocytic anemia",
]
HEALTHY_CLASS = "Healthy"
TASK_B_EXCLUDED_CLASSES = [
    "Leukemia",
    "Leukemia with thrombocytopenia",
    "Thrombocytopenia",
]

# ---------------------------------------------------------------------------
# APPROVED EXCLUSION LISTS (from Phase 3 suspicious record review)
#
# IMPORTANT: These are ORIGINAL DataFrame indices (0-indexed) from the
# full, unmodified DataFrame produced by pd.read_csv().
# Each index was cross-verified against the actual CSV values and diagnosis
# labels before approval. Do not add, remove, or modify entries here
# without re-running the verification step.
#
# CSV row number = df_index + 2.
# ---------------------------------------------------------------------------

# Category A: Demonstrably invalid values — approved for exclusion in both scenarios.
# Verification: ALL 8 rows confirmed by running cross-check against the raw CSV.
CATEGORY_A_DF_INDICES = [4, 216, 245, 344, 375, 383, 465, 472]

# Human-readable details for the cleaning log and documentation.
CATEGORY_A_DETAILS = {
    4: {
        "csv_row": 6,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "HCT",
        "offending_value": 316.0,
        "reason": (
            "HCT = 316.0%. Haematocrit is a percentage; values above 100% "
            "are physically impossible. This is a data-entry or unit error."
        ),
    },
    216: {
        "csv_row": 218,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "NEUTp",
        "offending_value": 5317.0,
        "reason": (
            "NEUTp = 5317.0%. Neutrophil percentage cannot exceed 100%. "
            "A raw count may have been entered instead of a percentage."
        ),
    },
    245: {
        "csv_row": 247,
        "diagnosis": "Iron deficiency anemia",
        "offending_feature": "HGB",
        "offending_value": -10.0,
        "reason": (
            "HGB = -10.0 g/dL. Haemoglobin concentration cannot be negative. "
            "This is physically impossible."
        ),
    },
    344: {
        "csv_row": 346,
        "diagnosis": "Iron deficiency anemia",
        "offending_feature": "MCV",
        "offending_value": -79.3,
        "reason": (
            "MCV = -79.3 fL. Cell volume cannot be negative. "
            "This is a data-entry error."
        ),
    },
    375: {
        "csv_row": 377,
        "diagnosis": "Iron deficiency anemia",
        "offending_feature": "HCT",
        "offending_value": 741.0,
        "reason": (
            "HCT = 741.0%. Haematocrit is a percentage; "
            "values above 100% are physically impossible."
        ),
    },
    383: {
        "csv_row": 385,
        "diagnosis": "Healthy",
        "offending_feature": "NEUTp",
        "offending_value": 671.0,
        "reason": (
            "NEUTp = 671.0%. Neutrophil percentage cannot exceed 100%. "
            "A raw count may have been entered instead of a percentage."
        ),
    },
    465: {
        "csv_row": 467,
        "diagnosis": "Normocytic normochromic anemia",
        "offending_feature": "MCH",
        "offending_value": 3117.0,
        "reason": (
            "MCH = 3117.0 pg. Physiological maximum is ~45 pg. "
            "This value is ~95x the maximum and is certainly a data-entry error."
        ),
    },
    472: {
        "csv_row": 474,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "HCT; MCH",
        "offending_value": "3715.0; 2511.0",
        "reason": (
            "HCT = 3715.0% (impossible percentage) AND MCH = 2511.0 pg "
            "(~78x physiological maximum). Multiple impossible values. "
            "This record cannot be corrected without the original source data."
        ),
    },
}

# Category B: Suspicious / unverifiable — retained in Scenario 1, excluded in Scenario 2.
# All 21 df_indices are verified against the original CSV.
# Record at df_index 345 (CSV row 347) is included here per the Phase 3 review.
# Per Phase 4 instructions: keep in Scenario 1, exclude only in Scenario 2.
CATEGORY_B_DF_INDICES = [
    5,    # CSV row   7 — Other microcytic anemia              — MCHC=79.6
    91,   # CSV row  93 — Normocytic hypochromic anemia        — MCHC=61.3
    96,   # CSV row  98 — Normocytic normochromic anemia       — MCHC=61.6
    201,  # CSV row 203 — Normocytic hypochromic anemia        — PCT=11.3
    206,  # CSV row 208 — Other microcytic anemia              — MCH=76, PCT=12.8
    237,  # CSV row 239 — Leukemia with thrombocytopenia       — NEUTn=79
    241,  # CSV row 243 — Other microcytic anemia              — RBC=41.1, HGB=85.1
    252,  # CSV row 254 — Thrombocytopenia                     — PDW=97
    271,  # CSV row 273 — Normocytic normochromic anemia       — MCH=90
    272,  # CSV row 274 — Healthy                              — MCHC=92.8
    273,  # CSV row 275 — Normocytic hypochromic anemia        — RBC=90.8
    345,  # CSV row 347 — Iron deficiency anemia               — LYMn=41.8, WBC=45.7
    351,  # CSV row 353 — Normocytic hypochromic anemia        — MCH=241
    356,  # CSV row 358 — Thrombocytopenia                     — MCV=990
    384,  # CSV row 386 — Other microcytic anemia              — MCH=88.2, PCT=13.6
    404,  # CSV row 406 — Thrombocytopenia                     — RBC=10.9, HGB=35.2, HCT=81.2
    438,  # CSV row 440 — Normocytic normochromic anemia       — MCH=275
    446,  # CSV row 448 — Other microcytic anemia              — RBC=29.7, HGB=87.1
    453,  # CSV row 455 — Normocytic hypochromic anemia        — RBC=13.1, HGB=41
    473,  # CSV row 475 — Normocytic hypochromic anemia        — PCT=12.0
    1134, # CSV row 1136 — Other microcytic anemia             — MCHC=50.2
]

# Detailed metadata for Category B records (logged during Scenario 2 exclusions)
CATEGORY_B_DETAILS = {
    5: {
        "csv_row": 7,
        "diagnosis": "Other microcytic anemia",
        "offending_feature": "MCHC",
        "offending_value": 79.6,
        "reason": (
            "MCHC = 79.6 g/dL. Clinical hyperchromia rarely exceeds 40 g/dL; "
            ">50 g/dL is physiologically implausible. Likely data-entry error, unverifiable."
        ),
    },
    91: {
        "csv_row": 93,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "MCHC",
        "offending_value": 61.3,
        "reason": (
            "MCHC = 61.3 g/dL. Exceeds clinical upper limit (~38 g/dL). "
            "Likely data-entry error, unverifiable."
        ),
    },
    96: {
        "csv_row": 98,
        "diagnosis": "Normocytic normochromic anemia",
        "offending_feature": "MCHC",
        "offending_value": 61.6,
        "reason": (
            "MCHC = 61.6 g/dL. Exceeds clinical upper limit (~38 g/dL). "
            "Likely data-entry error, unverifiable."
        ),
    },
    201: {
        "csv_row": 203,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "PCT",
        "offending_value": 11.3,
        "reason": (
            "PCT = 11.3%. Plateletcrit normal range is 0.1-0.5%; values >5% are extreme. "
            "Likely data-entry error, unverifiable."
        ),
    },
    206: {
        "csv_row": 208,
        "diagnosis": "Other microcytic anemia",
        "offending_feature": "MCH; PCT",
        "offending_value": "76.0; 12.8",
        "reason": (
            "MCH = 76.0 pg (upper clinical limit ~45 pg) and PCT = 12.8% (extreme). "
            "Likely unit/data-entry errors, unverifiable."
        ),
    },
    237: {
        "csv_row": 239,
        "diagnosis": "Leukemia with thrombocytopenia",
        "offending_feature": "NEUTn",
        "offending_value": 79.0,
        "reason": (
            "NEUTn = 79.0 x10^3/uL. Extreme neutrophilia (>50 x10^3/uL) is unusual even in severe infection. "
            "Associated with leukemia diagnosis; requires review."
        ),
    },
    241: {
        "csv_row": 243,
        "diagnosis": "Other microcytic anemia",
        "offending_feature": "RBC; HGB; MCHC",
        "offending_value": "41.1; 85.1; 12.4",
        "reason": (
            "RBC = 41.1 x10^6/uL (max plausible ~9-10), HGB = 85.1 g/dL (max plausible ~23), "
            "MCHC = 12.4 g/dL. Likely unit/data-entry errors, unverifiable."
        ),
    },
    252: {
        "csv_row": 254,
        "diagnosis": "Thrombocytopenia",
        "offending_feature": "PDW",
        "offending_value": 97.0,
        "reason": (
            "PDW = 97.0 fL. PDW above 40 is highly unusual. "
            "Appears to be a data-entry error, unverifiable."
        ),
    },
    271: {
        "csv_row": 273,
        "diagnosis": "Normocytic normochromic anemia",
        "offending_feature": "MCH",
        "offending_value": 90.0,
        "reason": (
            "MCH = 90.0 pg. Normal range is 27-33 pg; clinical upper limit ~45 pg. "
            "Likely unit/data-entry error, unverifiable."
        ),
    },
    272: {
        "csv_row": 274,
        "diagnosis": "Healthy",
        "offending_feature": "MCHC",
        "offending_value": 92.8,
        "reason": (
            "MCHC = 92.8 g/dL. Severe hyperchromia >50 g/dL does not occur physiologically. "
            "Likely data-entry error, unverifiable."
        ),
    },
    273: {
        "csv_row": 275,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "RBC",
        "offending_value": 90.8,
        "reason": (
            "RBC = 90.8 x10^6/uL. Maximum physiological RBC count is ~9-10 x10^6/uL. "
            "Likely unit error (e.g. x10^4), unverifiable."
        ),
    },
    345: {
        "csv_row": 347,
        "diagnosis": "Iron deficiency anemia",
        "offending_feature": "LYMn; WBC; LYMp",
        "offending_value": "41.8; 45.7; 91.4",
        "reason": (
            "Record 16. LYMn = 41.8 x10^3/uL, WBC = 45.7 x10^3/uL, LYMp = 91.4%. "
            "Extreme lymphocytosis in iron deficiency anemia. Kept in Scenario 1 and flagged for review; "
            "excluded in Scenario 2 for sensitivity analysis."
        ),
    },
    351: {
        "csv_row": 353,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "MCH",
        "offending_value": 241.0,
        "reason": (
            "MCH = 241.0 pg. Upper clinical limit ~45 pg; ~8x expected normal. "
            "Likely unit/data-entry error, unverifiable."
        ),
    },
    356: {
        "csv_row": 358,
        "diagnosis": "Thrombocytopenia",
        "offending_feature": "MCV",
        "offending_value": 990.0,
        "reason": (
            "MCV = 990.0 fL. Clinical maximum is ~130 fL even in severe macrocytosis. "
            "Likely data-entry error, unverifiable."
        ),
    },
    384: {
        "csv_row": 386,
        "diagnosis": "Other microcytic anemia",
        "offending_feature": "MCH; PCT",
        "offending_value": "88.2; 13.6",
        "reason": (
            "MCH = 88.2 pg and PCT = 13.6%. Both well above physiological maximums. "
            "Likely unit/data-entry errors, unverifiable."
        ),
    },
    404: {
        "csv_row": 406,
        "diagnosis": "Thrombocytopenia",
        "offending_feature": "RBC; HGB; HCT; MCV",
        "offending_value": "10.9; 35.2; 81.2; 25.1",
        "reason": (
            "RBC = 10.9 x10^6/uL, HGB = 35.2 g/dL, HCT = 81.2%, MCV = 25.1 fL. "
            "Multiple extreme values outside physiological ranges, unverifiable."
        ),
    },
    438: {
        "csv_row": 440,
        "diagnosis": "Normocytic normochromic anemia",
        "offending_feature": "MCH",
        "offending_value": 275.0,
        "reason": (
            "MCH = 275.0 pg. Upper clinical limit ~45 pg; ~9x normal. "
            "Likely unit/data-entry error, unverifiable."
        ),
    },
    446: {
        "csv_row": 448,
        "diagnosis": "Other microcytic anemia",
        "offending_feature": "RBC; HGB; MCHC",
        "offending_value": "29.7; 87.1; 11.5",
        "reason": (
            "RBC = 29.7 x10^6/uL, HGB = 87.1 g/dL, MCHC = 11.5 g/dL. "
            "Multiple impossible/extreme values; likely unit errors, unverifiable."
        ),
    },
    453: {
        "csv_row": 455,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "RBC; HGB; HCT",
        "offending_value": "13.1; 41.0; 2.0",
        "reason": (
            "RBC = 13.1 x10^6/uL, HGB = 41.0 g/dL, HCT = 2.0%. "
            "Extreme values outside reference ranges; likely unit errors, unverifiable."
        ),
    },
    473: {
        "csv_row": 475,
        "diagnosis": "Normocytic hypochromic anemia",
        "offending_feature": "PCT",
        "offending_value": 12.0,
        "reason": (
            "PCT = 12.0%. Plateletcrit >5% is extreme; typical range 0.1-0.5%. "
            "Likely data-entry error, unverifiable."
        ),
    },
    1134: {
        "csv_row": 1136,
        "diagnosis": "Other microcytic anemia",
        "offending_feature": "MCHC",
        "offending_value": 50.2,
        "reason": (
            "MCHC = 50.2 g/dL. Values above 50 g/dL virtually never occur physiologically. "
            "Likely data-entry error, unverifiable."
        ),
    },
}


# =============================================================================
# SECTION 2: Core Loading and Validation
# =============================================================================

def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the CSV dataset from the given path.

    Parameters
    ----------
    path : Path
        Path to the CSV file. Defaults to the project dataset.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset exactly as it appears in the CSV.

    Raises
    ------
    FileNotFoundError : If the file does not exist.
    ValueError        : If expected columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at:\n  {path}\n"
            f"Run from the project root:\n"
            f"  python ml_model/src/preprocessing.py\n"
        )

    df = pd.read_csv(path)
    validate_columns(df)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Raise a ValueError if any expected column is missing from the DataFrame.
    This guards against accidentally loading a different CSV or a corrupted file.
    """
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"[ERROR] Missing expected columns: {sorted(missing)}\n"
            f"Found columns: {df.columns.tolist()}"
        )


# =============================================================================
# SECTION 3: Verify Exclusion Lists Against the Raw DataFrame
# =============================================================================

def verify_exclusion_lists(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Cross-check that each approved df_index in the Category A and B lists
    actually corresponds to the expected diagnosis label and impossible value.

    This must be called BEFORE any rows are removed.
    Raises a RuntimeError if any mismatch is found — do not proceed if it fails.

    Parameters
    ----------
    df      : The raw (unmodified) DataFrame loaded from the CSV.
    verbose : If True, print a table of verification results.

    Returns
    -------
    bool : True if all checks pass, raises otherwise.
    """
    if verbose:
        print("\n--- Verifying Category A exclusion indices ---")

    all_ok = True

    for df_idx, details in CATEGORY_A_DETAILS.items():
        if df_idx not in df.index:
            all_ok = False
            print(f"  [FAIL] df_index {df_idx}: not found in DataFrame index.")
            continue
        actual_diag = df.loc[df_idx, TARGET_COLUMN]
        expected_diag = details["diagnosis"]
        expected_csv  = df_idx + 2
        diag_ok = actual_diag == expected_diag
        csv_ok  = expected_csv == details["csv_row"]
        ok = diag_ok and csv_ok
        if not ok:
            all_ok = False
        if verbose:
            status = "OK" if ok else "MISMATCH!"
            print(f"  df_index={df_idx:4d}  csv_row={details['csv_row']:4d}  "
                  f"[{status}]  {actual_diag}")

    if verbose:
        print(f"\n--- Verifying Category B exclusion indices ({len(CATEGORY_B_DF_INDICES)} rows) ---")

    for df_idx, details in CATEGORY_B_DETAILS.items():
        if df_idx not in df.index:
            all_ok = False
            if verbose:
                print(f"  [FAIL] df_index {df_idx}: not found in DataFrame index.")
            continue
        actual_diag = df.loc[df_idx, TARGET_COLUMN]
        expected_diag = details["diagnosis"]
        expected_csv  = df_idx + 2
        diag_ok = actual_diag == expected_diag
        csv_ok  = expected_csv == details["csv_row"]
        ok = diag_ok and csv_ok
        if not ok:
            all_ok = False
        if verbose:
            status = "OK" if ok else "MISMATCH!"
            print(f"  df_index={df_idx:4d}  csv_row={details['csv_row']:4d}  "
                  f"[{status}]  {actual_diag}")

    if not all_ok:
        raise RuntimeError(
            "[CRITICAL] Exclusion list verification FAILED. "
            "Indices do not match expected records. "
            "Do not proceed with exclusions until this is resolved."
        )

    if verbose:
        print(f"\n  All {len(CATEGORY_B_DF_INDICES)} Category B indices verified against expected diagnosis and CSV row.")
        print("  Verification passed. Safe to proceed with exclusions.")

    return True


# =============================================================================
# SECTION 4: Cleaning Steps
# =============================================================================

def remove_exact_duplicates(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list]:
    """
    Remove exact duplicate rows, keeping the first occurrence.

    WHY: Duplicates that span the train/test split allow the model to
    'remember' training rows when evaluating on the test set (data leakage).
    This must happen BEFORE the split.

    NOTE: We work on a copy; the original DataFrame passed in is not modified.

    Parameters
    ----------
    df      : The raw DataFrame (before any cleaning).
    verbose : Print a summary of what was removed.

    Returns
    -------
    df_clean : DataFrame with duplicates removed.
    dup_log  : List of dicts describing each removed duplicate.
    """
    duplicate_mask = df.duplicated(keep="first")
    n_dups = duplicate_mask.sum()

    dup_log = []
    for df_idx in df[duplicate_mask].index:
        dup_log.append({
            "step"            : "deduplication",
            "original_df_index": df_idx,
            "csv_row_number"  : df_idx + 2,
            "diagnosis"       : df.loc[df_idx, TARGET_COLUMN],
            "offending_feature": "ALL (exact duplicate)",
            "offending_value" : "ALL",
            "reason"          : "Exact duplicate of an earlier row. Removed to prevent data leakage across train/test split.",
            "scenario"        : "both",
        })

    df_clean = df[~duplicate_mask].copy()

    if verbose:
        print(f"\n  Deduplication: removed {n_dups} exact duplicate rows.")
        print(f"  Rows before: {len(df):4d}  |  Rows after: {len(df_clean):4d}")

    return df_clean, dup_log


def apply_record_exclusions(
    df: pd.DataFrame,
    scenario: str = "scenario1",
    verbose: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Exclude approved records based on the selected preprocessing scenario.

    IMPORTANT: This function uses ORIGINAL DataFrame indices (from the full,
    unmodified CSV) — NOT positions in the already-deduplicated DataFrame.
    Using original indices prevents accidental exclusion of wrong rows.

    Scenarios
    ---------
    "scenario1" (conservative):
        Exclude only the 8 Category A rows (demonstrably invalid values).
        Retain all 21 Category B rows.

    "scenario2" (aggressive / sensitivity):
        Exclude the 8 Category A rows AND the 21 Category B rows.
        Used to measure the effect of suspicious records on model performance.

    Parameters
    ----------
    df       : Deduplicated DataFrame (still carrying original df_index values).
    scenario : "scenario1" or "scenario2".
    verbose  : Print exclusion summary.

    Returns
    -------
    df_clean    : DataFrame after approved exclusions.
    exclusion_log : List of dicts describing each excluded record.
    """
    valid_scenarios = {"scenario1", "scenario2"}
    if scenario not in valid_scenarios:
        raise ValueError(
            f"[ERROR] Unknown scenario '{scenario}'. "
            f"Valid options: {valid_scenarios}"
        )

    exclusion_log = []

    # -- Build the set of indices to exclude for this scenario --
    indices_to_exclude = set(CATEGORY_A_DF_INDICES)  # always excluded
    if scenario == "scenario2":
        indices_to_exclude.update(CATEGORY_B_DF_INDICES)

    # -- Only exclude rows that are still present (some may have been deduped) --
    present_indices = set(df.index)
    to_remove = indices_to_exclude & present_indices
    already_gone = indices_to_exclude - present_indices  # removed during dedup

    # -- Build the exclusion log --
    for df_idx in sorted(to_remove):
        row = df.loc[df_idx]
        is_cat_a = df_idx in set(CATEGORY_A_DF_INDICES)
        category = "A" if is_cat_a else "B"

        if is_cat_a:
            details = CATEGORY_A_DETAILS[df_idx]
            feat    = details["offending_feature"]
            val     = str(details["offending_value"])
            reason  = details["reason"]
        else:
            details = CATEGORY_B_DETAILS[df_idx]
            feat    = details["offending_feature"]
            val     = str(details["offending_value"])
            reason  = details["reason"]

        exclusion_log.append({
            "step"             : f"record_exclusion_{scenario}",
            "category"        : category,
            "original_df_index": df_idx,
            "csv_row_number"  : df_idx + 2,
            "diagnosis"       : row[TARGET_COLUMN],
            "offending_feature": feat,
            "offending_value" : val,
            "reason"          : reason,
            "scenario"        : scenario,
        })

    df_clean = df[~df.index.isin(to_remove)].copy()

    # Special protocol check: Record 16 (CSV row 347, df_index 345)
    # Kept in Scenario 1 and flagged for review; excluded in Scenario 2.
    if scenario == "scenario1" and 345 in df_clean.index:
        rec16_row = df_clean.loc[345]
        exclusion_log.append({
            "step"             : "record_16_flagged_for_review",
            "category"        : "B (retained)",
            "original_df_index": 345,
            "csv_row_number"  : 347,
            "diagnosis"       : rec16_row[TARGET_COLUMN],
            "offending_feature": "LYMn; WBC; LYMp",
            "offending_value" : f"LYMn={rec16_row['LYMn']}; WBC={rec16_row['WBC']}; LYMp={rec16_row['LYMp']}%",
            "reason"          : (
                "Record 16 kept in Scenario 1 and flagged for clinical review. "
                "Not relabeled or excluded solely for extreme lymphocytosis/leukocytosis. "
                "Excluded only in Scenario 2 sensitivity analysis as part of verified Category B set."
            ),
            "scenario"        : "scenario1",
        })

    if verbose:
        n_cat_a = len(set(CATEGORY_A_DF_INDICES) & to_remove)
        n_cat_b = len(to_remove) - n_cat_a
        print(f"\n  Record exclusions ({scenario}):")
        print(f"    Category A excluded : {n_cat_a}")
        print(f"    Category B excluded : {n_cat_b}")
        if already_gone:
            print(f"    Already removed by dedup : {len(already_gone)} (not double-counted)")
        print(f"    Total rows removed  : {len(to_remove)}")
        print(f"    Rows remaining      : {len(df_clean)}")

    return df_clean, exclusion_log


# =============================================================================
# SECTION 5: Task Target Preparation
# =============================================================================

def prepare_task_a(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for Task A — 9-class multiclass classification.

    Uses all 9 Diagnosis labels as-is.
    No rows are excluded here (exclusions happen before this call).

    Parameters
    ----------
    df      : Cleaned DataFrame (after dedup + approved exclusions).
    verbose : Print class distribution.

    Returns
    -------
    X : DataFrame of 14 CBC feature columns.
    y : Series of string Diagnosis labels.
    """
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"[ERROR] Missing feature columns: {missing_features}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    if verbose:
        print("\n  Task A (multiclass) target distribution:")
        dist = y.value_counts()
        for label, count in dist.items():
            pct = count / len(y) * 100
            print(f"    {label:<45} {count:>4}  ({pct:.1f}%)")
        print(f"    {'TOTAL':<45} {len(y):>4}")

    return X, y


def prepare_task_b(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and binary target for Task B — anemia screening.

    Label mapping (documented and approved in Phase 3):
      Anemia classes (5 subtypes)  → 1
      Healthy                      → 0
      Leukemia, Thrombocytopenia
      and Leukemia+Thrombocytopenia → EXCLUDED from Task B

    WHY exclude those 3 classes?
    Thrombocytopenia is a platelet disorder, not anaemia.
    Leukemia is a blood malignancy, not anaemia.
    Labelling either as 'not anaemia' in a binary classifier is clinically
    misleading. Labelling them as 'anaemia' is clinically wrong.
    Exclusion is the only defensible choice for a binary screening task.

    Parameters
    ----------
    df      : Cleaned DataFrame (after dedup + approved exclusions).
    verbose : Print class distribution and exclusion count.

    Returns
    -------
    X : DataFrame of 14 CBC feature columns (rows retained for Task B only).
    y : Series of integer binary labels (0 = healthy, 1 = anemia).
    """
    # Mark which rows to keep for Task B
    valid_mask = df[TARGET_COLUMN].isin(ANEMIA_CLASSES + [HEALTHY_CLASS])
    excluded_mask = df[TARGET_COLUMN].isin(TASK_B_EXCLUDED_CLASSES)

    n_excluded = excluded_mask.sum()
    df_b = df[valid_mask].copy()

    # Map labels to 0 / 1
    label_map = {cls: 1 for cls in ANEMIA_CLASSES}
    label_map[HEALTHY_CLASS] = 0
    y_b = df_b[TARGET_COLUMN].map(label_map)

    # Verify no unmapped labels crept in
    unmapped = y_b.isnull().sum()
    if unmapped > 0:
        raise ValueError(
            f"[ERROR] {unmapped} rows have labels not in the Task B mapping. "
            f"Check ANEMIA_CLASSES and HEALTHY_CLASS constants."
        )

    X_b = df_b[FEATURE_COLUMNS].copy()

    if verbose:
        print(f"\n  Task B (binary) — excluded classes (not anemia, not healthy):")
        for cls in TASK_B_EXCLUDED_CLASSES:
            n = (df[TARGET_COLUMN] == cls).sum()
            print(f"    {cls:<45} {n:>4} rows excluded")
        print(f"    {'TOTAL EXCLUDED':<45} {n_excluded:>4}")
        print(f"\n  Task B label distribution:")
        b_dist = y_b.value_counts().sort_index()
        label_names = {0: "Healthy (0)", 1: "Anemia (1)"}
        for lbl, count in b_dist.items():
            pct = count / len(y_b) * 100
            print(f"    {label_names[lbl]:<45} {count:>4}  ({pct:.1f}%)")
        print(f"    {'TOTAL':<45} {len(y_b):>4}")

    return X_b, y_b


# =============================================================================
# SECTION 6: Train / Test Split
# =============================================================================

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    Split features and target into training and test sets.

    Uses stratified splitting (class proportions preserved in both sets)
    wherever the smallest class has enough samples. Falls back to unstratified
    splitting with a clear warning if stratification fails.

    WHY stratified?
    With imbalanced classes, a random split can put all samples of a rare
    class into training, leaving the test set unable to evaluate that class.
    Stratification ensures both sets reflect the true class distribution.

    Parameters
    ----------
    X            : Feature DataFrame.
    y            : Target Series.
    test_size    : Fraction reserved for testing. Default 0.20 (80/20 split).
    random_state : Seed for reproducibility. Default 42.
    stratify     : Whether to use stratified splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    stratify_arr = y if stratify else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr,
        )
    except ValueError as e:
        # Stratification fails if a class has fewer samples than the number
        # of splits. In that case, fall back to unstratified.
        print(
            f"\n  [WARNING] Stratified split failed: {e}\n"
            f"  Falling back to non-stratified split. "
            f"This may produce uneven class distributions.\n"
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    if verbose:
        print(f"\n  Train/Test split (test_size={test_size}, seed={random_state}):")
        print(f"    Training rows : {len(X_train):4d}")
        print(f"    Test rows     : {len(X_test):4d}")
        print(f"    Total         : {len(X_train) + len(X_test):4d}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 7: Feature Preprocessing Pipeline
# =============================================================================

def build_feature_pipeline() -> Pipeline:
    """
    Build a scikit-learn Pipeline for CBC feature preprocessing.

    Current steps:
      1. SimpleImputer (median strategy): handles any NaN that may appear
         in future data. No NaN exist in the current training set, but
         this makes the pipeline safe for new prediction inputs.
      2. StandardScaler: transforms each feature to mean=0, std=1.
         Required by Logistic Regression and other distance-based models.
         Decision trees and Random Forests work without scaling, but
         including it in the pipeline keeps inference consistent.

    WHY a Pipeline instead of applying steps manually?
      - The pipeline ensures the scaler is ALWAYS fit on training data only.
      - When we call pipeline.predict(new_input), the same transformations
        are applied automatically. There is no risk of forgetting a step.
      - The fitted pipeline is saved as a single object, making deployment safe.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline (must be fit on training data only).
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    return pipeline


# =============================================================================
# SECTION 8: Report Generation
# =============================================================================

def save_cleaning_log(log_entries: list, out_path: Path = None) -> Path:
    """
    Save a detailed CSV log of every record removed during cleaning.

    Columns:
        step, category, original_df_index, csv_row_number, diagnosis,
        offending_feature, offending_value, reason, scenario

    Parameters
    ----------
    log_entries : List of dicts, one per removed record.
    out_path    : Output path. Defaults to ml_model/reports/cleaning_log.csv.

    Returns
    -------
    Path : The path where the log was saved.
    """
    if out_path is None:
        out_path = REPORTS_DIR / "cleaning_log.csv"

    df_log = pd.DataFrame(log_entries)
    df_log.to_csv(out_path, index=False)
    return out_path


def _class_dist_str(y: pd.Series, indent: int = 4, label_names: dict = None) -> str:
    """Helper: format a class distribution as an indented string."""
    pad = " " * indent
    lines = []
    dist = y.value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(y) * 100
        disp_name = label_names[label] if label_names and label in label_names else str(label)
        lines.append(f"{pad}{disp_name:<45} {count:>4}  ({pct:.1f}%)")
    lines.append(f"{pad}{'TOTAL':<45} {len(y):>4}")
    return "\n".join(lines)


def save_preprocessing_summary(
    summaries: dict,
    out_path: Path = None,
) -> Path:
    """
    Save a human-readable preprocessing summary to a text file.

    Reports actual row counts and diagnosis distributions at every stage:
    raw dataset, after deduplication, after Category A exclusions,
    after Category B exclusions (Scenario 2), and after Task B non-anemia exclusions.

    Parameters
    ----------
    summaries : Dict produced by run_full_preprocessing().
    out_path  : Output path. Defaults to ml_model/reports/preprocessing_summary.txt.

    Returns
    -------
    Path : The path where the summary was saved.
    """
    if out_path is None:
        out_path = REPORTS_DIR / "preprocessing_summary.txt"

    lines = []
    def L(text=""): lines.append(str(text))

    L("=" * 72)
    L("PREPROCESSING SUMMARY — AI-Based Anemia Screening")
    L(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"Dataset   : {DATASET_PATH.relative_to(PROJECT_ROOT)}")
    L("=" * 72)
    L()

    for scenario_key, s in summaries.items():
        L(f"{'─' * 72}")
        L(f"SCENARIO: {scenario_key.upper()}")
        L(f"{'─' * 72}")
        L()

        L(f"  Step 0 — Raw Dataset")
        L(f"    Rows: {s['n_raw']}  |  Columns: {s['n_cols']}")
        L(f"    Raw Diagnosis Distribution:")
        L(_class_dist_str(s["dist_raw"], indent=6))
        L()

        L(f"  Step 1 — Deduplication")
        L(f"    Duplicates removed : {s['n_duplicates_removed']}")
        L(f"    Rows remaining     : {s['n_after_dedup']}")
        L(f"    Diagnosis Distribution after Deduplication:")
        L(_class_dist_str(s["dist_dedup"], indent=6))
        L()

        L(f"  Step 2 — Category A exclusions (demonstrably invalid, both scenarios)")
        L(f"    Records excluded   : {s['n_cat_a_excluded']} (CSV rows 6, 218, 247, 346, 377, 385, 467, 474)")
        L(f"    Rows remaining     : {s['n_after_cat_a']}")
        L(f"    Diagnosis Distribution after Category A exclusions:")
        L(_class_dist_str(s["dist_after_cat_a"], indent=6))
        L()

        if scenario_key == "scenario1":
            L(f"  Step 3 — Scenario 1 Protocol: Record 16 (CSV row 347, df index 345)")
            L(f"    Status: RETAINED and flagged for clinical review.")
            L(f"    Diagnosis: Iron deficiency anemia | WBC=45.7 | LYMp=91.4% | LYMn=41.8")
            L(f"    Reason: Kept in primary analysis; not relabeled or excluded solely for extreme lymphocytosis.")
            L()

        if scenario_key == "scenario2":
            L(f"  Step 3 — Category B exclusions (suspicious, Scenario 2 sensitivity analysis)")
            L(f"    Records excluded   : {s['n_cat_b_excluded']} (all 21 verified Category B rows, including Record 16)")
            L(f"    Rows remaining     : {s['n_after_cat_b']}")
            L(f"    Diagnosis Distribution after Category B exclusions:")
            L(_class_dist_str(s["dist_after_cat_b"], indent=6))
            L()

        L(f"  TASK A (Multiclass — 9 classes)")
        L(f"    Total rows for training/testing : {s['task_a_n_total']}")
        L(f"    Class distribution:")
        L(_class_dist_str(s["task_a_y"], indent=6))
        L(f"    Train rows : {s['task_a_n_train']}  |  Test rows : {s['task_a_n_test']}")
        L()

        L(f"  TASK B (Binary Anemia Screening)")
        L(f"    Additional exclusions (non-anemia classes): {s['task_b_n_excluded']}")
        L(f"    Breakdown of excluded non-anemia classes:")
        for cls, count in s["task_b_excluded_breakdown"].items():
            L(f"      {cls:<43} {count:>4} rows")
        L(f"    Total rows for training/testing : {s['task_b_n_total']}")
        L(f"    Label distribution:")
        task_b_labels = {0: "Healthy (0)", 1: "Anemia subtypes (1)"}
        L(_class_dist_str(s["task_b_y"], indent=6, label_names=task_b_labels))
        L(f"    Train rows : {s['task_b_n_train']}  |  Test rows : {s['task_b_n_test']}")
        L()

        L(f"  DATA QUALITY NOTES:")
        L(f"    LYMp: {s['lymp_mode_pct']:.1f}% of rows have mode value "
          f"{s['lymp_mode_val']} (suspected pre-imputation).")
        L(f"    LYMn: {s['lymn_mode_pct']:.1f}% of rows have mode value "
          f"{s['lymn_mode_val']} (suspected pre-imputation).")
        L(f"    Both features are kept. This is documented for honest reporting.")
        L()

    L("=" * 72)
    L("DISCLAIMER")
    L("=" * 72)
    L("This is an educational project. Results are NOT clinically validated.")
    L("Do not use model outputs for medical decisions.")
    L("Consult a qualified healthcare professional for interpretation.")
    L()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


# =============================================================================
# SECTION 9: Full Preprocessing Pipeline Runner
# =============================================================================

def run_full_preprocessing(
    scenario: str = "scenario1",
    verbose: bool = True,
) -> dict:
    """
    Run the complete preprocessing pipeline for one scenario.

    Steps (in order):
      1. Load and validate the dataset.
      2. Verify exclusion indices against the raw DataFrame.
      3. Remove exact duplicate rows.
      4. Apply approved record exclusions for the chosen scenario.
      5. Prepare Task A (multiclass) features and targets.
      6. Prepare Task B (binary screening) features and targets.
      7. Split each task into train/test sets.
      8. Build (but do NOT fit) the feature preprocessing pipeline.

    The pipeline is returned unfitted. Fitting happens in train.py using
    training data only, to prevent data leakage from the test set.

    Parameters
    ----------
    scenario : "scenario1" or "scenario2".
    verbose  : Print step-by-step progress.

    Returns
    -------
    dict with keys:
        scenario, n_raw, n_cols, n_duplicates_removed, n_after_dedup,
        n_cat_a_excluded, n_after_cat_a, n_cat_b_excluded (scenario2 only),
        n_after_cat_b (scenario2 only),
        task_a_X_train, task_a_X_test, task_a_y_train, task_a_y_test,
        task_a_n_total, task_a_n_train, task_a_n_test, task_a_y,
        task_b_X_train, task_b_X_test, task_b_y_train, task_b_y_test,
        task_b_n_total, task_b_n_train, task_b_n_test, task_b_n_excluded, task_b_y,
        feature_pipeline (unfitted),
        cleaning_log (list of dicts),
        lymp_mode_val, lymp_mode_pct, lymn_mode_val, lymn_mode_pct,
        n_cols,
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  PREPROCESSING — {scenario.upper()}")
        print(f"{'=' * 60}")

    cleaning_log = []

    # Step 1: Load
    if verbose: print("\n[1/7] Loading dataset ...")
    df_raw = load_dataset()
    n_raw  = len(df_raw)
    n_cols = df_raw.shape[1]
    if verbose: print(f"  Loaded {n_raw} rows, {n_cols} columns.")

    # Step 2: Verify exclusion lists
    if verbose: print("\n[2/7] Verifying exclusion indices ...")
    verify_exclusion_lists(df_raw, verbose=verbose)

    # Step 3: Deduplicate
    if verbose: print("\n[3/7] Removing duplicate rows ...")
    df_deduped, dup_log = remove_exact_duplicates(df_raw, verbose=verbose)
    cleaning_log.extend(dup_log)
    n_after_dedup = len(df_deduped)

    # Step 4: Apply record exclusions
    if verbose: print(f"\n[4/7] Applying record exclusions ({scenario}) ...")
    df_clean, excl_log = apply_record_exclusions(df_deduped, scenario=scenario, verbose=verbose)
    cleaning_log.extend(excl_log)

    n_cat_a_excluded = len([e for e in excl_log if e["category"] == "A"])
    n_cat_b_excluded = len([e for e in excl_log if e["category"] == "B"])
    n_after_cat_a    = n_after_dedup - n_cat_a_excluded
    n_after_cat_b    = n_after_cat_a - n_cat_b_excluded  # same as len(df_clean)

    # Capture stage-by-stage distributions
    dist_raw = df_raw[TARGET_COLUMN].copy()
    dist_dedup = df_deduped[TARGET_COLUMN].copy()
    df_after_cat_a = df_deduped[~df_deduped.index.isin(CATEGORY_A_DF_INDICES)]
    dist_after_cat_a = df_after_cat_a[TARGET_COLUMN].copy()
    dist_after_cat_b = df_clean[TARGET_COLUMN].copy() if scenario == "scenario2" else None

    # Step 5: Fair Split Protocol for Task A (Shared Clean Held-out Test Set)
    # -------------------------------------------------------------------------
    # To ensure a strictly fair, benchmarked comparison between Scenario 1 and
    # Scenario 2, the held-out test set is partitioned exclusively from the
    # 1,203 clean records (0 Category A, 0 Category B).
    # Both scenarios evaluate on the EXACT SAME 241 test records.
    # Scenario 2 train set = 962 clean records.
    # Scenario 1 train set = 962 clean records + 21 Category B records = 983 records.
    if verbose: print("\n[5/7] Preparing Task A (multiclass) with Fair Split Protocol ...")
    df_clean_pool = df_deduped[~df_deduped.index.isin(CATEGORY_A_DF_INDICES + CATEGORY_B_DF_INDICES)]
    X_a_clean, y_a_clean = prepare_task_a(df_clean_pool, verbose=False)
    X_a_tr_clean, X_a_test, y_a_tr_clean, y_a_test = split_data(
        X_a_clean, y_a_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True, verbose=False
    )

    if scenario == "scenario1":
        # Add 21 Category B records to Scenario 1 training set only
        cat_b_df = df_deduped[df_deduped.index.isin(CATEGORY_B_DF_INDICES)]
        X_cat_b = cat_b_df[FEATURE_COLUMNS].copy()
        y_cat_b = cat_b_df[TARGET_COLUMN].copy()
        X_a_train = pd.concat([X_a_tr_clean, X_cat_b]).sort_index()
        y_a_train = pd.concat([y_a_tr_clean, y_cat_b]).sort_index()
    else:
        X_a_train = X_a_tr_clean.sort_index()
        y_a_train = y_a_tr_clean.sort_index()

    # Reconstitute Task A total records for this scenario
    X_a = pd.concat([X_a_train, X_a_test]).sort_index()
    y_a = pd.concat([y_a_train, y_a_test]).sort_index()

    if verbose:
        print(f"  Task A test set (shared, clean): {len(X_a_test)} rows")
        print(f"  Task A train set ({scenario})   : {len(X_a_train)} rows")
        print(f"  Category B in test set          : 0 (guaranteed)")

    # Step 6: Fair Split Protocol for Task B (Shared Clean Held-out Test Set)
    # -------------------------------------------------------------------------
    # Candidate clean pool has 1,080 records (non-anemia excluded).
    # Test set = exactly 216 clean records (identical for S1 and S2).
    # Scenario 2 train set = 864 clean records.
    # Scenario 1 train set = 864 clean records + 17 eligible Category B records = 881 records.
    if verbose: print("\n[6/7] Preparing Task B (binary screening) with Fair Split Protocol ...")
    df_clean_b_pool = df_clean_pool[df_clean_pool[TARGET_COLUMN].isin(ANEMIA_CLASSES + [HEALTHY_CLASS])]
    X_b_clean, y_b_clean = prepare_task_b(df_clean_b_pool, verbose=False)
    X_b_tr_clean, X_b_test, y_b_tr_clean, y_b_test = split_data(
        X_b_clean, y_b_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=True, verbose=False
    )

    # Breakdown of excluded classes for this scenario
    task_b_n_excluded = (
        df_clean[TARGET_COLUMN].isin(TASK_B_EXCLUDED_CLASSES).sum()
    )
    task_b_excluded_breakdown = {
        cls: int((df_clean[TARGET_COLUMN] == cls).sum())
        for cls in TASK_B_EXCLUDED_CLASSES
    }

    if scenario == "scenario1":
        # Cat B records eligible for Task B (exclude non-anemia disorders)
        cat_b_b_df = cat_b_df[cat_b_df[TARGET_COLUMN].isin(ANEMIA_CLASSES + [HEALTHY_CLASS])]
        X_cat_b_b, y_cat_b_b = prepare_task_b(cat_b_b_df, verbose=False)
        X_b_train = pd.concat([X_b_tr_clean, X_cat_b_b]).sort_index()
        y_b_train = pd.concat([y_b_tr_clean, y_cat_b_b]).sort_index()
    else:
        X_b_train = X_b_tr_clean.sort_index()
        y_b_train = y_b_tr_clean.sort_index()

    X_b = pd.concat([X_b_train, X_b_test]).sort_index()
    y_b = pd.concat([y_b_train, y_b_test]).sort_index()

    if verbose:
        print(f"  Task B test set (shared, clean): {len(X_b_test)} rows")
        print(f"  Task B train set ({scenario})   : {len(X_b_train)} rows")
        print(f"  Non-anemia classes excluded     : {task_b_n_excluded} rows")

    # Step 7: Build pipeline (unfitted)
    if verbose: print("\n[7/7] Building feature preprocessing pipeline (unfitted) ...")
    pipeline = build_feature_pipeline()
    if verbose:
        print("  Pipeline steps:")
        for name, step in pipeline.steps:
            print(f"    {name}: {type(step).__name__}")
        print("  Pipeline NOT fitted yet — will be fit on training data in train.py.")

    # LYMp/LYMn pre-imputation metrics (computed on cleaned data)
    lymp_mode_val = df_clean["LYMp"].mode()[0]
    lymp_mode_pct = (df_clean["LYMp"] == lymp_mode_val).sum() / len(df_clean) * 100
    lymn_mode_val = df_clean["LYMn"].mode()[0]
    lymn_mode_pct = (df_clean["LYMn"] == lymn_mode_val).sum() / len(df_clean) * 100

    return {
        "scenario"                  : scenario,
        "n_raw"                     : n_raw,
        "n_cols"                    : n_cols,
        "n_duplicates_removed"      : n_raw - n_after_dedup,
        "n_after_dedup"             : n_after_dedup,
        "n_cat_a_excluded"          : n_cat_a_excluded,
        "n_after_cat_a"             : n_after_cat_a,
        "n_cat_b_excluded"          : n_cat_b_excluded,
        "n_after_cat_b"             : n_after_cat_b,
        "dist_raw"                  : dist_raw,
        "dist_dedup"                : dist_dedup,
        "dist_after_cat_a"          : dist_after_cat_a,
        "dist_after_cat_b"          : dist_after_cat_b,
        "task_b_excluded_breakdown" : task_b_excluded_breakdown,
        # Task A
        "task_a_X_train"            : X_a_train,
        "task_a_X_test"             : X_a_test,
        "task_a_y_train"            : y_a_train,
        "task_a_y_test"             : y_a_test,
        "task_a_y"                  : y_a,
        "task_a_n_total"            : len(X_a),
        "task_a_n_train"            : len(X_a_train),
        "task_a_n_test"             : len(X_a_test),
        # Task B
        "task_b_X_train"            : X_b_train,
        "task_b_X_test"             : X_b_test,
        "task_b_y_train"            : y_b_train,
        "task_b_y_test"             : y_b_test,
        "task_b_y"                  : y_b,
        "task_b_n_total"            : len(X_b),
        "task_b_n_train"            : len(X_b_train),
        "task_b_n_test"             : len(X_b_test),
        "task_b_n_excluded"         : task_b_n_excluded,
        # Pipeline and metadata
        "feature_pipeline"          : pipeline,
        "cleaning_log"              : cleaning_log,
        "feature_columns"           : FEATURE_COLUMNS,
        "lymp_mode_val"             : lymp_mode_val,
        "lymp_mode_pct"             : lymp_mode_pct,
        "lymn_mode_val"             : lymn_mode_val,
        "lymn_mode_pct"             : lymn_mode_pct,
    }


# =============================================================================
# SECTION 10: Main Entry Point
# =============================================================================

def main():
    """
    Run both scenarios and save the cleaning log and preprocessing summary.
    """
    print("=" * 70)
    print("AI-Based Anemia Screening — Phase 4: Preprocessing")
    print("=" * 70)

    summaries = {}

    for scenario in ["scenario1", "scenario2"]:
        result = run_full_preprocessing(scenario=scenario, verbose=True)
        summaries[scenario] = result
        print()

    # Build clean combined audit log (dedup once, then scenario1, then scenario2)
    dedup_entries = [e for e in summaries["scenario1"]["cleaning_log"] if e["step"] == "deduplication"]
    scen1_entries = [e for e in summaries["scenario1"]["cleaning_log"] if e["step"] != "deduplication"]
    scen2_entries = [e for e in summaries["scenario2"]["cleaning_log"] if e["step"] != "deduplication"]
    all_logs = dedup_entries + scen1_entries + scen2_entries

    # Save cleaning log (combined, both scenarios)
    log_path = save_cleaning_log(all_logs)
    print(f"\nCleaning log saved     -> {log_path.relative_to(PROJECT_ROOT)}")

    # Save preprocessing summary
    summary_path = save_preprocessing_summary(summaries)
    print(f"Preprocessing summary  -> {summary_path.relative_to(PROJECT_ROOT)}")

    # Final overview
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for scenario, s in summaries.items():
        print(f"\n  {scenario.upper()}:")
        print(f"    Raw → dedup → excl → Task A : "
              f"{s['n_raw']} → {s['n_after_dedup']} → {s['task_a_n_total']}")
        print(f"    Task A  — train: {s['task_a_n_train']:4d}  test: {s['task_a_n_test']:4d}")
        print(f"    Task B  — train: {s['task_b_n_train']:4d}  test: {s['task_b_n_test']:4d}")

    print()
    print("NEXT STEP: Phase 5 — Run tests, then implement train.py")
    print()
    print("DISCLAIMER: Educational project. Not clinically validated.")


if __name__ == "__main__":
    main()
