"""
test_preprocessing.py — Unit Tests for Data Preprocessing Pipeline
===================================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Phase 4 Validation

Tests cover:
  1. Target mapping (Task A 9-class, Task B binary 0/1, and non-anemia class exclusion)
  2. Duplicate handling (49 exact duplicate rows removed, original preserved)
  3. Row exclusions (8 Category A excluded in both scenarios, 21 Category B in Scenario 2 only)
  4. Record 16 protocol (CSV row 347, df index 345: kept in Scenario 1, excluded in Scenario 2)
  5. Scenario selection ('scenario1', 'scenario2', invalid scenario handling)
  6. Reproducible splitting (seed=42 reproducibility, train/test index disjointness, no data leakage)
  7. Pipeline safety (unfitted pipeline, median imputer, standard scaler)
  8. CSV file integrity (raw CSV unchanged, MD5 checksum verified)

HOW TO RUN:
  source .venv/bin/activate
  python -m unittest ml_model/src/test_preprocessing.py -v
"""

import unittest
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

from ml_model.src.preprocessing import (
    DATASET_PATH,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    ANEMIA_CLASSES,
    HEALTHY_CLASS,
    TASK_B_EXCLUDED_CLASSES,
    CATEGORY_A_DF_INDICES,
    CATEGORY_A_DETAILS,
    CATEGORY_B_DF_INDICES,
    CATEGORY_B_DETAILS,
    load_dataset,
    validate_columns,
    verify_exclusion_lists,
    remove_exact_duplicates,
    apply_record_exclusions,
    prepare_task_a,
    prepare_task_b,
    split_data,
    build_feature_pipeline,
    run_full_preprocessing,
)

# Known MD5 checksum of raw anemia_types.csv
RAW_CSV_MD5 = "c6375e974d730a42b53160614237ddf0"


class TestDatasetIntegrity(unittest.TestCase):
    """Verify raw CSV file integrity and loading."""

    def test_raw_csv_md5_hash_preserved(self):
        """Ensure raw CSV has not been modified in place."""
        hasher = hashlib.md5()
        with open(DATASET_PATH, "rb") as f:
            hasher.update(f.read())
        self.assertEqual(
            hasher.hexdigest(),
            RAW_CSV_MD5,
            "Raw CSV file MD5 does not match expected hash! File may have been altered."
        )

    def test_load_dataset_shape_and_columns(self):
        """Verify load_dataset returns 1281 rows, 15 columns with expected names."""
        df = load_dataset()
        self.assertEqual(len(df), 1281)
        self.assertEqual(df.shape[1], 15)
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, df.columns)
        self.assertIn(TARGET_COLUMN, df.columns)

    def test_validate_columns_raises_on_missing(self):
        """Ensure validate_columns raises ValueError when required column is missing."""
        df = load_dataset()
        corrupted = df.drop(columns=["HGB"])
        with self.assertRaises(ValueError):
            validate_columns(corrupted)


class TestDuplicateHandling(unittest.TestCase):
    """Verify duplicate row identification and removal."""

    def test_exact_duplicates_count(self):
        """Verify exactly 49 duplicate rows are identified and removed."""
        df = load_dataset()
        df_clean, dup_log = remove_exact_duplicates(df, verbose=False)
        self.assertEqual(len(dup_log), 49)
        self.assertEqual(len(df_clean), 1232)
        # Ensure no duplicates remain in df_clean
        self.assertEqual(df_clean.duplicated().sum(), 0)

    def test_category_records_not_in_duplicates(self):
        """Ensure none of the approved Cat A or Cat B records are inside duplicate rows."""
        df = load_dataset()
        dup_mask = df.duplicated(keep="first")
        for idx in CATEGORY_A_DF_INDICES:
            self.assertFalse(dup_mask.iloc[idx], f"Category A df_index {idx} was flagged as duplicate!")
        for idx in CATEGORY_B_DF_INDICES:
            self.assertFalse(dup_mask.iloc[idx], f"Category B df_index {idx} was flagged as duplicate!")


class TestExclusionListsAndRecord16(unittest.TestCase):
    """Verify Category A, Category B, and Record 16 mapping and exclusion behavior."""

    def test_category_a_mapping(self):
        """Verify all 8 Category A rows have csv_row = df_index + 2 and correct diagnoses."""
        df = load_dataset()
        self.assertEqual(len(CATEGORY_A_DF_INDICES), 8)
        for df_idx in CATEGORY_A_DF_INDICES:
            details = CATEGORY_A_DETAILS[df_idx]
            self.assertEqual(details["csv_row"], df_idx + 2)
            self.assertEqual(df.loc[df_idx, TARGET_COLUMN], details["diagnosis"])

    def test_category_b_mapping(self):
        """Verify all 21 Category B rows have csv_row = df_index + 2 and correct diagnoses."""
        df = load_dataset()
        self.assertEqual(len(CATEGORY_B_DF_INDICES), 21)
        for df_idx in CATEGORY_B_DF_INDICES:
            details = CATEGORY_B_DETAILS[df_idx]
            self.assertEqual(details["csv_row"], df_idx + 2)
            self.assertEqual(df.loc[df_idx, TARGET_COLUMN], details["diagnosis"])

    def test_record_16_protocol(self):
        """
        Record 16 (CSV row 347, df index 345):
        - Kept in Scenario 1 and flagged for review
        - Excluded in Scenario 2 as part of Category B
        """
        df = load_dataset()
        df_dedup, _ = remove_exact_duplicates(df, verbose=False)

        # Scenario 1
        df_s1, log_s1 = apply_record_exclusions(df_dedup, scenario="scenario1", verbose=False)
        self.assertIn(345, df_s1.index, "Record 16 (df index 345) was improperly excluded in Scenario 1!")
        self.assertEqual(df_s1.loc[345, TARGET_COLUMN], "Iron deficiency anemia")
        # Check flagged for review log entry exists
        flag_entries = [e for e in log_s1 if e["original_df_index"] == 345 and e["step"] == "record_16_flagged_for_review"]
        self.assertEqual(len(flag_entries), 1)

        # Scenario 2
        df_s2, log_s2 = apply_record_exclusions(df_dedup, scenario="scenario2", verbose=False)
        self.assertNotIn(345, df_s2.index, "Record 16 (df index 345) was not excluded in Scenario 2!")
        excl_entries = [e for e in log_s2 if e["original_df_index"] == 345 and e["step"] == "record_exclusion_scenario2"]
        self.assertEqual(len(excl_entries), 1)

    def test_verify_exclusion_lists_succeeds(self):
        """Ensure verify_exclusion_lists passes on unmodified DataFrame."""
        df = load_dataset()
        self.assertTrue(verify_exclusion_lists(df, verbose=False))


class TestScenarioSelection(unittest.TestCase):
    """Verify scenario counts and invalid scenario rejection."""

    def test_invalid_scenario_raises_value_error(self):
        """apply_record_exclusions and run_full_preprocessing must raise ValueError on bad scenario."""
        df = load_dataset()
        with self.assertRaises(ValueError):
            apply_record_exclusions(df, scenario="invalid_scenario", verbose=False)
        with self.assertRaises(ValueError):
            run_full_preprocessing(scenario="unknown_3", verbose=False)

    def test_scenario_1_and_2_row_counts(self):
        """
        Scenario 1: 1281 raw -> 1232 dedup -> 1224 after Cat A
        Scenario 2: 1281 raw -> 1232 dedup -> 1203 after Cat A + Cat B
        """
        res1 = run_full_preprocessing(scenario="scenario1", verbose=False)
        self.assertEqual(res1["n_raw"], 1281)
        self.assertEqual(res1["n_duplicates_removed"], 49)
        self.assertEqual(res1["n_after_dedup"], 1232)
        self.assertEqual(res1["n_cat_a_excluded"], 8)
        self.assertEqual(res1["task_a_n_total"], 1224)
        self.assertEqual(res1["task_a_n_train"], 983)
        self.assertEqual(res1["task_a_n_test"], 241)

        res2 = run_full_preprocessing(scenario="scenario2", verbose=False)
        self.assertEqual(res2["n_raw"], 1281)
        self.assertEqual(res2["n_duplicates_removed"], 49)
        self.assertEqual(res2["n_after_dedup"], 1232)
        self.assertEqual(res2["n_cat_a_excluded"], 8)
        self.assertEqual(res2["n_cat_b_excluded"], 21)
        self.assertEqual(res2["task_a_n_total"], 1203)
        self.assertEqual(res2["task_a_n_train"], 962)
        self.assertEqual(res2["task_a_n_test"], 241)


class TestFairComparisonProtocol(unittest.TestCase):
    """Verify that held-out test sets are 100% identical and contain zero Category B records."""

    def test_identical_test_indices_across_scenarios(self):
        """Task A and Task B test sets must be exactly identical in both scenarios."""
        res1 = run_full_preprocessing(scenario="scenario1", verbose=False)
        res2 = run_full_preprocessing(scenario="scenario2", verbose=False)

        # Task A
        self.assertEqual(list(res1["task_a_X_test"].index), list(res2["task_a_X_test"].index))
        pd.testing.assert_frame_equal(res1["task_a_X_test"], res2["task_a_X_test"])
        pd.testing.assert_series_equal(res1["task_a_y_test"], res2["task_a_y_test"])

        # Task B
        self.assertEqual(list(res1["task_b_X_test"].index), list(res2["task_b_X_test"].index))
        pd.testing.assert_frame_equal(res1["task_b_X_test"], res2["task_b_X_test"])
        pd.testing.assert_series_equal(res1["task_b_y_test"], res2["task_b_y_test"])

    def test_zero_category_b_in_test_set(self):
        """Test sets must contain ZERO Category B records."""
        res1 = run_full_preprocessing(scenario="scenario1", verbose=False)
        test_a_indices = set(res1["task_a_X_test"].index)
        test_b_indices = set(res1["task_b_X_test"].index)
        cat_b_set = set(CATEGORY_B_DF_INDICES)

        self.assertEqual(len(test_a_indices & cat_b_set), 0)
        self.assertEqual(len(test_b_indices & cat_b_set), 0)

    def test_disjoint_train_and_test(self):
        """In both scenarios, train and test index sets must have zero overlap."""
        for sc in ["scenario1", "scenario2"]:
            res = run_full_preprocessing(scenario=sc, verbose=False)
            overlap_a = set(res["task_a_X_train"].index) & set(res["task_a_X_test"].index)
            overlap_b = set(res["task_b_X_train"].index) & set(res["task_b_X_test"].index)
            self.assertEqual(len(overlap_a), 0, f"Task A train/test overlap in {sc}!")
            self.assertEqual(len(overlap_b), 0, f"Task B train/test overlap in {sc}!")


class TestTargetMapping(unittest.TestCase):
    """Verify Task A (9 classes) and Task B (binary screening) target mappings."""

    def test_task_a_preserves_all_nine_classes(self):
        """Task A retains all 9 classes and 14 feature columns."""
        res1 = run_full_preprocessing(scenario="scenario1", verbose=False)
        y_a = res1["task_a_y"]
        self.assertEqual(y_a.nunique(), 9)
        self.assertEqual(res1["task_a_X_train"].shape[1], 14)
        self.assertEqual(res1["task_a_X_test"].shape[1], 14)

    def test_task_b_binary_mapping_and_exclusions(self):
        """
        Task B:
        - Labels are strictly {0, 1}
        - Label 0 corresponds to Healthy
        - Label 1 corresponds to 5 anemia subtypes
        - Excluded classes (Leukemia, Leukemia with thrombocytopenia, Thrombocytopenia)
          are NEVER in Task B.
        """
        for sc in ["scenario1", "scenario2"]:
            res = run_full_preprocessing(scenario=sc, verbose=False)
            y_b = res["task_b_y"]
            unique_labels = set(y_b.unique())
            self.assertEqual(unique_labels, {0, 1}, f"Task B in {sc} must only contain 0 and 1!")
            self.assertEqual(y_b.isna().sum(), 0)

            # Check that excluded classes are absent
            # In Task B y, indices come from df_clean, check none of those rows have excluded diagnosis
            raw_diag = res["task_a_y"].loc[y_b.index]
            for bad_cls in TASK_B_EXCLUDED_CLASSES:
                self.assertFalse((raw_diag == bad_cls).any(), f"{bad_cls} found in Task B {sc}!")


class TestReproducibleSplittingAndPipeline(unittest.TestCase):
    """Verify reproducible splitting, absence of data leakage, and unfitted pipeline."""

    def test_split_reproducibility(self):
        """Multiple split calls with random_state=42 must produce identical train/test splits."""
        df = load_dataset()
        df_clean, _ = remove_exact_duplicates(df, verbose=False)
        df_clean, _ = apply_record_exclusions(df_clean, scenario="scenario1", verbose=False)
        X, y = prepare_task_a(df_clean, verbose=False)

        X_tr1, X_te1, y_tr1, y_te1 = split_data(X, y, test_size=0.2, random_state=42, verbose=False)
        X_tr2, X_te2, y_tr2, y_te2 = split_data(X, y, test_size=0.2, random_state=42, verbose=False)

        pd.testing.assert_frame_equal(X_tr1, X_tr2)
        pd.testing.assert_frame_equal(X_te1, X_te2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)
        pd.testing.assert_series_equal(y_te1, y_te2)

    def test_no_data_leakage_between_train_and_test(self):
        """Train and test sets must have disjoint index sets."""
        res = run_full_preprocessing(scenario="scenario1", verbose=False)
        train_indices = set(res["task_a_X_train"].index)
        test_indices = set(res["task_a_X_test"].index)
        overlap = train_indices & test_indices
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping indices: {overlap}")
        self.assertEqual(len(train_indices) + len(test_indices), res["task_a_n_total"])

    def test_pipeline_is_unfitted(self):
        """Pipeline returned by build_feature_pipeline must not be fitted."""
        pipe = build_feature_pipeline()
        self.assertEqual(len(pipe.steps), 2)
        self.assertEqual(pipe.steps[0][0], "imputer")
        self.assertEqual(pipe.steps[1][0], "scaler")
        # An unfitted StandardScaler does not have mean_ attribute
        scaler = pipe.named_steps["scaler"]
        self.assertFalse(hasattr(scaler, "mean_"), "Scaler in pipeline should not be fitted yet!")


class TestNonAnemiaExclusionReconciliation(unittest.TestCase):
    """Verify exact count tracking of non-anemia classes from 131 to 127 to 123."""

    def test_non_anemia_stage_counts(self):
        """
        Verify exact counts of Leukemia, Leukemia w/ thrombocytopenia, Thrombocytopenia:
        Raw: 131 (73 + 47 + 11)
        Deduplication removes 4: 127 remain (72 + 44 + 11)
        Category A has 0: Scenario 1 Task B exclusions = 127
        Category B has 4: Scenario 2 Task B exclusions = 123 (69 + 44 + 10)
        """
        df = load_dataset()
        raw_na = df[df[TARGET_COLUMN].isin(TASK_B_EXCLUDED_CLASSES)]
        self.assertEqual(len(raw_na), 131)

        # 4 duplicates in non-anemia
        dup_mask = df.duplicated(keep="first")
        na_dups = df[df[TARGET_COLUMN].isin(TASK_B_EXCLUDED_CLASSES) & dup_mask]
        self.assertEqual(len(na_dups), 4)

        # Scenario 1 exclusions for Task B
        res1 = run_full_preprocessing(scenario="scenario1", verbose=False)
        self.assertEqual(res1["task_b_n_excluded"], 127)
        self.assertEqual(res1["task_b_excluded_breakdown"]["Thrombocytopenia"], 72)
        self.assertEqual(res1["task_b_excluded_breakdown"]["Leukemia"], 44)
        self.assertEqual(res1["task_b_excluded_breakdown"]["Leukemia with thrombocytopenia"], 11)

        # Scenario 2 exclusions for Task B
        res2 = run_full_preprocessing(scenario="scenario2", verbose=False)
        self.assertEqual(res2["task_b_n_excluded"], 123)
        self.assertEqual(res2["task_b_excluded_breakdown"]["Thrombocytopenia"], 69)
        self.assertEqual(res2["task_b_excluded_breakdown"]["Leukemia"], 44)
        self.assertEqual(res2["task_b_excluded_breakdown"]["Leukemia with thrombocytopenia"], 10)


class TestCleaningLogStructure(unittest.TestCase):
    """Verify cleaning log file integrity and absence of double-counting."""

    def test_cleaning_log_entries(self):
        """Verify cleaning_log.csv has 87 rows with exact step breakdown."""
        from ml_model.src.preprocessing import REPORTS_DIR
        log_file = REPORTS_DIR / "cleaning_log.csv"
        self.assertTrue(log_file.exists())
        df_log = pd.read_csv(log_file)
        self.assertEqual(len(df_log), 87)

        counts = df_log["step"].value_counts()
        self.assertEqual(counts["deduplication"], 49)
        self.assertEqual(counts["record_exclusion_scenario1"], 8)
        self.assertEqual(counts["record_16_flagged_for_review"], 1)
        self.assertEqual(counts["record_exclusion_scenario2"], 29)


if __name__ == "__main__":
    unittest.main()

