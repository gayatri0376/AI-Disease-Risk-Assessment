# Phase 5.1 — Independent Machine Learning Benchmark Audit Report

**Project**: AI-Based Anemia Screening and Blood Report Interpretation System  
**Date**: October 11, 2026  
**Auditor**: ML Engineering & Quality Assurance  
**Status**: **PASSED — READY FOR NEXT PHASE** (Zero critical defects, 0 test leakage, 100% metric reproducibility)  
**Dataset Integrity**: Checksum `c6375e974d730a42b53160614237ddf0` verified unmodified.

---

## Executive Summary

An exhaustive independent audit was conducted on the Phase 4 and Phase 5 machine-learning codebase, datasets, pipelines, and evaluation outputs. All 12 serialized model pipelines, metrics tables, cross-validation mechanisms, threshold-selection functions, and splitting algorithms were tested and verified against the raw dataset.

### Audit Findings Overview
* **Passed Checks**: 15 / 18 checks passed with zero defects.
* **Informational / Methodological Items**: 3 items documented (clinical diagnostic separation in Task B, rare class evaluation variance in Task A, and synthetic feature concentration).
* **Failed Checks**: **0** (No critical or major failures).
* **Decision**: **Approved to proceed to Phase 6 (Inference / Prediction Module & Application Development)**.

---

## 1. Audit of the Fair Comparison Protocol

The splitting implementation in [`ml_model/src/preprocessing.py`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/src/preprocessing.py) (lines 1183–1248) was inspected line-by-line:

| Requirement | Audit Verification Result | Evidence & File Location |
| :--- | :--- | :--- |
| **Task A Test Set Equality** | **PASS**: Exactly 241 test records. Identical indices and feature values across Scenario 1 and Scenario 2. | `preprocessing.py:1196` |
| **Task B Test Set Equality** | **PASS**: Exactly 216 test records. Identical indices and feature values across Scenario 1 and Scenario 2. | `preprocessing.py:1232` |
| **Zero Category A or B in Test Sets** | **PASS**: The test partitions were drawn exclusively from `df_clean_pool` (which strips out all 8 Category A and all 21 Category B records before splitting). Overlap count = 0. | `preprocessing.py:1194` |
| **Zero Train/Test Overlap** | **PASS**: `len(Train ∩ Test) == 0` verified for both tasks in both scenarios. | `test_preprocessing.py:220` |
| **Scenario 1 Training Eligibility** | **PASS**: 21 Category B records are appended strictly to the training pool (`X_a_train = concat([X_a_tr_clean, X_cat_b])`). | `preprocessing.py:1202` |
| **Scenario 2 Complete Exclusion** | **PASS**: Category B records are excluded from both training and test partitions. | `preprocessing.py:1209` |
| **Task A Target Classes** | **PASS**: All 9 diagnosis categories preserved in features and targets. | `preprocessing.py:1212` |
| **Task B Target Classes** | **PASS**: Only `Healthy` (0) and the 5 designated anemia subtypes (1) are present. The 3 non-anemia classes (`Leukemia`, `Leukemia with thrombocytopenia`, `Thrombocytopenia`) are excluded. | `preprocessing.py:1229` |
| **No Target Leakage in Splitting** | **PASS**: Only the 14 CBC features are passed to `X`. The target column `Diagnosis` is never used as a predictive input. | `preprocessing.py:67` |

---

## 2. Audit of Data Leakage & Duplicate Risks

| Potential Leakage Vector | Investigation Method & Finding | Status |
| :--- | :--- | :--- |
| **Target Feature Leakage** | Feature lists were inspected. Predictors consist solely of 14 continuous laboratory features: `['WBC', 'LYMp', 'NEUTp', 'LYMn', 'NEUTn', 'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'PLT', 'PDW', 'PCT']`. | **PASS (No Leakage)** |
| **Row Index / Identifier Leakage** | Indices are reset or preserved as pure DataFrame index keys. No index or ID is passed as a column to the estimators. | **PASS (No Leakage)** |
| **Transform Fitting Timing** | Inspected [`ml_model/src/train.py`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/src/train.py) lines 145–155 and 215–225. Preprocessing pipelines are embedded inside scikit-learn `Pipeline` objects. Transformations (`SimpleImputer` and `StandardScaler`) are fit exclusively during `pipe.fit(X_train, y_train)` on training data. | **PASS (No Leakage)** |
| **Exact Duplicate Leakage** | 49 exact duplicate rows were dropped in Step 1 prior to splitting. No exact duplicates exist between train and test partitions (`min_distance > 0` for all 241 test samples). | **PASS (No Leakage)** |
| **Near-Duplicate Density** | Computed pairwise normalized Euclidean distance from test samples to nearest training samples. **Finding**: Minimum normalized Euclidean distance is 0.0065, median is 0.8605, and mean is 1.1050. Exactly 13 test samples (5.3%) have a normalized distance < 0.1 to their closest training record. **Cause**: Traced to the pre-imputed mode values in `LYMp` (25.845) and `LYMn` (1.88076) present in the original dataset, rather than duplicate patient records. | **INFO (Documented)** |
| **Validation Fold Separation** | In training cross-validation (`find_optimal_threshold_cv` and `cv_macro_f1`), `StratifiedKFold(n_splits=5)` operates strictly within `X_train`. The held-out test partition is never exposed to fold fitting. | **PASS (No Leakage)** |

---

## 3. Independent Recalculation & Metric Verification

All 12 saved `.joblib` pipelines in [`ml_model/models/`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/models/) were loaded and re-evaluated against the exact test partitions (`task_a_X_test`, `task_b_X_test`).

### Independent Verification Results

```
Task A (Multiclass — 9 Classes, 241 Test Records):
  • scenario1_logistic_regression : Accuracy=0.7178 | BalAcc=0.7134 | Macro F1=0.6541 | Log Loss=1.0006 (Exact match)
  • scenario1_decision_tree       : Accuracy=0.9917 | BalAcc=0.9858 | Macro F1=0.9858 | Log Loss=0.2991 (Exact match)
  • scenario1_random_forest       : Accuracy=0.9668 | BalAcc=0.9389 | Macro F1=0.9262 | Log Loss=0.2016 (Exact match)
  • scenario2_logistic_regression : Accuracy=0.7925 | BalAcc=0.7915 | Macro F1=0.7217 | Log Loss=0.5958 (Exact match)
  • scenario2_decision_tree       : Accuracy=0.9751 | BalAcc=0.9686 | Macro F1=0.9689 | Log Loss=0.7514 (Exact match)
  • scenario2_random_forest       : Accuracy=0.9502 | BalAcc=0.9218 | Macro F1=0.9135 | Log Loss=0.2047 (Exact match)

Task B (Binary Screening — 216 Test Records):
  • scenario1_logistic_regression : Threshold=0.390 | Sens=0.9605 | Spec=0.8438 | F1=0.9481 | ROC-AUC=0.9616 (Exact match)
  • scenario1_decision_tree       : Threshold=0.100 | Sens=1.0000 | Spec=1.0000 | F1=1.0000 | ROC-AUC=1.0000 (Exact match)
  • scenario1_random_forest       : Threshold=0.410 | Sens=1.0000 | Spec=1.0000 | F1=1.0000 | ROC-AUC=1.0000 (Exact match)
  • scenario2_logistic_regression : Threshold=0.340 | Sens=1.0000 | Spec=0.9375 | F1=0.9870 | ROC-AUC=0.9974 (Exact match)
  • scenario2_decision_tree       : Threshold=0.100 | Sens=1.0000 | Spec=1.0000 | F1=1.0000 | ROC-AUC=1.0000 (Exact match)
  • scenario2_random_forest       : Threshold=0.370 | Sens=1.0000 | Spec=1.0000 | F1=1.0000 | ROC-AUC=1.0000 (Exact match)
```

**Discrepancy**: **0.000000** across all 12 models. Positive class is confirmed as **`1` (Anemia subtypes)**.

---

## 4. Threshold Selection & Cross-Validation Audit

Inspected `find_optimal_threshold_cv()` in [`ml_model/src/train.py`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/src/train.py) (lines 104–130):

1. **Test Set Isolation**: The function accepts `X_train` and `y_train` exclusively. Test labels are never accessed during threshold determination.
2. **Objective Function**: Sweeps 81 candidate thresholds from `0.10` to `0.90` (step `0.01`) and selects the threshold maximizing cross-validated training F1 score for class `1`.
3. **Tie-Breaking Rule**: Selects the first threshold encountered with strict inequality (`score > best_f1`).
4. **Leakage Assessment**:
   * `cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")` evaluates out-of-fold predictions.
   * Inside `cross_val_predict`, each fold executes the pipeline's `fit` method on 4/5 of training data and transforms the held-out 1/5. Preprocessing parameters are not leaked across folds.
   * Tuned thresholds (e.g., `0.340` for Scenario 2 Logistic Regression) reflect training optimization only.

---

## 5. Investigation of Perfect (100%) Scores in Task B

In Task B, Decision Tree and Random Forest achieved 100% Sensitivity, 100% Specificity, and 1.000 ROC-AUC on the held-out test partition. An empirical investigation was conducted to determine the underlying cause:

### Empirical Evidence & Findings
1. **Feature Importance Analysis**:
   * In Decision Tree, **Hemoglobin (HGB)** accounts for **98.69%** of feature importance.
   * In Random Forest, **HGB** accounts for **43.92%**, followed by **RBC (20.65%)**, **MCV (9.01%)**, and **PLT (7.57%)**.
2. **Clinical Cutoff Structure in Dataset**:
   * In the clean test set (216 rows):
     * `Healthy` (Class 0): Minimum HGB = **13.0 g/dL**, Maximum HGB = **16.2 g/dL**.
     * `Anemia Subtypes` (Class 1): Minimum HGB = **5.0 g/dL**, Maximum HGB = **12.9 g/dL**.
   * In this dataset's clean records, there is **zero overlap** in Hemoglobin between Healthy patients and Anemia patients. Anemia by medical definition is diagnosed when hemoglobin drops below standard clinical thresholds (~12.0–13.0 g/dL depending on sex/age).
3. **Conclusion**:
   * The 100% score is **not caused by label leakage or coding error**. It reflects that a simple tree split on `HGB <= 12.95 g/dL` cleanly separates Healthy from Anemia in this curated dataset.
   * **Real-World Limitation**: In true clinical practice, borderline cases (e.g., mild anemia vs. low-normal healthy individuals) overlap due to biological variability, dehydration, or differing demographic reference intervals. This limitation is documented for honest reporting.

### Rare-Class Uncertainty in Task A (Multiclass)
In Task A, the held-out clean test partition contains **241 samples** across 9 classes. Sample distribution:
* `Healthy`: 64 test samples (26.6%)
* `Normocytic hypochromic anemia`: 53 test samples (22.0%)
* `Normocytic normochromic anemia`: 50 test samples (20.7%)
* `Iron deficiency anemia`: 36 test samples (14.9%)
* `Thrombocytopenia`: 14 test samples (5.8%)
* `Other microcytic anemia`: 10 test samples (4.1%)
* `Leukemia`: 9 test samples (3.7%)
* **`Macrocytic anemia`**: **3 test samples (1.2%)**
* **`Leukemia with thrombocytopenia`**: **2 test samples (0.8%)**

**Assessment**:
* For `Macrocytic anemia`, Random Forest correctly classified 2 of 3 samples (Precision=0.400, Recall=0.667). A single misclassification swings class recall by **33.3%**.
* For `Leukemia with thrombocytopenia`, Random Forest correctly classified 2 of 2 samples (Precision=1.000, Recall=1.000). A single misclassification swings class recall by **50.0%**.
* Macro F1 and Balanced Accuracy remain the appropriate metrics, but confidence intervals for these rare classes are inherently wide.

---

## 6. Audit of Saved Artifacts & Integrity

1. **12 Serialized Model Pipelines**: All present in [`ml_model/models/`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/models/) and verified loadable via `joblib.load()`:
   * `task_a_scenario1_logistic_regression.joblib`
   * `task_a_scenario1_decision_tree.joblib`
   * `task_a_scenario1_random_forest.joblib`
   * `task_a_scenario2_logistic_regression.joblib`
   * `task_a_scenario2_decision_tree.joblib`
   * `task_a_scenario2_random_forest.joblib`
   * `task_b_scenario1_logistic_regression.joblib`
   * `task_b_scenario1_decision_tree.joblib`
   * `task_b_scenario1_random_forest.joblib`
   * `task_b_scenario2_logistic_regression.joblib`
   * `task_b_scenario2_decision_tree.joblib`
   * `task_b_scenario2_random_forest.joblib`
2. **Metadata Integrity**: [`ml_model/models/model_metadata.json`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/models/model_metadata.json) accurately matches model hyperparameters, file paths, test sample counts, and threshold values.
3. **Report Regeneration**: Re-running [`ml_model/src/evaluate.py`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/src/evaluate.py) reproduced the entire table in [`ml_model/reports/evaluation_results.csv`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/reports/evaluation_results.csv) with zero discrepancies.
4. **No Hardcoded Values**: Inspected code for hardcoded evaluation outputs; all metrics are computed dynamically from `sklearn.metrics`.

---

## 7. Audit Findings Table

The findings table has been written to [`ml_model/reports/phase5_audit_findings.csv`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/reports/phase5_audit_findings.csv) with the following records:

| Audit ID | Category | Check Description | Status | Severity | Evidence Location |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **AUD-01** | Fair Protocol | Task A Test Set Equality (241 rows) | **PASS** | NONE | `preprocessing.py:1196` |
| **AUD-02** | Fair Protocol | Task B Test Set Equality (216 rows) | **PASS** | NONE | `preprocessing.py:1232` |
| **AUD-03** | Fair Protocol | Zero Cat A/B Records in Test Sets | **PASS** | NONE | `preprocessing.py:1194` |
| **AUD-04** | Fair Protocol | Zero Train/Test Index Overlap | **PASS** | NONE | `preprocessing.py:1213` |
| **AUD-05** | Fair Protocol | Scenario 1 Cat B Training Inclusion | **PASS** | NONE | `preprocessing.py:1202` |
| **AUD-06** | Fair Protocol | Scenario 2 Cat B Total Exclusion | **PASS** | NONE | `preprocessing.py:1209` |
| **AUD-07** | Fair Protocol | Task A 9 Diagnosis Classes Preserved | **PASS** | NONE | `preprocessing.py:1212` |
| **AUD-08** | Fair Protocol | Task B Target Definitions & Exclusions | **PASS** | NONE | `preprocessing.py:1229` |
| **AUD-09** | Leakage Risk | Absence of Target Predictive Leakage | **PASS** | NONE | `preprocessing.py:67` |
| **AUD-10** | Leakage Risk | Absence of Row ID / Index Leakage | **PASS** | NONE | `preprocessing.py:67` |
| **AUD-11** | Leakage Risk | Preprocessing Transform Fitting Isolation | **PASS** | NONE | `train.py:149` |
| **AUD-12** | Leakage Risk | Exact Duplicate Overlap Elimination | **PASS** | NONE | `preprocessing.py:345` |
| **AUD-13** | Leakage Risk | Near-Duplicate Density via Pre-Imputation | **INFO** | LOW | `anemia_types.csv` |
| **AUD-14** | Metrics Verification | Metric Calculation Reproducibility | **PASS** | NONE | `evaluation_results.csv` |
| **AUD-15** | Threshold Audit | Training-Only Threshold Tuning | **PASS** | NONE | `train.py:111` |
| **AUD-16** | High Score Audit | Task B 100% Score Cutoff Identification | **INFO** | LOW | `train.py:228` |
| **AUD-17** | Class Distribution | Rare Class Small Test Sample Variance | **INFO** | MEDIUM | `model_comparison_report.txt` |
| **AUD-18** | Artifact Integrity | 12 Model Pipelines Exist & Load | **PASS** | NONE | `ml_model/models/` |
| **AUD-19** | Dataset Integrity | Raw Dataset MD5 Checksum Intact | **PASS** | NONE | `anemia_types.csv` |

---

## 8. Unit Test Suite Re-Run

All 21 unit tests in [`ml_model/src/test_preprocessing.py`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/src/test_preprocessing.py) were executed to ensure full test suite compliance:

```bash
$ .venv/bin/python -m unittest ml_model/src/test_preprocessing.py -v

test_cleaning_log_entries ... ok
test_load_dataset_shape_and_columns ... ok
test_raw_csv_md5_hash_preserved ... ok
test_validate_columns_raises_on_missing ... ok
test_category_records_not_in_duplicates ... ok
test_exact_duplicates_count ... ok
test_category_a_mapping ... ok
test_category_b_mapping ... ok
test_record_16_protocol ... ok
test_verify_exclusion_lists_succeeds ... ok
test_disjoint_train_and_test ... ok
test_identical_test_indices_across_scenarios ... ok
test_zero_category_b_in_test_set ... ok
test_non_anemia_stage_counts ... ok
test_no_data_leakage_between_train_and_test ... ok
test_pipeline_is_unfitted ... ok
test_split_reproducibility ... ok
test_invalid_scenario_raises_value_error ... ok
test_scenario_1_and_2_row_counts ... ok
test_task_a_preserves_all_nine_classes ... ok
test_task_b_binary_mapping_and_exclusions ... ok

----------------------------------------------------------------------
Ran 21 tests in 0.186s

OK
```

---

## 9. Final Decision & Recommendations

### Final Audit Decision: **READY FOR NEXT IMPLEMENTATION PHASE**
* **Zero Defects Identified**: The machine learning pipeline adheres strictly to fair comparison protocols, zero test leakage, training-only transformation fitting, and reproducible artifact storage.
* **Original CSV Intact**: Checksum `c6375e974d730a42b53160614237ddf0` matches original raw dataset state.
* **Next Implementation Steps (Phase 6)**:
  1. Implement inference module ([`ml_model/src/predict.py`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/src/predict.py)) to accept patient CBC blood profiles, run validation checks on feature ranges, apply the trained preprocessing pipeline, and output screening predictions with confidence estimates and clinical disclaimers.
  2. Implement backend integration / interactive report interpretation service.
