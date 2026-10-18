# ML Model Selection and Limitations Report

**Project:** AI-Based Anemia Screening and Blood Report Interpretation System  
**Module:** Module 2 — Machine Learning and Model Evaluation  
**Date:** October 13, 2026  
**Document Status:** Formal Documentation & Architectural Review Only  
**System Status:** **ML Module Complete, Integration Pending** — `ml_model/src/predict.py` is fully implemented (620 lines, 51 tests passing). The Django REST API backend and React frontend are **not yet implemented**. Models are strictly provisional research baselines and have **not** been approved or validated for real-world clinical use.

---

## 1. Executive Summary

This report establishes the model selection rationale, algorithmic trade-offs, and dataset limitations for the machine learning pipelines trained and evaluated in Module 2. 

Following the completion of the Phase 4 preprocessing protocols and the Phase 5 benchmark audits, this document evaluates the three standard classifier architectures—**Logistic Regression**, **Decision Tree**, and **Random Forest**—across:
1. **Task A:** 9-class multiclass diagnosis of anemia subtypes and related hematological conditions ($N=241$ clean test samples).
2. **Task B:** Binary anemia screening distinguishing Healthy individuals from Anemia subtypes ($N=216$ clean test samples).

The analysis focuses on **Scenario 2** (the primary analysis scenario where all 21 noisy, unverifiable Category B records are excluded from training), while maintaining strict awareness of dataset-specific artifacts, rare-class statistical instability, and missing patient demographic variables.

---

## 2. Quantitative Comparison of Models in Scenario 2

All models were evaluated on the exact, deterministic held-out clean test partitions established by the Fair Split Protocol (`test_size=0.20`, `random_state=42`). Test partitions contain zero Category A records (physically impossible values) and zero Category B records (unverifiable values).

### 2.1 Overall Performance Summary (Scenario 2)

| Task | Model | Tuned Thresh | Train / Test | Test Acc | Bal Acc | Macro F1 | Weighted F1 | Log Loss | Sensitivity (Recall) | Specificity | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task A** (9-Class) | Logistic Regression | N/A | 962 / 241 | 0.7925 | 0.7915 | 0.7217 | 0.7994 | 0.5958 | — | — | — | — |
| **Task A** (9-Class) | Decision Tree | N/A | 962 / 241 | 0.9751 | 0.9686 | 0.9689 | 0.9750 | 0.7514 | — | — | — | — |
| **Task A** (9-Class) | Random Forest | N/A | 962 / 241 | 0.9502 | 0.9218 | 0.9135 | 0.9520 | **0.2047** | — | — | — | — |
| **Task B** (Binary) | Logistic Regression | 0.340 | 864 / 216 | 0.9815 | 0.9688 | 0.9774 | 0.9813 | N/A | 1.0000 | 0.9375 | 0.9974 | 0.9989 |
| **Task B** (Binary) | Decision Tree | 0.100 | 864 / 216 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | N/A | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Task B** (Binary) | Random Forest | 0.370 | 864 / 216 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | N/A | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Note: For Task B, Class 1 = Anemia Subtypes ($N=152$), Class 0 = Healthy ($N=64$). Tuned thresholds were determined via 5-fold cross-validation on training data only.*

---

### 2.2 Task A: Class-Wise Performance Breakdown (Scenario 2)

| Class Name | Test Support | Logistic Regression Recall | Logistic Regression Precision | Decision Tree Recall | Decision Tree Precision | Random Forest Recall | Random Forest Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 64 | 0.7969 | 0.9444 | 1.0000 | 1.0000 | **1.0000** | 0.9846 |
| **Iron deficiency anemia** | 36 | 0.9167 | 0.9167 | 0.8889 | 0.9697 | **0.9167** | 0.9706 |
| **Leukemia** | 9 | 0.8889 | 0.6667 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **Leukemia with thrombocytopenia** | **2** | 1.0000 | 0.6667 | 1.0000 | 1.0000 | **1.0000** | 1.0000 |
| **Macrocytic anemia** | **3** | 0.3333 | 0.2000 | 1.0000 | 1.0000 | **0.6667** | 0.4000 |
| **Normocytic hypochromic anemia** | 53 | 0.6792 | 0.7059 | 1.0000 | 0.9464 | **0.9245** | 0.9074 |
| **Normocytic normochromic anemia** | 50 | 0.7800 | 0.8864 | 1.0000 | 1.0000 | **0.9600** | 0.9600 |
| **Other microcytic anemia** | 10 | 0.8000 | 0.4706 | 0.9000 | 0.8182 | **0.9000** | 1.0000 |
| **Thrombocytopenia** | 14 | 0.9286 | 0.6842 | 0.9286 | 1.0000 | **0.9286** | 1.0000 |

---

## 3. Algorithmic Trade-Offs (Why Accuracy Alone Is Misleading)

Selecting a machine learning model based strictly on raw test accuracy or unweighted F1-score is methodologically flawed, especially in clinical diagnostics. Each model family presents distinct architectural advantages and failure modes:

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 MODEL ARCHITECTURE TRADE-OFFS           │
                  └─────────────────────────────────────────────────────────┘

     Logistic Regression                Decision Tree                    Random Forest
  ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
  │ • Linear Boundaries     │    │ • Single Flowchart      │    │ • Ensemble of 150 Trees │
  │ • Transparent Odds      │    │ • Highly Brittle        │    │ • Variance Reduction    │
  │ • Underfits Non-Linear  │    │ • Overconfident Probs   │    │ • Low Log Loss (0.2047) │
  │   CBC Interactions      │    │ • High Log Loss (0.7514)│    │ • Smooth Boundaries     │
  └─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
```

### 3.1 Logistic Regression
* **Advantages:**
  * Highly interpretable: model coefficients represent log-odds contributions per laboratory feature.
  * Direct convex optimization without stochastic training variance.
  * Fast inference with negligible computational overhead.
* **Limitations & Failure Modes:**
  * Assumes linear decision boundaries in log-odds space. Complete blood count interpretation involves non-linear physiologic ratios (e.g., Mentzer Index $\text{MCV}/\text{RBC}$, Green-King Index, and non-linear interactions between MCV and MCH).
  * In Task A, it achieved only **79.25% accuracy** and **0.7217 Macro F1**.
  * Poor performance on rare and subtle classes: `Macrocytic anemia` recall was only **33.33%** (1 out of 3 samples detected, with 4 false positives yielding a precision of 20.0%). For `Normocytic hypochromic anemia`, it missed 17 patients (recall 67.9%).

### 3.2 Decision Tree
* **Advantages:**
  * High symbolic interpretability: can be visualized directly as a clinical flowchart with explicit split points (e.g., `HGB <= 12.95`, `MCV <= 80.5`).
  * Captures non-linear thresholds and interactions without requiring manual feature engineering.
* **Limitations & Failure Modes:**
  * **Extreme Variance and Overconfidence:** A single tree (configured here with `max_depth=8`) partitions feature space into rigid orthogonal hypercubes. Small perturbations in patient measurements near threshold boundaries cause dramatic diagnostic shifts.
  * **Severe Log-Loss Penalty:** In Task A, the Decision Tree achieved an apparent accuracy of **97.51%**, but had a poor **Log Loss of 0.7514**—significantly worse than Random Forest (0.2047) and even worse than Logistic Regression (0.5958). Because tree leaves output empirical step-function probabilities ($0.0$ or $1.0$), when the tree makes an error (e.g., misclassifying 4 Iron Deficiency Anemia patients), it is maximally penalized for overconfidence.
  * Susceptible to memorizing dataset-specific threshold artifacts.

### 3.3 Random Forest
* **Advantages:**
  * **Ensemble Stabilization:** Averages predictions across 150 randomized decision trees (`n_estimators=150`, `max_depth=12`), substantially dampening the high variance of individual trees.
  * **Superior Probabilistic Calibration:** Achieved the lowest **Log Loss (0.2047)** in Task A by a wide margin, demonstrating that its ensemble vote distributions are far better calibrated than a single tree.
  * **Robust Feature Integration:** In Task B, rather than relying exclusively on a single feature, Random Forest distributes importance across physiological correlates: **HGB (43.92%)**, **RBC (20.65%)**, **MCV (9.01%)**, and **PLT (7.57%)**.
  * Strong generalization: achieved **0.9502 accuracy**, **0.9218 Balanced Accuracy**, and **0.9135 Macro F1** in Task A.
* **Limitations:**
  * Loss of direct flowchart transparency (requires post-hoc feature importance or SHAP values for clinical explanation).
  * Output scores represent tree vote fractions rather than formally calibrated posterior probabilities (Brier score / calibration scaling still required).
  * Modest precision on `Macrocytic anemia` ($0.4000$) due to 3 false alarms on a tiny support set ($N=3$).

---

## 4. In-Depth Analysis: The 100% Binary Screening Result

In Task B, both Decision Tree and Random Forest achieved **100% Sensitivity, 100% Specificity, 1.0000 Accuracy, 1.0000 ROC-AUC, and 1.0000 PR-AUC** across the 216 clean test records.

> [!WARNING]
> **Critical Clinical Finding:** This 100% benchmark score is a **dataset artifact**, NOT proof that machine learning can diagnose anemia with flawless clinical accuracy. It must never be marketed or represented as infallible clinical performance.

### 4.1 The Empirical Cutoff Mechanism
Inspection of feature distributions in the test set ($N=216$) reveals a completely non-overlapping distribution in **Hemoglobin (`HGB`)**:
* **Healthy Patient Cohort ($N=64$):**
  * Minimum HGB: **13.00 g/dL**
  * Maximum HGB: **16.20 g/dL**
  * Mean: 14.00 g/dL, Median: 13.90 g/dL
* **Anemia Subtypes Cohort ($N=152$):**
  * Minimum HGB: **5.00 g/dL**
  * Maximum HGB: **12.90 g/dL**
  * Mean: 10.89 g/dL, Median: 11.35 g/dL
* **Boundary Gap:** There is an artificial gap between **12.90 g/dL** and **13.00 g/dL**.

Because of this gap, a trivial rule (`HGB <= 12.95 g/dL`) cleanly separates every single healthy patient from every anemia patient in this test set. In the Decision Tree, `HGB` accounted for **98.69%** of total split importance.

### 4.2 Why This Fails in Real-World Clinical Practice
1. **Biological Sex Differences:**
   * World Health Organization (WHO) diagnostic guidelines establish sex-stratified thresholds:
     * Adult Males: $\text{HGB} < 13.0$ g/dL
     * Non-Pregnant Adult Females: $\text{HGB} < 12.0$ g/dL
     * Pregnant Females: $\text{HGB} < 11.0$ g/dL
   * A healthy female patient with $\text{HGB} = 12.5$ g/dL is completely normal by hematological standards, but would be flagged as **false-positive anemic** by a model relying on an unstratified $12.95$ g/dL threshold.
2. **Hydration and Hemoconcentration:**
   * Severe plasma volume contraction (dehydration, diuresis) artificially elevates measured hemoglobin concentration, masking true red blood cell mass deficits.
3. **Latent / Early Iron Deficiency:**
   * In stage 1 and stage 2 iron deficiency, bone marrow iron stores and serum ferritin are depleted while hemoglobin remains within normal limits. A binary screening model relying strictly on HGB cannot detect pre-anemic iron deficiency.
4. **Instrument Measurement Drift:**
   * Standard automated hematology analyzers (photometric cyanmethemoglobin or SLS methods) have typical analytical coefficients of variation yielding $\pm 0.2\text{--}0.4$ g/dL measurement variance. Borderline patients at 12.8–13.1 g/dL cannot be categorized with absolute certainty from a single blood draw.

---

## 5. Rare-Class Statistical Instability in Task A

The clean test partition for Task A ($N=241$) has severe class imbalance:

| Diagnosis Category | Test Count ($N$) | % of Partition | Statistical Limitation |
| :--- | :---: | :---: | :--- |
| **Healthy** | 64 | 26.56% | Sufficient support for reliable evaluation. |
| **Normocytic hypochromic anemia** | 53 | 21.99% | Sufficient support for reliable evaluation. |
| **Normocytic normochromic anemia** | 50 | 20.75% | Sufficient support for reliable evaluation. |
| **Iron deficiency anemia** | 36 | 14.94% | Adequate support for reliable evaluation. |
| **Thrombocytopenia** | 14 | 5.81% | Moderate sample size. |
| **Other microcytic anemia** | 10 | 4.15% | Small sample size; sensitive to individual errors. |
| **Leukemia** | 9 | 3.73% | Small sample size; sensitive to individual errors. |
| **Macrocytic anemia** | **3** | **1.24%** | **Critically small; recall steps in units of 33.3%.** |
| **Leukemia with thrombocytopenia** | **2** | **0.83%** | **Critically small; recall steps in units of 50.0%.** |

### Impact of Small Sample Sizes:
* For `Leukemia with thrombocytopenia` ($N=2$), each patient represents **50% of the class recall**. A single error drops sensitivity from 100% to 50%.
* For `Macrocytic anemia` ($N=3$), each patient represents **33.3% of the class recall**. 
  * Random Forest identified 2 of 3 patients (recall = 66.67%), but incurred 3 false positives, causing precision to drop to 40.0%.
  * Logistic Regression identified only 1 of 3 patients (recall = 33.33%), with 4 false positives (precision = 20.0%).
* **Takeaway:** While Macro F1 ($0.9135$ for Random Forest) appropriately weights each class equally, single-point estimates for classes with $N \le 3$ possess wide confidence intervals and must be communicated with explicit uncertainty caveats.

---

## 6. Missing Clinical and Demographic Context

The current dataset (`anemia_types.csv`) has substantial structural gaps that prevent real-world clinical deployment:

1. **Absence of Patient Age:**
   * Hematological reference intervals vary drastically between neonates (HGB 14–24 g/dL), infants (HGB 10–13 g/dL), children (HGB 11–14 g/dL), adults, and geriatric populations. Without age metadata, age-adjusted interpretation is impossible.
2. **Absence of Biological Sex:**
   * Because male and female adult hemoglobin baselines differ by roughly $1.0\text{--}2.0$ g/dL, an unstratified model cannot distinguish a normal female hemoglobin ($12.2$ g/dL) from male mild anemia ($12.2$ g/dL).
3. **Absence of Pregnancy Status:**
   * Physiological hemodilution during pregnancy expands plasma volume, lowering the normal hemoglobin cutoff to $11.0$ g/dL.
4. **Single-Source Data Provenance:**
   * The dataset originates from a single laboratory distribution. It has not undergone cross-laboratory validation across different hematology instrumentation (e.g., Sysmex XN, Beckman Coulter DxH, Abbott Alinity).
5. **Pre-Imputation Mode Artifacts:**
   * As documented in Phase 3 audits, $30.6\%$ of records contain an identical mode value in `LYMp` (25.845) and $30.8\%$ in `LYMn` (1.88076), indicating that synthetic or mean/median pre-imputation was performed prior to dataset distribution.

---

## 7. Provisional Model Recommendations

These recommendations designate **provisional architectural candidates** for downstream prototype engineering. They do **not** constitute clinical validation or regulatory approval.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROVISIONAL CANDIDATE SUMMARY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Task A (Multiclass Diagnosis):                                             │
│    -> Primary Candidate: Scenario 2 Random Forest                           │
│    -> Pipeline Artifact: task_a_scenario2_random_forest.joblib               │
│    -> Key Strength: Lowest Log Loss (0.2047), Robust Ensemble (Macro F1 0.91)│
│                                                                             │
│  Task B (Binary Screening):                                                 │
│    -> Primary Candidate: Scenario 2 Random Forest                           │
│    -> Pipeline Artifact: task_b_scenario2_random_forest.joblib               │
│    -> Secondary / Comparative: Scenario 2 Logistic Regression                │
│    -> Pipeline Artifact: task_b_scenario2_logistic_regression.joblib         │
│    -> Key Strength: Multi-feature integration (HGB + RBC + MCV + PLT)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.1 Task A (9-Class Multiclass Diagnosis)
* **Provisional Recommendation:** **Scenario 2 Random Forest** (`task_a_scenario2_random_forest.joblib`)
* **Supporting Evidence:**
  1. **Log-Loss Superiority:** Achieved a log loss of **0.2047**, compared to 0.7514 for Decision Tree and 0.5958 for Logistic Regression, indicating far superior class probability distributions.
  2. **Balanced Performance:** Macro F1 of **0.9135**, Balanced Accuracy of **0.9218**, and Weighted F1 of **0.9520**.
  3. **High Diagnostic Sensitivity on Major Anemias:** Recall of $91.67\%$ on Iron Deficiency Anemia, $92.45\%$ on Normocytic Hypochromic Anemia, $96.00\%$ on Normocytic Normochromic Anemia, and $100.0\%$ on Healthy.
* **Limitations to Document:**
  * Modest precision on `Macrocytic anemia` ($0.4000$) due to small sample size ($N=3$).
  * Uncalibrated voting probabilities (requires post-hoc calibration before clinical presentation).
  * Feature interactions must be interpreted via SHAP / feature importance rather than a single flowchart.

### 7.2 Task B (Binary Anemia Screening)
* **Provisional Recommendation:** **Scenario 2 Random Forest** (`task_b_scenario2_random_forest.joblib`), with **Scenario 2 Logistic Regression** maintained as a configurable comparative baseline.
* **Supporting Evidence:**
  1. **Multi-Marker Redundancy:** Unlike the Decision Tree, which placed $98.7\%$ of split reliance solely on HGB, Random Forest distributes predictive weight across physiological red cell parameters: HGB ($43.9\%$), RBC ($20.7\%$), MCV ($9.0\%$), and PLT ($7.6\%$).
  2. **Cross-Validated Threshold Tuning:** Utilizes an optimal decision threshold of **0.370**, tuned via 5-fold cross-validation on training data only.
  3. **Role of Logistic Regression Baseline:** Scenario 2 Logistic Regression (`tuned_threshold=0.340`) achieved **100% Sensitivity (152/152)** and **93.75% Specificity (60/64)** with ROC-AUC **0.9974**. It offers direct linear log-odds transparency and continuous sigmoidal scores, making it a valuable comparative model.
* **Limitations to Document:**
  * The 100% benchmark score on Random Forest is inflated by the unstratified $12.95$ g/dL cutoff.
  * Real-world deployment will yield false positives in healthy adult females unless threshold adjustments are made.

---

## 8. Architectural Strategy: Configurable Model Selection for Inference

To ensure the software system remains adaptable as clinical requirements evolve, **model selection must NOT be hard-coded** into upcoming inference and application layers.

### 8.1 Design Principles for `ml_model/src/predict.py`
When implementing the prediction module in future phases, the following architecture should be enforced:

1. **Configurable Model Dispatcher:**
   * Support dynamic selection via a configuration object or environment variable:
     ```python
     # Concept: Configurable selection (to be implemented in future phases)
     DEFAULT_TASK_A_MODEL = "scenario2_random_forest"
     DEFAULT_TASK_B_MODEL = "scenario2_random_forest"
     ```
   * Allow downstream services to toggle between `random_forest`, `logistic_regression`, and `decision_tree` for benchmarking and sensitivity analysis.
2. **Explicit Decision Threshold Enforcement:**
   * Always load the cross-validation tuned threshold from `model_metadata.json` (e.g., $0.370$ for Scenario 2 Random Forest) rather than using a default $0.50$ threshold.
3. **Rigid Pre-Inference Physiological Validation:**
   * Intercept incoming laboratory inputs and reject physiologically impossible values before model evaluation:
     * Flag negative values ($\text{HGB} < 0$, $\text{MCV} < 0$).
     * Flag impossible percentages ($\text{HCT} > 100\%$, $\text{NEUTp} > 100\%$, $\text{LYMp} > 100\%$).
     * Flag extreme cell counts ($\text{RBC} > 15 \times 10^6/\mu\text{L}$, $\text{WBC} > 300 \times 10^3/\mu\text{L}$).
4. **Mandatory Disclaimer Embedding:**
   * Every prediction output dictionary must contain explicit medical disclaimers stating that predictions are educational research outputs and require clinician verification.

---

## 9. Remaining Gaps and Unresolved Questions

1. **Bootstrap Confidence Intervals:**
   * Metric point estimates on classes with 2 or 3 test samples carry substantial uncertainty. Generating 95% empirical bootstrap intervals (1,000 iterations) is recommended to formally quantify evaluation variance.
2. **Probability Calibration Post-Processing:**
   * Raw model probabilities from Decision Trees (step functions) and Random Forests (vote fractions) require calibration curves (e.g., via `CalibratedClassifierCV` or Platt scaling) before being interpreted as probabilistic risk levels.
3. **Absence of Demographic Metadata:**
   * Because patient age and biological sex cannot be retroactively added to `anemia_types.csv`, should the inference service incorporate a rule-based post-processing layer applying WHO demographic cutoffs to incoming patient reports?
4. **Integration Readiness:**
   * Neither the backend Django REST framework API nor the frontend web application is functional. The next development phase must focus on `ml_model/src/predict.py` input validation and inference handling before API endpoints can be wired up.

---
*(End of Formal ML Model Selection and Limitations Report)*
