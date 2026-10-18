# Prediction Service Architecture & Usage Guide

**Module**: `backend.apps.ml_service`  
**Purpose**: Reusable Python inference service integrating Phase 5 audited machine-learning models for binary anemia screening and multiclass diagnosis prediction.  
**Audience**: Backend API engineers, pair programmers, academic evaluators.  
**Status**: Completed and unit-tested (19/19 service tests passing; 21/21 preprocessing tests passing).

---

## 1. How the Service Loads Models

The service loads serialized scikit-learn `Pipeline` objects from [`ml_model/models/`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/models/) using Python's `joblib`:

1. **Lazy Loading & In-Memory Caching**: Models are loaded on-demand and cached in `_MODEL_CACHE` by model key (e.g. `task_b_scenario2_random_forest`). Subsequent prediction calls execute with zero file I/O overhead.
2. **Metadata Synchronization**: Operational parameters (such as the exact cross-validated threshold for Task B) are loaded from [`ml_model/models/model_metadata.json`](file:///Users/gayatri/Documents/AI-Anemia-Screening/ml_model/models/model_metadata.json).
3. **Embedded Preprocessing**: Each saved pipeline encapsulates both feature preprocessing (`SimpleImputer(strategy='median')` and `StandardScaler()`) and the trained estimator (`RandomForestClassifier`, `DecisionTreeClassifier`, or `LogisticRegression`). Input features are passed directly as standard continuous floats; no external scaling or transformation is needed outside the pipeline.

### Model Identifiers & Default Settings
* **Default Scenario**: `scenario_2` (canonical: `scenario2`). In Scenario 2, all 21 suspicious Category B records were excluded from training, yielding the cleanest benchmark performance.
* **Default Model**: `random_forest` (achieved the highest balanced accuracy and test F1 across Phase 5 evaluations).
* **Available Models**: `random_forest`, `decision_tree`, `logistic_regression`.
* **Available Scenarios**: `scenario_2` (or `scenario2`), `scenario_1` (or `scenario1`).

---

## 2. Expected Features, Units, and Permissible Ranges

The input must be a dictionary containing **exactly the 14 CBC features**. No target labels (`Diagnosis`) or surrogate record IDs are permitted.

| Feature Name | Clinical Meaning | Standard Unit | Physical Sanity Bounds | Clinical Notes |
| :--- | :--- | :---: | :---: | :--- |
| `WBC` | White Blood Cell Count | $\times 10^3 / \mu\text{L}$ | `[0.1, 300.0]` | Total leukocyte count |
| `LYMp` | Lymphocyte Differential Percentage | $\%$ | `[0.0, 100.0]` | Must satisfy $0 \le \text{LYMp} \le 100$ |
| `NEUTp` | Neutrophil Differential Percentage | $\%$ | `[0.0, 100.0]` | Must satisfy $0 \le \text{NEUTp} \le 100$ |
| `LYMn` | Absolute Lymphocyte Count | $\times 10^3 / \mu\text{L}$ | `[0.0, 150.0]` | Absolute count |
| `NEUTn` | Absolute Neutrophil Count | $\times 10^3 / \mu\text{L}$ | `[0.0, 250.0]` | Absolute count |
| `RBC` | Red Blood Cell (Erythrocyte) Count | $\times 10^6 / \mu\text{L}$ | `[0.5, 15.0]` | Physiological upper limit $\sim 10$ |
| `HGB` | Hemoglobin Concentration | $\text{g/dL}$ | `[0.0, 30.0]` | Cannot be negative; key indicator |
| `HCT` | Hematocrit (Packed Cell Volume) | $\%$ | `[0.0, 100.0]` | Percentage volume; cannot exceed 100% |
| `MCV` | Mean Corpuscular Volume | $\text{fL}$ | `[20.0, 200.0]` | Average erythrocyte volume |
| `MCH` | Mean Corpuscular Hemoglobin | $\text{pg}$ | `[5.0, 100.0]` | Average hemoglobin content per cell |
| `MCHC` | Mean Corpuscular HGB Concentration | $\text{g/dL}$ | `[10.0, 60.0]` | Cellular hemoglobin concentration |
| `PLT` | Platelet (Thrombocyte) Count | $\times 10^3 / \mu\text{L}$ | `[1.0, 2500.0]` | Essential clotting element count |
| `PDW` | Platelet Distribution Width | $\text{fL}$ | `[1.0, 60.0]` | Platelet size variation |
| `PCT` | Plateletcrit (Platelet Volume Ratio) | $\%$ | `[0.0, 5.0]` | Typical normal range is $0.1–0.5\%$ |

*Note on Sanity Bounds: These bounds are conservative mathematical guards designed to reject corrupt, impossible, or misplaced units (e.g. negative HGB, percentage > 100%). They do NOT represent diagnostic reference ranges and do NOT validate physiological normality.*

---

## 3. How to Invoke the Prediction Functions

### Python API Usage

```python
from backend.apps.ml_service import predict_anemia_screening, predict_diagnosis

sample_cbc = {
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

# 1. Binary Anemia Screening (Task B)
screening_result = predict_anemia_screening(
    cbc_values=sample_cbc,
    scenario="scenario_2",       # default
    model_name="random_forest",  # default
)

# 2. Multiclass Diagnosis Classification (Task A)
diagnosis_result = predict_diagnosis(
    cbc_values=sample_cbc,
    scenario="scenario_2",       # default
    model_name="random_forest",  # default
)
```

### Example Output: Task B (Binary Anemia Screening)

```json
{
  "task": "Task B (Binary Anemia Screening)",
  "predicted_class": 0,
  "screening_result": "Healthy (Non-Anemic)",
  "probability_anemia": 0.0,
  "probability_healthy": 1.0,
  "applied_threshold": 0.37,
  "model_used": "random_forest",
  "scenario": "scenario2",
  "model_path": "ml_model/models/task_b_scenario2_random_forest.joblib",
  "training_rows": 864,
  "features_analyzed": ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"],
  "clinical_disclaimer": "DISCLAIMER: This prediction is generated by an academic machine-learning model for educational and screening demonstration purposes only. It is NOT clinically validated, does NOT constitute a medical diagnosis, and must NOT be used for patient care decisions. Consult a qualified healthcare professional or laboratory physician for proper medical interpretation."
}
```

### Example Output: Task A (Multiclass Diagnosis Classification)

```json
{
  "task": "Task A (Multiclass Diagnosis Classification)",
  "predicted_diagnosis": "Healthy",
  "class_probabilities": {
    "Healthy": 0.985,
    "Iron deficiency anemia": 0.005,
    "Leukemia": 0.0,
    "Leukemia with thrombocytopenia": 0.0,
    "Macrocytic anemia": 0.0,
    "Normocytic hypochromic anemia": 0.01,
    "Normocytic normochromic anemia": 0.0,
    "Other microcytic anemia": 0.0,
    "Thrombocytopenia": 0.0
  },
  "model_used": "random_forest",
  "scenario": "scenario2",
  "model_path": "ml_model/models/task_a_scenario2_random_forest.joblib",
  "training_rows": 962,
  "features_analyzed": ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"],
  "rare_class_note": "Note: Certain rare hematologic subtypes (e.g. Macrocytic anemia, Leukemia with thrombocytopenia) have small sample representations in the dataset. Output probabilities represent uncalibrated model scores interpreted as probabilities and must not be interpreted as absolute clinical risk.",
  "clinical_disclaimer": "DISCLAIMER: This prediction is generated by an academic machine-learning model for educational and screening demonstration purposes only. It is NOT clinically validated, does NOT constitute a medical diagnosis, and must NOT be used for patient care decisions. Consult a qualified healthcare professional or laboratory physician for proper medical interpretation."
}
```

---

## 4. Difference Between Binary Screening and Multiclass Classification

* **Task B (Binary Screening)**:
  * Answers: *Does this patient present signs consistent with anemia?*
  * Target: Binary classification between Healthy (`0`) and Anemia Subtypes (`1`).
  * Non-anemia hematologic conditions (Leukemia and Thrombocytopenia) were strictly excluded during training to avoid clinical misrepresentation.
  * Uses an audited, training-tuned decision threshold (e.g. `0.370` for Scenario 2 Random Forest, `0.340` for Scenario 2 Logistic Regression) applied to uncalibrated model scores.
* **Task A (Multiclass Classification)**:
  * Answers: *Which specific hematologic subtype or anemia category best characterizes this CBC profile?*
  * Target: Differentiates across all 9 original diagnosis classes in the dataset.
  * Provides a complete vector of uncalibrated model scores interpreted as probabilities across all 9 classes, mapped to their verified class names.

---

## 5. Known Dataset Limitations

1. **Synthetic / Pre-Imputed Feature Artifacts**:
   * Approximately 61–62% of records in the dataset share identical values for `LYMp` (25.845%) and `LYMn` (1.88076 $\times 10^3/\mu\text{L}$). This artifact from prior dataset imputation artificially clusters certain differential features.
2. **Sharp Diagnostic Hemoglobin Boundary in Task B**:
   * In clean records, Healthy individuals have $\text{HGB} \ge 13.0\text{ g/dL}$ while Anemia patients have $\text{HGB} \le 12.9\text{ g/dL}$. This produces near-perfect binary tree separation that will not reflect real-world clinical overlap among borderline cases.
3. **Small Sample Representation for Rare Disorders**:
   * `Macrocytic anemia` (only 3 test samples) and `Leukemia with thrombocytopenia` (only 2 test samples) have wide uncertainty bands.
4. **Probability Calibration**:
   * The models output raw, uncalibrated model probabilities (or tree frequency scores) rather than calibrated clinical risk probabilities. They must not be interpreted as absolute clinical likelihood.
5. **Clinical Validity**:
   * This is an academic research prototype. Predictions must never be used for real-world diagnosis or patient triage without physician oversight.

---

## 6. Django REST Framework API Endpoints

The prediction service is exposed via Django REST Framework endpoints:

### Base URL: `/api/`

### 1. Health Status: `GET /api/health/`
Checks service health and returns available endpoints.
* **Request**: `GET /api/health/`
* **Response (HTTP 200)**:
  ```json
  {
    "status": "healthy",
    "service": "AI Anemia Screening Prediction API",
    "version": "1.0.0",
    "tasks": {
      "screening": "/api/screening/",
      "diagnosis": "/api/diagnosis/"
    }
  }
  ```

### 2. Binary Screening: `POST /api/screening/`
Executes Task B binary screening.
* **Request Headers**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "scenario": "scenario_2",
    "model_name": "random_forest",
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
    "PCT": 0.25
  }
  ```
* **Response (HTTP 200)**:
  ```json
  {
    "task": "Task B (Binary Anemia Screening)",
    "predicted_class": 0,
    "screening_result": "Healthy (Non-Anemic)",
    "probability_anemia": 0.0,
    "probability_healthy": 1.0,
    "applied_threshold": 0.37,
    "model_used": "random_forest",
    "scenario": "scenario2",
    "model_path": "ml_model/models/task_b_scenario2_random_forest.joblib",
    "training_rows": 864,
    "features_analyzed": ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"],
    "clinical_disclaimer": "DISCLAIMER: This prediction is generated by an academic machine-learning model for educational and screening demonstration purposes only. It is NOT clinically validated, does NOT constitute a medical diagnosis, and must NOT be used for patient care decisions. Consult a qualified healthcare professional or laboratory physician for proper medical interpretation."
  }
  ```

### 3. Multiclass Diagnosis: `POST /api/diagnosis/`
Executes Task A 9-class multiclass diagnosis classification.
* **Request Headers**: `Content-Type: application/json`
* **Request Body**: Same format as screening.
* **Response (HTTP 200)**:
  ```json
  {
    "task": "Task A (Multiclass Diagnosis Classification)",
    "predicted_diagnosis": "Healthy",
    "class_probabilities": {
      "Healthy": 0.985,
      "Iron deficiency anemia": 0.005,
      "Leukemia": 0.0,
      "Leukemia with thrombocytopenia": 0.0,
      "Macrocytic anemia": 0.0,
      "Normocytic hypochromic anemia": 0.01,
      "Normocytic normochromic anemia": 0.0,
      "Other microcytic anemia": 0.0,
      "Thrombocytopenia": 0.0
    },
    "model_used": "random_forest",
    "scenario": "scenario2",
    "model_path": "ml_model/models/task_a_scenario2_random_forest.joblib",
    "training_rows": 962,
    "features_analyzed": ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"],
    "rare_class_note": "Note: Certain rare hematologic subtypes (e.g. Macrocytic anemia, Leukemia with thrombocytopenia) have small sample representations in the dataset. Output probabilities represent uncalibrated model scores interpreted as probabilities and must not be interpreted as absolute clinical risk.",
    "clinical_disclaimer": "DISCLAIMER: This prediction is generated by an academic machine-learning model for educational and screening demonstration purposes only. It is NOT clinically validated, does NOT constitute a medical diagnosis, and must NOT be used for patient care decisions. Consult a qualified healthcare professional or laboratory physician for proper medical interpretation."
  }
  ```

### Error Responses
* **HTTP 400 Bad Request**: Invalid inputs, missing features, unexpected extra fields, target column injection, non-numeric values, or values violating physical bounds.
* **HTTP 503 Service Unavailable**: Model artifact temporarily unavailable or unreadable.
* **HTTP 500 Internal Server Error**: Generic safe message for unexpected backend errors (tracebacks are logged server-side and never returned to clients).
