# Complete Class-Wise Evaluation and Model Interpretation Report

**Project:** AI-Based Anemia Screening and Blood Report Interpretation System  
**Module:** Module 2 — Machine Learning and Model Evaluation  
**Date Generated:** 2026-10-12 21:07:54  
**Evaluation Protocol:** Clean Held-Out Test Partitions (Fair Split Protocol, Zero Category B in Test Set)  

---

## 1. Executive Summary

This report provides the exhaustive class-wise performance breakdown for all 12 trained machine learning pipelines:
- **3 Algorithms:** Logistic Regression, Decision Tree, Random Forest
- **2 Clinical Tasks:** Task A (9-Class Multiclass Diagnosis) and Task B (Binary Anemia Screening)
- **2 Experimental Scenarios:** Scenario 1 (Category B records retained in training) and Scenario 2 (Category B records excluded from training)
- **Evaluation Partitions:** Exactly 241 clean records for Task A and 216 clean records for Task B. Identical test partitions are evaluated across both scenarios.

---

## 2. Task A: 9-Class Multiclass Diagnosis (Class-Wise Metrics)

Test Set Size: **241 records** across 9 classes. Classes are listed in standardized alphabetical order.

### Scenario 1 (Retains 21 Cat B Rows in Train)

#### Model: Logistic Regression
- **Overall Accuracy:** `0.7178` | **Balanced Accuracy:** `0.7134` | **Macro F1:** `0.6541` | **Weighted F1:** `0.7280`

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 0.9216 | 0.7344 | 0.8174 | 64 | 47 | 4 | 17 | 173 |
| **Iron deficiency anemia** | 0.8158 | 0.8611 | 0.8378 | 36 | 31 | 7 | 5 | 198 |
| **Leukemia** | 0.7778 | 0.7778 | 0.7778 | 9 | 7 | 2 | 2 | 230 |
| **Leukemia with thrombocytopenia** | 0.6667 | 1.0000 | 0.8000 | 2 | 2 | 1 | 0 | 238 |
| **Macrocytic anemia** | 0.0000 | 0.0000 | 0.0000 | 3 | 0 | 8 | 3 | 230 |
| **Normocytic hypochromic anemia** | 0.6170 | 0.5472 | 0.5800 | 53 | 29 | 18 | 24 | 170 |
| **Normocytic normochromic anemia** | 0.7955 | 0.7000 | 0.7447 | 50 | 35 | 9 | 15 | 182 |
| **Other microcytic anemia** | 0.4211 | 0.8000 | 0.5517 | 10 | 8 | 11 | 2 | 220 |
| **Thrombocytopenia** | 0.6364 | 1.0000 | 0.7778 | 14 | 14 | 8 | 0 | 219 |

<details>
<summary>Click to view 9x9 Confusion Matrix for Logistic Regression (scenario1)</summary>

| True \ Pred | **C1** | **C2** | **C3** | **C4** | **C5** | **C6** | **C7** | **C8** | **C9** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Healthy** | 47 | 1 | 1 | 0 | 0 | 3 | 4 | 3 | 5 |
| **C2: Iron deficiency anemia** | 0 | 31 | 0 | 0 | 0 | 4 | 0 | 1 | 0 |
| **C3: Leukemia** | 1 | 0 | 7 | 0 | 1 | 0 | 0 | 0 | 0 |
| **C4: Leukemia with thrombocytopenia** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **C5: Macrocytic anemia** | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 |
| **C6: Normocytic hypochromic anemia** | 2 | 6 | 0 | 0 | 4 | 29 | 4 | 6 | 2 |
| **C7: Normocytic normochromic anemia** | 1 | 0 | 1 | 1 | 3 | 7 | 35 | 1 | 1 |
| **C8: Other microcytic anemia** | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 8 | 0 |
| **C9: Thrombocytopenia** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
</details>

#### Model: Decision Tree
- **Overall Accuracy:** `0.9917` | **Balanced Accuracy:** `0.9858` | **Macro F1:** `0.9858` | **Weighted F1:** `0.9917`

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 177 |
| **Iron deficiency anemia** | 0.9722 | 0.9722 | 0.9722 | 36 | 35 | 1 | 1 | 204 |
| **Leukemia** | 1.0000 | 1.0000 | 1.0000 | 9 | 9 | 0 | 0 | 232 |
| **Leukemia with thrombocytopenia** | 1.0000 | 1.0000 | 1.0000 | 2 | 2 | 0 | 0 | 239 |
| **Macrocytic anemia** | 1.0000 | 1.0000 | 1.0000 | 3 | 3 | 0 | 0 | 238 |
| **Normocytic hypochromic anemia** | 1.0000 | 1.0000 | 1.0000 | 53 | 53 | 0 | 0 | 188 |
| **Normocytic normochromic anemia** | 1.0000 | 1.0000 | 1.0000 | 50 | 50 | 0 | 0 | 191 |
| **Other microcytic anemia** | 0.9000 | 0.9000 | 0.9000 | 10 | 9 | 1 | 1 | 230 |
| **Thrombocytopenia** | 1.0000 | 1.0000 | 1.0000 | 14 | 14 | 0 | 0 | 227 |

<details>
<summary>Click to view 9x9 Confusion Matrix for Decision Tree (scenario1)</summary>

| True \ Pred | **C1** | **C2** | **C3** | **C4** | **C5** | **C6** | **C7** | **C8** | **C9** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Healthy** | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C2: Iron deficiency anemia** | 0 | 35 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| **C3: Leukemia** | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C4: Leukemia with thrombocytopenia** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **C5: Macrocytic anemia** | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **C6: Normocytic hypochromic anemia** | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 0 |
| **C7: Normocytic normochromic anemia** | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | 0 |
| **C8: Other microcytic anemia** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| **C9: Thrombocytopenia** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
</details>

#### Model: Random Forest
- **Overall Accuracy:** `0.9668` | **Balanced Accuracy:** `0.9389` | **Macro F1:** `0.9262` | **Weighted F1:** `0.9686`

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 177 |
| **Iron deficiency anemia** | 0.9730 | 1.0000 | 0.9863 | 36 | 36 | 1 | 0 | 204 |
| **Leukemia** | 1.0000 | 1.0000 | 1.0000 | 9 | 9 | 0 | 0 | 232 |
| **Leukemia with thrombocytopenia** | 1.0000 | 1.0000 | 1.0000 | 2 | 2 | 0 | 0 | 239 |
| **Macrocytic anemia** | 0.4000 | 0.6667 | 0.5000 | 3 | 2 | 3 | 1 | 235 |
| **Normocytic hypochromic anemia** | 0.9434 | 0.9434 | 0.9434 | 53 | 50 | 3 | 3 | 185 |
| **Normocytic normochromic anemia** | 0.9792 | 0.9400 | 0.9592 | 50 | 47 | 1 | 3 | 190 |
| **Other microcytic anemia** | 1.0000 | 0.9000 | 0.9474 | 10 | 9 | 0 | 1 | 231 |
| **Thrombocytopenia** | 1.0000 | 1.0000 | 1.0000 | 14 | 14 | 0 | 0 | 227 |

<details>
<summary>Click to view 9x9 Confusion Matrix for Random Forest (scenario1)</summary>

| True \ Pred | **C1** | **C2** | **C3** | **C4** | **C5** | **C6** | **C7** | **C8** | **C9** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Healthy** | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C2: Iron deficiency anemia** | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C3: Leukemia** | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C4: Leukemia with thrombocytopenia** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **C5: Macrocytic anemia** | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| **C6: Normocytic hypochromic anemia** | 0 | 0 | 0 | 0 | 2 | 50 | 1 | 0 | 0 |
| **C7: Normocytic normochromic anemia** | 0 | 0 | 0 | 0 | 1 | 2 | 47 | 0 | 0 |
| **C8: Other microcytic anemia** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| **C9: Thrombocytopenia** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
</details>

### Scenario 2 (Excludes Cat B Rows Entirely)

#### Model: Logistic Regression
- **Overall Accuracy:** `0.7925` | **Balanced Accuracy:** `0.7915` | **Macro F1:** `0.7217` | **Weighted F1:** `0.7994`

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 0.9444 | 0.7969 | 0.8644 | 64 | 51 | 3 | 13 | 174 |
| **Iron deficiency anemia** | 0.9167 | 0.9167 | 0.9167 | 36 | 33 | 3 | 3 | 202 |
| **Leukemia** | 0.6667 | 0.8889 | 0.7619 | 9 | 8 | 4 | 1 | 228 |
| **Leukemia with thrombocytopenia** | 0.6667 | 1.0000 | 0.8000 | 2 | 2 | 1 | 0 | 238 |
| **Macrocytic anemia** | 0.2000 | 0.3333 | 0.2500 | 3 | 1 | 4 | 2 | 234 |
| **Normocytic hypochromic anemia** | 0.7059 | 0.6792 | 0.6923 | 53 | 36 | 15 | 17 | 173 |
| **Normocytic normochromic anemia** | 0.8864 | 0.7800 | 0.8298 | 50 | 39 | 5 | 11 | 186 |
| **Other microcytic anemia** | 0.4706 | 0.8000 | 0.5926 | 10 | 8 | 9 | 2 | 222 |
| **Thrombocytopenia** | 0.6842 | 0.9286 | 0.7879 | 14 | 13 | 6 | 1 | 221 |

<details>
<summary>Click to view 9x9 Confusion Matrix for Logistic Regression (scenario2)</summary>

| True \ Pred | **C1** | **C2** | **C3** | **C4** | **C5** | **C6** | **C7** | **C8** | **C9** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Healthy** | 51 | 0 | 3 | 0 | 0 | 3 | 1 | 0 | 6 |
| **C2: Iron deficiency anemia** | 0 | 33 | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| **C3: Leukemia** | 0 | 0 | 8 | 0 | 1 | 0 | 0 | 0 | 0 |
| **C4: Leukemia with thrombocytopenia** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **C5: Macrocytic anemia** | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 0 |
| **C6: Normocytic hypochromic anemia** | 1 | 2 | 0 | 0 | 3 | 36 | 4 | 7 | 0 |
| **C7: Normocytic normochromic anemia** | 1 | 0 | 1 | 1 | 0 | 7 | 39 | 1 | 0 |
| **C8: Other microcytic anemia** | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 8 | 0 |
| **C9: Thrombocytopenia** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
</details>

#### Model: Decision Tree
- **Overall Accuracy:** `0.9751` | **Balanced Accuracy:** `0.9686` | **Macro F1:** `0.9689` | **Weighted F1:** `0.9750`

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 177 |
| **Iron deficiency anemia** | 0.9697 | 0.8889 | 0.9275 | 36 | 32 | 1 | 4 | 204 |
| **Leukemia** | 1.0000 | 1.0000 | 1.0000 | 9 | 9 | 0 | 0 | 232 |
| **Leukemia with thrombocytopenia** | 1.0000 | 1.0000 | 1.0000 | 2 | 2 | 0 | 0 | 239 |
| **Macrocytic anemia** | 1.0000 | 1.0000 | 1.0000 | 3 | 3 | 0 | 0 | 238 |
| **Normocytic hypochromic anemia** | 0.9464 | 1.0000 | 0.9725 | 53 | 53 | 3 | 0 | 185 |
| **Normocytic normochromic anemia** | 1.0000 | 1.0000 | 1.0000 | 50 | 50 | 0 | 0 | 191 |
| **Other microcytic anemia** | 0.8182 | 0.9000 | 0.8571 | 10 | 9 | 2 | 1 | 229 |
| **Thrombocytopenia** | 1.0000 | 0.9286 | 0.9630 | 14 | 13 | 0 | 1 | 227 |

<details>
<summary>Click to view 9x9 Confusion Matrix for Decision Tree (scenario2)</summary>

| True \ Pred | **C1** | **C2** | **C3** | **C4** | **C5** | **C6** | **C7** | **C8** | **C9** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Healthy** | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C2: Iron deficiency anemia** | 0 | 32 | 0 | 0 | 0 | 3 | 0 | 1 | 0 |
| **C3: Leukemia** | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C4: Leukemia with thrombocytopenia** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **C5: Macrocytic anemia** | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **C6: Normocytic hypochromic anemia** | 0 | 0 | 0 | 0 | 0 | 53 | 0 | 0 | 0 |
| **C7: Normocytic normochromic anemia** | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 0 | 0 |
| **C8: Other microcytic anemia** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| **C9: Thrombocytopenia** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 13 |
</details>

#### Model: Random Forest
- **Overall Accuracy:** `0.9502` | **Balanced Accuracy:** `0.9218` | **Macro F1:** `0.9135` | **Weighted F1:** `0.9520`

| Class Name | Precision | Recall | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy** | 0.9846 | 1.0000 | 0.9922 | 64 | 64 | 1 | 0 | 176 |
| **Iron deficiency anemia** | 0.9706 | 0.9167 | 0.9429 | 36 | 33 | 1 | 3 | 204 |
| **Leukemia** | 1.0000 | 1.0000 | 1.0000 | 9 | 9 | 0 | 0 | 232 |
| **Leukemia with thrombocytopenia** | 1.0000 | 1.0000 | 1.0000 | 2 | 2 | 0 | 0 | 239 |
| **Macrocytic anemia** | 0.4000 | 0.6667 | 0.5000 | 3 | 2 | 3 | 1 | 235 |
| **Normocytic hypochromic anemia** | 0.9074 | 0.9245 | 0.9159 | 53 | 49 | 5 | 4 | 183 |
| **Normocytic normochromic anemia** | 0.9600 | 0.9600 | 0.9600 | 50 | 48 | 2 | 2 | 189 |
| **Other microcytic anemia** | 1.0000 | 0.9000 | 0.9474 | 10 | 9 | 0 | 1 | 231 |
| **Thrombocytopenia** | 1.0000 | 0.9286 | 0.9630 | 14 | 13 | 0 | 1 | 227 |

<details>
<summary>Click to view 9x9 Confusion Matrix for Random Forest (scenario2)</summary>

| True \ Pred | **C1** | **C2** | **C3** | **C4** | **C5** | **C6** | **C7** | **C8** | **C9** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1: Healthy** | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C2: Iron deficiency anemia** | 0 | 33 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| **C3: Leukemia** | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| **C4: Leukemia with thrombocytopenia** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| **C5: Macrocytic anemia** | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 |
| **C6: Normocytic hypochromic anemia** | 0 | 0 | 0 | 0 | 2 | 49 | 2 | 0 | 0 |
| **C7: Normocytic normochromic anemia** | 0 | 0 | 0 | 0 | 1 | 1 | 48 | 0 | 0 |
| **C8: Other microcytic anemia** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| **C9: Thrombocytopenia** | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 13 |
</details>

---

## 3. Task B: Binary Anemia Screening (Class-Wise & Diagnostic Metrics)

Test Set Size: **216 records** (Healthy: 64, Anemia Subtypes: 152).
Decisions are evaluated using the optimal decision thresholds tuned via 5-fold cross-validation on training data only.

### Scenario 1

#### Model: Logistic Regression (Decision Threshold = `0.390`)
- **Accuracy:** `0.9259` | **Balanced Accuracy:** `0.9021` | **Macro F1:** `0.9095`
- **Sensitivity (Recall Class 1):** `0.9605` | **Specificity (Recall Class 0):** `0.8438`
- **ROC-AUC:** `0.9616` | **PR-AUC:** `0.9819`

| Class Name | Precision (PPV / NPV) | Recall (Sens / Spec) | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 0.9000 | 0.8438 | 0.8710 | 64 | 54 | 6 | 10 | 146 |
| **Anemia Subtypes (1)** | 0.9359 | 0.9605 | 0.9481 | 152 | 146 | 10 | 6 | 54 |

**Confusion Matrix:** TN = `54`, FP = `10`, FN = `6`, TP = `146`

#### Model: Decision Tree (Decision Threshold = `0.100`)
- **Accuracy:** `1.0000` | **Balanced Accuracy:** `1.0000` | **Macro F1:** `1.0000`
- **Sensitivity (Recall Class 1):** `1.0000` | **Specificity (Recall Class 0):** `1.0000`
- **ROC-AUC:** `1.0000` | **PR-AUC:** `1.0000`

| Class Name | Precision (PPV / NPV) | Recall (Sens / Spec) | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 152 |
| **Anemia Subtypes (1)** | 1.0000 | 1.0000 | 1.0000 | 152 | 152 | 0 | 0 | 64 |

**Confusion Matrix:** TN = `64`, FP = `0`, FN = `0`, TP = `152`

#### Model: Random Forest (Decision Threshold = `0.410`)
- **Accuracy:** `1.0000` | **Balanced Accuracy:** `1.0000` | **Macro F1:** `1.0000`
- **Sensitivity (Recall Class 1):** `1.0000` | **Specificity (Recall Class 0):** `1.0000`
- **ROC-AUC:** `1.0000` | **PR-AUC:** `1.0000`

| Class Name | Precision (PPV / NPV) | Recall (Sens / Spec) | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 152 |
| **Anemia Subtypes (1)** | 1.0000 | 1.0000 | 1.0000 | 152 | 152 | 0 | 0 | 64 |

**Confusion Matrix:** TN = `64`, FP = `0`, FN = `0`, TP = `152`

### Scenario 2

#### Model: Logistic Regression (Decision Threshold = `0.340`)
- **Accuracy:** `0.9815` | **Balanced Accuracy:** `0.9688` | **Macro F1:** `0.9774`
- **Sensitivity (Recall Class 1):** `1.0000` | **Specificity (Recall Class 0):** `0.9375`
- **ROC-AUC:** `0.9974` | **PR-AUC:** `0.9989`

| Class Name | Precision (PPV / NPV) | Recall (Sens / Spec) | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 1.0000 | 0.9375 | 0.9677 | 64 | 60 | 0 | 4 | 152 |
| **Anemia Subtypes (1)** | 0.9744 | 1.0000 | 0.9870 | 152 | 152 | 4 | 0 | 60 |

**Confusion Matrix:** TN = `60`, FP = `4`, FN = `0`, TP = `152`

#### Model: Decision Tree (Decision Threshold = `0.100`)
- **Accuracy:** `1.0000` | **Balanced Accuracy:** `1.0000` | **Macro F1:** `1.0000`
- **Sensitivity (Recall Class 1):** `1.0000` | **Specificity (Recall Class 0):** `1.0000`
- **ROC-AUC:** `1.0000` | **PR-AUC:** `1.0000`

| Class Name | Precision (PPV / NPV) | Recall (Sens / Spec) | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 152 |
| **Anemia Subtypes (1)** | 1.0000 | 1.0000 | 1.0000 | 152 | 152 | 0 | 0 | 64 |

**Confusion Matrix:** TN = `64`, FP = `0`, FN = `0`, TP = `152`

#### Model: Random Forest (Decision Threshold = `0.370`)
- **Accuracy:** `1.0000` | **Balanced Accuracy:** `1.0000` | **Macro F1:** `1.0000`
- **Sensitivity (Recall Class 1):** `1.0000` | **Specificity (Recall Class 0):** `1.0000`
- **ROC-AUC:** `1.0000` | **PR-AUC:** `1.0000`

| Class Name | Precision (PPV / NPV) | Recall (Sens / Spec) | F1-Score | Support | TP | FP | FN | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Healthy (0)** | 1.0000 | 1.0000 | 1.0000 | 64 | 64 | 0 | 0 | 152 |
| **Anemia Subtypes (1)** | 1.0000 | 1.0000 | 1.0000 | 152 | 152 | 0 | 0 | 64 |

**Confusion Matrix:** TN = `64`, FP = `0`, FN = `0`, TP = `152`

---

## 4. In-Depth Analysis: Rare-Class Support & Statistical Uncertainty

### Critical Sample Size Breakdown in Task A (Test Set $N=241$):

| Diagnosis Category | Test Support | % of Test Partition | Observation |
| :--- | :---: | :---: | :--- |
| **Healthy** | 64 | 26.56% | Sufficient support for stable estimates |
| **Normocytic hypochromic anemia** | 53 | 21.99% | Robust test representation |
| **Normocytic normochromic anemia** | 50 | 20.75% | Robust test representation |
| **Iron deficiency anemia** | 36 | 14.94% | Adequate test representation |
| **Thrombocytopenia** | 14 | 5.81% | Moderate representation |
| **Other microcytic anemia** | 10 | 4.15% | Small sample size ($N=10$) |
| **Leukemia** | 9 | 3.73% | Small sample size ($N=9$) |
| **Macrocytic anemia** | **3** | **1.24%** | **Extremely small ($N=3$)** |
| **Leukemia with thrombocytopenia** | **2** | **0.83%** | **Extremely small ($N=2$)** |

### Why Small Sample Sizes Cause Severe Metric Instability:
1. **Recall Granularity for `Leukemia with thrombocytopenia` ($N=2$):**
   - Class recall can only take values of `0.0` (0/2), `0.5` (1/2), or `1.0` (2/2).
   - A single misclassification drops sensitivity by **50.0 percentage points**.
2. **Recall Granularity for `Macrocytic anemia` ($N=3$):**
   - Class recall can only take values of `0.0` (0/3), `0.3333` (1/3), `0.6667` (2/3), or `1.0` (3/3).
   - A single misclassified patient shifts recall by **33.33 percentage points**.
   - Notice how Logistic Regression in Scenario 1 got `0.0%` recall (0/3), while in Scenario 2 it got `33.33%` (1/3), and Random Forest got `66.67%` (2/3).
3. **Precision Volatility:**
   - For `Macrocytic anemia`, Random Forest achieved precision of `0.4000` because it predicted 5 samples in total, 2 of which were true positives. Predicting 3 false positives out of 241 patients dropped precision to 40%.
4. **Clinical Recommendation:**
   - Macro F1 and Balanced Accuracy must be reported alongside raw class support counts.
   - Performance on `Macrocytic anemia` and `Leukemia with thrombocytopenia` carries wide confidence intervals and must be explicitly qualified in educational documentation.

---

## 5. In-Depth Analysis: The 100% Binary Test Results & Hemoglobin Separation

In Task B, Decision Tree and Random Forest achieved **100% Sensitivity, 100% Specificity, 1.0000 Accuracy, 1.0000 ROC-AUC, and 1.0000 PR-AUC** across both scenarios.

### The Empirical Reason for Perfect Separation:
Inspection of the test set features reveals a complete, non-overlapping threshold on **Hemoglobin (`HGB`)**:
- **Healthy patients (Class 0, $N=64$):** Minimum HGB = **13.0 g/dL**, Maximum HGB = **16.2 g/dL** (Mean = 14.00, Median = 13.90).
- **Anemia patients (Class 1, $N=152$):** Minimum HGB = **5.0 g/dL**, Maximum HGB = **12.9 g/dL** (Mean = 10.89, Median = 11.35).
- **Separation Margin:** There is a clean gap between `12.90 g/dL` and `13.00 g/dL` in this test set.

A single decision stump `HGB <= 12.95 g/dL` perfectly separates every single healthy patient from every anemia patient in this test set.

### Why This Is NOT Proof of Clinical Performance:
> [!WARNING]
> **Critical Educational Disclaimer:** Achieving 100% test accuracy on this benchmark is a direct artifact of how the underlying dataset was curated and labeled, NOT proof that machine learning can diagnose anemia with 100% real-world accuracy.

1. **Biological and Demographic Overlap:** In true clinical populations, normal hemoglobin cutoffs depend on biological sex (e.g., WHO guidelines: $\ge 13.0$ g/dL for adult men, $\ge 12.0$ g/dL for non-pregnant adult women), age, pregnancy, and altitude. A woman with HGB = 12.5 g/dL is healthy by standard female reference intervals, but would be misclassified or borderline in an unstratified population.
2. **Dehydration and Hemoconcentration:** A patient with mild anemia who is severely dehydrated may show falsely elevated hemoglobin (hemoconcentration), masking their anemia.
3. **Early-Stage Anemia:** In early iron deficiency without anemia (latent iron deficiency), hemoglobin remains normal while MCV, MCH, and serum ferritin decline.
4. **Analytical Variability:** Different clinical hematology analyzers carry small measurement variances ($\pm 0.2$ to $0.4$ g/dL), meaning borderline patients at 12.8–13.1 g/dL cannot be diagnosed with 100% certainty from a single blood draw.

---

## 6. Verification and Compliance Checklist

- [x] **Original Dataset MD5 Intact:** `c6375e974d730a42b53160614237ddf0`
- [x] **Zero Test Leakage:** Transformation pipelines fitted strictly on training data
- [x] **Identical Test Partitions:** Scenario 1 and Scenario 2 evaluated on identical clean held-out test records
- [x] **Zero Category B in Test Set:** Test partition contains 0 Category A and 0 Category B rows
- [x] **Consistent Class Ordering:** Standardized alphabetical ordering across all tables
- [x] **Artifacts Preserved:** All existing models, metadata, and reports untouched

---
*(End of Report)*