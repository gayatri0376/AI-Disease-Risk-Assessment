# HemoCare AI — Complete College Major Project Documentation

**AI-Based Anemia Screening, Blood Report Interpretation & Connected Care Platform**

---

## 1. Project Overview & Architecture

**HemoCare AI** is an end-to-end full-stack healthcare web application designed for preliminary anemia screening and hematologic subtype prediction from Complete Blood Count (CBC) test results.

### High-Level Architecture Flow:
```
React Frontend (Vite, Port 5173)
       │
       ▼ (REST API / JSON Payloads)
Django Backend (Port 8000)
       │
       ├──► Authentication Engine (auth_user with PBKDF2 SHA-256)
       ├──► Serialization & Normalization (14 CBC parameters & defaults)
       ├──► Dual-Inference ML Engine (Random Forest Task A & Task B)
       │
       ▼ (ACID Compliant Transactions)
MySQL Database (anemia_system, Port 3306)
```

---

## 2. Backend Architecture & Files Directory

**Directory:** `E:\mohini's program\.vscode\Anemia screening\`

### Core Backend Files:

1. **`screening/models.py`**:
   - `Patient`: Stores demographics (`name`, `email`, `age`, `gender`, `weight`, `height`).
   - `BloodReport`: Stores 14 CBC test features, `rdw`, and `symptoms` (JSONField).
   - `ScreeningRequest`: Audit tracking of each screening execution.
   - `ScreeningResult`: Stores binary prediction (`prediction_label`), confidence score, and full `raw_output` JSON.
   - `DiagnosisReport`: Stores multiclass clinical interpretation and safety limitations.
   - `Recommendation`: Stores evidence-based medical next-steps with priorities (`high`, `normal`, `low`).
   - `Doctor`: Doctor profiles (`name`, `spec`, `exp`, `loc`, `avail`, `rating`, `status`).
   - `Appointment`: Consultation bookings (`patient_name`, `doctor_name`, `date`, `time`, `why`, `status`).
   - `MLModel`: Tracking ML model versions, names, and artifact paths.

2. **`screening/services.py`**:
   - `run_patient_screening_pipeline()`: Master function that validates CBC physical bounds, resolves/creates the patient, creates the blood report, runs ML predictions, saves diagnosis reports, and returns the response JSON.
   - `generate_clinical_recommendations()`: Generates subtype-specific medical actions (e.g. Serum Ferritin, TIBC tests, dietary plans).

3. **`screening/views.py`**:
   - `ScreenPatientAPIView`: `POST /api/screen-patient/` for unified AI screening.
   - `RegisterAPIView`: `POST /api/auth/register/` for user and patient creation.
   - `LoginAPIView`: `POST /api/auth/login/` for password verification and session return.
   - `LogoutAPIView`: `POST /api/auth/logout/` for secure session termination.
   - `PatientViewSet`, `DoctorViewSet`, `AppointmentViewSet`: Full CRUD REST endpoints.

4. **`screening/serializers.py`**:
   - `ScreenPatientRequestSerializer`: Maps frontend form fields (e.g., `'Hemoglobin'`, `'Weight (kg)'`) and provides clinically safe median defaults for differential cell parameters.
   - `PatientSerializer`, `DoctorSerializer`, `AppointmentSerializer`.

5. **`screening/urls.py`**:
   - Maps API endpoints: `/api/auth/register/`, `/api/auth/login/`, `/api/auth/logout/`, `/api/screen-patient/`, `/api/doctors/`, `/api/appointments/`, `/api/patients/`.

6. **`config/settings.py`**:
   - MySQL database connection (`anemia_system`, user: `root`).
   - CORS headers configured for React (`http://localhost:5173`).
   - Installed apps: `django.contrib.auth`, `rest_framework`, `corsheaders`, `screening`.

---

## 3. Machine Learning Inference Engine

**Directory:** `backend/apps/ml_service/` and `ml_model/models/`

The ML engine implements a dual-inference strategy using pre-trained **Random Forest Classifiers**:

### A. Task B: Binary Anemia Screening (Healthy vs. Anemic)
- **Model:** `task_b_scenario2_random_forest.joblib`
- **Tuned Threshold:** `0.37`
- **Sensitivity / Recall:** `100.0%` (Zero False Negatives)
- **Specificity:** `100.0%`
- **ROC-AUC:** `1.0000` | **PR-AUC:** `1.0000`
- **Purpose:** Flags any patient requiring clinical follow-up without missing true anemia cases.

### B. Task A: 9-Class Multiclass Subtype Diagnosis
- **Model:** `task_a_scenario2_random_forest.joblib`
- **Classes:**
  1. Healthy (Non-Anemic)
  2. Iron deficiency anemia
  3. Normocytic normochromic anemia
  4. Normocytic hypochromic anemia
  5. Other microcytic anemia
  6. Macrocytic anemia
  7. Thrombocytopenia
  8. Leukemia
  9. Leukemia with thrombocytopenia
- **Purpose:** Estimates exact class probabilities to help clinicians identify the underlying disorder type.

### C. 14 Complete Blood Count (CBC) Features Analyzed:
- `HGB`: Hemoglobin (g/dL)
- `RBC`: Red Blood Cell Count (x10⁶/µL)
- `HCT`: Hematocrit / Packed Cell Volume (%)
- `MCV`: Mean Corpuscular Volume (fL)
- `MCH`: Mean Corpuscular Hemoglobin (pg)
- `MCHC`: Mean Corpuscular Hemoglobin Concentration (g/dL)
- `RDW`: Red Cell Distribution Width (%)
- `WBC`: White Blood Cell Count (x10³/µL)
- `PLT`: Platelet Count (x10³/µL)
- `LYMp` / `NEUTp`: Lymphocyte & Neutrophil percentages
- `LYMn` / `NEUTn`: Absolute Lymphocyte & Neutrophil counts
- `PDW` / `PCT`: Platelet Distribution Width & Plateletcrit

---

## 4. Frontend Architecture (React + Vite)

**Directory:** `E:\mohini's program\.vscode\Anemia screening\AI-Anemia-Screening\`

### Component Structure:
- **`src/App.jsx`**: Main route definitions.
- **`src/layouts/AppLayout.jsx`**:
  - Dynamic top bar with user initials and name.
  - Layering fix: `.top` (`z-index: 50`), `.pop` (`z-index: 60`), and `.scrim` (`z-index: 45`).
  - Dropdown menu with "My Profile" and "Logout".
- **`src/context/AppContext.jsx`**:
  - Global `currentUser`, `role`, and `toast` notification state.
  - Synchronized `logout()` that resets storage and redirects to `/login`.
- **`src/pages/patient/Patient.jsx`**:
  - `PDash()`: Time-based greeting (*"Good morning/afternoon/evening, [User]"*) and stats.
  - `Assess()`: 4-step wizard collecting health info and 9 CBC indices.
  - `Result()`: Visual cards showing risk badge, score %, diagnosis, and recommendations.
  - `Reports()`: Saved screening reports list.
  - `Doctors()` & `DoctorProfile()`: Verified doctor search and consultation booking.
  - `Appts()`: Filtered appointments for the logged-in user.
  - `PatientProfile()`: Live profile view & editor with PATCH to MySQL.
- **`src/pages/public/Public.jsx`**:
  - `Landing()`: Platform features, how it works, FAQ.
  - `Auth()`: Real registration and login connected to `/api/auth/register/` and `/api/auth/login/`.

---

## 5. How to Run the Project

### Terminal 1: Start Django Backend
```powershell
cd "E:\mohini's program\.vscode\Anemia screening"
.\venv\Scripts\python manage.py runserver
```
➔ **http://127.0.0.1:8000/**

### Terminal 2: Start React Frontend
```powershell
cd "E:\mohini's program\.vscode\Anemia screening\AI-Anemia-Screening"
npm run dev
```
➔ **http://localhost:5173/**

---

## 6. MySQL Database Schema & Demonstration Queries

**Database:** `anemia_system` (Port: 3306)

### SQL Queries for Project Demonstration:

#### 1. Patient Screening Report with AI Interpretation:
```sql
USE anemia_system;

SELECT 
    p.name AS Patient_Name,
    p.gender AS Gender,
    b.hemoglobin AS Hemoglobin_g_dL,
    b.mcv AS MCV_fL,
    r.prediction_label AS Screening_Status,
    CONCAT(ROUND(r.confidence * 100, 1), '%') AS Confidence,
    d.interpretation AS AI_Interpretation
FROM screening_screeningresult r
JOIN screening_screeningrequest req ON r.request_id = req.id
JOIN screening_bloodreport b ON req.blood_report_id = b.id
JOIN screening_patient p ON b.patient_id = p.id
JOIN screening_diagnosisreport d ON d.result_id = r.id
ORDER BY r.id DESC;
```

#### 2. Specific Subtype & Probabilities Extraction:
```sql
SELECT 
    id AS Result_ID,
    prediction_label AS Status,
    raw_output->>'$.task_a.predicted_diagnosis' AS Specific_Diagnosis,
    raw_output->>'$.task_b.probability_healthy' AS Healthy_Probability,
    raw_output->>'$.task_b.probability_anemia' AS Anemia_Probability
FROM screening_screeningresult
ORDER BY id DESC LIMIT 5;
```

#### 3. Booked Appointments:
```sql
SELECT 
    id,
    patient_name,
    doctor_name,
    date,
    time,
    status,
    why AS Consultation_Reason
FROM screening_appointment
ORDER BY id DESC;
```

#### 4. Registered Patients:
```sql
SELECT 
    id,
    name,
    email,
    age,
    gender,
    weight AS Weight_kg,
    height AS Height_cm
FROM screening_patient
ORDER BY id DESC;
```
