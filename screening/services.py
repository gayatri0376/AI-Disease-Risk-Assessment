"""
screening/services.py — End-to-End Patient Screening Orchestration
==================================================================
Combines MySQL database records (Patient, BloodReport) with the ML
prediction service (Task B Screening + Task A Multiclass Diagnosis)
and automatically generates clinical diagnosis reports and recommendations.
"""

import logging
from django.db import transaction
from django.utils import timezone

from .models import (
    Patient,
    BloodReport,
    ScreeningRequest,
    ScreeningResult,
    DiagnosisReport,
    Recommendation,
    MLModel,
)
from backend.apps.ml_service.predictor import (
    predict_anemia_screening,
    predict_diagnosis,
    CLINICAL_DISCLAIMER,
)
from backend.apps.ml_service.validators import validate_cbc_input, ValidationError

logger = logging.getLogger(__name__)


def get_or_create_ml_model_record(model_name: str = "random_forest", scenario: str = "scenario_2") -> MLModel:
    """Ensures an entry for the ML model exists in the database for tracking and auditing."""
    normalized_name = f"{model_name.replace('_', ' ').title()} ({scenario})"
    version = scenario
    artifact = f"ml_model/models/task_b_{scenario.replace('_', '')}_{model_name}.joblib"

    ml_model_obj, _ = MLModel.objects.get_or_create(
        name=normalized_name,
        version=version,
        defaults={
            "artifact_path": artifact,
            "is_active": True,
        },
    )
    return ml_model_obj


def generate_clinical_recommendations(diagnosis_name: str, is_anemic: bool, cbc_values: dict) -> list:
    """
    Generates evidence-based clinical recommendations and next steps based on
    the ML predictions and CBC blood indices.
    """
    recommendations = []

    if not is_anemic and diagnosis_name.lower() == "healthy":
        recommendations.append({
            "title": "Routine Health Maintenance",
            "description": "All primary hemoglobin and erythrocyte parameters are within expected ranges. Continue a balanced diet rich in iron, folate, and vitamin B12.",
            "priority": "low",
        })
        recommendations.append({
            "title": "Periodic Preventive Checkup",
            "description": "Schedule a routine Complete Blood Count (CBC) annually or during routine medical checkups.",
            "priority": "low",
        })
        return recommendations

    # Specific Anemia Subtype Recommendations
    diag_lower = diagnosis_name.lower()

    if "iron deficiency" in diag_lower:
        recommendations.append({
            "title": "Serum Ferritin & Iron Studies Confirmation",
            "description": "Microcytic hypochromic features strongly suggest iron deficiency. Confirm with Serum Ferritin, Total Iron Binding Capacity (TIBC), and Serum Iron.",
            "priority": "high",
        })
        recommendations.append({
            "title": "Dietary & Supplemental Iron Plan",
            "description": "Consult a healthcare provider for appropriate oral elemental iron therapy. Enhance dietary intake with dark leafy greens, legumes, and citrus fruits (Vitamin C aids iron absorption).",
            "priority": "normal",
        })
        recommendations.append({
            "title": "Identify Etiology of Blood/Iron Loss",
            "description": "Investigate underlying sources of blood loss (e.g. gastrointestinal bleeding, heavy menstrual bleeding, or malabsorption conditions).",
            "priority": "high",
        })

    elif "macrocytic" in diag_lower:
        recommendations.append({
            "title": "Serum Vitamin B12 and Folate Evaluation",
            "description": "Elevated MCV indicates macrocytosis. Recommend measuring serum Vitamin B12, serum/RBC folate, and methylmalonic acid to distinguish megaloblastic anemia from non-megaloblastic causes.",
            "priority": "high",
        })
        recommendations.append({
            "title": "Review Medication & Hepatic Profile",
            "description": "Evaluate current medication profile (e.g., methotrexate, hydroxyurea) and assess liver function tests and alcohol consumption history.",
            "priority": "normal",
        })

    elif "normocytic" in diag_lower:
        recommendations.append({
            "title": "Reticulocyte Count & Hemolysis Workup",
            "description": "Normocytic anemia can stem from decreased production or increased destruction. Check reticulocyte count, LDH, haptoglobin, and indirect bilirubin.",
            "priority": "high",
        })
        recommendations.append({
            "title": "Renal & Inflammatory Marker Screening",
            "description": "Assess serum creatinine and eGFR (renal function), along with CRP or ESR to rule out Anemia of Chronic Disease.",
            "priority": "normal",
        })

    elif "leukemia" in diag_lower or "thrombocytopenia" in diag_lower:
        recommendations.append({
            "title": "Urgent Hematology Consultation",
            "description": "Significant alterations in leukocyte and/or platelet counts require immediate evaluation by a hematologist.",
            "priority": "urgent",
        })
        recommendations.append({
            "title": "Peripheral Blood Smear Examination",
            "description": "Perform an urgent manual peripheral blood smear review to evaluate cell morphology and blast presence.",
            "priority": "urgent",
        })

    else:
        # General Anemia fallback
        recommendations.append({
            "title": "Comprehensive Anemia Profile",
            "description": "Blood indices indicate anemia. Recommend further diagnostic testing including iron studies, Vitamin B12, folate, and peripheral blood smear review.",
            "priority": "high",
        })
        recommendations.append({
            "title": "Primary Care Physician Consultation",
            "description": "Schedule a clinical consultation with a physician for correlation with patient symptoms (fatigue, pallor, dyspnea).",
            "priority": "high",
        })

    return recommendations


@transaction.atomic
def run_patient_screening_pipeline(
    patient_data: dict,
    cbc_values: dict,
    scenario: str = "scenario_2",
    model_name: str = "random_forest",
) -> dict:
    """
    Executes the entire end-to-end pipeline:
    1. Validates CBC input parameters with physical bounds.
    2. Resolves or creates the Patient record in MySQL.
    3. Saves the 14-parameter BloodReport in MySQL.
    4. Creates a ScreeningRequest entry.
    5. Runs ML Task B (Binary Anemia Screening) and Task A (9-Class Multiclass Diagnosis).
    6. Saves ScreeningResult, DiagnosisReport, and clinical Recommendations in MySQL.
    7. Returns complete serialized JSON ready for the frontend.
    """
    # 1. Validate CBC input
    validate_cbc_input(cbc_values)

    # 2. Resolve or create Patient
    patient_id = patient_data.get("id") or patient_data.get("patient_id")
    patient = None
    if patient_id:
        try:
            patient = Patient.objects.get(pk=patient_id)
        except Patient.DoesNotExist:
            patient = None

    if not patient and patient_data.get("email"):
        patient = Patient.objects.filter(email=patient_data["email"]).first()

    if patient:
        if patient_data.get("name") and patient_data["name"] != "Anonymous Patient":
            patient.name = patient_data["name"]
        if patient_data.get("age"):
            patient.age = int(patient_data["age"])
        if patient_data.get("gender"):
            patient.gender = patient_data["gender"]
        if patient_data.get("weight"):
            patient.weight = float(patient_data["weight"])
        if patient_data.get("height"):
            patient.height = float(patient_data["height"])
        if patient_data.get("email"):
            patient.email = patient_data["email"]
        patient.save()
    else:
        patient = Patient.objects.create(
            name=patient_data.get("name", "Anonymous Patient"),
            age=int(patient_data.get("age", 25)),
            gender=patient_data.get("gender", "Female"),
            weight=patient_data.get("weight"),
            height=patient_data.get("height"),
            email=patient_data.get("email"),
        )

    # 3. Create BloodReport
    blood_report = BloodReport.objects.create(
        patient=patient,
        wbc=cbc_values.get("WBC"),
        lymp=cbc_values.get("LYMp"),
        neutp=cbc_values.get("NEUTp"),
        lymn=cbc_values.get("LYMn"),
        neutn=cbc_values.get("NEUTn"),
        rbc_count=cbc_values.get("RBC"),
        hemoglobin=cbc_values.get("HGB"),
        hct=cbc_values.get("HCT"),
        mcv=cbc_values.get("MCV"),
        mch=cbc_values.get("MCH"),
        mchc=cbc_values.get("MCHC"),
        plt=cbc_values.get("PLT"),
        pdw=cbc_values.get("PDW"),
        pct=cbc_values.get("PCT"),
        rdw=patient_data.get("rdw"),
        symptoms=patient_data.get("symptoms"),
    )

    # 4. Create ScreeningRequest
    screening_req = ScreeningRequest.objects.create(
        blood_report=blood_report,
        status="in_progress",
    )

    # 5. Execute ML Predictions
    ml_model_record = get_or_create_ml_model_record(model_name, scenario)

    # Task B: Binary Anemia Screening
    task_b_result = predict_anemia_screening(
        cbc_values=cbc_values,
        scenario=scenario,
        model_name=model_name,
    )
    is_anemic = bool(task_b_result["predicted_class"] == 1)
    screening_label = task_b_result["screening_result"]
    screening_confidence = task_b_result.get("probability_anemia") if is_anemic else task_b_result.get("probability_healthy")

    # Task A: 9-Class Multiclass Diagnosis
    task_a_result = predict_diagnosis(
        cbc_values=cbc_values,
        scenario=scenario,
        model_name=model_name,
    )
    predicted_diagnosis = task_a_result["predicted_diagnosis"]
    class_probabilities = task_a_result.get("class_probabilities", {})

    # Combine raw output
    combined_raw_output = {
        "task_b": task_b_result,
        "task_a": task_a_result,
    }

    # 6. Save ScreeningResult
    result_obj = ScreeningResult.objects.create(
        request=screening_req,
        model=ml_model_record,
        prediction_label=screening_label,
        confidence=screening_confidence,
        raw_output=combined_raw_output,
    )

    # 7. Generate & Save DiagnosisReport
    if is_anemic:
        interpretation = (
            f"Screening indicates presence of anemia ({screening_label}) with an estimated confidence of "
            f"{(screening_confidence or 0.0) * 100:.1f}%. Multiclass analysis characterizes the profile as "
            f"most consistent with '{predicted_diagnosis}' (top class probability: "
            f"{class_probabilities.get(predicted_diagnosis, 0.0) * 100:.1f}%)."
        )
    else:
        interpretation = (
            f"Screening indicates healthy hematologic profile ({screening_label}) with an estimated confidence of "
            f"{(screening_confidence or 0.0) * 100:.1f}%. Multiclass analysis confirms '{predicted_diagnosis}'."
        )

    limitations = (
        "Academic machine learning model for screening demonstration only. "
        "Not a clinical diagnosis. Laboratory review and physician oversight required."
    )

    diagnosis_report = DiagnosisReport.objects.create(
        result=result_obj,
        interpretation=interpretation,
        limitations=limitations,
    )

    # 8. Generate & Save Recommendations
    recs_data = generate_clinical_recommendations(
        diagnosis_name=predicted_diagnosis,
        is_anemic=is_anemic,
        cbc_values=cbc_values,
    )

    created_recs = []
    for r in recs_data:
        rec_obj = Recommendation.objects.create(
            diagnosis=diagnosis_report,
            title=r["title"],
            description=r["description"],
            priority=r["priority"],
        )
        created_recs.append({
            "id": rec_obj.id,
            "title": rec_obj.title,
            "description": rec_obj.description,
            "priority": rec_obj.priority,
        })

    # 9. Mark ScreeningRequest as completed
    screening_req.status = "completed"
    screening_req.save()

    # 10. Format frontend-aligned values
    frontend_status = "Needs Attention" if is_anemic else "Low Risk"
    if is_anemic:
        risk_score = min(98, max(65, round((screening_confidence or 0.85) * 100)))
    else:
        risk_score = max(5, min(35, round((task_b_result.get("probability_anemia", 0.15)) * 100)))

    cbc_summary = [
        ["Hemoglobin", f"{cbc_values.get('HGB', 0.0):.1f} g/dL"],
        ["RBC Count", f"{cbc_values.get('RBC', 0.0):.2f} M/uL"],
        ["Hematocrit", f"{cbc_values.get('HCT', 0.0):.1f} %"],
        ["MCV", f"{cbc_values.get('MCV', 0.0):.1f} fL"],
        ["MCH", f"{cbc_values.get('MCH', 0.0):.1f} pg"],
        ["MCHC", f"{cbc_values.get('MCHC', 0.0):.1f} g/dL"],
        ["RDW", f"{float(patient_data.get('rdw') or 13.0):.1f} %"],
        ["WBC Count", f"{cbc_values.get('WBC', 0.0):.1f} K/uL"],
        ["Platelets", f"{cbc_values.get('PLT', 0.0):.0f} K/uL"],
    ]

    # 11. Return unified payload (matching both React frontend and rich API)
    return {
        "status": frontend_status,
        "score": risk_score,
        "assessment_id": f"AS-{screening_req.id + 1000}",
        "date": timezone.now().strftime("%d %b %Y"),
        "predicted_diagnosis": predicted_diagnosis,
        "cbc": cbc_summary,
        "patient": {
            "id": patient.id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "weight": patient.weight,
            "height": patient.height,
            "email": patient.email,
        },
        "blood_report": {
            "id": blood_report.id,
            "created_at": blood_report.created_at.isoformat(),
            "cbc_values": cbc_values,
        },
        "screening_result": {
            "result_id": result_obj.id,
            "is_anemic": is_anemic,
            "screening_status": screening_label,
            "confidence": screening_confidence,
            "predicted_diagnosis": predicted_diagnosis,
            "class_probabilities": class_probabilities,
            "applied_threshold": task_b_result.get("applied_threshold"),
            "model_used": task_b_result.get("model_used"),
            "scenario": task_b_result.get("scenario"),
        },
        "diagnosis_report": {
            "report_id": diagnosis_report.id,
            "interpretation": diagnosis_report.interpretation,
            "limitations": diagnosis_report.limitations,
            "recommendations": created_recs,
        },
        "clinical_disclaimer": CLINICAL_DISCLAIMER,
    }
