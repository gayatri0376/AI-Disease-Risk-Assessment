from rest_framework import serializers
from .models import (
    Patient,
    PatientVisit,
    BloodReport,
    MLModel,
    ReportAttachment,
    ScreeningRequest,
    ScreeningResult,
    DiagnosisReport,
    Recommendation,
    FollowUp,
    Doctor,
    Appointment,
    ConsultationNote,
    Notification,
    PlatformSetting,
)


class RecommendationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recommendation
        fields = '__all__'


class DiagnosisReportSerializer(serializers.ModelSerializer):
    recommendations = RecommendationSerializer(many=True, read_only=True)

    class Meta:
        model = DiagnosisReport
        fields = '__all__'


class ScreeningResultSerializer(serializers.ModelSerializer):
    diagnosis_report = DiagnosisReportSerializer(read_only=True)

    class Meta:
        model = ScreeningResult
        fields = '__all__'


class ScreeningRequestSerializer(serializers.ModelSerializer):
    result = ScreeningResultSerializer(read_only=True)

    class Meta:
        model = ScreeningRequest
        fields = '__all__'


class ReportAttachmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReportAttachment
        fields = '__all__'


class BloodReportSerializer(serializers.ModelSerializer):
    attachments = ReportAttachmentSerializer(many=True, read_only=True)
    screening_requests = ScreeningRequestSerializer(many=True, read_only=True)

    class Meta:
        model = BloodReport
        fields = '__all__'


class PatientVisitSerializer(serializers.ModelSerializer):
    blood_reports = BloodReportSerializer(many=True, read_only=True)

    class Meta:
        model = PatientVisit
        fields = '__all__'


class PatientSerializer(serializers.ModelSerializer):
    visits = PatientVisitSerializer(many=True, read_only=True)
    blood_reports = BloodReportSerializer(many=True, read_only=True)

    class Meta:
        model = Patient
        fields = '__all__'


class MLModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = MLModel
        fields = '__all__'


class FollowUpSerializer(serializers.ModelSerializer):
    class Meta:
        model = FollowUp
        fields = '__all__'


class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = '__all__'


class AppointmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Appointment
        fields = '__all__'


class ConsultationNoteSerializer(serializers.ModelSerializer):
    class Meta:
        model = ConsultationNote
        fields = '__all__'


class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = '__all__'


class PlatformSettingSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlatformSetting
        fields = '__all__'





# --- Input Serializer for the Screening API ---

class ScreenPatientRequestSerializer(serializers.Serializer):
    """
    Serializer for the unified end-to-end patient screening endpoint.
    Accepts both standard medical abbreviations (HGB, RBC, MCV)
    and frontend form field labels ('Hemoglobin', 'RBC Count', 'Age', etc.).
    """
    # Patient Demographics
    patient_id = serializers.IntegerField(required=False, allow_null=True)
    name = serializers.CharField(max_length=100, required=False, default="Anonymous Patient")
    age = serializers.IntegerField(min_value=0, max_value=125, required=False, default=25)
    gender = serializers.CharField(max_length=20, required=False, default='Female')
    weight = serializers.FloatField(required=False, allow_null=True)
    height = serializers.FloatField(required=False, allow_null=True)
    email = serializers.EmailField(required=False, allow_null=True)
    symptoms = serializers.JSONField(required=False, allow_null=True)

    # Model Configuration
    scenario = serializers.CharField(required=False, default='scenario_2')
    model_name = serializers.CharField(required=False, default='random_forest')

    # Primary Lab Parameters (Required for accurate clinical prediction)
    hemoglobin = serializers.FloatField(required=False, help_text="Hemoglobin (HGB) in g/dL")
    rbc = serializers.FloatField(required=False, help_text="RBC Count in x10^6/uL")
    mcv = serializers.FloatField(required=False, help_text="Mean Corpuscular Volume (MCV) in fL")
    mch = serializers.FloatField(required=False, help_text="Mean Corpuscular Hemoglobin (MCH) in pg")
    mchc = serializers.FloatField(required=False, help_text="Mean Corpuscular HGB Conc (MCHC) in g/dL")
    hct = serializers.FloatField(required=False, help_text="Hematocrit (HCT) in %")
    wbc = serializers.FloatField(required=False, help_text="WBC Count in x10^3/uL")
    plt = serializers.FloatField(required=False, help_text="Platelet Count in x10^3/uL")
    rdw = serializers.FloatField(required=False, help_text="Red Cell Distribution Width (RDW) in %")

    # Additional Differentials (optional, defaulted to population median if absent)
    lymp = serializers.FloatField(required=False, default=28.0)
    neutp = serializers.FloatField(required=False, default=62.0)
    lymn = serializers.FloatField(required=False, default=2.0)
    neutn = serializers.FloatField(required=False, default=4.5)
    pdw = serializers.FloatField(required=False, default=12.5)
    pct = serializers.FloatField(required=False, default=0.25)

    def to_internal_value(self, data):
        """
        Flexibly maps frontend dictionary keys ('Hemoglobin', 'RBC Count', etc.)
        to internal serializer fields.
        """
        normalized = {}
        # Map patient info
        normalized["name"] = data.get("name") or data.get("Full Name") or "Anonymous Patient"
        normalized["age"] = int(data.get("age") or data.get("Age") or 25)
        normalized["gender"] = data.get("gender") or data.get("Gender") or "Female"
        normalized["weight"] = data.get("weight") or data.get("Weight (kg)") or None
        normalized["height"] = data.get("height") or data.get("Height (cm)") or None
        normalized["email"] = data.get("email") or data.get("Email") or None
        normalized["patient_id"] = data.get("patient_id")
        normalized["symptoms"] = data.get("symptoms")

        # Map laboratory parameters
        normalized["hemoglobin"] = float(data.get("HGB") or data.get("Hemoglobin") or data.get("hemoglobin") or 12.0)
        normalized["rbc"] = float(data.get("RBC") or data.get("RBC Count") or data.get("rbc") or 4.5)
        normalized["mcv"] = float(data.get("MCV") or data.get("mcv") or 85.0)
        normalized["mch"] = float(data.get("MCH") or data.get("mch") or 28.0)
        normalized["mchc"] = float(data.get("MCHC") or data.get("mchc") or 33.0)
        normalized["hct"] = float(data.get("HCT") or data.get("Hematocrit") or data.get("hct") or 38.0)
        normalized["wbc"] = float(data.get("WBC") or data.get("WBC Count") or data.get("wbc") or 7.0)
        normalized["plt"] = float(data.get("PLT") or data.get("Platelet Count") or data.get("plt") or 250.0)
        normalized["rdw"] = float(data.get("RDW") or data.get("rdw") or 13.0)

        # Map differential parameters
        normalized["lymp"] = float(data.get("LYMp") or data.get("lymp") or 28.0)
        normalized["neutp"] = float(data.get("NEUTp") or data.get("neutp") or 62.0)
        normalized["lymn"] = float(data.get("LYMn") or data.get("lymn") or 2.0)
        normalized["neutn"] = float(data.get("NEUTn") or data.get("neutn") or 4.5)
        normalized["pdw"] = float(data.get("PDW") or data.get("pdw") or 12.5)
        normalized["pct"] = float(data.get("PCT") or data.get("pct") or 0.25)

        normalized["scenario"] = data.get("scenario", "scenario_2")
        normalized["model_name"] = data.get("model_name", "random_forest")

        return super().to_internal_value(normalized)

    def extract_cbc_dict(self):
        """Converts internal fields into the exact 14 features expected by the ML model."""
        d = self.validated_data
        return {
            "WBC": float(d["wbc"]),
            "LYMp": float(d["lymp"]),
            "NEUTp": float(d["neutp"]),
            "LYMn": float(d["lymn"]),
            "NEUTn": float(d["neutn"]),
            "RBC": float(d["rbc"]),
            "HGB": float(d["hemoglobin"]),
            "HCT": float(d["hct"]),
            "MCV": float(d["mcv"]),
            "MCH": float(d["mch"]),
            "MCHC": float(d["mchc"]),
            "PLT": float(d["plt"]),
            "PDW": float(d["pdw"]),
            "PCT": float(d["pct"]),
        }

    def extract_patient_dict(self):
        """Extracts patient data."""
        d = self.validated_data
        return {
            "id": d.get("patient_id"),
            "name": d.get("name"),
            "age": d.get("age"),
            "gender": d.get("gender"),
            "weight": d.get("weight"),
            "height": d.get("height"),
            "email": d.get("email"),
            "symptoms": d.get("symptoms"),
            "rdw": d.get("rdw"),
        }