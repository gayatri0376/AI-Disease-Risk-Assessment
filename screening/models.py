from django.db import models
from django.contrib.auth.models import User


class Patient(models.Model):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
    ]

    name = models.CharField(max_length=100)
    email = models.EmailField(blank=True, null=True)
    age = models.PositiveIntegerField()
    gender = models.CharField(max_length=20, choices=GENDER_CHOICES)
    weight = models.FloatField(blank=True, null=True, help_text="Weight in kg")
    height = models.FloatField(blank=True, null=True, help_text="Height in cm")

    class Meta:
        managed = True
        db_table = 'screening_patient'

    def __str__(self):
        return f"{self.name} ({self.gender}, {self.age}y)"


class PatientVisit(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, db_column='patient_id', related_name='visits', null=True, blank=True)
    visit_date = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'screening_patientvisit'

    def __str__(self):
        return f"Visit for {self.patient.name} on {self.visit_date.strftime('%Y-%m-%d')}"


class BloodReport(models.Model):
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, db_column='patient_id', related_name='blood_reports', null=True, blank=True)
    visit = models.ForeignKey(PatientVisit, on_delete=models.SET_NULL, db_column='visit_id', blank=True, null=True, related_name='blood_reports')
    
    # --- 14 CBC Parameters for ML Screening ---
    wbc = models.FloatField(blank=True, null=True, help_text="x10^3/uL (White Blood Cell Count)")
    lymp = models.FloatField(blank=True, null=True, help_text="% (Lymphocyte %)")
    neutp = models.FloatField(blank=True, null=True, help_text="% (Neutrophil %)")
    lymn = models.FloatField(blank=True, null=True, help_text="x10^3/uL (Absolute Lymphocyte Count)")
    neutn = models.FloatField(blank=True, null=True, help_text="x10^3/uL (Absolute Neutrophil Count)")
    rbc_count = models.FloatField(blank=True, null=True, help_text="x10^6/uL (RBC Count)")
    hemoglobin = models.FloatField(help_text="g/dL (Hemoglobin / HGB)")
    hct = models.FloatField(blank=True, null=True, help_text="% (Hematocrit / PCV)")
    mcv = models.FloatField(blank=True, null=True, help_text="fL (Mean Corpuscular Volume)")
    mch = models.FloatField(blank=True, null=True, help_text="pg (Mean Corpuscular Hemoglobin)")
    mchc = models.FloatField(blank=True, null=True, help_text="g/dL (Mean Corpuscular HGB Conc)")
    plt = models.FloatField(blank=True, null=True, help_text="x10^3/uL (Platelet Count)")
    pdw = models.FloatField(blank=True, null=True, help_text="fL (Platelet Distribution Width)")
    pct = models.FloatField(blank=True, null=True, help_text="% (Plateletcrit)")
    
    # Additional fields from frontend assessment
    rdw = models.FloatField(blank=True, null=True, help_text="% (Red Cell Distribution Width)")
    symptoms = models.JSONField(blank=True, null=True, help_text="Reported symptoms")
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_bloodreport'

    def __str__(self):
        return f"Report #{self.id} for {self.patient.name} (Hb: {self.hemoglobin})"

    def to_ml_dict(self):
        """Converts blood report to the exact dictionary expected by the ML model."""
        return {
            "WBC": self.wbc,
            "LYMp": self.lymp,
            "NEUTp": self.neutp,
            "LYMn": self.lymn,
            "NEUTn": self.neutn,
            "RBC": self.rbc_count,
            "HGB": self.hemoglobin,
            "HCT": self.hct,
            "MCV": self.mcv,
            "MCH": self.mch,
            "MCHC": self.mchc,
            "PLT": self.plt,
            "PDW": self.pdw,
            "PCT": self.pct,
        }


class ReportAttachment(models.Model):
    blood_report = models.ForeignKey(BloodReport, on_delete=models.CASCADE, db_column='blood_report_id', related_name='attachments', null=True, blank=True)
    file_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    content_type = models.CharField(max_length=100, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_reportattachment'


class ScreeningRequest(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    blood_report = models.ForeignKey(BloodReport, on_delete=models.CASCADE, db_column='blood_report_id', related_name='screening_requests', null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    requested_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'screening_screeningrequest'

    def __str__(self):
        return f"Request #{self.id} - {self.status}"


class MLModel(models.Model):
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    artifact_path = models.CharField(max_length=500, blank=True, null=True)
    is_active = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_mlmodel'

    def __str__(self):
        return f"{self.name} v{self.version}"


class ScreeningResult(models.Model):
    request = models.OneToOneField(ScreeningRequest, on_delete=models.CASCADE, db_column='request_id', related_name='result', null=True, blank=True)
    model = models.ForeignKey(MLModel, on_delete=models.PROTECT, db_column='model_id', related_name='screening_results', null=True, blank=True)
    prediction_label = models.CharField(max_length=100)
    confidence = models.FloatField(blank=True, null=True)
    raw_output = models.JSONField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_screeningresult'

    def __str__(self):
        return f"Result: {self.prediction_label} ({self.confidence * 100:.1f}%)" if self.confidence else self.prediction_label


class DiagnosisReport(models.Model):
    result = models.OneToOneField(ScreeningResult, on_delete=models.CASCADE, db_column='result_id', related_name='diagnosis_report', null=True, blank=True)
    interpretation = models.TextField()
    limitations = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_diagnosisreport'


class Recommendation(models.Model):
    PRIORITY_CHOICES = [
        ('low', 'Low'),
        ('normal', 'Normal'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ]

    diagnosis = models.ForeignKey(DiagnosisReport, on_delete=models.CASCADE, db_column='diagnosis_id', related_name='recommendations', null=True, blank=True)
    title = models.CharField(max_length=150)
    description = models.TextField()
    priority = models.CharField(max_length=20, choices=PRIORITY_CHOICES, default='normal')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_recommendation'


class FollowUp(models.Model):
    STATUS_CHOICES = [
        ('planned', 'Planned'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]

    patient = models.ForeignKey(Patient, on_delete=models.CASCADE, db_column='patient_id', related_name='followups', null=True, blank=True)
    diagnosis = models.ForeignKey(DiagnosisReport, on_delete=models.SET_NULL, db_column='diagnosis_id', blank=True, null=True, related_name='followups')
    scheduled_date = models.DateField(blank=True, null=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='planned')
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_followup'


class Doctor(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=150, blank=True, null=True, help_text="Doctor login/contact email")
    spec = models.CharField(max_length=100, default='Hematologist')
    exp = models.PositiveIntegerField(default=5, help_text="Years of experience")
    loc = models.CharField(max_length=100, default='Pune')
    avail = models.CharField(max_length=150, default='Mon–Fri, 10:00 - 17:00')
    rating = models.FloatField(default=4.8)
    status = models.CharField(max_length=20, default='Verified')
    patients_count = models.PositiveIntegerField(default=0)

    class Meta:
        managed = True
        db_table = 'screening_doctor'

    def __str__(self):
        return f"{self.name} ({self.spec})"


class Appointment(models.Model):
    patient_name = models.CharField(max_length=100)
    doctor_name = models.CharField(max_length=100)
    date = models.CharField(max_length=50)
    time = models.CharField(max_length=20)
    status = models.CharField(max_length=20, default='Upcoming')
    why = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_appointment'

    def __str__(self):
        return f"{self.patient_name} with {self.doctor_name} on {self.date}"


class ConsultationNote(models.Model):
    doctor = models.ForeignKey(Doctor, on_delete=models.SET_NULL, null=True, blank=True, related_name='consultation_notes', db_constraint=False)
    patient = models.ForeignKey(Patient, on_delete=models.SET_NULL, null=True, blank=True, related_name='consultation_notes', db_constraint=False)
    doctor_name = models.CharField(max_length=150)
    patient_name = models.CharField(max_length=150)
    date = models.CharField(max_length=50, default='2026-10-16')
    notes = models.TextField(help_text="Clinical observations & diagnosis")
    prescription = models.TextField(blank=True, null=True, help_text="Medications & supplements")
    dietary_advice = models.TextField(blank=True, null=True, help_text="Dietary recommendations")
    follow_up = models.CharField(max_length=100, default='Yes, in 4 weeks')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_consultationnote'
        ordering = ['-id']

    def __str__(self):
        return f"Consultation for {self.patient_name} by {self.doctor_name} ({self.date})"


class Notification(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='notifications', db_constraint=False)
    recipient_email = models.EmailField(max_length=150, blank=True, null=True)
    role = models.CharField(max_length=20, default='all')
    title = models.CharField(max_length=150)
    message = models.TextField()
    type = models.CharField(max_length=30, default='reminder')  # 'reminder', 'screening', 'appointment', 'info'
    is_read = models.BooleanField(default=False)
    link = models.CharField(max_length=200, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed = True
        db_table = 'screening_notification'
        ordering = ['-created_at']

    def __str__(self):
        return f"[{self.role}] {self.title} - {'Read' if self.is_read else 'Unread'}"


class PlatformSetting(models.Model):
    platform_name = models.CharField(max_length=150, default='HemoCare AI')
    support_email = models.EmailField(max_length=150, default='support@hemocare.ai')
    contact_phone = models.CharField(max_length=30, default='+91 98765 43210', blank=True)
    institution = models.CharField(max_length=200, default='AI Health Research Center', blank=True)
    allow_patient_registration = models.BooleanField(default=True)
    enable_email_alerts = models.BooleanField(default=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True
        db_table = 'screening_platformsetting'

    def __str__(self):
        return f"{self.platform_name} Platform Settings"