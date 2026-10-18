from django.contrib import admin
from .models import (Patient, PatientVisit, BloodReport, ReportAttachment,
                     ScreeningRequest, MLModel, ScreeningResult,
                     DiagnosisReport, Recommendation, FollowUp)

for m in (Patient, PatientVisit, BloodReport, ReportAttachment, ScreeningRequest,
          MLModel, ScreeningResult, DiagnosisReport, Recommendation, FollowUp):
    admin.site.register(m)