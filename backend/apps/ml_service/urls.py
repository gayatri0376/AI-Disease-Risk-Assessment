"""
urls.py — ML Service URL Configuration
======================================
"""

from django.urls import path
from backend.apps.ml_service.views import (
    HealthCheckView,
    AnemiaScreeningView,
    DiagnosisClassificationView,
)

urlpatterns = [
    path("health/", HealthCheckView.as_view(), name="api-health"),
    path("screening/", AnemiaScreeningView.as_view(), name="api-screening"),
    path("diagnosis/", DiagnosisClassificationView.as_view(), name="api-diagnosis"),
]
