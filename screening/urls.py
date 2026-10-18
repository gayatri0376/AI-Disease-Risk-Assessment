from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    PatientViewSet,
    PatientVisitViewSet,
    BloodReportViewSet,
    MLModelViewSet,
    ScreeningResultViewSet,
    DiagnosisReportViewSet,
    RecommendationViewSet,
    DoctorViewSet,
    AppointmentViewSet,
    ConsultationNoteViewSet,
    NotificationViewSet,
    ScreenPatientAPIView,
    RegisterAPIView,
    LoginAPIView,
    LogoutAPIView,
    PlatformSettingAPIView,
)

router = DefaultRouter()
router.register('patients', PatientViewSet)
router.register('visits', PatientVisitViewSet)
router.register('blood-reports', BloodReportViewSet)
router.register('ml-models', MLModelViewSet)
router.register('screening-results', ScreeningResultViewSet)
router.register('diagnosis-reports', DiagnosisReportViewSet)
router.register('recommendations', RecommendationViewSet)
router.register('doctors', DoctorViewSet)
router.register('appointments', AppointmentViewSet)
router.register('consultation-notes', ConsultationNoteViewSet)
router.register('notifications', NotificationViewSet)

urlpatterns = [
    # Authentication Endpoints
    path('auth/register/', RegisterAPIView.as_view(), name='auth-register'),
    path('auth/login/', LoginAPIView.as_view(), name='auth-login'),
    path('auth/logout/', LogoutAPIView.as_view(), name='auth-logout'),

    # Platform Settings Endpoint (MySQL-backed)
    path('settings/', PlatformSettingAPIView.as_view(), name='platform-settings'),

    # Unified End-to-End Screening Endpoint (Saves to MySQL + Runs ML + Generates Report)
    path('screen-patient/', ScreenPatientAPIView.as_view(), name='screen-patient'),
    # Standard CRUD Router Endpoints
    path('', include(router.urls)),
]