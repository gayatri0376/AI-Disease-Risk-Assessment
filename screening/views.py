import logging
from rest_framework import viewsets, status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import action

from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login as django_login, logout as django_logout
from django.db.models import Q

from .models import (
    Patient,
    PatientVisit,
    BloodReport,
    MLModel,
    ScreeningRequest,
    ScreeningResult,
    DiagnosisReport,
    Recommendation,
    Doctor,
    Appointment,
    ConsultationNote,
    Notification,
    PlatformSetting,
)
from .serializers import (
    PatientSerializer,
    PatientVisitSerializer,
    BloodReportSerializer,
    MLModelSerializer,
    ScreeningResultSerializer,
    DiagnosisReportSerializer,
    RecommendationSerializer,
    DoctorSerializer,
    AppointmentSerializer,
    ConsultationNoteSerializer,
    NotificationSerializer,
    PlatformSettingSerializer,
    ScreenPatientRequestSerializer,
)
from .services import run_patient_screening_pipeline
from backend.apps.ml_service.validators import ValidationError as MLValidationError

logger = logging.getLogger(__name__)


class PatientViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Patients in MySQL database.
    GET /api/patients/
    POST /api/patients/
    GET /api/patients/{id}/
    """
    queryset = Patient.objects.all().order_by('-id')
    serializer_class = PatientSerializer


class PatientVisitViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Patient Visits.
    GET /api/visits/
    POST /api/visits/
    """
    queryset = PatientVisit.objects.all().order_by('-id')
    serializer_class = PatientVisitSerializer


class BloodReportViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Blood Reports.
    GET /api/blood-reports/
    POST /api/blood-reports/
    POST /api/blood-reports/{id}/screen/ (Run ML screening on an existing report)
    """
    queryset = BloodReport.objects.all().order_by('-id')
    serializer_class = BloodReportSerializer

    @action(detail=True, methods=['post'], url_path='screen')
    def screen(self, request, pk=None):
        """
        Executes ML screening on an existing BloodReport in the database.
        """
        blood_report = self.get_object()
        cbc_dict = blood_report.to_ml_dict()

        # Check for any missing values
        missing_fields = [k for k, v in cbc_dict.items() if v is None]
        if missing_fields:
            return Response(
                {
                    "error": f"BloodReport #{blood_report.id} is missing required CBC fields: {missing_fields}",
                    "status_code": 400,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        scenario = request.data.get("scenario", "scenario_2")
        model_name = request.data.get("model_name", "random_forest")

        try:
            result = run_patient_screening_pipeline(
                patient_data={"id": blood_report.patient.id},
                cbc_values=cbc_dict,
                scenario=scenario,
                model_name=model_name,
            )
            return Response(result, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error("Error screening blood report #%s: %s", blood_report.id, str(e), exc_info=True)
            return Response(
                {"error": f"Failed to execute screening: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class MLModelViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/ml-models/
    View available/active ML models recorded in the database.
    """
    queryset = MLModel.objects.all().order_by('-id')
    serializer_class = MLModelSerializer


class ScreeningResultViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/screening-results/
    View historical screening results.
    """
    queryset = ScreeningResult.objects.all().order_by('-id')
    serializer_class = ScreeningResultSerializer


class DiagnosisReportViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/diagnosis-reports/
    View clinical diagnosis reports with attached recommendations.
    """
    queryset = DiagnosisReport.objects.all().order_by('-id')
    serializer_class = DiagnosisReportSerializer


class RecommendationViewSet(viewsets.ReadOnlyModelViewSet):
    """
    GET /api/recommendations/
    View clinical recommendations.
    """
    queryset = Recommendation.objects.all().order_by('-id')
    serializer_class = RecommendationSerializer


class DoctorViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Doctors.
    GET /api/doctors/
    POST /api/doctors/
    """
    queryset = Doctor.objects.all().order_by('-id')
    serializer_class = DoctorSerializer


class AppointmentViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Appointments.
    GET /api/appointments/
    POST /api/appointments/
    """
    queryset = Appointment.objects.all().order_by('-id')
    serializer_class = AppointmentSerializer


class ConsultationNoteViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Doctor Consultation Notes in MySQL.
    GET /api/consultation-notes/
    POST /api/consultation-notes/
    """
    queryset = ConsultationNote.objects.all().order_by('-id')
    serializer_class = ConsultationNoteSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        doc_name = self.request.query_params.get('doctor_name')
        patient_name = self.request.query_params.get('patient_name')
        if doc_name:
            queryset = queryset.filter(doctor_name__icontains=doc_name)
        if patient_name:
            queryset = queryset.filter(patient_name__icontains=patient_name)
        return queryset


class NotificationViewSet(viewsets.ModelViewSet):
    """
    CRUD API for Dynamic Clinical Notifications and Reminders in MySQL.
    GET /api/notifications/?role=patient&email=user@example.com
    POST /api/notifications/
    PATCH /api/notifications/{id}/
    DELETE /api/notifications/{id}/
    POST /api/notifications/mark-all-read/
    """
    queryset = Notification.objects.all().order_by('-created_at')
    serializer_class = NotificationSerializer

    def get_queryset(self):
        queryset = super().get_queryset()
        role = self.request.query_params.get('role')
        email = self.request.query_params.get('email')

        # Auto-seed initial notifications if none exist for this role
        if role:
            role_clean = role.lower()
            role_q = Q(role=role_clean) | Q(role='all')
            if email:
                role_q = role_q | Q(recipient_email=email)
            filtered = queryset.filter(role_q)
            if not filtered.exists():
                self._seed_default_notifications(role_clean, email)
                filtered = Notification.objects.filter(role_q).order_by('-created_at')
            return filtered

        return queryset

    def _seed_default_notifications(self, role, email=None):
        """Creates initial clinical reminders and alerts tailored to user role"""
        seed_data = []
        if role == 'patient':
            seed_data = [
                {
                    "title": "4-Week CBC Re-Test Reminder",
                    "message": "It has been 3 weeks since your last Hemoglobin screening. Schedule your follow-up CBC test to track red blood cell indices.",
                    "type": "reminder",
                    "link": "/patient/history",
                    "role": "patient",
                },
                {
                    "title": "Iron & Vitamin C Absorption Tip",
                    "message": "Remember to take prescribed Iron supplements with vitamin C (citrus/amla) on an empty stomach for 40% higher absorption.",
                    "type": "info",
                    "link": "/patient/diet",
                    "role": "patient",
                },
                {
                    "title": "Doctor Consultation Available",
                    "message": "Dr. Aarav Mehta is available for video consultation this week. Book your follow-up slot.",
                    "type": "appointment",
                    "link": "/patient/consult",
                    "role": "patient",
                },
            ]
        elif role == 'doctor':
            seed_data = [
                {
                    "title": "Pending Consultation Notes",
                    "message": "You have 2 completed patient screenings awaiting clinical diagnosis and prescription notes.",
                    "type": "reminder",
                    "link": "/doctor/consultation",
                    "role": "doctor",
                },
                {
                    "title": "Weekly Availability Check",
                    "message": "Ensure your weekly consultation hours and time slots are up to date for patient booking.",
                    "type": "info",
                    "link": "/doctor/availability",
                    "role": "doctor",
                },
                {
                    "title": "AI Screening Flag",
                    "message": "Recent screening for Sunita Sharma showed microcytic hypochromic pattern (HGB: 8.4 g/dL). Review recommended.",
                    "type": "screening",
                    "link": "/doctor/patients",
                    "role": "doctor",
                },
            ]
        elif role == 'admin':
            seed_data = [
                {
                    "title": "Doctor Credential Verification",
                    "message": "New doctor registered: Dr. Mohini Jadhav. Medical license and credentials pending review.",
                    "type": "reminder",
                    "link": "/admin/doctors",
                    "role": "admin",
                },
                {
                    "title": "AI Model Retraining Milestone",
                    "message": "Random Forest pipeline has achieved 98.4% diagnostic accuracy across 150+ screened patient records.",
                    "type": "info",
                    "link": "/admin/models",
                    "role": "admin",
                },
                {
                    "title": "Health Education Article Draft",
                    "message": "New clinical nutrition guide ready for publication review in the Knowledge Center.",
                    "type": "info",
                    "link": "/admin/education",
                    "role": "admin",
                },
            ]
        else:
            seed_data = [
                {
                    "title": "Welcome to HemoCare AI",
                    "message": "Your intelligent AI-assisted anemia diagnostic and management platform.",
                    "type": "info",
                    "link": "/",
                    "role": "all",
                }
            ]

        for item in seed_data:
            Notification.objects.create(
                role=item.get("role", role),
                title=item["title"],
                message=item["message"],
                type=item.get("type", "reminder"),
                link=item.get("link"),
                recipient_email=email if email else None,
                is_read=False
            )

    @action(detail=False, methods=['post'], url_path='mark-all-read')
    def mark_all_read(self, request):
        role = request.data.get('role')
        email = request.data.get('email')
        qs = Notification.objects.filter(is_read=False)
        if role:
            qs = qs.filter(Q(role=role.lower()) | Q(role='all'))
        if email:
            qs = qs.filter(Q(recipient_email=email) | Q(recipient_email__isnull=True))
        count = qs.update(is_read=True)
        return Response({"status": "success", "marked_read": count})



class ScreenPatientAPIView(generics.GenericAPIView):
    serializer_class = ScreenPatientRequestSerializer
    """
    POST /api/screen-patient/
    =========================
    Unified End-to-End Patient Screening API.

    Performs the complete workflow:
    1. Validates all 14 Complete Blood Count (CBC) test features.
    2. Creates or links the Patient in MySQL.
    3. Saves the Complete Blood Report in MySQL.
    4. Runs ML Binary Screening (Task B: Healthy vs Anemic).
    5. Runs ML Multiclass Diagnosis (Task A: 9-class specific subtype).
    6. Saves ScreeningResult & DiagnosisReport in MySQL.
    7. Generates clinical Recommendations based on findings.
    8. Returns complete structured JSON for the frontend.
    """

    def post(self, request, *args, **kwargs):
        serializer = ScreenPatientRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        patient_data = serializer.extract_patient_dict()
        cbc_values = serializer.extract_cbc_dict()
        scenario = serializer.validated_data.get("scenario", "scenario_2")
        model_name = serializer.validated_data.get("model_name", "random_forest")

        try:
            response_data = run_patient_screening_pipeline(
                patient_data=patient_data,
                cbc_values=cbc_values,
                scenario=scenario,
                model_name=model_name,
            )
            return Response(response_data, status=status.HTTP_201_CREATED)

        except Patient.DoesNotExist:
            return Response(
                {"error": f"Patient with ID {patient_data.get('id')} does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except (MLValidationError, ValueError) as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            logger.critical("Failed to process screening pipeline: %s", str(e), exc_info=True)
            return Response(
                {"error": "An internal server error occurred while processing the patient screening."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class RegisterAPIView(APIView):
    """
    POST /api/auth/register/
    Registers a new user (patient, doctor, admin) in Django auth_user and corresponding MySQL table.
    """
    def post(self, request, *args, **kwargs):
        data = request.data
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        name = data.get("name") or data.get("Full Name") or "Demo User"
        role = data.get("role", "patient").lower()
        age = int(data.get("age", 25))
        gender = data.get("gender", "Female")

        if not email or not password:
            return Response({"error": "Email and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Check if user already exists
        user = User.objects.filter(username=email).first() or User.objects.filter(email=email).first()
        if user:
            user.set_password(password)
            user.first_name = name
            user.save()
        else:
            user = User.objects.create_user(username=email, email=email, password=password, first_name=name)

        patient_id = None
        doctor_id = None

        if role == "patient":
            patient = Patient.objects.filter(email=email).first()
            if not patient:
                patient = Patient.objects.create(name=name, email=email, age=age, gender=gender)
            else:
                patient.name = name
                patient.age = age
                patient.gender = gender
                patient.save()
            patient_id = patient.id

        elif role == "doctor":
            spec = data.get("spec", "Hematologist")
            exp = int(data.get("exp", 5))
            loc = data.get("loc", "Pune")
            doc = Doctor.objects.filter(email=email).first() or Doctor.objects.filter(name=name).first()
            if not doc:
                doc = Doctor.objects.create(name=name, email=email, spec=spec, exp=exp, loc=loc, avail="Today", status="Verified")
            else:
                doc.name = name
                doc.email = email
                doc.spec = spec
                doc.exp = exp
                doc.loc = loc
                doc.save()
            doctor_id = doc.id

        return Response({
            "status": "success",
            "message": "User registered successfully.",
            "user": {
                "id": user.id,
                "name": name,
                "email": email,
                "role": role,
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "spec": doc.spec if role == "doctor" and doc else "Hematologist",
                "exp": doc.exp if role == "doctor" and doc else 5,
                "loc": doc.loc if role == "doctor" and doc else "Pune",
            }
        }, status=status.HTTP_201_CREATED)


class LoginAPIView(APIView):
    """
    POST /api/auth/login/
    Authenticates a user against Django auth_user.
    Supports login via Email, Username, or Doctor Name.
    """
    def post(self, request, *args, **kwargs):
        data = request.data
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        role = data.get("role", "patient").lower()

        if not email or not password:
            return Response({"error": "Email and password are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Authenticate with Django auth (support email, username, or doctor full name)
        user = User.objects.filter(username=email).first() or User.objects.filter(email=email).first()
        if not user and role == "doctor":
            doc_record = Doctor.objects.filter(email__iexact=email).first() or Doctor.objects.filter(name__iexact=email).first()
            if doc_record and doc_record.email:
                user = User.objects.filter(email=doc_record.email).first() or User.objects.filter(username=doc_record.email).first()
        
        # If user doesn't exist yet but it's demo credentials, auto-provision
        if not user and ("demo" in email or "demo1234" in password):
            name = "Demo User"
            if role == "doctor":
                name = "Dr. Aarav Mehta"
            elif role == "admin":
                name = "Admin User"
            user = User.objects.create_user(username=email, email=email, password=password, first_name=name)

        if not user or not user.check_password(password):
            return Response({"error": "Invalid email or password. Please check your credentials."}, status=status.HTTP_401_UNAUTHORIZED)

        patient_id = None
        doctor_id = None
        if role == "patient":
            patient = Patient.objects.filter(email=email).first()
            if not patient:
                patient = Patient.objects.create(name=user.first_name or "Demo Patient", email=email, age=25, gender="Female")
            patient_id = patient.id
        elif role == "doctor":
            doc = Doctor.objects.filter(email=email).first() or Doctor.objects.filter(name=user.first_name).first()
            if not doc:
                doc = Doctor.objects.filter(email__iexact=user.email).first() or Doctor.objects.first()
            doctor_id = doc.id if doc else None
            spec = doc.spec if doc else "Hematologist"
            exp = doc.exp if doc else 5
            loc = doc.loc if doc else "Pune"

        django_login(request, user)

        return Response({
            "status": "success",
            "message": "Login successful.",
            "user": {
                "id": user.id,
                "name": user.first_name or user.username,
                "email": user.email,
                "role": role,
                "patient_id": patient_id,
                "doctor_id": doctor_id,
                "spec": doc.spec if role == "doctor" and doc else "Hematologist",
                "exp": doc.exp if role == "doctor" and doc else 5,
                "loc": doc.loc if role == "doctor" and doc else "Pune",
            }
        }, status=status.HTTP_200_OK)


class LogoutAPIView(APIView):
    """
    POST /api/auth/logout/
    Logs out the authenticated user.
    """
    def post(self, request, *args, **kwargs):
        django_logout(request)
        return Response({
            "status": "success",
            "message": "Logged out successfully."
        }, status=status.HTTP_200_OK)


class PlatformSettingAPIView(APIView):
    """
    GET /api/settings/ - Fetch active platform settings from MySQL
    POST /api/settings/ - Update platform settings in MySQL
    """
    def get(self, request):
        setting = PlatformSetting.objects.first()
        if not setting:
            setting = PlatformSetting.objects.create(
                platform_name='HemoCare AI',
                support_email='support@hemocare.ai',
                contact_phone='+91 98765 43210',
                institution='AI Health Research Center',
                allow_patient_registration=True,
                enable_email_alerts=True,
            )
        serializer = PlatformSettingSerializer(setting)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        setting = PlatformSetting.objects.first()
        if not setting:
            setting = PlatformSetting.objects.create()
        serializer = PlatformSettingSerializer(setting, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({
                "status": "success",
                "message": "Platform settings successfully updated in MySQL database.",
                "data": serializer.data
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)