"""
views.py — API Views for ML Prediction and Service Health
==========================================================
AI-Based Anemia Screening and Blood Report Interpretation System
College Project — Educational Use Only (Not Clinically Validated)

ENDPOINTS:
----------
1. POST /api/screening/
   Calls predict_anemia_screening() (Task B: Binary Screening)
2. POST /api/diagnosis/
   Calls predict_diagnosis() (Task A: 9-Class Diagnosis Classification)
3. GET /api/health/
   Returns basic service health status without leaking system paths or credentials.
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from backend.apps.ml_service.serializers import CBCPredictionRequestSerializer
from backend.apps.ml_service.predictor import (
    predict_anemia_screening,
    predict_diagnosis,
)
from backend.apps.ml_service.validators import ValidationError as MLValidationError

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """
    GET /api/health/
    Returns service operational health status.
    Does NOT leak internal filesystem paths, configurations, or credentials.
    """

    def get(self, request, *args, **kwargs):
        return Response(
            {
                "status": "healthy",
                "service": "AI Anemia Screening Prediction API",
                "version": "1.0.0",
                "tasks": {
                    "screening": "/api/screening/",
                    "diagnosis": "/api/diagnosis/",
                },
            },
            status=status.HTTP_200_OK,
        )


class AnemiaScreeningView(APIView):
    """
    POST /api/screening/
    Executes Task B binary screening on patient CBC parameters.
    """

    def post(self, request, *args, **kwargs):
        serializer = CBCPredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        cbc_values = validated_data["cbc_values"]
        scenario = validated_data["scenario"]
        model_name = validated_data["model_name"]

        try:
            result = predict_anemia_screening(
                cbc_values=cbc_values,
                scenario=scenario,
                model_name=model_name,
            )
            return Response(result, status=status.HTTP_200_OK)

        except (FileNotFoundError, KeyError) as e:
            logger.error("Model artifact or metadata unavailable: %s", str(e), exc_info=True)
            return Response(
                {
                    "error": "The requested machine learning model artifact is temporarily unavailable.",
                    "status_code": 503,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        except (ValueError, MLValidationError) as e:
            logger.warning("Invalid configuration or validation error: %s", str(e))
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.critical("Unexpected internal error during screening prediction: %s", str(e), exc_info=True)
            return Response(
                {
                    "error": "An internal server error occurred while processing the prediction request.",
                    "status_code": 500,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DiagnosisClassificationView(APIView):
    """
    POST /api/diagnosis/
    Executes Task A 9-class multiclass diagnosis classification on patient CBC parameters.
    """

    def post(self, request, *args, **kwargs):
        serializer = CBCPredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        cbc_values = validated_data["cbc_values"]
        scenario = validated_data["scenario"]
        model_name = validated_data["model_name"]

        try:
            result = predict_diagnosis(
                cbc_values=cbc_values,
                scenario=scenario,
                model_name=model_name,
            )
            return Response(result, status=status.HTTP_200_OK)

        except (FileNotFoundError, KeyError) as e:
            logger.error("Model artifact or metadata unavailable: %s", str(e), exc_info=True)
            return Response(
                {
                    "error": "The requested machine learning model artifact is temporarily unavailable.",
                    "status_code": 503,
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        except (ValueError, MLValidationError) as e:
            logger.warning("Invalid configuration or validation error: %s", str(e))
            return Response(
                {"error": str(e)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        except Exception as e:
            logger.critical("Unexpected internal error during diagnosis prediction: %s", str(e), exc_info=True)
            return Response(
                {
                    "error": "An internal server error occurred while processing the prediction request.",
                    "status_code": 500,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
