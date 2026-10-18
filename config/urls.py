
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("screening.urls")),
     path("api/ml/", include("backend.apps.ml_service.urls")),  
]
