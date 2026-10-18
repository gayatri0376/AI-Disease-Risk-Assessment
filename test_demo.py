import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from rest_framework.test import APIClient

client = APIClient()

print("=" * 60)
print("TEST 1: HEALTHY PERSON SAMPLE (Normal Hb, Normal MCV)")
print("=" * 60)

healthy_patient = {
    "name": "Aarav Sharma",
    "age": 26,
    "gender": "Male",
    "WBC": 7.2,
    "LYMp": 32.0,
    "NEUTp": 58.0,
    "LYMn": 2.3,
    "NEUTn": 4.1,
    "RBC": 5.1,
    "HGB": 15.2,       # Normal Hemoglobin (Male: 13.8 - 17.2)
    "HCT": 45.0,       # Normal Hematocrit (40 - 50%)
    "MCV": 88.0,       # Normal MCV (80 - 100 fL)
    "MCH": 29.8,       # Normal MCH (27 - 33 pg)
    "MCHC": 33.8,      # Normal MCHC (32 - 36 g/dL)
    "PLT": 240.0,      # Normal Platelets (150 - 450)
    "PDW": 11.5,
    "PCT": 0.22,
}

r1 = client.post('/api/screen-patient/', healthy_patient, format='json')
res1 = r1.json()

print(f"Patient Name:       {res1['patient']['name']}")
print(f"Screening Result:   {res1['screening_result']['screening_status']}")
print(f"Predicted Disease:  {res1['screening_result']['predicted_diagnosis']}")
print(f"AI Confidence:      {res1['screening_result']['confidence'] * 100:.1f}%")
print(f"Recommendation:     {res1['diagnosis_report']['recommendations'][0]['title']}")
print(f"Details:            {res1['diagnosis_report']['recommendations'][0]['description']}")

print("\n" + "=" * 60)
print("TEST 2: ANEMIC PERSON SAMPLE (Low Hb: 8.8, Low MCV: 71.0)")
print("=" * 60)

anemic_patient = {
    "name": "Priya Verma",
    "age": 22,
    "gender": "Female",
    "WBC": 6.5,
    "LYMp": 26.0,
    "NEUTp": 64.0,
    "LYMn": 1.7,
    "NEUTn": 4.2,
    "RBC": 3.6,
    "HGB": 8.8,        # Very Low Hemoglobin (Normal Female: 12.1 - 15.1)
    "HCT": 28.0,       # Low Hematocrit
    "MCV": 71.0,       # Low MCV (Microcytic)
    "MCH": 23.0,       # Low MCH (Hypochromic)
    "MCHC": 30.0,
    "PLT": 340.0,
    "PDW": 15.0,
    "PCT": 0.33,
}

r2 = client.post('/api/screen-patient/', anemic_patient, format='json')
res2 = r2.json()

print(f"Patient Name:       {res2['patient']['name']}")
print(f"Screening Result:   {res2['screening_result']['screening_status']}")
print(f"Predicted Disease:  {res2['screening_result']['predicted_diagnosis']}")
print(f"AI Confidence:      {res2['screening_result']['confidence'] * 100:.1f}%")
print(f"Recommendation:     {res2['diagnosis_report']['recommendations'][0]['title']}")
print(f"Details:            {res2['diagnosis_report']['recommendations'][0]['description']}")
print("=" * 60)
