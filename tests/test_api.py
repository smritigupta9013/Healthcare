import pytest
from fastapi.testclient import TestClient
import joblib
import json
import os

from src.api.main import app
import src.api.main as api_module


@pytest.fixture(scope="module")
def client():
    """
    Initializes FastAPI TestClient with loaded model artifacts.
    """
    model_path = os.path.join("models", "xgboost_model.joblib")
    features_path = os.path.join("models", "feature_columns.json")
    metrics_path = os.path.join("artifacts", "xgboost_metrics.json")

    if os.path.exists(model_path) and os.path.exists(features_path):
        api_module.MODEL = joblib.load(model_path)
        with open(features_path, "r", encoding="utf-8") as f:
            api_module.FEATURE_COLUMNS = json.load(f)

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            api_module.METRICS = json.load(f)

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["features_count"] == 209


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "roc_auc_score" in data
    assert data["roc_auc_score"] > 0.60


def test_predict_endpoint(client):
    payload = {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 4,
        "medical_specialty": "Cardiology",
        "num_lab_procedures": 52,
        "num_procedures": 1,
        "num_medications": 16,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "diag_1": "428",
        "diag_2": "250.02",
        "diag_3": "401",
        "number_diagnoses": 8,
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "insulin": "Down",
        "metformin": "No",
        "change": "Ch",
        "diabetesMed": "Yes",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "readmission_probability" in data
    assert "readmission_prediction" in data
    assert "risk_tier" in data
    assert "clinical_recommendation" in data
    assert 0.0 <= data["readmission_probability"] <= 1.0
    assert data["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
