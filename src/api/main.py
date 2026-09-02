import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.api.schemas import PatientInput, PredictionResponse
from src.preprocessing.missing import MissingValueHandler
from src.feature_engineering.create_features import FeatureEngineer
from src.preprocessing.encoding import DataEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global model state
MODEL = None
FEATURE_COLUMNS = []
METRICS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads machine learning model artifacts on startup.
    """
    global MODEL, FEATURE_COLUMNS, METRICS
    model_path = os.path.join("models", "xgboost_model.joblib")
    features_path = os.path.join("models", "feature_columns.json")
    metrics_path = os.path.join("artifacts", "xgboost_metrics.json")

    logger.info("Initializing Healthcare ML Inference Service...")

    if not os.path.exists(model_path) or not os.path.exists(features_path):
        logger.warning(
            f"Artifacts not found! Expected model at '{model_path}' and features at '{features_path}'."
        )
    else:
        MODEL = joblib.load(model_path)
        with open(features_path, "r", encoding="utf-8") as f:
            FEATURE_COLUMNS = json.load(f)
        logger.info(
            f"Successfully loaded XGBoost model and {len(FEATURE_COLUMNS)} feature columns."
        )

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            METRICS = json.load(f)

    yield
    logger.info("Shutting down Healthcare ML Inference Service.")


app = FastAPI(
    title="🏥 Healthcare ML: Hospital Readmission Prediction API",
    description="Production REST API for real-time 30-day diabetic patient readmission risk scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_single_patient(patient_data: dict) -> pd.DataFrame:
    """
    Transforms a single patient record into the exact 209-feature numeric matrix required by the model.
    """
    df = pd.DataFrame([patient_data])

    # 1. Missing handling
    df = MissingValueHandler(df).handle_missing()

    # 2. Feature engineering
    df = FeatureEngineer(df).create_features()

    # 3. Encoding steps
    encoder = DataEncoder(df)
    df = encoder.encode_age()
    df = encoder.convert_id_cols_to_str()
    df = encoder.drop_raw_columns()
    df = encoder.one_hot_encode()

    # 4. Align with training feature columns (fill missing one-hot dummies with 0)
    df_aligned = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df_aligned


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Healthcare ML Readmission Prediction API",
        "status": "online",
        "model_loaded": MODEL is not None,
        "docs_url": "/docs",
    }


@app.get("/health", tags=["General"])
def health_check():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Please run the training pipeline first.",
        )
    return {
        "status": "healthy",
        "model": "XGBoost Classifier",
        "features_count": len(FEATURE_COLUMNS),
    }


@app.get("/metrics", tags=["Model Auditing"])
def get_model_metrics():
    """
    Returns latest model validation metrics and performance scores.
    """
    if not METRICS:
        return {"message": "Metrics not found. Train the model to generate artifacts."}
    return METRICS


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_readmission(patient: PatientInput):
    """
    Predicts 30-day readmission probability for an individual patient encounter.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Please run the training pipeline first.",
        )

    try:
        # Preprocess patient input
        patient_dict = patient.model_dump(by_alias=True)
        features_df = preprocess_single_patient(patient_dict)

        # Predict continuous probability
        risk_probability = float(MODEL.predict_proba(features_df)[0, 1])

        # Optimal F2 Clinical Decision Threshold (0.44 threshold achieves 71.36% recall)
        is_readmitted = int(risk_probability >= 0.44)

        # Determine clinical risk tier & guidance
        if risk_probability >= 0.44:
            risk_tier = "High Risk"
            recommendation = (
                "High readmission risk detected! Recommend scheduling a post-discharge "
                "nurse phone check within 48 hours and pharmacy medication titration review."
            )
        elif risk_probability >= 0.25:
            risk_tier = "Moderate Risk"
            recommendation = (
                "Moderate readmission risk. Recommend primary care follow-up call within "
                "7 days and medication reconciliation."
            )
        else:
            risk_tier = "Low Risk"
            recommendation = (
                "Low readmission risk. Provide standard post-discharge home recovery guide."
            )

        return PredictionResponse(
            readmission_probability=round(risk_probability, 4),
            readmission_prediction=is_readmitted,
            risk_tier=risk_tier,
            clinical_recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
