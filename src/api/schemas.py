from typing import Optional
from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """
    Patient Clinical Input Payload for 30-Day Readmission Risk Scoring.
    """

    race: str = Field(default="Caucasian", example="Caucasian")
    gender: str = Field(default="Female", example="Female")
    age: str = Field(default="[70-80)", example="[70-80)")
    admission_type_id: int = Field(default=1, example=1)
    discharge_disposition_id: int = Field(default=1, example=1)
    admission_source_id: int = Field(default=7, example=7)
    time_in_hospital: int = Field(default=3, ge=1, le=14, example=3)
    medical_specialty: str = Field(default="InternalMedicine", example="InternalMedicine")
    num_lab_procedures: int = Field(default=45, ge=1, example=45)
    num_procedures: int = Field(default=0, ge=0, example=0)
    num_medications: int = Field(default=12, ge=1, example=12)
    number_outpatient: int = Field(default=0, ge=0, example=0)
    number_emergency: int = Field(default=0, ge=0, example=0)
    number_inpatient: int = Field(default=1, ge=0, example=1)
    diag_1: str = Field(default="428", example="428")
    diag_2: str = Field(default="250.01", example="250.01")
    diag_3: str = Field(default="401", example="401")
    number_diagnoses: int = Field(default=7, ge=1, example=7)
    max_glu_serum: str = Field(default="None", example="None")
    A1Cresult: str = Field(default="None", example="None")
    metformin: str = Field(default="Steady", example="Steady")
    repaglinide: str = Field(default="No", example="No")
    nateglinide: str = Field(default="No", example="No")
    chlorpropamide: str = Field(default="No", example="No")
    glimepiride: str = Field(default="No", example="No")
    acetohexamide: str = Field(default="No", example="No")
    glipizide: str = Field(default="No", example="No")
    glyburide: str = Field(default="No", example="No")
    tolbutamide: str = Field(default="No", example="No")
    pioglitazone: str = Field(default="No", example="No")
    rosiglitazone: str = Field(default="No", example="No")
    acarbose: str = Field(default="No", example="No")
    miglitol: str = Field(default="No", example="No")
    troglitazone: str = Field(default="No", example="No")
    tolazamide: str = Field(default="No", example="No")
    insulin: str = Field(default="Down", example="Down")
    glyburide_metformin: str = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: str = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: str = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: str = Field(default="No", alias="metformin-pioglitazone")
    change: str = Field(default="Ch", example="Ch")
    diabetesMed: str = Field(default="Yes", example="Yes")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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
        },
    }


class PredictionResponse(BaseModel):
    """
    Model Readmission Risk Prediction Response.
    """

    readmission_probability: float = Field(..., description="Continuous risk probability between 0.0 and 1.0")
    readmission_prediction: int = Field(..., description="Binary prediction: 1 = Readmitted <30 days, 0 = Safe")
    risk_tier: str = Field(..., description="Clinical risk tier: Low Risk, Moderate Risk, or High Risk")
    clinical_recommendation: str = Field(..., description="Actionable post-discharge clinical intervention guide")
