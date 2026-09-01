import pandas as pd
import pytest
from src.feature_engineering.create_features import FeatureEngineer, map_icd9


def test_map_icd9_categories():
    assert map_icd9("250.01") == "Diabetes"
    assert map_icd9("428") == "Circulatory"
    assert map_icd9("785") == "Circulatory"
    assert map_icd9("486") == "Respiratory"
    assert map_icd9("530") == "Digestive"
    assert map_icd9("590") == "Genitourinary"
    assert map_icd9("162") == "Neoplasms"
    assert map_icd9("715") == "Musculoskeletal"
    assert map_icd9("820") == "Injury"
    assert map_icd9("V45") == "Other"
    assert map_icd9("E876") == "Other"
    assert map_icd9("?") == "Other"
    assert map_icd9(None) == "Other"


@pytest.fixture
def sample_features_df():
    return pd.DataFrame({
        "number_outpatient": [2, 0, 1],
        "number_emergency": [1, 0, 0],
        "number_inpatient": [3, 0, 1],
        "diag_1": ["428", "250.01", "715"],
        "diag_2": ["250.02", "401", "V45"],
        "diag_3": ["486", "530", "820"],
        "metformin": ["Steady", "No", "No"],
        "insulin": ["Down", "Steady", "No"],
        "glipizide": ["No", "Steady", "No"],
        "glyburide": ["No", "No", "No"],
        "pioglitazone": ["No", "No", "No"],
        "rosiglitazone": ["No", "No", "No"],
        "acarbose": ["No", "No", "No"],
        "miglitol": ["No", "No", "No"],
        "tolbutamide": ["No", "No", "No"],
        "tolazamide": ["No", "No", "No"],
        "chlorpropamide": ["No", "No", "No"],
        "glimepiride": ["No", "No", "No"],
        "acetohexamide": ["No", "No", "No"],
        "troglitazone": ["No", "No", "No"],
        "repaglinide": ["No", "No", "No"],
        "nateglinide": ["No", "No", "No"],
        "glyburide-metformin": ["No", "No", "No"],
        "glipizide-metformin": ["No", "No", "No"],
        "glimepiride-pioglitazone": ["No", "No", "No"],
        "metformin-rosiglitazone": ["No", "No", "No"],
        "metformin-pioglitazone": ["No", "No", "No"],
    })


def test_create_healthcare_utilization(sample_features_df):
    fe = FeatureEngineer(sample_features_df)
    df = fe.create_healthcare_utilization()
    assert "total_visits" in df.columns
    # Row 0: 2+1+3 = 6
    assert df.loc[0, "total_visits"] == 6
    # Row 1: 0+0+0 = 0
    assert df.loc[1, "total_visits"] == 0
    # Row 2: 1+0+1 = 2
    assert df.loc[2, "total_visits"] == 2


def test_create_medication_count(sample_features_df):
    fe = FeatureEngineer(sample_features_df)
    df = fe.create_medication_count()
    assert "num_diabetes_meds" in df.columns
    # Row 0: metformin + insulin = 2 active
    assert df.loc[0, "num_diabetes_meds"] == 2
    # Row 1: insulin + glipizide = 2 active
    assert df.loc[1, "num_diabetes_meds"] == 2
    # Row 2: 0 active
    assert df.loc[2, "num_diabetes_meds"] == 0


def test_create_comorbidity_flags(sample_features_df):
    fe = FeatureEngineer(sample_features_df)
    fe.create_diagnosis_groups()
    df = fe.create_comorbidity_flags()
    assert "has_circulatory" in df.columns
    assert "has_diabetes_diag" in df.columns
    # Row 0: diag_1=428 (Circulatory), diag_2=250.02 (Diabetes) -> both 1
    assert df.loc[0, "has_circulatory"] == 1
    assert df.loc[0, "has_diabetes_diag"] == 1
    # Row 2: diag_1=715 (Musculoskeletal), diag_2=V45 (Other), diag_3=820 (Injury) -> both 0
    assert df.loc[2, "has_circulatory"] == 0
    assert df.loc[2, "has_diabetes_diag"] == 0
