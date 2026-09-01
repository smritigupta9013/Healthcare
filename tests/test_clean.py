import pandas as pd
import pytest
from src.preprocessing.clean import DataCleaner


@pytest.fixture
def sample_raw_df():
    """
    Creates a synthetic mini-dataframe to test data cleaning edge cases.
    """
    return pd.DataFrame({
        "encounter_id": [1, 2, 3, 4, 5, 6],
        "patient_nbr": [101, 101, 102, 103, 104, 105],
        "discharge_disposition_id": [1, 1, 11, 1, 13, 1],  # 11 and 13 are expired/hospice
        "gender": ["Female", "Female", "Male", "Unknown/Invalid", "Female", "Male"],
        "weight": ["?", "?", "?", "?", "?", "?"],
        "payer_code": ["MC", "MC", "MC", "MC", "MC", "MC"],
        "examide": ["No", "No", "No", "No", "No", "No"],
        "citoglipton": ["No", "No", "No", "No", "No", "No"],
        "readmitted": ["<30", ">30", "NO", "<30", "NO", ">30"],
        "time_in_hospital": [3, 2, 5, 1, 4, 3],
    })


def test_remove_deceased_and_hospice(sample_raw_df):
    cleaner = DataCleaner(sample_raw_df)
    df = cleaner.remove_deceased_and_hospice()
    # IDs 11 and 13 should be removed
    assert 11 not in df["discharge_disposition_id"].values
    assert 13 not in df["discharge_disposition_id"].values
    assert len(df) == 4


def test_remove_invalid_gender(sample_raw_df):
    cleaner = DataCleaner(sample_raw_df)
    df = cleaner.remove_invalid_gender()
    assert "Unknown/Invalid" not in df["gender"].values
    assert len(df) == 5


def test_remove_duplicate_patients(sample_raw_df):
    cleaner = DataCleaner(sample_raw_df)
    df = cleaner.remove_duplicate_patients()
    # patient 101 appeared twice, only first encounter (id=1) should remain
    assert len(df) == 5
    assert df[df["patient_nbr"] == 101]["encounter_id"].values[0] == 1


def test_drop_unnecessary_columns(sample_raw_df):
    cleaner = DataCleaner(sample_raw_df)
    df = cleaner.drop_unnecessary_columns()
    for col in ["weight", "payer_code", "examide", "citoglipton"]:
        assert col not in df.columns


def test_create_binary_target(sample_raw_df):
    cleaner = DataCleaner(sample_raw_df)
    df = cleaner.create_binary_target()
    assert "readmitted30_target" in df.columns
    # Row 0 had '<30' -> 1, Row 1 had '>30' -> 0, Row 2 had 'NO' -> 0
    assert df.loc[0, "readmitted30_target"] == 1
    assert df.loc[1, "readmitted30_target"] == 0
    assert df.loc[2, "readmitted30_target"] == 0


def test_full_clean_pipeline(sample_raw_df):
    cleaner = DataCleaner(sample_raw_df)
    df = cleaner.clean()
    # Expected survivors: row 0 (patient 101), row 5 (patient 105)
    # row 1 dropped as duplicate of 101, row 2 dropped as expired (11),
    # row 3 dropped as invalid gender, row 4 dropped as hospice (13)
    assert len(df) == 2
    assert "readmitted30_target" in df.columns
    for col in ["weight", "payer_code", "examide", "citoglipton"]:
        assert col not in df.columns
