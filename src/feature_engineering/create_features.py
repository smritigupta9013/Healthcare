import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def map_icd9(val) -> str:
    """
    Maps raw alphanumeric ICD-9 diagnosis codes into 9 high-level
    clinical disease categories.
    """
    if pd.isna(val) or val == "?" or val == "Unknown":
        return "Other"

    val_str = str(val).strip()

    # Supplementary / External cause codes
    if val_str.startswith(("V", "E")):
        return "Other"

    # Diabetes codes
    if val_str.startswith("250"):
        return "Diabetes"

    try:
        code = float(val_str)
    except ValueError:
        return "Other"

    # Standard clinical ICD-9 body system ranges
    if (390 <= code <= 459) or code == 785:
        return "Circulatory"
    elif (460 <= code <= 519) or code == 786:
        return "Respiratory"
    elif (520 <= code <= 579) or code == 787:
        return "Digestive"
    elif (580 <= code <= 629) or code == 788:
        return "Genitourinary"
    elif 140 <= code <= 239:
        return "Neoplasms"
    elif 710 <= code <= 739:
        return "Musculoskeletal"
    elif 800 <= code <= 999:
        return "Injury"
    else:
        return "Other"


class FeatureEngineer:
    """
    Engineers high-impact clinical risk features:
    1. ICD-9 disease body system grouping (diag_1, diag_2, diag_3).
    2. Healthcare service utilization (total_visits).
    3. Active diabetes medication burden (num_diabetes_meds).
    4. Comorbidity risk flags (has_circulatory, has_diabetes_diag).
    """

    MEDICATION_COLS = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_diagnosis_groups(self) -> pd.DataFrame:
        """
        Groups raw 700+ ICD-9 codes into 9 clinical categories.
        """
        for col in ["diag_1", "diag_2", "diag_3"]:
            if col in self.df.columns:
                self.df[f"{col}_group"] = self.df[col].apply(map_icd9)
                logger.info(f"Grouped '{col}' into '{col}_group' (9 clinical categories).")
        return self.df

    def create_healthcare_utilization(self) -> pd.DataFrame:
        """
        Calculates total prior healthcare utilization: outpatient + emergency + inpatient.
        """
        util_cols = ["number_outpatient", "number_emergency", "number_inpatient"]
        present_cols = [c for c in util_cols if c in self.df.columns]
        if present_cols:
            self.df["total_visits"] = self.df[present_cols].sum(axis=1)
            logger.info("Created 'total_visits' (total prior healthcare utilization).")
        return self.df

    def create_medication_count(self) -> pd.DataFrame:
        """
        Counts how many active diabetes medications (value != 'No') the patient is prescribed.
        """
        present_meds = [col for col in self.MEDICATION_COLS if col in self.df.columns]
        if present_meds:
            self.df["num_diabetes_meds"] = (self.df[present_meds] != "No").sum(axis=1)
            logger.info(
                f"Created 'num_diabetes_meds' counting active medications across {len(present_meds)} drugs."
            )
        return self.df

    def create_comorbidity_flags(self) -> pd.DataFrame:
        """
        Creates binary comorbidity flags for circulatory disease and active diabetes complications.
        """
        group_cols = [c for c in ["diag_1_group", "diag_2_group", "diag_3_group"] if c in self.df.columns]
        if group_cols:
            self.df["has_circulatory"] = (
                (self.df[group_cols] == "Circulatory").any(axis=1)
            ).astype(int)
            self.df["has_diabetes_diag"] = (
                (self.df[group_cols] == "Diabetes").any(axis=1)
            ).astype(int)
            logger.info(
                f"Created comorbidity flags: 'has_circulatory' ({self.df['has_circulatory'].sum()} positive), "
                f"'has_diabetes_diag' ({self.df['has_diabetes_diag'].sum()} positive)."
            )
        return self.df

    def create_features(self) -> pd.DataFrame:
        """
        Executes all feature engineering steps sequentially.
        """
        logger.info("Starting Feature Engineering pipeline...")
        self.create_diagnosis_groups()
        self.create_healthcare_utilization()
        self.create_medication_count()
        self.create_comorbidity_flags()
        logger.info(
            f"Feature Engineering completed successfully. Final shape: {self.df.shape}"
        )
        return self.df


if __name__ == "__main__":
    from src.ingestion.ingest import DataIngestion
    from src.preprocessing.clean import DataCleaner
    from src.preprocessing.missing import MissingValueHandler

    # Test the entire pipeline up to Feature Engineering
    data_path = "data/raw/diabetic_data.csv"
    raw_df = DataIngestion(data_path).load_data()
    cleaned_df = DataCleaner(raw_df).clean()
    imputed_df = MissingValueHandler(cleaned_df).handle_missing()

    fe = FeatureEngineer(imputed_df)
    features_df = fe.create_features()

    print("\nEngineered Features Preview:")
    print(features_df[["total_visits", "num_diabetes_meds", "has_circulatory", "has_diabetes_diag", "diag_1_group"]].head())
    print("\nFeatures DataFrame Shape:", features_df.shape)
