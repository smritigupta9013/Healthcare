import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans raw healthcare data by removing noise, data leakage,
    deceased patients, duplicates, and redundant zero-variance columns.
    """

    DECEASED_OR_HOSPICE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]
    COLUMNS_TO_DROP = ["examide", "citoglipton", "weight", "payer_code"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_deceased_and_hospice(self) -> pd.DataFrame:
        """
        Removes patients who expired or entered hospice care, as they
        cannot be readmitted (prevents target leakage).
        """
        initial_rows = len(self.df)
        self.df = self.df[
            ~self.df["discharge_disposition_id"].isin(
                self.DECEASED_OR_HOSPICE_DISCHARGE_IDS
            )
        ]
        removed = initial_rows - len(self.df)
        logger.info(
            f"Removed {removed} deceased/hospice patient records. Remaining: {len(self.df)}"
        )
        return self.df

    def remove_invalid_gender(self) -> pd.DataFrame:
        """
        Removes invalid or unknown gender records.
        """
        initial_rows = len(self.df)
        self.df = self.df[self.df["gender"] != "Unknown/Invalid"]
        removed = initial_rows - len(self.df)
        logger.info(
            f"Removed {removed} invalid gender records. Remaining: {len(self.df)}"
        )
        return self.df

    def remove_duplicate_patients(self) -> pd.DataFrame:
        """
        Keeps only the first encounter per unique patient (patient_nbr)
        to prevent train/test data leakage.
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=["patient_nbr"], keep="first")
        removed = initial_rows - len(self.df)
        logger.info(
            f"Removed {removed} duplicate encounters. Unique patients: {len(self.df)}"
        )
        return self.df

    def drop_unnecessary_columns(self) -> pd.DataFrame:
        """
        Drops zero-variance and excessive missing columns (weight, payer_code, examide, citoglipton).
        """
        cols_present = [
            col for col in self.COLUMNS_TO_DROP if col in self.df.columns
        ]
        self.df = self.df.drop(columns=cols_present)
        logger.info(
            f"Dropped {len(cols_present)} redundant columns: {cols_present}. Remaining columns: {self.df.shape[1]}"
        )
        return self.df

    def create_binary_target(self) -> pd.DataFrame:
        """
        Creates binary target 'readmitted30_target' (1 for '<30' days, 0 otherwise).
        """
        if "readmitted" in self.df.columns:
            self.df["readmitted30_target"] = (
                self.df["readmitted"] == "<30"
            ).astype(int)
            pos_rate = self.df["readmitted30_target"].mean() * 100
            logger.info(
                f"Created binary target 'readmitted30_target'. Positive class rate: {pos_rate:.2f}%"
            )
        return self.df

    def clean(self) -> pd.DataFrame:
        """
        Executes all cleaning steps sequentially.
        """
        logger.info("Starting Data Cleaning pipeline...")
        self.remove_deceased_and_hospice()
        self.remove_invalid_gender()
        self.remove_duplicate_patients()
        self.drop_unnecessary_columns()
        self.create_binary_target()
        logger.info(
            f"Data Cleaning completed successfully. Final shape: {self.df.shape}"
        )
        return self.df


if __name__ == "__main__":
    from src.ingestion.ingest import DataIngestion

    # Test the DataCleaner module
    data_path = "data/raw/diabetic_data.csv"
    ingestion = DataIngestion(data_path)
    raw_df = ingestion.load_data()

    cleaner = DataCleaner(raw_df)
    cleaned_df = cleaner.clean()

    print("\nCleaned DataFrame Preview:")
    print(cleaned_df.head())
    print("\nCleaned DataFrame Shape:", cleaned_df.shape)
