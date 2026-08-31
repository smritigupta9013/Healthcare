import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MissingValueHandler:
    """
    Handles missing values and '?' placeholder strings across
    categorical columns and clinical lab test results.
    """

    CATEGORICAL_UNKNOWN_COLS = ["race", "medical_specialty"]
    LAB_TEST_COLS = ["A1Cresult", "max_glu_serum"]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_categorical_missing(self) -> pd.DataFrame:
        """
        Replaces '?' placeholders and NaN with 'Unknown' in categorical columns.
        """
        for col in self.CATEGORICAL_UNKNOWN_COLS:
            if col in self.df.columns:
                q_count = (self.df[col] == "?").sum()
                nan_count = self.df[col].isna().sum()
                self.df[col] = self.df[col].replace("?", "Unknown").fillna("Unknown")
                logger.info(
                    f"Column '{col}': Replaced {q_count} '?' and {nan_count} NaNs with 'Unknown'."
                )
        return self.df

    def handle_lab_test_missing(self) -> pd.DataFrame:
        """
        Fills missing lab test results with 'None' (indicating test was not performed).
        """
        for col in self.LAB_TEST_COLS:
            if col in self.df.columns:
                missing_count = self.df[col].isna().sum() + (self.df[col] == "null").sum()
                self.df[col] = self.df[col].replace("null", "None").fillna("None")
                logger.info(
                    f"Lab test column '{col}': Imputed {missing_count} unmeasured records with 'None'."
                )
        return self.df

    def handle_missing(self) -> pd.DataFrame:
        """
        Executes all missing value handling steps.
        """
        logger.info("Starting Missing Value Imputation pipeline...")
        self.handle_categorical_missing()
        self.handle_lab_test_missing()
        logger.info("Missing Value Imputation completed successfully.")
        return self.df


if __name__ == "__main__":
    from src.ingestion.ingest import DataIngestion
    from src.preprocessing.clean import DataCleaner

    # Test the MissingValueHandler module
    data_path = "data/raw/diabetic_data.csv"
    ingestion = DataIngestion(data_path)
    raw_df = ingestion.load_data()

    cleaner = DataCleaner(raw_df)
    cleaned_df = cleaner.clean()

    missing_handler = MissingValueHandler(cleaned_df)
    processed_df = missing_handler.handle_missing()

    print("\nMissing Values Count after Imputation:")
    print(processed_df[MissingValueHandler.CATEGORICAL_UNKNOWN_COLS + MissingValueHandler.LAB_TEST_COLS].isnull().sum())
    print("\nProcessed DataFrame Shape:", processed_df.shape)
