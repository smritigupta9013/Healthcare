from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


class DataIngestion:

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:

        logger.info("Starting data ingestion")

        # Check if dataset exists
        if not self.data_path.exists():
            logger.error(
                f"Dataset not found at: {self.data_path}"
            )
            raise FileNotFoundError(
                f"Dataset not found at: {self.data_path}"
            )

        logger.info(
            f"Reading dataset from: {self.data_path}"
        )

        # Load dataset
        df = pd.read_csv(self.data_path)

        logger.info(
            f"Dataset loaded successfully. "
            f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        )

        return df


if __name__ == "__main__":

    data_path = "data/raw/diabetic_data.csv"

    ingestion = DataIngestion(data_path)

    df = ingestion.load_data()

    print("\nDataset Preview:")
    print(df.head())

    print("\nDataset Shape:")
    print(df.shape)