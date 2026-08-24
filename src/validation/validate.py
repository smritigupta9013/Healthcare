import pandas as pd

from src.utils.logger import get_logger
from src.validation.schema import REQUIRED_COLUMNS


logger = get_logger(__name__)


class DataValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate_columns(self):

        logger.info("Starting column validation")

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in self.df.columns
        ]

        if missing_columns:
            logger.error(
                f"Missing required columns: {missing_columns}"
            )
            return False

        logger.info("Column validation passed")

        return True

    def validate_duplicates(self):

        logger.info("Checking duplicate records")

        duplicate_count = self.df.duplicated().sum()

        logger.info(
            f"Duplicate records found: {duplicate_count}"
        )

        return duplicate_count

    def validate_missing_values(self):

        logger.info("Checking missing values")

        missing_values = self.df.isnull().sum()

        missing_columns = missing_values[
            missing_values > 0
        ]

        logger.info(
            f"Columns with missing values: "
            f"{len(missing_columns)}"
        )

        return missing_columns

    def validate_target(self):

        logger.info("Validating target column")

        valid_targets = ["NO", ">30", "<30"]

        invalid_targets = self.df[
            ~self.df["readmitted"].isin(valid_targets)
        ]

        if len(invalid_targets) > 0:

            logger.error(
                f"Invalid target values found: "
                f"{len(invalid_targets)}"
            )

            return False

        logger.info("Target validation passed")

        return True


if __name__ == "__main__":

    data_path = "data/raw/diabetic_data.csv"

    df = pd.read_csv(data_path)

    validator = DataValidator(df)

    print("\nColumn Validation:")
    print(validator.validate_columns())

    print("\nDuplicate Records:")
    print(validator.validate_duplicates())

    print("\nMissing Values:")
    print(validator.validate_missing_values())

    print("\nTarget Validation:")
    print(validator.validate_target())