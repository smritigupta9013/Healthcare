import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataEncoder:
    """
    Transforms clinical features into a numeric matrix ready for ML models:
    1. Ordinal midpoint mapping for age brackets.
    2. Conversion of numeric ID columns to categorical strings.
    3. Dropping raw IDs and unneeded columns.
    4. One-Hot Encoding for all nominal categories.
    5. Separation into feature matrix X and target vector y.
    """

    AGE_MAP = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }

    ID_CATEGORICAL_COLS = [
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]

    RAW_COLS_TO_DROP = [
        "encounter_id",
        "patient_nbr",
        "diag_1",
        "diag_2",
        "diag_3",
        "readmitted",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def encode_age(self) -> pd.DataFrame:
        """
        Maps age brackets to numerical decade midpoints (5, 15, ..., 95).
        """
        if "age" in self.df.columns:
            self.df["age"] = self.df["age"].map(self.AGE_MAP).fillna(55)
            logger.info("Mapped 'age' brackets to ordinal decade midpoints.")
        return self.df

    def convert_id_cols_to_str(self) -> pd.DataFrame:
        """
        Converts administrative ID columns to string to treat them as nominal categories.
        """
        for col in self.ID_CATEGORICAL_COLS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)
        logger.info(
            f"Converted {self.ID_CATEGORICAL_COLS} to string categorical types."
        )
        return self.df

    def drop_raw_columns(self) -> pd.DataFrame:
        """
        Drops identifier keys and raw columns replaced by engineered features.
        """
        cols_to_drop = [
            c for c in self.RAW_COLS_TO_DROP if c in self.df.columns
        ]
        self.df = self.df.drop(columns=cols_to_drop)
        logger.info(
            f"Dropped {len(cols_to_drop)} raw columns: {cols_to_drop}. Remaining: {self.df.shape[1]}"
        )
        return self.df

    def one_hot_encode(self) -> pd.DataFrame:
        """
        One-hot encodes all categorical/string columns (drop_first=True, dtype=int).
        """
        initial_cols = self.df.shape[1]
        self.df = pd.get_dummies(self.df, drop_first=True, dtype=int)
        logger.info(
            f"Applied One-Hot Encoding: columns expanded from {initial_cols} to {self.df.shape[1]} numeric features."
        )
        return self.df

    def get_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separates the encoded dataframe into X (feature matrix) and y (target vector).
        """
        if "readmitted30_target" in self.df.columns:
            X = self.df.drop(columns=["readmitted30_target"])
            y = self.df["readmitted30_target"]
        else:
            X = self.df
            y = None
        logger.info(
            f"Separated dataset: X shape = {X.shape}, y shape = {y.shape if y is not None else 'None'}"
        )
        return X, y

    def encode(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Executes the full encoding and data preparation pipeline.
        """
        logger.info("Starting Data Encoding pipeline...")
        self.encode_age()
        self.convert_id_cols_to_str()
        self.drop_raw_columns()
        self.one_hot_encode()
        X, y = self.get_features_and_target()
        logger.info("Data Encoding pipeline completed successfully.")
        return X, y


if __name__ == "__main__":
    from src.ingestion.ingest import DataIngestion
    from src.preprocessing.clean import DataCleaner
    from src.preprocessing.missing import MissingValueHandler
    from src.feature_engineering.create_features import FeatureEngineer

    # Test the entire data processing chain up to encoding
    data_path = "data/raw/diabetic_data.csv"
    raw_df = DataIngestion(data_path).load_data()
    cleaned_df = DataCleaner(raw_df).clean()
    imputed_df = MissingValueHandler(cleaned_df).handle_missing()
    features_df = FeatureEngineer(imputed_df).create_features()

    encoder = DataEncoder(features_df)
    X, y = encoder.encode()

    print("\nFeature Matrix X Shape:", X.shape)
    print("Target Vector y Shape:", y.shape)
    print("Target Positive Rate (%):", f"{y.mean() * 100:.2f}%")
    print("\nSample Columns in X:", list(X.columns[:10]))
