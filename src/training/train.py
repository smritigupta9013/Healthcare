import json
import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains, validates, and serializes machine learning models for hospital readmission prediction.
    Supports XGBoost, LightGBM, Random Forest, and Logistic Regression with automated class balancing.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "xgboost",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.X = X
        self.y = y
        self.model_type = model_type.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self):
        """
        Performs stratified 80/20 train-test split.
        """
        logger.info(
            f"Splitting dataset: test_size={self.test_size}, random_state={self.random_state}, stratify=y"
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y,
        )
        logger.info(
            f"Train split shape: {self.X_train.shape}, Test split shape: {self.X_test.shape}"
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        """
        Builds and fits the selected model architecture with class balancing.
        """
        if self.X_train is None:
            self.split_data()

        # Calculate positive class imbalance weight
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        logger.info(
            f"Class balance in training set: Class 0={neg_count}, Class 1={pos_count}. Imbalance weight={scale_pos_weight:.2f}"
        )

        logger.info(f"Initializing and training model: '{self.model_type}'...")

        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss",
                n_jobs=-1,
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "logistic_regression":
            self.model = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ])
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'")

        self.model.fit(self.X_train, self.y_train)
        logger.info(f"Model '{self.model_type}' trained successfully.")
        return self.model

    def save_artifacts(self, output_dir: str = "models") -> dict:
        """
        Saves trained model artifact (.joblib) and feature column names (.json) for deployment.
        """
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"{self.model_type}_model.joblib")
        features_path = os.path.join(output_dir, "feature_columns.json")

        joblib.dump(self.model, model_path)
        logger.info(f"Saved trained model artifact to: '{model_path}'")

        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(list(self.X.columns), f, indent=4)
        logger.info(
            f"Saved {len(self.X.columns)} feature column names to: '{features_path}'"
        )

        return {"model_path": model_path, "features_path": features_path}


if __name__ == "__main__":
    from src.ingestion.ingest import DataIngestion
    from src.preprocessing.clean import DataCleaner
    from src.preprocessing.missing import MissingValueHandler
    from src.feature_engineering.create_features import FeatureEngineer
    from src.preprocessing.encoding import DataEncoder

    # End-to-end data preparation and training
    data_path = "data/raw/diabetic_data.csv"
    raw_df = DataIngestion(data_path).load_data()
    cleaned_df = DataCleaner(raw_df).clean()
    imputed_df = MissingValueHandler(cleaned_df).handle_missing()
    features_df = FeatureEngineer(imputed_df).create_features()
    X, y = DataEncoder(features_df).encode()

    trainer = ModelTrainer(X, y, model_type="xgboost")
    trainer.split_data()
    trainer.train()
    saved_paths = trainer.save_artifacts()

    print("\nTraining completed successfully!")
    print(f"Model saved to: {saved_paths['model_path']}")
    print(f"Feature columns saved to: {saved_paths['features_path']}")
