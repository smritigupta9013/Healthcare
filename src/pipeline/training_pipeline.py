import argparse
import time
from src.utils.logger import get_logger
from src.ingestion.ingest import DataIngestion
from src.validation.validate import DataValidator
from src.preprocessing.clean import DataCleaner
from src.preprocessing.missing import MissingValueHandler
from src.feature_engineering.create_features import FeatureEngineer
from src.preprocessing.encoding import DataEncoder
from src.training.train import ModelTrainer
from src.evaluation.evaluate import ModelEvaluator

logger = get_logger(__name__)


class TrainingPipeline:
    """
    End-to-End automated training pipeline orchestrator:
    Ingestion -> Validation -> Cleaning -> Missing Imputation ->
    Feature Engineering -> Encoding -> Training -> Evaluation -> Serialization.
    """

    def __init__(
        self,
        data_path: str = "data/raw/diabetic_data.csv",
        model_type: str = "xgboost",
        test_size: float = 0.2,
        random_state: int = 42,
        models_dir: str = "models",
        artifacts_dir: str = "artifacts",
    ):
        self.data_path = data_path
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.models_dir = models_dir
        self.artifacts_dir = artifacts_dir

    def run(self) -> dict:
        """
        Executes all pipeline stages sequentially and logs execution duration.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STARTING HEALTHCARE ML END-TO-END TRAINING PIPELINE")
        logger.info("=" * 60)

        # 1. Ingestion
        logger.info("[Step 1/8] Running Data Ingestion...")
        ingestion = DataIngestion(self.data_path)
        raw_df = ingestion.load_data()

        # 2. Validation
        logger.info("[Step 2/8] Running Data Validation...")
        validator = DataValidator(raw_df)
        is_valid = validator.validate_all()
        if not is_valid:
            raise ValueError("Data validation failed! Check application logs for details.")

        # 3. Cleaning
        logger.info("[Step 3/8] Running Data Cleaning...")
        cleaner = DataCleaner(raw_df)
        cleaned_df = cleaner.clean()

        # 4. Missing Imputation
        logger.info("[Step 4/8] Running Missing Value Imputation...")
        missing_handler = MissingValueHandler(cleaned_df)
        imputed_df = missing_handler.handle_missing()

        # 5. Feature Engineering
        logger.info("[Step 5/8] Running Clinical Feature Engineering...")
        feature_engineer = FeatureEngineer(imputed_df)
        features_df = feature_engineer.create_features()

        # 6. Encoding
        logger.info("[Step 6/8] Running Data Encoding and Matrix Preparation...")
        encoder = DataEncoder(features_df)
        X, y = encoder.encode()

        # 7. Training & Artifact Export
        logger.info(f"[Step 7/8] Training Model Architecture: '{self.model_type}'...")
        trainer = ModelTrainer(
            X,
            y,
            model_type=self.model_type,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        X_train, X_test, y_train, y_test = trainer.split_data()
        model = trainer.train()
        saved_artifacts = trainer.save_artifacts(output_dir=self.models_dir)

        # 8. Evaluation & Metrics Export
        logger.info("[Step 8/8] Evaluating Model on Unseen Test Cohort...")
        evaluator = ModelEvaluator(
            model,
            X_test,
            y_test,
            model_name=self.model_type,
        )
        metrics = evaluator.evaluate()
        evaluator.print_summary()
        metrics_file = evaluator.save_metrics(output_dir=self.artifacts_dir)

        elapsed_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(
            f"TRAINING PIPELINE COMPLETED IN {elapsed_time:.2f} SECONDS | ROC-AUC: {metrics['roc_auc_score']}"
        )
        logger.info("=" * 60)

        return {
            "model_type": self.model_type,
            "metrics": metrics,
            "metrics_file": metrics_file,
            "saved_artifacts": saved_artifacts,
            "elapsed_seconds": round(elapsed_time, 2),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Healthcare ML Training Pipeline")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/diabetic_data.csv",
        help="Path to raw dataset CSV",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest", "logistic_regression"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    pipeline = TrainingPipeline(
        data_path=args.data_path,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    pipeline.run()
