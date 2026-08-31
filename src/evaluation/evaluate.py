import json
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates trained healthcare classification models using domain-relevant
    metrics (ROC-AUC, Precision, Recall, F1-score, and Confusion Matrix).
    """

    def __init__(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "xgboost",
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.metrics = {}

    def evaluate(self) -> dict:
        """
        Computes all evaluation metrics on unseen test data.
        """
        logger.info(f"Evaluating model '{self.model_name}' on {len(self.X_test)} test patients...")

        # 1. Predict probabilities and binary class labels
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = self.model.predict(self.X_test)

        # 2. Compute metrics
        roc_auc = float(roc_auc_score(self.y_test, y_pred_proba))
        report_dict = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "model_name": self.model_name,
            "test_patients_count": int(len(self.X_test)),
            "roc_auc_score": round(roc_auc, 4),
            "class_1_readmitted": {
                "recall": round(report_dict["1"]["recall"], 4),
                "precision": round(report_dict["1"]["precision"], 4),
                "f1_score": round(report_dict["1"]["f1-score"], 4),
                "support": int(report_dict["1"]["support"]),
            },
            "class_0_not_readmitted": {
                "recall": round(report_dict["0"]["recall"], 4),
                "precision": round(report_dict["0"]["precision"], 4),
                "f1_score": round(report_dict["0"]["f1-score"], 4),
                "support": int(report_dict["0"]["support"]),
            },
            "overall_accuracy": round(report_dict["accuracy"], 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

        logger.info(f"Evaluation complete. ROC-AUC: {roc_auc:.4f}, Recall (Class 1): {self.metrics['class_1_readmitted']['recall']:.2%}")
        return self.metrics

    def save_metrics(self, output_dir: str = "artifacts") -> str:
        """
        Exports evaluation metrics to JSON for auditing and tracking.
        """
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{self.model_name}_metrics.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)
        logger.info(f"Saved evaluation metrics to: '{file_path}'")
        return file_path

    def print_summary(self):
        """
        Prints a clean formatted clinical evaluation summary to console.
        """
        if not self.metrics:
            self.evaluate()

        c1 = self.metrics["class_1_readmitted"]
        cm = self.metrics["confusion_matrix"]

        print(f"\n{'='*20} {self.model_name.upper()} EVALUATION SUMMARY {'='*20}")
        print(f"• ROC-AUC Score:             {self.metrics['roc_auc_score']}")
        print(f"• Overall Accuracy:          {self.metrics['overall_accuracy']:.2%}")
        print(f"• Readmission Recall (Hit %): {c1['recall']:.2%} ({cm['true_positives']} of {c1['support']} readmissions caught)")
        print(f"• Readmission Precision:     {c1['precision']:.2%}")
        print(f"• Readmission F1-Score:      {c1['f1_score']}")
        print("\n--- Confusion Matrix Breakdown ---")
        print(f"  True Positives (Correctly caught readmissions):  {cm['true_positives']}")
        print(f"  False Negatives (Missed readmissions):           {cm['false_negatives']}")
        print(f"  True Negatives (Correctly identified safe):       {cm['true_negatives']}")
        print(f"  False Positives (False alarms):                  {cm['false_positives']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import joblib
    from src.ingestion.ingest import DataIngestion
    from src.preprocessing.clean import DataCleaner
    from src.preprocessing.missing import MissingValueHandler
    from src.feature_engineering.create_features import FeatureEngineer
    from src.preprocessing.encoding import DataEncoder
    from src.training.train import ModelTrainer

    # Run data prep, training, and evaluation
    data_path = "data/raw/diabetic_data.csv"
    raw_df = DataIngestion(data_path).load_data()
    cleaned_df = DataCleaner(raw_df).clean()
    imputed_df = MissingValueHandler(cleaned_df).handle_missing()
    features_df = FeatureEngineer(imputed_df).create_features()
    X, y = DataEncoder(features_df).encode()

    trainer = ModelTrainer(X, y, model_type="xgboost")
    X_train, X_test, y_train, y_test = trainer.split_data()
    model = trainer.train()

    evaluator = ModelEvaluator(model, X_test, y_test, model_name="xgboost")
    evaluator.evaluate()
    evaluator.print_summary()
    saved_json = evaluator.save_metrics()
