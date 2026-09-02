import json
import logging
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Finds optimal decision thresholds for imbalanced healthcare classification.
    Supports F2-score (recall-biased), Youden's J-statistic, and Target Recall thresholding.
    """

    def __init__(self, y_true: np.ndarray, y_probs: np.ndarray):
        self.y_true = np.array(y_true)
        self.y_probs = np.array(y_probs)

    def find_best_fbeta_threshold(self, beta: float = 2.0, num_thresholds: int = 100) -> dict:
        """
        Finds threshold maximizing F-beta score.
        beta=2.0 weights Recall twice as much as Precision (ideal for clinical readmission).
        """
        thresholds = np.linspace(0.1, 0.9, num_thresholds)
        best_threshold = 0.5
        best_fbeta = -1.0
        best_metrics = {}

        for t in thresholds:
            y_pred = (self.y_probs >= t).astype(int)
            fb = fbeta_score(self.y_true, y_pred, beta=beta, zero_division=0)
            if fb > best_fbeta:
                best_fbeta = fb
                best_threshold = t
                cm = confusion_matrix(self.y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                best_metrics = {
                    "optimal_threshold": round(float(best_threshold), 4),
                    f"max_f{int(beta)}_score": round(float(best_fbeta), 4),
                    "recall": round(float(recall), 4),
                    "precision": round(float(precision), 4),
                    "specificity": round(float(specificity), 4),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_negatives": int(tn),
                }

        logger.info(
            f"F{int(beta)} Optimal Threshold: {best_metrics['optimal_threshold']} "
            f"| Recall: {best_metrics['recall']:.2%} | Precision: {best_metrics['precision']:.2%}"
        )
        return best_metrics

    def find_youden_j_threshold(self) -> dict:
        """
        Finds threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1)
        on the ROC curve.
        """
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]

        y_pred = (self.y_probs >= best_threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "optimal_threshold": round(float(best_threshold), 4),
            "max_youden_j": round(float(j_scores[best_idx]), 4),
            "recall": round(float(tp / (tp + fn)), 4),
            "specificity": round(float(tn / (tn + fp)), 4),
            "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
        }
        logger.info(
            f"Youden's J Optimal Threshold: {metrics['optimal_threshold']} "
            f"| Recall: {metrics['recall']:.2%} | Specificity: {metrics['specificity']:.2%}"
        )
        return metrics

    def find_threshold_for_target_recall(self, target_recall: float = 0.70) -> dict:
        """
        Finds the highest threshold that guarantees at least `target_recall` (e.g. 70% or 80%).
        """
        thresholds = np.linspace(0.9, 0.05, 150)
        selected_threshold = 0.5
        for t in thresholds:
            y_pred = (self.y_probs >= t).astype(int)
            cm = confusion_matrix(self.y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if recall >= target_recall:
                selected_threshold = t
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                return {
                    "target_recall_requested": target_recall,
                    "achieved_recall": round(float(recall), 4),
                    "threshold": round(float(selected_threshold), 4),
                    "precision": round(float(precision), 4),
                    "patients_flagged": int(tp + fp),
                }

        return {"error": f"Could not achieve target recall of {target_recall}"}


class HyperparameterTuner:
    """
    Conducts stratified cross-validated random search across XGBoost hyperparameter space.
    """

    def __init__(
        self,
        n_iter: int = 20,
        cv_folds: int = 5,
        random_state: int = 42,
        scoring: str = "roc_auc",
    ):
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        self.best_model = None
        self.best_params = {}
        self.search_results = None

    def get_param_distributions(self) -> dict:
        """
        Defines search space for XGBoost.
        """
        return {
            "n_estimators": [100, 150, 200, 250, 300],
            "max_depth": [3, 4, 5, 6, 7],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
            "subsample": [0.7, 0.8, 0.85, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7, 10],
            "gamma": [0.0, 0.1, 0.2, 0.5, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0, 5.0],
            "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        }

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
        """
        Executes RandomizedSearchCV with StratifiedKFold.
        """
        logger.info(f"Starting Hyperparameter Tuning ({self.n_iter} iterations, {self.cv_folds}-Fold CV)...")

        # Automatically calculate scale_pos_weight
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight = num_neg / num_pos
        logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")

        base_xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )

        param_dist = self.get_param_distributions()
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        random_search = RandomizedSearchCV(
            estimator=base_xgb,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring=self.scoring,
            cv=cv,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1,
        )

        random_search.fit(X_train, y_train)

        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.search_results = random_search.cv_results_

        logger.info(f"Hyperparameter Tuning Complete! Best CV {self.scoring}: {random_search.best_score_:.4f}")
        logger.info(f"Best Parameters: {json.dumps(self.best_params, indent=2)}")

        return self.best_model

    def evaluate_and_optimize_thresholds(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        artifacts_dir: str = "artifacts",
        models_dir: str = "models",
    ) -> dict:
        """
        Evaluates best tuned model on holdout test set and finds optimal thresholds.
        """
        if self.best_model is None:
            raise ValueError("Model has not been tuned yet. Call tune() first.")

        y_probs = self.best_model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probs)

        # Baseline threshold (0.50) evaluation
        y_pred_50 = (y_probs >= 0.50).astype(int)
        report_50 = classification_report(y_test, y_pred_50, output_dict=True)

        # Optimize thresholds
        optimizer = ThresholdOptimizer(y_test, y_probs)
        f2_metrics = optimizer.find_best_fbeta_threshold(beta=2.0)
        youden_metrics = optimizer.find_youden_j_threshold()
        target_70_metrics = optimizer.find_threshold_for_target_recall(0.70)
        target_80_metrics = optimizer.find_threshold_for_target_recall(0.80)

        tuning_summary = {
            "tuned_model": "XGBoost_Tuned",
            "test_roc_auc": round(float(roc_auc), 4),
            "best_hyperparameters": self.best_params,
            "baseline_threshold_0_50": {
                "threshold": 0.50,
                "recall_class_1": round(report_50["1"]["recall"], 4),
                "precision_class_1": round(report_50["1"]["precision"], 4),
                "accuracy": round(report_50["accuracy"], 4),
            },
            "optimal_f2_threshold (balanced clinical)": f2_metrics,
            "youden_j_threshold": youden_metrics,
            "target_70_percent_recall": target_70_metrics,
            "target_80_percent_recall": target_80_metrics,
        }

        # Save artifacts
        os.makedirs(artifacts_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        summary_path = os.path.join(artifacts_dir, "hyperparameter_tuning_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(tuning_summary, f, indent=2)

        # Save tuned model
        model_path = os.path.join(models_dir, "xgboost_tuned_model.joblib")
        joblib.dump(self.best_model, model_path)
        logger.info(f"Saved tuned model to {model_path}")
        logger.info(f"Saved tuning & threshold report to {summary_path}")

        return tuning_summary


if __name__ == "__main__":
    from src.preprocessing.clean import DataCleaner
    from src.preprocessing.missing import MissingValueHandler
    from src.feature_engineering.create_features import FeatureEngineer
    from src.preprocessing.encoding import DataEncoder
    from sklearn.model_selection import train_test_split

    logger.info("Running standalone Hyperparameter Tuning & Threshold Optimization...")

    # Load & Preprocess
    df_raw = pd.read_csv("data/raw/diabetic_data.csv")
    df_clean = DataCleaner(df_raw).clean()
    df_missing = MissingValueHandler(df_clean).handle_missing()
    df_features = FeatureEngineer(df_missing).create_features()

    encoder = DataEncoder(df_features)
    X, y = encoder.encode()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    tuner = HyperparameterTuner(n_iter=15, cv_folds=5, random_state=42)
    tuner.tune(X_train, y_train)
    results = tuner.evaluate_and_optimize_thresholds(X_test, y_test)

    print("\n" + "=" * 50)
    print("TUNING & OPTIMAL THRESHOLD SUMMARY:")
    print("=" * 50)
    print(json.dumps(results, indent=2))
