import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import training_logger
from src.exception import CustomException
from src.utils import load_object, save_json

class ModelEvaluation:
    def __init__(self):
        self.model_path = "models/diabetes_model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.metrics_path = "metrics.json"

    def evaluate_model(self, X_test_scaled, y_test, best_model_name: str = "Unknown"):
        training_logger.info("Starting Model Evaluation component")
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
                
            model = load_object(self.model_path)
            
            # Predict
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = float(accuracy_score(y_test, preds))
            precision = float(precision_score(y_test, preds))
            recall = float(recall_score(y_test, preds))
            f1 = float(f1_score(y_test, preds))
            roc_auc = float(roc_auc_score(y_test, probs))
            
            metrics = {
                "model_name": best_model_name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "ROC AUC": round(roc_auc, 4)
            }
            
            # Save to json
            save_json(self.metrics_path, metrics)
            training_logger.info(f"Evaluation metrics written to {self.metrics_path}: {metrics}")
            
            return metrics
            
        except Exception as e:
            training_logger.error(f"Exception raised in Model Evaluation: {str(e)}")
            raise CustomException(e, sys)
