import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.logger import training_logger
from src.exception import CustomException
from src.utils import load_yaml, save_object

class ModelTrainer:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_yaml(config_path)
        self.model_config = self.config["models"]
        self.model_path = "models/diabetes_model.pkl"

    def initiate_model_trainer(self, X_train_scaled, y_train, X_test_scaled, y_test):
        training_logger.info("Starting Model Trainer component")
        try:
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=self.model_config["logistic_regression"]["max_iter"],
                    random_state=self.model_config["logistic_regression"]["random_state"]
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=self.model_config["random_forest"]["n_estimators"],
                    random_state=self.model_config["random_forest"]["random_state"]
                ),
                "XGBoost": XGBClassifier(
                    eval_metric=self.model_config["xgboost"]["eval_metric"],
                    random_state=self.model_config["xgboost"]["random_state"],
                    use_label_encoder=False
                )
            }
            
            best_model_name = ""
            best_model = None
            best_roc_auc = -1
            
            for name, model in models.items():
                training_logger.info(f"Training model: {name}")
                model.fit(X_train_scaled, y_train)
                
                probs = model.predict_proba(X_test_scaled)[:, 1]
                roc_auc = roc_auc_score(y_test, probs)
                
                training_logger.info(f"{name} ROC-AUC: {roc_auc:.4f}")
                
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_name = name
                    best_model = model
            
            training_logger.info(f"Selected Best Model: {best_model_name} with ROC-AUC {best_roc_auc:.4f}")
            
            # Save best model
            save_object(self.model_path, best_model)
            training_logger.info(f"Model saved successfully at {self.model_path}")
            
            return self.model_path, best_model_name
            
        except Exception as e:
            training_logger.error(f"Exception raised in Model Trainer: {str(e)}")
            raise CustomException(e, sys)
