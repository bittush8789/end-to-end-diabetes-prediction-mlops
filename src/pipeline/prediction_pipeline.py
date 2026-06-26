import os
import sys
import pandas as pd
from src.logger import prediction_logger
from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("models", "diabetes_model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
        self.model = None
        self.scaler = None
        self.medians = None
        self.features = None
        
        self._load_resources()

    def _load_resources(self):
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError("Model or Preprocessor files not found. Ensure training pipeline is run.")
                
            self.model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            
            self.scaler = preprocessor["scaler"]
            self.medians = preprocessor["medians"]
            self.features = preprocessor["features"]
            
            prediction_logger.info("Prediction resources loaded successfully")
        except Exception as e:
            prediction_logger.error(f"Error loading resources: {str(e)}")
            raise CustomException(e, sys)

    def predict(self, features_dict: dict) -> dict:
        try:
            prediction_logger.info(f"Running prediction for single payload: {features_dict}")
            
            # 1. Parse and validate inputs
            parsed_data = {}
            for feature in self.features:
                val = features_dict.get(feature)
                if val is None:
                    raise ValueError(f"Missing feature: {feature}")
                parsed_data[feature] = float(val)
                
            # 2. Impute zeros
            for col, median_val in self.medians.items():
                if parsed_data[col] == 0:
                    parsed_data[col] = median_val
                    
            # 3. Format as DataFrame
            df_row = pd.DataFrame([parsed_data])[self.features]
            
            # 4. Scale features
            scaled_features = self.scaler.transform(df_row)
            
            # 5. Predict
            prob = float(self.model.predict_proba(scaled_features)[0][1])
            pred_class = int(self.model.predict(scaled_features)[0])
            
            prediction_label = "Diabetic" if pred_class == 1 else "Non-Diabetic"
            
            prediction_logger.info(f"Prediction result: Class {prediction_label}, Probability {prob:.4f}")
            
            return {
                "prediction": prediction_label,
                "probability": round(prob, 4)
            }
            
        except Exception as e:
            prediction_logger.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)
            
    def predict_batch(self, batch_list: list) -> list:
        try:
            prediction_logger.info(f"Running prediction for batch size: {len(batch_list)}")
            results = []
            for item in batch_list:
                try:
                    res = self.predict(item)
                    results.append(res)
                except Exception as item_err:
                    results.append({"error": str(item_err)})
            return results
        except Exception as e:
            prediction_logger.error(f"Error during batch prediction: {str(e)}")
            raise CustomException(e, sys)
