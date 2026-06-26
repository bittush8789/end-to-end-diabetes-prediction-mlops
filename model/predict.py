import os
import joblib
import pandas as pd
import numpy as np

# Path to the serialized model bundle
MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")

class DiabetesPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run training first.")
        
        # Load model bundle
        self.bundle = joblib.load(MODEL_PATH)
        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.medians = self.bundle["medians"]
        self.model_name = self.bundle.get("model_name", "Unknown Model")
        self.metrics = self.bundle.get("metrics", {})

    def predict(self, data_dict):
        """
        Expects a dictionary with the following keys:
        - Pregnancies
        - Glucose
        - BloodPressure
        - SkinThickness
        - Insulin
        - BMI
        - DiabetesPedigreeFunction
        - Age
        """
        # Convert inputs to float
        features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        
        # Parse inputs
        parsed_data = {}
        for feature in features:
            val = data_dict.get(feature)
            if val is None:
                raise ValueError(f"Missing required feature: {feature}")
            try:
                parsed_data[feature] = float(val)
            except ValueError:
                raise ValueError(f"Invalid numeric value for {feature}: {val}")
        
        # Impute zeros for columns where 0 is invalid
        for col, median_val in self.medians.items():
            if parsed_data[col] == 0:
                parsed_data[col] = median_val
                
        # Prepare for scaler
        df_row = pd.DataFrame([parsed_data])[features]
        scaled_features = self.scaler.transform(df_row)
        
        # Predict probability & class
        prob = float(self.model.predict_proba(scaled_features)[0][1])
        prediction_val = int(self.model.predict(scaled_features)[0])
        
        prediction_label = "Diabetic" if prediction_val == 1 else "Non-Diabetic"
        
        # Risk classification
        if prob < 0.40:
            risk_level = "Low Risk"
        elif prob <= 0.70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        return {
            "prediction": prediction_label,
            "probability": round(prob, 4),
            "risk_level": risk_level
        }

# Singleton instance
_predictor = None

def predict_diabetes(data_dict):
    global _predictor
    if _predictor is None:
        _predictor = DiabetesPredictor()
    return _predictor.predict(data_dict)

def get_model_info():
    global _predictor
    if _predictor is None:
        _predictor = DiabetesPredictor()
    return {
        "model_name": _predictor.model_name,
        "metrics": _predictor.metrics
    }
