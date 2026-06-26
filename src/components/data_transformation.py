import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.logger import training_logger
from src.exception import CustomException
from src.utils import load_yaml, save_object

class DataTransformation:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_yaml(config_path)
        self.train_path = self.config["data"]["train_path"]
        self.test_path = self.config["data"]["test_path"]
        self.impute_cols = self.config["imputation"]["cols"]
        self.preprocessor_obj_path = "artifacts/preprocessor.pkl"

    def initiate_data_transformation(self):
        training_logger.info("Starting Data Transformation component")
        try:
            if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
                raise FileNotFoundError("Train or test dataset missing. Please run Ingestion stage first.")
                
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            # Separate Features and Targets
            X_train = train_df.drop(columns=["Outcome"])
            y_train = train_df["Outcome"]
            X_test = test_df.drop(columns=["Outcome"])
            y_test = test_df["Outcome"]
            
            # Compute medians of non-zero values on training set
            training_logger.info("Calculating column medians from training split to prevent leakage")
            medians = {}
            for col in self.impute_cols:
                # Median of non-zero training values
                median_val = X_train[X_train[col] != 0][col].median()
                medians[col] = median_val
                
                # Apply imputation
                X_train[col] = X_train[col].replace(0, median_val)
                X_test[col] = X_test[col].replace(0, median_val)
            
            training_logger.info("Fitting and transforming StandardScaler on training features")
            scaler = StandardScaler()
            
            # Ensure columns order matches expectations
            features_list = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            X_train = X_train[features_list]
            X_test = X_test[features_list]
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Build Preprocessor Bundle
            preprocessor_bundle = {
                "scaler": scaler,
                "medians": medians,
                "features": features_list
            }
            
            # Save preprocessor object
            save_object(self.preprocessor_obj_path, preprocessor_bundle)
            training_logger.info(f"Preprocessor object saved at {self.preprocessor_obj_path}")
            
            # Return scaled arrays along with target series
            return X_train_scaled, y_train, X_test_scaled, y_test, self.preprocessor_obj_path
            
        except Exception as e:
            training_logger.error(f"Exception raised in Data Transformation: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation()
