import os
import sys
import pandas as pd
from src.logger import training_logger
from src.exception import CustomException
from src.utils import load_yaml, save_yaml

class DataValidation:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_yaml(config_path)
        self.raw_path = self.config["data"]["raw_path"]
        self.report_path = "validation_report.yaml"
        # Expected Schema
        self.expected_cols = {
            "Pregnancies": "int64",
            "Glucose": "int64",
            "BloodPressure": "int64",
            "SkinThickness": "int64",
            "Insulin": "int64",
            "BMI": "float64",
            "DiabetesPedigreeFunction": "float64",
            "Age": "int64",
            "Outcome": "int64"
        }

    def validate_dataset(self) -> bool:
        training_logger.info("Starting Data Validation component")
        try:
            if not os.path.exists(self.raw_path):
                raise FileNotFoundError(f"Raw data file not found at {self.raw_path}")
                
            df = pd.read_csv(self.raw_path)
            
            report = {
                "dataset_shape": list(df.shape),
                "null_values_count": df.isnull().sum().to_dict(),
                "duplicate_records": int(df.duplicated().sum()),
                "schema_validation": {},
                "datatype_validation": {},
                "overall_status": "Passed"
            }
            
            # Schema Validation (Column names check)
            for col in self.expected_cols.keys():
                if col not in df.columns:
                    report["schema_validation"][col] = "Missing"
                    report["overall_status"] = "Failed"
                else:
                    report["schema_validation"][col] = "Valid"
                    
            # Data Type Validation
            for col, expected_type in self.expected_cols.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    # Relax check slightly (e.g. float vs int is fine as long as numeric)
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    report["datatype_validation"][col] = {
                        "expected": expected_type,
                        "actual": actual_type,
                        "status": "Valid" if is_numeric else "Invalid"
                    }
                    if not is_numeric:
                        report["overall_status"] = "Failed"
            
            # Save Report
            save_yaml(self.report_path, report)
            training_logger.info(f"Data Validation report saved at {self.report_path} with status: {report['overall_status']}")
            
            return report["overall_status"] == "Passed"
            
        except Exception as e:
            training_logger.error(f"Exception raised in Data Validation: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataValidation()
    obj.validate_dataset()
