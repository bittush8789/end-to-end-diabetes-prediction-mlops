import os
import sys
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from src.logger import training_logger
from src.exception import CustomException
from src.utils import load_yaml

class DataIngestion:
    def __init__(self, config_path: str = "params.yaml"):
        self.config = load_yaml(config_path)
        self.ingestion_config = self.config["data"]
        self.split_config = self.config["train_test_split"]

    def initiate_data_ingestion(self):
        training_logger.info("Starting Data Ingestion component")
        try:
            raw_path = self.ingestion_config["raw_path"]
            train_path = self.ingestion_config["train_path"]
            test_path = self.ingestion_config["test_path"]
            
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            
            # Download dataset if not exists
            if not os.path.exists(raw_path):
                url = self.ingestion_config["url"]
                training_logger.info(f"Downloading raw dataset from: {url}")
                urllib.request.urlretrieve(url, raw_path)
                training_logger.info("Download completed successfully")
            
            df = pd.read_csv(raw_path)
            training_logger.info(f"Dataset loaded with shape: {df.shape}")
            
            # Train/Test Split (stratified by Outcome)
            training_logger.info("Executing stratified Train-Test split")
            train_set, test_set = train_test_split(
                df, 
                test_size=self.split_config["test_size"], 
                random_state=self.split_config["random_state"],
                stratify=df["Outcome"]
            )
            
            train_set.to_csv(train_path, index=False, header=True)
            test_set.to_csv(test_path, index=False, header=True)
            training_logger.info(f"Train/Test sets saved successfully at {train_path} and {test_path}")
            
            return raw_path, train_path, test_path
            
        except Exception as e:
            training_logger.error(f"Exception raised in Data Ingestion: {str(e)}")
            raise CustomException(e, sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
