import argparse
import sys
import pandas as pd
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import training_logger
from src.exception import CustomException

class TrainingPipeline:
    def __init__(self):
        pass

    def run_ingestion(self):
        ingestion = DataIngestion()
        return ingestion.initiate_data_ingestion()

    def run_validation(self):
        validation = DataValidation()
        return validation.validate_dataset()

    def run_transformation(self):
        transformation = DataTransformation()
        return transformation.initiate_data_transformation()

    def run_training(self, X_train, y_train, X_test, y_test):
        trainer = ModelTrainer()
        return trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

    def run_evaluation(self, X_test, y_test, best_model_name):
        evaluation = ModelEvaluation()
        return evaluation.evaluate_model(X_test, y_test, best_model_name)

    def run_full_pipeline(self):
        training_logger.info("Executing full Training Pipeline")
        try:
            # 1. Ingestion
            _, train_path, test_path = self.run_ingestion()
            
            # 2. Validation
            self.run_validation()
            
            # 3. Transformation
            X_train, y_train, X_test, y_test, _ = self.run_transformation()
            
            # 4. Training
            _, best_model_name = self.run_training(X_train, y_train, X_test, y_test)
            
            # 5. Evaluation
            self.run_evaluation(X_test, y_test, best_model_name)
            
            training_logger.info("Training Pipeline completed successfully")
            
        except Exception as e:
            training_logger.error(f"Error in training pipeline run: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diabetes Prediction System Training Pipeline")
    parser.add_argument(
        "--stage", 
        type=str, 
        default="all", 
        choices=["ingestion", "validation", "transformation", "training", "evaluation", "all"],
        help="Stage to execute"
    )
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    try:
        if args.stage == "ingestion":
            pipeline.run_ingestion()
        elif args.stage == "validation":
            pipeline.run_validation()
        elif args.stage == "transformation":
            pipeline.run_transformation()
        elif args.stage == "training":
            # Requires loading transformed datasets, so we execute transformation first
            X_train, y_train, X_test, y_test, _ = pipeline.run_transformation()
            pipeline.run_training(X_train, y_train, X_test, y_test)
        elif args.stage == "evaluation":
            # Requires loading transformed datasets & model, so we run transformation first
            X_train, y_train, X_test, y_test, _ = pipeline.run_transformation()
            pipeline.run_evaluation(X_test, y_test, "Model")
        else:
            pipeline.run_full_pipeline()
    except Exception as e:
        print("Pipeline execution failed:", str(e))
        sys.exit(1)
