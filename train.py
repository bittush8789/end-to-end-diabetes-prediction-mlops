import sys
from src.pipeline.training_pipeline import TrainingPipeline
from src.logger import training_logger

if __name__ == "__main__":
    training_logger.info("Initializing train.py pipeline runner")
    try:
        pipeline = TrainingPipeline()
        pipeline.run_full_pipeline()
        print("Training pipeline run successful!")
    except Exception as e:
        print("Training pipeline run failed. Check logs/training.log for details.", str(e))
        sys.exit(1)
