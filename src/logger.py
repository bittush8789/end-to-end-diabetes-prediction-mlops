import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add them again
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    log_filepath = os.path.join(LOG_DIR, filename)
    
    # Create formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")
    
    # File Handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Preconfigured loggers
training_logger = get_logger("training", "training.log")
prediction_logger = get_logger("prediction", "prediction.log")
api_logger = get_logger("api", "api.log")
