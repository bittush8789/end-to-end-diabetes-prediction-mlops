import os
import sys
import json
import joblib
import yaml
from src.exception import CustomException

def save_object(file_path: str, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Object file not found at {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)

def save_yaml(file_path: str, data: dict):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
    except Exception as e:
        raise CustomException(e, sys)

def load_yaml(file_path: str) -> dict:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at {file_path}")
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise CustomException(e, sys)

def save_json(file_path: str, data: dict):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise CustomException(e, sys)
