import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def download_dataset():
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, "diabetes.csv")
    
    if not os.path.exists(csv_path):
        url = "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv"
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, csv_path)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists locally.")
    return csv_path

def main():
    # 1. Load Dataset
    csv_path = download_dataset()
    df = pd.read_csv(csv_path)
    
    print("\n--- Data Exploration & Validation ---")
    print(f"Dataset Shape: {df.shape}")
    print("\nNull Values Count:")
    print(df.isnull().sum())
    print(f"\nDuplicate Records: {df.duplicated().sum()}")
    print("\nData Types:")
    print(df.dtypes)
    
    # 2. Data Splitting (Before Imputation/Scaling to prevent data leakage)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Data Cleaning (Impute zero values with median of training set)
    # Zeros in these columns are invalid/missing
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    medians = {}
    
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    print("\n--- Data Cleaning & Imputation ---")
    for col in zero_cols:
        # Calculate median of non-zero values on training set
        median_val = X_train[X_train[col] != 0][col].median()
        medians[col] = median_val
        print(f"Imputing 0s in '{col}' with training median: {median_val}")
        
        X_train_clean[col] = X_train_clean[col].replace(0, median_val)
        X_test_clean[col] = X_test_clean[col].replace(0, median_val)
        
    # 4. Feature Engineering (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # 5. Model Training & Comparison
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    }
    
    results = {}
    best_roc_auc = -1
    best_model_name = ""
    best_model = None
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        # Predict
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
        
        print(f"\n{name} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_model = model
            
    print(f"\nBest Model Selected: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")
    
    # 6. Model Serialization
    model_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the bundle containing model, scaler, imputation medians, and model name/metrics
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "medians": medians,
        "metrics": results
    }
    
    bundle_path = os.path.join(model_dir, "diabetes_model.pkl")
    joblib.dump(model_bundle, bundle_path)
    print(f"\nSaved model bundle successfully to '{bundle_path}'")

if __name__ == "__main__":
    main()
