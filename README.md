# 🧪 End-to-End Diabetes Prediction MLOps Platform

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI Version](https://img.shields.io/badge/fastapi-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![DVC Version](https://img.shields.io/badge/DVC-tracked-orange.svg)](https://dvc.org/)
[![Docker Support](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-KServe--ready-blue.svg)](https://kubernetes.io/)

This is a production-grade, end-to-end Machine Learning MLOps Platform built to predict diabetes risk using the Pima Indians Diabetes Dataset. It features modular training scripts, DVC version tracking, FastAPI backend APIs (with Prometheus metrics), a beautiful custom glassmorphism web frontend, and Kubernetes/KServe and ArgoCD GitOps deployment manifests.

---

## 🏛️ Architecture Flow

```text
Developer Push Code ──► GitHub Repository ──► GitHub Actions (CI) ──► Train Model (OOP Pipeline)
                                                                            │
                                                                            ▼
                                                                     DVC Track Model
                                                                            │
                                                                            ▼
ArgoCD Sync ◄── Sync K8s Cluster ◄── Update Inference YAML ◄── Push Model to S3 Remote
```

---

## 📂 Project Structure

```text
end-to-end-diabetes-prediction-mlops/
│
├── .github/workflows/
│   └── mlops-pipeline.yaml      # CI/CD pipeline automation
│
├── argocd/
│   └── application.yaml         # ArgoCD GitOps deployment config
│
├── k8s/
│   ├── namespace.yaml           # Deployment namespace
│   ├── serviceaccount.yaml      # Authentication credentials
│   └── inference.yaml           # KServe InferenceService definition
│
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml       # Prometheus scrapers configuration
│   └── grafana/
│       └── dashboards.yaml      # Grafana provisioning setups
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py    # Downloads and splits dataset
│   │   ├── data_validation.py   # Checks Schema & data types
│   │   ├── data_transformation.py # Feature scaling and zero imputation
│   │   ├── model_trainer.py     # Fits LR, RF, XGBoost classifiers
│   │   └── model_evaluation.py  # Evaluates and dumps metrics.json
│   ├── pipeline/
│   │   ├── training_pipeline.py # Pipelines stage driver
│   │   └── prediction_pipeline.py # Production inference wrapper
│   ├── logger.py                # Logging system (api.log, training.log)
│   ├── exception.py             # Custom Exception handler
│   └── utils.py                 # File IO helpers (pkl, yaml, json)
│
├── frontend/                    # Premium healthcare UI (SPA layout)
│   ├── index.html
│   ├── about.html
│   ├── style.css
│   └── script.js
│
├── artifacts/                   # DVC-ignored training cache
├── models/                      # Trained model directory
├── logs/                        # App log directory
├── app.py                       # FastAPI application entrypoint
├── train.py                     # Training execution entrypoint
├── dvc.yaml                     # DVC stages manifest
├── params.yaml                  # Parameters and hyperparameters
├── Dockerfile                   # ASGI Container config
├── requirements.txt             # Python packages manifest
└── README.md                    # Platform documentation
```

---

## 🚀 Machine Learning Pipeline & Metrics

Models are evaluated on a holdout test split (20%) after stratified train-test splitting. Zero-values are resolved via training medians, and features scaled with a standardized scaler.

The pipeline outputs:
- **Metrics**: `metrics.json`
- **Validation Report**: `validation_report.yaml`
- **Preprocessing Bundle**: `artifacts/preprocessor.pkl`
- **Best Classifier**: `models/diabetes_model.pkl` (Random Forest, ROC-AUC: **81.79%**)

---

## ⚙️ Setup & Local Execution

### 1. Install Packages
```bash
pip install -r requirements.txt
```

### 2. Execute Training Pipeline
```bash
python train.py
```
This downloads the dataset, runs validations, pre-processes the data, trains Logistic Regression/Random Forest/XGBoost models, and saves the best bundle.

### 3. Run FastAPI Application
```bash
python app.py
```
FastAPI server will be running on `http://localhost:8000`.

---

## 📡 API Endpoint Reference

### `POST /predict`
Performs a single patient risk evaluation.
*   **Request Body:**
    ```json
    {
      "Pregnancies": 6, "Glucose": 180, "BloodPressure": 90, "SkinThickness": 35,
      "Insulin": 250, "BMI": 38.5, "DiabetesPedigreeFunction": 1.2, "Age": 52
    }
    ```
*   **Response:**
    ```json
    {
      "prediction": "Diabetic",
      "probability": 0.7933
    }
    ```

### `POST /batch-predict`
Performs evaluations for multiple patient payloads.
*   **Request Body:**
    ```json
    {
      "patients": [
        {"Pregnancies":6,"Glucose":180,"BloodPressure":90,"SkinThickness":35,"Insulin":250,"BMI":38.5,"DiabetesPedigreeFunction":1.2,"Age":52}
      ]
    }
    ```
*   **Response:**
    ```json
    {
      "results": [
        {"prediction":"Diabetic","probability":0.7933}
      ]
    }
    ```

### `GET /metrics`
Exposes real-time API requests counters and latency histograms to Prometheus.

---

## ☸️ Kubernetes & GitOps Deployment

### KIND Cluster Setup
1. **Initialize KIND cluster:**
   ```bash
   kind create cluster --name diabetes-cluster
   ```
2. **Verify cluster availability:**
   ```bash
   kubectl cluster-info
   ```

### Deploying KServe
1. **Install KServe operator:**
   ```bash
   kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.11.0/kserve.yaml
   ```
2. **Deploy the manifests:**
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/serviceaccount.yaml
   kubectl apply -f k8s/inference.yaml
   ```

### GitOps Setup via ArgoCD
1. **Deploy ArgoCD namespace and controller:**
   ```bash
   kubectl create namespace argocd
   kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
   ```
2. **Launch Application GitOps controller:**
   ```bash
   kubectl apply -f argocd/application.yaml
   ```

---

## 📈 Monitoring & Dashboards

1. **Prometheus:** Pulls `/metrics` from the container port `8000` as configured in `monitoring/prometheus/prometheus.yml`.
2. **Grafana:** Visualizes API counts, inference latency distribution histograms, and cluster node states using the provisioned configs in `monitoring/grafana/`.

---

## 📄 License
Licensed under the MIT License.
