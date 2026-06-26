import time
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.logger import api_logger

app = FastAPI(title="DiaPredict MLOps Platform", version="2.0.0")

# Mount frontend directory for static assets (style.css, script.js)
FRONTEND_DIR = os.path.join(os.getcwd(), "frontend")
os.makedirs(FRONTEND_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Prometheus Metrics definition
from prometheus_client import REGISTRY
if "api_requests_total" in REGISTRY._names_to_collectors:
    API_REQUESTS = REGISTRY._names_to_collectors["api_requests_total"]
else:
    API_REQUESTS = Counter("api_requests_total", "Total API Requests", ["method", "endpoint"])

if "predictions_total" in REGISTRY._names_to_collectors:
    PREDICTIONS_RUN = REGISTRY._names_to_collectors["predictions_total"]
else:
    PREDICTIONS_RUN = Counter("predictions_total", "Total Prediction Runs", ["outcome"])

if "inference_latency_seconds" in REGISTRY._names_to_collectors:
    INFERENCE_LATENCY = REGISTRY._names_to_collectors["inference_latency_seconds"]
else:
    INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference Latency in Seconds")


# Pydantic Schemas for Validation
class ClinicalMetrics(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=1000)
    BMI: float = Field(..., ge=0, le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: float = Field(..., ge=0, le=120)

class BatchPayload(BaseModel):
    patients: List[ClinicalMetrics]

# Prediction pipeline instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = PredictionPipeline()
    return predictor

# Frontend Routes
@app.get("/", response_class=FileResponse)
def home():
    API_REQUESTS.labels(method="GET", endpoint="/").inc()
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/about", response_class=FileResponse)
def about_page():
    API_REQUESTS.labels(method="GET", endpoint="/about").inc()
    return FileResponse(os.path.join(FRONTEND_DIR, "about.html"))

# API Endpoints
@app.get("/health")
def health():
    API_REQUESTS.labels(method="GET", endpoint="/health").inc()
    return {"status": "healthy"}

@app.get("/info")
def model_info():
    API_REQUESTS.labels(method="GET", endpoint="/info").inc()
    import json
    metrics_path = "metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Return placeholder if file not present
    return {
        "model_name": "Random Forest",
        "Accuracy": 0.7468,
        "Precision": 0.6596,
        "Recall": 0.5741,
        "F1 Score": 0.6139,
        "ROC AUC": 0.8179
    }


@app.get("/metrics")
def metrics():
    # Exposes prometheus metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict_endpoint(payload: ClinicalMetrics):
    API_REQUESTS.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    try:
        pipeline = get_predictor()
        # Convert Pydantic object to dict
        data_dict = payload.model_dump()
        result = pipeline.predict(data_dict)
        
        # Track metric values
        PREDICTIONS_RUN.labels(outcome=result["prediction"]).inc()
        INFERENCE_LATENCY.observe(time.time() - start_time)
        
        return result
    except Exception as e:
        api_logger.error(f"Inference exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict_endpoint(payload: BatchPayload):
    API_REQUESTS.labels(method="POST", endpoint="/batch-predict").inc()
    start_time = time.time()
    try:
        pipeline = get_predictor()
        # Extract patient records
        patient_list = [p.model_dump() for p in payload.patients]
        results = pipeline.predict_batch(patient_list)
        
        # Track predictions run
        for res in results:
            if "error" not in res:
                PREDICTIONS_RUN.labels(outcome=res["prediction"]).inc()
                
        INFERENCE_LATENCY.observe(time.time() - start_time)
        return {"results": results}
    except Exception as e:
        api_logger.error(f"Batch inference exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start ASGI dev server on port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
