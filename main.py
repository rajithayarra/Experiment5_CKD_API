from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import time
import os


# =========================================================
# 1. CREATE FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="CKD Prediction API",
    description="REST API for Chronic Kidney Disease Prediction",
    version="1.0.0"
)


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

MODEL_PATH = "model/ckd_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Please run train_model.py first."
    )

model = joblib.load(MODEL_PATH)
METRICS_PATH = "model/metrics.joblib"
metrics_data = joblib.load(METRICS_PATH)

# =========================================================
# 3. REQUEST DATA MODEL
# =========================================================

class CKDInput(BaseModel):

    age: float = Field(..., ge=0, le=120)
    bp: float = Field(..., ge=0, le=250)
    sg: float = Field(..., ge=1.0, le=1.1)
    al: float = Field(..., ge=0, le=5)
    su: float = Field(..., ge=0, le=5)

    rbc: str
    pc: str
    pcc: str
    ba: str

    bgr: float = Field(..., ge=0)
    bu: float = Field(..., ge=0)
    sc: float = Field(..., ge=0)
    sod: float = Field(..., ge=0)
    pot: float = Field(..., ge=0)
    hemo: float = Field(..., ge=0)

    pcv: str
    wc: str
    rc: str

    htn: str
    dm: str
    cad: str
    appet: str
    pe: str
    ane: str


# =========================================================
# 4. API METRICS
# =========================================================

total_requests = 0
successful_predictions = 0
failed_requests = 0
total_response_time = 0.0


# =========================================================
# 5. ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "CKD Prediction API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }


# =========================================================
# 6. HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =========================================================
# 7. READY ENDPOINT
# =========================================================

@app.get("/ready")
def ready():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )

    return {
        "status": "ready",
        "model_loaded": True
    }


# =========================================================
# 8. MODEL INFO ENDPOINT
# =========================================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": "CKD Prediction Model",
        "model_version": "1.0",
        "algorithm": "Random Forest Classifier",
        "number_of_features": 24,
        "target": "classification",
        "classes": ["notckd", "ckd"],
        "evaluation_metrics": {
            "accuracy": metrics_data["accuracy"],
            "precision": metrics_data["precision"],
            "recall": metrics_data["recall"],
            "f1_score": metrics_data["f1_score"]
        }
    }


# =========================================================
# 9. PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: CKDInput):

    global total_requests
    global successful_predictions
    global failed_requests
    global total_response_time

    start_time = time.time()

    total_requests += 1

    try:

        # Convert input into dictionary
        input_data = data.model_dump()

        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Get probability
        probabilities = model.predict_proba(input_df)[0]

        # Probability of predicted class
        predicted_probability = float(
            max(probabilities)
        )

        # Convert numerical output to readable label
        if prediction == 1:
            result = "CKD"
        else:
            result = "NOT CKD"

        successful_predictions += 1

        response_time = time.time() - start_time
        total_response_time += response_time

        return {
            "prediction": result,
            "model_probability": round(
                predicted_probability, 4
            ),
            "message": (
                "This is a machine learning model output "
                "and not a medical diagnosis."
            )
        }

    except Exception as e:

        failed_requests += 1

        response_time = time.time() - start_time
        total_response_time += response_time

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# =========================================================
# 10. METRICS ENDPOINT
# =========================================================

@app.get("/metrics")
def metrics():

    if total_requests > 0:
        success_rate = (
            successful_predictions / total_requests
        ) * 100
    else:
        success_rate = 0

    if total_requests > 0:
        average_response_time = (
            total_response_time / total_requests
        )
    else:
        average_response_time = 0

    return {
        "total_requests": total_requests,
        "successful_predictions": successful_predictions,
        "failed_requests": failed_requests,
        "success_rate_percent": round(
            success_rate, 2
        ),
        "average_response_time_seconds": round(
            average_response_time, 4
        )
    }