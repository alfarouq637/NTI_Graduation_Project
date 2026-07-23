import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="IEEE CIS Fraud Detection API",
    description="Real time fraud prediction API for IEEE CIS transaction data",
    version="1.0.0",
)

# fraud threshold value max f1
FRAUD_THRESHOLD = 0.77

# load model startup
try:
    model = joblib.load("models/fraud_model.pkl")
except Exception as e:
    model = None
    print(f"Failed to load model {e}")


class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_array = np.array(data.features).reshape(1, -1)

    # predict probabilities & class labels
    raw_output = model.predict(input_array)

    if hasattr(model, "predict_proba"):
        prediction = int(raw_output[0])
        proba = float(model.predict_proba(input_array)[0][1])
    elif 0 <= raw_output[0] <= 1 and not isinstance(raw_output[0], (int, np.integer)):
        proba = float(raw_output[0])
        prediction = 1 if proba >= FRAUD_THRESHOLD else 0
    else:
        prediction = int(raw_output[0])
        proba = None

    result = {"fraud": prediction}
    if proba is not None:
        result["probability"] = round(proba, 6)
        result["threshold"] = FRAUD_THRESHOLD
    return result
