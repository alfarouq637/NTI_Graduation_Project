from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("models/fraud_model.pkl")

class Transaction(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Transaction):
    input_array = np.array(data.features).reshape(1, -1)
    result = model.predict(input_array)
    return {"fraud": int(result[0])}
