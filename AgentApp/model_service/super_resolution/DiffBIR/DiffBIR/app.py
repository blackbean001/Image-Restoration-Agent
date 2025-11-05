# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict
from inference import *

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict_api():
    result = predict(data.features)
    return {"prediction": result}

