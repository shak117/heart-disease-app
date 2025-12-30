from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: HeartInput):

    features = np.array([[ 
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]])

    features_scaled = scaler.transform(features)

    prediction = int(model.predict(features_scaled)[0])

    prob = float(model.predict_proba(features_scaled)[0][1])

    return {
        "prediction": prediction,
        "risk_percentage": round(prob * 100, 2)
    }
