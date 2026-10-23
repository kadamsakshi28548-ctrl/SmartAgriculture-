from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Smart Agriculture ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("irrigation_model.pkl")


class FarmData(BaseModel):
    Crop: str
    Region: str
    Soil_Type: str
    Soil_pH: float
    Rainfall_mm: float
    Temperature_C: float
    Humidity_pct: float
    Fertilizer_Used_kg: float
    Pesticides_Used_kg: float
    Planting_Density: float
    Previous_Crop: str


@app.get("/")
def home():
    return {
        "message": "Smart Agriculture ML API is running"
    }


@app.post("/predict")
def predict(data: FarmData):
    input_data = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_data)[0]

    return {
        "prediction": str(prediction),
        "model": "Random Forest Classifier"
    }