import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load("agriculture_model.pkl")

app = FastAPI(title="Smart Agriculture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class FarmData(BaseModel):
    Crop: str
    Region: str
    Soil_Type: str
    Soil_pH: float
    Rainfall_mm: float
    Temperature_C: float
    Humidity_pct: float
    Fertilizer_Used_kg: float
    Irrigation: str | None = None
    Pesticides_Used_kg: float
    Planting_Density: float
    Previous_Crop: str | None = None

@app.get("/")
def home():
    return {"message": "Smart Agriculture API is running"}

@app.post("/predict")
def predict(data: FarmData):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)[0]

    return {
        "predicted_crop_yield_ton_per_ha": round(float(prediction), 2)
    }