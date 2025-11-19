# app/app.py

from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# Load model
model = joblib.load("../src/model.pkl")

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to House Price Prediction API"}

# Prediction route
@app.post("/predict")
def predict(data: dict):
    """
    data = {
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    """
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"prediction": float(prediction[0])}
