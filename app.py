from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle, os
# -----------------------------
# Step 4: FastAPI service
# -----------------------------
app = FastAPI(title="Weather Forecast API")

class ForecastRequest(BaseModel):
    exog_future: list | None = None  # optional future exogenous values

@app.post("/predict")
def predict(req: ForecastRequest):
    exog_future = pd.DataFrame(req.exog_future) if req.exog_future else None
    result = forecast(exog_future)
    return result

@app.post("/predict")
def predict(req: ForecastRequest):
    exog_future = pd.DataFrame(req.exog_future) if req.exog_future else None
    result = forecast(exog_future)
    return result

@app.post("/health")
def health_check():
    """
    Health check endpoint to verify that the API service is running.
    Returns a simple JSON response with status information.
    """
    return {
        "status": "ok",
        "service": "Running!"
    }

@app.post("/")
def home():
    """
    Home endpoint to verify API accessibility.
    Returns a welcome message and basic service info.
    """
    return {
        "status": "ok",
        "message": "Welcome to the Weather Forecast API",
        "available_endpoints": ["/predict", "/health", "/home"],
        "model_loaded": os.path.exists("sarima_model.pkl")
    }

def forecast(exog_future: pd.DataFrame = None):
    steps = int(7)
    with open("model/sarima_model.pkl", "rb") as f:
        model = pickle.load(f)

    forecast_res = model.get_forecast(steps=steps, exog=exog_future)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    return {
        "forecast": mean_forecast.tolist(),
        "lower_bound": conf_int.iloc[:, 0].tolist(),
        "upper_bound": conf_int.iloc[:, 1].tolist()
    }
