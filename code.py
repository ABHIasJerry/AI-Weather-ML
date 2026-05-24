# weather_sarima.py
import pandas as pd
import numpy as np
import pickle
import json
import os
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Step 1: Preprocessing
# -----------------------------
def preprocess_data(csv_path: str, target_column: str, seasonal_period: int = 12):
    # Create preprocessing folder
    os.makedirs("preprocessing", exist_ok=True)

    # Load dataset
    df = pd.read_csv(csv_path)

    # Ensure target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    series = df[target_column].dropna()

    # Train-test split (80/20)
    split_idx = int(len(series) * 0.8)
    train, test = series[:split_idx], series[split_idx:]

    # Save train/test sets
    train.to_csv("preprocessing/train.csv", index=False)
    test.to_csv("preprocessing/test.csv", index=False)

    return train, test, seasonal_period


# -----------------------------
# Step 2: Train SARIMA model
# -----------------------------
def train_sarima(train: pd.Series, test: pd.Series, seasonal_period: int = 12):
    # Auto hyperparameter search
    model_auto = auto_arima(
        train,
        seasonal=True,
        m=seasonal_period,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True
    )

    order = model_auto.order
    seasonal_order = model_auto.seasonal_order

    # Fit SARIMA with best params
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Evaluate on test set
    forecast_res = results.get_forecast(steps=len(test))
    preds = forecast_res.predicted_mean
    mse = mean_squared_error(test, preds)
    mae = mean_absolute_error(test, preds)

    # Save model
    with open("sarima_model.pkl", "wb") as f:
        pickle.dump(results, f)

    # Save hyperparameters
    hyperparams = {
        "order": order,
        "seasonal_order": seasonal_order,
        "seasonal_period": seasonal_period
    }
    with open("preprocessing/hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=4)

    # Save metrics
    metrics = {
        "mse": mse,
        "mae": mae,
        "aic": results.aic,
        "bic": results.bic
    }
    with open("preprocessing/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return results, hyperparams, metrics


# -----------------------------
# Step 3: Prediction function
# -----------------------------
def forecast(steps: int = 7):
    with open("sarima_model.pkl", "rb") as f:
        model = pickle.load(f)

    forecast_res = model.get_forecast(steps=steps)
    mean_forecast = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()

    return {
        "forecast": mean_forecast.tolist(),
        "lower_bound": conf_int.iloc[:, 0].tolist(),
        "upper_bound": conf_int.iloc[:, 1].tolist()
    }


# -----------------------------
# Step 4: FastAPI service
# -----------------------------
app = FastAPI(title="Weather Forecast API")

class ForecastRequest(BaseModel):
    steps: int = 7

@app.post("/predict")
def predict(req: ForecastRequest):
    result = forecast(req.steps)
    return result


# -----------------------------
# Example usage (training)
# -----------------------------
if __name__ == "__main__":
    # Example: train using a CSV file with a column "temperature"
    train, test, seasonal_period = preprocess_data("weather_data.csv", target_column="temperature", seasonal_period=7)
    model, hyperparams, metrics = train_sarima(train, test, seasonal_period)

    print("Model trained and saved.")
    print("Hyperparameters:", hyperparams)
    print("Metrics:", metrics)
