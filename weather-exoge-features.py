"""
===============================================================================
 Weather Forecast API (SARIMAX Model)
===============================================================================

 Author: Abhinaba
 Date: May 2026
 Location: Kolkata, India

 Description:
    This module implements a complete pipeline for weather forecasting using
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous
    variables). It supports:
      - CSV dataset ingestion (temperature + optional exogenous predictors
        such as humidity, wind speed, pressure).
      - Automated preprocessing with 80/20 train-test split.
      - Auto hyperparameter tuning across multiple seasonal periods.
      - Model training, evaluation, and persistence.
      - Saving hyperparameters and metrics into JSON files.
      - FastAPI service for serving forecasts with confidence intervals.

 Workflow:
    1. Preprocessing:
       - Load dataset from CSV.
       - Split into train/test sets (80/20).
       - Save train/test and exogenous sets into `preprocessing/` folder.

    2. Training:
       - Run auto_arima to determine optimal (p,d,q) and (P,D,Q,m).
       - Fit SARIMAX model with exogenous regressors if provided.
       - Evaluate on test set (MSE, MAE, AIC, BIC).
       - Save model as `sarima_model.pkl`.
       - Save hyperparameters in `preprocessing/hyperparameters.json`.
       - Save metrics in `preprocessing/metrics.json`.

    3. Prediction:
       - Load trained model.
       - Forecast future values for N steps (controlled via config.json).
       - Return forecast with upper/lower confidence bounds.

    4. FastAPI Service:
       - Endpoint: POST /predict
       - Input: JSON with optional `exog_future` values.
       - Output: Forecast values + confidence intervals.

 Files Generated:
    - preprocessing/train.csv
    - preprocessing/test.csv
    - preprocessing/exog_train.csv (if exogenous variables provided)
    - preprocessing/exog_test.csv (if exogenous variables provided)
    - preprocessing/hyperparameters.json
    - preprocessing/metrics.json
    - sarima_model.pkl

 Config:
    - config.json controls forecast horizon and seasonal periods.
      Example:
        {
          "steps": 14,
          "seasonal_periods": [7, 365]
        }

 Usage:
    $ python weather_sarima.py
    # Trains the model and saves outputs

    $ uvicorn weather_sarima:app --reload --host 0.0.0.0 --port 8000
    # Runs FastAPI service
===============================================================================
"""
# Imports
import pandas as pd
import os, json, pickle, time
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Step 0: Load config
# -----------------------------
def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        # default config if not provided
        return {"steps": 7, "seasonal_periods": [7, 365]}
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()

# -----------------------------
# Step 1: Preprocessing
# -----------------------------
def preprocess_data(csv_path: str, target_column: str, exog_columns: list = None, seasonal_period: int = 12):
    os.makedirs("preprocessing", exist_ok=True)

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    series = df[target_column].dropna()
    exog = df[exog_columns] if exog_columns else None

    # Train-test split (80/20)
    split_idx = int(len(series) * 0.8)
    train, test = series[:split_idx], series[split_idx:]
    exog_train, exog_test = (exog[:split_idx], exog[split_idx:]) if exog is not None else (None, None)

    # Save train/test sets
    train.to_csv("preprocessing/train.csv", index=False)
    test.to_csv("preprocessing/test.csv", index=False)
    if exog is not None:
        exog_train.to_csv("preprocessing/exog_train.csv", index=False)
        exog_test.to_csv("preprocessing/exog_test.csv", index=False)

    return train, test, exog_train, exog_test, seasonal_period


# -----------------------------
# Step 2: Train SARIMAX model
# -----------------------------
def train_sarima_advanced(train: pd.Series, test: pd.Series, exog_train=None, exog_test=None, seasonal_periods=None):
    if seasonal_periods is None:
        seasonal_periods = CONFIG.get("seasonal_periods", [7, 365])

    best_model = None
    best_metrics = None
    best_params = None

    for m in seasonal_periods:
        model_auto = auto_arima(
            train,
            exogenous=exog_train,
            seasonal=True,
            m=m,
            start_p=0, start_q=0, max_p=3, max_q=3,
            start_P=0, start_Q=0, max_P=2, max_Q=2,
            d=None, D=None,
            trace=True,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True
        )

        order = model_auto.order
        seasonal_order = model_auto.seasonal_order

        model = SARIMAX(train, exog=exog_train, order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)

        forecast_res = results.get_forecast(steps=len(test), exog=exog_test)
        preds = forecast_res.predicted_mean
        mse = mean_squared_error(test, preds)
        mae = mean_absolute_error(test, preds)

        if best_metrics is None or mse < best_metrics["mse"]:
            best_model = results
            best_metrics = {"mse": mse, "mae": mae, "aic": results.aic, "bic": results.bic}
            best_params = {"order": order, "seasonal_order": seasonal_order, "seasonal_period": m}

    # Save best model
    with open("model/sarima_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save hyperparameters + metrics in JSON
    with open("parameters/hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=4)
    with open("parameters/metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=4)

    return best_model, best_params, best_metrics


# -----------------------------
# Step 3: Prediction function
# -----------------------------
def forecast(exog_future: pd.DataFrame = None):
    steps = CONFIG.get("steps", 7)  # read steps from config.json
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

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    """ MAIN ENTRY POINT"""
    start_time = time.time()
    train, test, exog_train, exog_test, seasonal_period = preprocess_data(
        "data/Indian_Climate_Dataset_2024_2025.csv",
        target_column="Temperature_Max (°C)",
        exog_columns=["Humidity (%)", "Wind_Speed (km/h)", "Pressure (hPa)"],
        seasonal_period=7
    )
    model, hyperparams, metrics = train_sarima_advanced(train, test, exog_train, exog_test)
    end_time = time.time()
    elapsed = end_time - start_time

    print(f"Trainng Complete! Execution time: {elapsed:.4f} seconds")
    print("Model saved in -> [model/]")
    print("Best Hyperparameters:", hyperparams)
    print("Metrics:", metrics)
    print("Hyperparameters & Metrics saved in -> [parameters/]")