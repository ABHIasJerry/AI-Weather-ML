# Weather Forecast API (SARIMA)

A FastAPI service for weather forecasting using a SARIMA model with automatic hyperparameter tuning.  
The model is trained locally on historical weather data (CSV file), saved as a `.pkl` file, and served via REST API.  
Preprocessing outputs (train/test splits, hyperparameters, metrics) are stored in a dedicated folder.

---

## 🚀 Features
- Load dataset from `.csv` file with a specified target column (e.g., temperature).
- Automatic **80/20 train–test split** saved as CSV files in `preprocessing/`.
- Auto hyperparameter selection using `pmdarima.auto_arima`.
- SARIMA forecasting with confidence intervals (upper & lower bounds).
- Model persistence (`sarima_model.pkl`).
- Hyperparameters saved in `preprocessing/hyperparameters.json`.
- Metrics saved in `preprocessing/metrics.json`.
- FastAPI endpoint `/predict` for serving forecasts.
- Dockerized for easy deployment.

---

## 📂 Project Structure


---

## 📂 Project Structure

├── weather_sarima.py   # Training + FastAPI service
├── requirements.txt    # Python dependencies
├── Dockerfile          # Containerization
└── README.md           # Documentation


---

## 🛠️ Setup

### 1. Train the model
Edit `weather_sarima.py` to load your historical weather data (e.g., temperature series).  
Run locally:

```bash
python weather_sarima.py


Run FastAPI locally
bash
uvicorn weather_sarima:app --reload --host 0.0.0.0 --port 8000
Access the API at:

Swagger docs → http://localhost:8000/docs

Prediction endpoint → POST http://localhost:8000/predict

Example request:

json
{
  "steps": 7
}
3. Docker Deployment
Build and run the container:

bash
docker build -t weather-sarima .
docker run -p 8000:8000 weather-sarima
📊 Example Response
json
{
  "forecast": [25.1, 25.3, 25.5, ...],
  "lower_bound": [24.5, 24.7, 24.9, ...],
  "upper_bound": [25.7, 25.9, 26.1, ...],
  "info": {
    "order": [1,1,1],
    "seasonal_order": [0,1,1,7],
    "mse": 0.45,
    "mae": 0.52,
    "aic": 1234.56,
    "bic": 1240.78
  }
}
📌 Notes
Replace synthetic data in weather_sarima.py with your actual historical weather dataset.

Adjust seasonal_period depending on your data frequency (e.g., 7 for weekly seasonality, 12 for monthly).

The .pkl file must exist before serving predictions.