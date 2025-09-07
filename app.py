import os
import numpy as np
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# === File paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "artifacts", "lstm_weather_model_fixed.h5")
SCALER_X_FILE = os.path.join(BASE_DIR, "artifacts", "scaler_X_fixed.pkl")
SCALER_Y_FILE = os.path.join(BASE_DIR, "artifacts", "scaler_y_fixed.pkl")
X_SCALED_FILE = os.path.join(BASE_DIR, "artifacts", "X_scaled_fixed.pkl")

# === Lazy-load globals ===
model = None
scaler_X = None
scaler_y = None
X_scaled = None
SEQ_LEN = 7

def load_artifacts():
    """Load model and scalers only once (lazy load)."""
    global model, scaler_X, scaler_y, X_scaled
    if model is None:
        print("🔄 Loading model and scalers...")
        model = load_model(MODEL_FILE, compile=False)
        scaler_X = joblib.load(SCALER_X_FILE)
        scaler_y = joblib.load(SCALER_Y_FILE)
        X_scaled = joblib.load(X_SCALED_FILE)
        print("✅ Model and scalers loaded!")

@app.route("/")
def home():
    return "☀️ Gridly Solar Forecast API is running (lazy load enabled)!"

@app.route("/predict", methods=["GET"])
def predict_energy():
    try:
        load_artifacts()  # ensure model/scalers are loaded

        city = request.args.get("city", "Patiala,IN")
        API_KEY = os.environ.get("OPENWEATHER_API_KEY")

        if not API_KEY:
            return jsonify({"error": "Set your OpenWeatherMap API key in env var OPENWEATHER_API_KEY"})

        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()

        tomorrow = datetime.now() + timedelta(days=1)
        tmr_date = tomorrow.date()

        temps, hums, winds, precs = [], [], [], []
        for entry in response.get("list", []):
            dt = pd.to_datetime(entry["dt"], unit="s")
            if dt.date() == tmr_date:
                temps.append(entry["main"]["temp"])
                hums.append(entry["main"]["humidity"])
                winds.append(entry["wind"]["speed"])
                precs.append(entry.get("rain", {}).get("3h", 0))

        if not temps:
            return jsonify({"error": "No forecast available for tomorrow."})

        tomorrow_features = np.array([[
            np.mean(temps),
            np.mean(hums),
            np.mean(winds),
            np.sum(precs)
        ]])

        # pad if fewer features than training
        expected_features = X_scaled.shape[1]
        if tomorrow_features.shape[1] < expected_features:
            pad = np.zeros((1, expected_features - tomorrow_features.shape[1]))
            tomorrow_features = np.hstack([tomorrow_features, pad])

        # scale input
        X_tomorrow_scaled = scaler_X.transform(tomorrow_features)
        recent_days = X_scaled[-(SEQ_LEN - 1):]
        seq_input = np.vstack([recent_days, X_tomorrow_scaled])
        seq_input = seq_input.reshape(1, SEQ_LEN, X_scaled.shape[1])

        # predict
        pred_scaled = model.predict(seq_input)
        pred_kWh = scaler_y.inverse_transform(pred_scaled)[0, 0]

        return jsonify({
            "city": city,
            "predicted_energy_kWh": round(float(pred_kWh), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7000)))
