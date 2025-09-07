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

# === Load files ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "lstm_weather_model.h5")
SCALER_X_FILE = os.path.join(BASE_DIR, "scaler_X.pkl")
SCALER_Y_FILE = os.path.join(BASE_DIR, "scaler_y.pkl")
X_SCALED_FILE = os.path.join(BASE_DIR, "X_scaled.pkl")

model = load_model(MODEL_FILE, compile=False)
scaler_X = joblib.load(SCALER_X_FILE)
scaler_y = joblib.load(SCALER_Y_FILE)
X_scaled = joblib.load(X_SCALED_FILE)

SEQ_LEN = 7

@app.route("/")
def home():
    return "☀️ Gridly Solar Forecast API is running!"

@app.route("/predict", methods=["GET"])
def predict_energy():
    try:
        city = request.args.get("city", "Patiala,IN")
        API_KEY = os.environ.get("OPENWEATHER_API_KEY")  

        if not API_KEY:
            return jsonify({"error": "Set your OpenWeatherMap API key in env var OPENWEATHER_API_KEY"})

        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()

        tomorrow = datetime.now() + timedelta(days=1)
        tmr_date = tomorrow.date()

        temps, hums, winds, precs = [], [], [], []
        for entry in response["list"]:
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

        expected_features = X_scaled.shape[1]
        if tomorrow_features.shape[1] < expected_features:
            pad = np.zeros((1, expected_features - tomorrow_features.shape[1]))
            tomorrow_features = np.hstack([tomorrow_features, pad])

        X_tomorrow_scaled = scaler_X.transform(tomorrow_features)
        recent_days = X_scaled[-(SEQ_LEN-1):]
        seq_input = np.vstack([recent_days, X_tomorrow_scaled])
        seq_input = seq_input.reshape(1, SEQ_LEN, X_scaled.shape[1])

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
