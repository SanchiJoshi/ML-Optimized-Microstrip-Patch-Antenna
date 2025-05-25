from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import pickle

app = Flask(__name__)
@app.route("/")
def home():
    return "✅ Antenna Frequency Prediction API is running. Use POST /predict."

# Load scaler and models
scaler = load("scaler.pkl")
svr_model = load("svr_model.pkl")
optimized_svr = load("optimized_svr.pkl")
rf_model = load("rf_model.pkl")
optimized_rf = load("optimized_rf.pkl")

# Model selector
models = {
    "svr": svr_model,
    "svr_ga": optimized_svr,
    "rf": rf_model,
    "rf_ga": optimized_rf
}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")
        model_key = data.get("model")

        if not features or model_key not in models:
            return jsonify({"error": "Invalid input"}), 400

        model = models[model_key]
        input_array = np.array(features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)

        return jsonify({
            "predicted_frequency": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)