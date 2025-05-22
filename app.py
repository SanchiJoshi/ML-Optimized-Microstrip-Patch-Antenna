from flask import Flask, request, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

# Load scaler and models
scaler = load("scaler.pkl")
svr_model = load("svr_model.pkl")
svr_ga_model = load("svr_ga_model.pkl")
rf_model = load("rf_model.pkl")
rf_ga_model = load("rf_ga_model.pkl")

# Model selector
models = {
    "svr": svr_model,
    "svr_ga": svr_ga_model,
    "rf": rf_model,
    "rf_ga": rf_ga_model
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

if __name__ == "_main_":
    app.run(host="0.0.0.0", port=5000)