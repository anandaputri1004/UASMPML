from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Customer Engagement Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
