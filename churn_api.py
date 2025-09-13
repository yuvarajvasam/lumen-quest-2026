# churn_api.py
from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Churn Prediction API is running!"

@app.route("/predict_churn", methods=["POST"])
def predict_churn():
    data = request.get_json()

    # Example input: {"tenure": 12, "monthly_charges": 800, "contract_type": "Monthly"}
    df = pd.DataFrame([data])

    # One-hot encode same as training
    df = pd.get_dummies(df, columns=["contract_type"], drop_first=True)

    # Ensure columns match model input
    for col in ["contract_type_Yearly"]:
        if col not in df.columns:
            df[col] = 0

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return jsonify({
        "churn": int(prediction),
        "probability": round(float(probability), 2)
    })

if __name__ == "__main__":
    app.run(port=5001, debug=True)
