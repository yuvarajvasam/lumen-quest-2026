from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("recommendation_model.pkl")
le_plan = joblib.load("le_plan.pkl")
le_type = joblib.load("le_type.pkl")
le_target = joblib.load("le_target.pkl")

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    usage_gb = data.get("usage_gb")
    current_plan = data.get("current_plan")
    plan_type = data.get("plan_type")
    price = data.get("price")

    # Encode inputs
    current_plan_enc = le_plan.transform([current_plan])[0]
    plan_type_enc = le_type.transform([plan_type])[0]

    # Prepare features
    X_new = pd.DataFrame([[usage_gb, current_plan_enc, plan_type_enc, price]],
                         columns=['usage_gb','current_plan_enc','plan_type_enc','price'])

    # Predict
    pred_enc = model.predict(X_new)[0]
    recommended_plan = le_target.inverse_transform([pred_enc])[0]

    return jsonify({
        "recommended_plan": recommended_plan,
        "reason": "Predicted using ML model based on your usage pattern"
    })

if __name__ == "__main__":
    app.run(debug=True)
