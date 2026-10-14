import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

model = None
scaler = None
feature_columns = []

BINARY_YES_NO_FIELDS = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea",
]

def _find_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "Housing.csv"),
        os.path.abspath(os.path.join(here, "..", "..", "Housing.csv")),
        os.environ.get("HOUSING_CSV", ""),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Housing.csv not found. Place it in backend/ or two levels up, "
        "or set the HOUSING_CSV env variable."
    )

def train_model():
    global model, scaler, feature_columns

    csv_path = _find_csv()
    print("[startup] Loading dataset from:", csv_path)

    df = pd.read_csv(csv_path)

    X = df.drop(["price"], axis=1)
    y = df["price"].astype(float)

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    from sklearn.metrics import r2_score
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    print("[startup] Model trained. R2 score on test set:", round(r2, 4))
    print("[startup] Features:", feature_columns)


def build_feature_row(data):
    row = {}
    for field in ["area", "bedrooms", "bathrooms", "stories", "parking"]:
        row[field] = float(data[field])
    for field in BINARY_YES_NO_FIELDS:
        row[field + "_yes"] = 1.0 if str(data[field]).lower() == "yes" else 0.0
    fs = str(data["furnishingstatus"]).lower()
    row["furnishingstatus_semi-furnished"] = 1.0 if fs == "semi-furnished" else 0.0
    row["furnishingstatus_unfurnished"]    = 1.0 if fs == "unfurnished"    else 0.0
    df_row = pd.DataFrame([row])
    df_row = df_row.reindex(columns=feature_columns, fill_value=0.0)
    return df_row


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_ready": model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not ready"}), 503

    data = request.get_json(force=True)
    required_keys = [
        "area", "bedrooms", "bathrooms", "stories", "parking",
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea", "furnishingstatus",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return jsonify({"error": "Missing fields: " + str(missing)}), 400

    try:
        df_row = build_feature_row(data)
        X_scaled = scaler.transform(df_row)
        predicted_price = float(model.predict(X_scaled)[0])
        predicted_price = max(0, round(predicted_price, 2))
        return jsonify({
            "predicted_price": predicted_price,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    train_model()
    app.run(host="0.0.0.0", debug=False, port=5001)