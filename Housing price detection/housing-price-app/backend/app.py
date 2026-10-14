"""
Flask backend for the Housing Price Classification model.

Trains a Logistic Regression model on startup using the same pipeline
from the Kaggle notebook, then serves predictions via a REST API.

Endpoints:
  GET  /api/health        — Health check
  POST /api/predict       — Predict price category (Low / Medium / High)
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow requests from the React dev server

# ---------------------------------------------------------------------------
# Global model artefacts (populated at startup)
# ---------------------------------------------------------------------------
model: LogisticRegression | None = None
scaler: StandardScaler | None = None
feature_columns: list[str] = []
label_map = {0: "Low", 1: "Medium", 2: "High"}

# ---------------------------------------------------------------------------
# Training helper — mirrors the notebook pipeline exactly
# ---------------------------------------------------------------------------
def _find_csv() -> str:
    """
    Search for Housing.csv in order of preference:
      1. Same directory as app.py  (backend/)
      2. Two levels up             (Housing price detection/)
      3. HOUSING_CSV env var       (custom path)
    Returns the first path that exists, or raises FileNotFoundError.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "Housing.csv"),                        # backend/Housing.csv
        os.path.abspath(os.path.join(here, "..", "..", "Housing.csv")),  # …/Housing price detection/Housing.csv
        os.environ.get("HOUSING_CSV", ""),                        # env override
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Housing.csv not found. Place it in the backend/ folder or in "
        "'Housing price detection/', or set the HOUSING_CSV env variable."
    )


def train_model() -> None:
    """Load Housing.csv, replicate the notebook pipeline, and fit the model."""
    global model, scaler, feature_columns

    csv_path = _find_csv()
    print(f"[startup] Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path)

    # ── Target: bin price into 3 equal-frequency classes ──────────────────
    df["price_category"] = pd.qcut(df["price"], q=3, labels=[0, 1, 2])

    X = df.drop(["price", "price_category"], axis=1)
    y = df["price_category"].astype(int)

    # ── One-hot encode all categorical columns (drop_first=True) ──────────
    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_columns = X_encoded.columns.tolist()

    # ── Train / test split ────────────────────────────────────────────────
    X_train, _, y_train, _ = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # ── Scale features ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ── Fit Logistic Regression (best model from notebook) ────────────────
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    print(f"[startup] Model trained. Features ({len(feature_columns)}): {feature_columns}")


# ---------------------------------------------------------------------------
# Request → feature vector helper
# ---------------------------------------------------------------------------
BINARY_YES_NO_FIELDS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]

FURNISHING_OPTIONS = ["furnished", "semi-furnished", "unfurnished"]


def build_feature_row(data: dict) -> pd.DataFrame:
    """
    Convert raw JSON payload from the frontend into a one-row DataFrame
    that matches the training feature columns exactly.

    Expected payload keys (all strings):
      area, bedrooms, bathrooms, stories, parking   → numeric
      mainroad, guestroom, basement,
      hotwaterheating, airconditioning, prefarea     → "yes" | "no"
      furnishingstatus                               → "furnished" |
                                                       "semi-furnished" |
                                                       "unfurnished"
    """
    row: dict = {}

    # Numeric fields
    for field in ["area", "bedrooms", "bathrooms", "stories", "parking"]:
        row[field] = float(data[field])

    # Binary yes/no → one-hot (drop_first=True drops the "no" level,
    # so the column is named <field>_yes)
    for field in BINARY_YES_NO_FIELDS:
        row[f"{field}_yes"] = 1.0 if str(data[field]).lower() == "yes" else 0.0

    # furnishingstatus → two dummy columns (furnished is the dropped reference)
    fs = str(data["furnishingstatus"]).lower()
    row["furnishingstatus_semi-furnished"] = 1.0 if fs == "semi-furnished" else 0.0
    row["furnishingstatus_unfurnished"] = 1.0 if fs == "unfurnished" else 0.0

    df_row = pd.DataFrame([row])

    # Align to training columns (add any missing cols as 0, drop extras)
    df_row = df_row.reindex(columns=feature_columns, fill_value=0.0)
    return df_row


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_ready": model is not None})


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Model not ready"}), 503

    data = request.get_json(force=True)

    # ── Validate required keys ────────────────────────────────────────────
    required_keys = [
        "area", "bedrooms", "bathrooms", "stories", "parking",
        "mainroad", "guestroom", "basement", "hotwaterheating",
        "airconditioning", "prefarea", "furnishingstatus",
    ]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        df_row = build_feature_row(data)
        X_scaled = scaler.transform(df_row)
        pred_idx = int(model.predict(X_scaled)[0])
        probas = model.predict_proba(X_scaled)[0].tolist()

        return jsonify({
            "prediction": pred_idx,
            "label": label_map[pred_idx],
            "probabilities": {
                label_map[i]: round(p * 100, 1)
                for i, p in enumerate(probas)
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_model()
    app.run(debug=True, port=5000)
