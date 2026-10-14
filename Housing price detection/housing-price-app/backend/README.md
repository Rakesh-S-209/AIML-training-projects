# Backend — Flask API & ML Model

This is the core of the web app. It trains a **Linear Regression** model on the housing dataset at startup and serves predictions through a REST API. The HTML UI is also served from here.

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.x | Runtime |
| Flask | 3.0.3 | Web framework & API server |
| flask-cors | 4.0.1 | Cross-origin request support |
| scikit-learn | 1.5.1 | LinearRegression, StandardScaler, train_test_split |
| pandas | 2.2.2 | Data loading and preprocessing |
| numpy | 1.26.4 | Numerical operations |

## File Structure

```
backend/
├── app.py              # Flask app: model training + API routes + UI serving
├── requirements.txt    # Python dependencies
└── templates/
    └── index.html      # Self-contained HTML/CSS/JS frontend
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
Put `Housing.csv` in **either** of these locations (searched in order):
- `backend/Housing.csv`
- Two levels up: `Housing price detection/Housing.csv`
- Or set the env variable: `HOUSING_CSV=/path/to/Housing.csv`

### 3. Start the server
```bash
python app.py
```
Open **http://localhost:5001** in your browser.

To stop the server press `Ctrl+C` in the terminal.

## API Reference

### `GET /`
Returns the HTML web UI.

---

### `GET /api/health`
Health check. Returns `200` when the model is loaded and ready.

**Response:**
```json
{ "status": "ok", "model_ready": true }
```

---

### `POST /api/predict`
Predicts the house price for a given set of features.

**Request body (JSON):**
```json
{
  "area": 7420,
  "bedrooms": 4,
  "bathrooms": 2,
  "stories": 3,
  "parking": 2,
  "mainroad": "yes",
  "guestroom": "no",
  "basement": "no",
  "hotwaterheating": "no",
  "airconditioning": "yes",
  "prefarea": "yes",
  "furnishingstatus": "furnished"
}
```

**Response:**
```json
{ "predicted_price": 4998904.37 }
```

## How the Model Works

1. **Load** `Housing.csv`
2. **Preprocess** — one-hot encode categorical columns (`drop_first=True`)
3. **Split** — 80% train / 20% test (`random_state=42`)
4. **Scale** — `StandardScaler` fit on the training set
5. **Train** — `LinearRegression` on the scaled training data
6. **Predict** — scale incoming request data, run `model.predict()`, return the value