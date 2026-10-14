# housing-price-app

Full-stack web application for predicting house prices. The Flask backend trains a **Linear Regression** model at startup and exposes a REST API. The HTML frontend is served directly by Flask — no Node.js required to run it.

## Structure

```
housing-price-app/
├── backend/    # Flask API, ML model, and HTML UI
└── frontend/   # Original React + Vite scaffold (not used at runtime)
```

## Running the App

See [`backend/README.md`](backend/README.md) for full instructions.

**Short version:**
```bash
cd backend
pip install -r requirements.txt
python app.py
# Open http://localhost:5001
```