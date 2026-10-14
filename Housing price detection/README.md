# Housing Price Detection

A machine learning project that predicts house prices using a **Linear Regression** model trained on the [Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset).

## Project Structure

```
Housing price detection/
├── House price detection.ipynb   # Exploratory data analysis & model experiments
├── Housing.csv                   # Dataset (not tracked in git)
└── housing-price-app/            # Full-stack web application
    ├── backend/                  # Flask API + ML model
    └── frontend/                 # React + Vite UI (original scaffold)
```

## What's Inside

### Notebook (`House price detection.ipynb`)
- Loads and explores the Housing dataset (545 rows, 13 columns)
- Visualises outliers with box plots
- Experiments with Logistic Regression, Random Forest, and Gradient Boosting classifiers
- Final decision: switched to **Linear Regression** for a more meaningful numeric price prediction

### Web App (`housing-price-app/`)
A Flask-based web application that serves both the REST API and the HTML UI from a single server.

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| `price` | int | House price (target) |
| `area` | int | Area in square feet |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `stories` | int | Number of stories |
| `parking` | int | Number of parking spaces |
| `mainroad` | yes/no | Connected to main road |
| `guestroom` | yes/no | Has a guest room |
| `basement` | yes/no | Has a basement |
| `hotwaterheating` | yes/no | Has hot water heating |
| `airconditioning` | yes/no | Has air conditioning |
| `prefarea` | yes/no | Located in a preferred area |
| `furnishingstatus` | categorical | furnished / semi-furnished / unfurnished |

## Quick Start

1. Clone the repo
2. Place `Housing.csv` in this folder (or in `housing-price-app/backend/`)
3. Follow the instructions in [`housing-price-app/backend/README.md`](housing-price-app/backend/README.md)