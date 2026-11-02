# California Housing — Regression Model

A machine learning project that predicts **median house values** in California using the classic [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from scikit-learn.

## Dataset

Fetched directly via `sklearn.datasets.fetch_california_housing()` — no external download needed.

| Property | Value |
|---|---|
| Rows | 20,640 |
| Features | 8 |
| Target | `MedHouseVal` (median house value in $100,000 units) |

### Features

| Feature | Description |
|---|---|
| `MedInc` | Median income of households in the block (in $10K units) |
| `HouseAge` | Median age of houses in the block (years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Total population of the block |
| `AveOccup` | Average number of household members |
| `Latitude` | Block latitude |
| `Longitude` | Block longitude |

> Each row represents a **census block** (neighbourhood aggregate), not a single house.

## Notebook

`housing_california_regression.ipynb` covers:

1. **Load & Explore** — dataset shape, head, data types, missing value check
2. **EDA** — target distribution, feature distributions, correlation heatmap
3. **Preprocessing** — train/test split (80/20), StandardScaler normalisation
4. **Model Training & Comparison** — four models evaluated side by side:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.5332 | 0.7456 | 0.5758 |
| Ridge Regression | 0.5332 | 0.7456 | 0.5758 |
| Random Forest | **0.3274** | **0.5051** | **0.8053** |
| Gradient Boosting | 0.3484 | 0.5114 | 0.8004 |

5. **Visualisations** — actual vs predicted scatter, residual plot, feature importance chart

## Results

**Random Forest** achieves the best performance with **R² = 0.805**, meaning it explains ~80% of the variance in house prices. `MedInc` (median income) is the most important feature by a large margin.

## Quick Start

```python
from sklearn.datasets import fetch_california_housing

cal = fetch_california_housing()
print(cal.feature_names)   # ['MedInc', 'HouseAge', 'AveRooms', ...]
print(cal.data.shape)      # (20640, 8)
```

No external data files needed — the dataset is bundled with scikit-learn.
