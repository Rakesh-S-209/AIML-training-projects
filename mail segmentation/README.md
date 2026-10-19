# Customer Segmentation — K-Means Clustering

A machine learning project that segments mall customers into distinct groups based on their **Annual Income** and **Spending Score** using K-Means clustering.

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle – Customer Segmentation Tutorial](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) |
| File | `Mall_Customers.csv` |
| Rows | 200 |
| Features | 5 |

### Features

| Feature | Description |
|---|---|
| `CustomerID` | Unique customer identifier |
| `Gender` | Male / Female |
| `Age` | Age of the customer |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned by the mall based on spending behaviour |

## Notebook

`customer-segmentation.ipynb` covers:

1. **Load & Explore** — dataset head, shape, missing value check (none found)
2. **Feature Selection** — uses `Annual Income` and `Spending Score` for clustering
3. **Elbow Method** — plots WCSS for k=1 to 10 to find the optimal number of clusters
4. **K-Means Clustering** — fits with **k=5** (optimal from elbow), predicts cluster labels
5. **Visualisation** — scatter plot showing the 5 customer segments with centroids marked

## Results

The elbow method identifies **5 optimal clusters**, each representing a distinct customer segment:

| Cluster | Annual Income | Spending Score | Profile |
|---|---|---|---|
| 1 | Low | High | Spenders on a budget |
| 2 | High | High | Top customers — high value |
| 3 | Medium | Medium | Average customers |
| 4 | High | Low | High earners, low spenders |
| 5 | Low | Low | Low income, low spend |

## Key Takeaway

**Cluster 2** (high income, high spending) is the most valuable customer segment for targeted marketing. **Cluster 4** (high income, low spending) represents an opportunity — high earners who aren't spending much and could be engaged with better offers.

## Quick Start

```python
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
```
