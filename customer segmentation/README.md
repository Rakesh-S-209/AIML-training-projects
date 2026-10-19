# Customer Segmentation — Unsupervised Clustering

A machine learning project that segments mall customers into distinct groups using **five unsupervised clustering algorithms**, compared side by side on the Mall Customers dataset.

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
| `Genre` | Male / Female |
| `Age` | Age of the customer |
| `Annual Income (k$)` | Annual income in thousands of dollars |
| `Spending Score (1-100)` | Score assigned by the mall based on spending behaviour |

> Clustering uses **Annual Income** and **Spending Score** — the two most discriminative features. Features are standardised with `StandardScaler`.

## Clustering Models

| Model | Type | Needs k? | Notes |
|---|---|---|---|
| K-Means | Centroid-based | Yes | Fast, assumes spherical clusters |
| DBSCAN | Density-based | No | Finds arbitrary shapes, labels noise as -1 |
| Agglomerative | Hierarchy-based | Yes | Builds a dendrogram bottom-up |
| Mean Shift | Density-based | No | Auto-detects number of clusters |
| Gaussian Mixture | Probabilistic | Yes | Soft cluster assignment with probabilities |

## Notebook

`customer-segmentation.ipynb` covers:

1. **Load & Explore** — head, shape, missing values, feature distributions, gender split
2. **Feature Selection & Scaling** — Annual Income + Spending Score, StandardScaler
3. **K-Means** — elbow method + silhouette score to pick k=5
4. **DBSCAN** — k-distance graph to tune `eps`, identifies noise points
5. **Agglomerative** — dendrogram (Ward linkage) to visualise cluster hierarchy
6. **Mean Shift** — auto-detects cluster count via bandwidth estimation
7. **GMM** — BIC/AIC to pick components, reports assignment confidence
8. **Side-by-Side Comparison** — scatter plots of all models + silhouette score bar chart

## Results

| Model | Clusters | Silhouette Score |
|---|---|---|
| **K-Means** | 5 | **0.5547** |
| Agglomerative | 5 | 0.5538 |
| GMM | 5 | 0.5537 |
| Mean Shift | 3 (auto) | 0.4641 |
| DBSCAN | 2 + 8 noise | 0.3876 |

> Silhouette score ranges from -1 to 1 — higher means better separated clusters.

### K-Means Customer Segments (k=5)

| Cluster | Annual Income | Spending Score | Profile |
|---|---|---|---|
| 1 | Low | High | Spenders on a budget |
| 2 | High | High | Top customers — high value |
| 3 | Medium | Medium | Average customers |
| 4 | High | Low | High earners, low spenders |
| 5 | Low | Low | Low income, low spend |

## Key Takeaways

- **K-Means, Agglomerative, and GMM** all converge on the same 5 natural segments with nearly identical silhouette scores (~0.55). Any of these is a solid choice.
- **Mean Shift** auto-detected 3 clusters, merging some of the 5 segments into broader groups.
- **DBSCAN** found only 2 dense clusters plus 8 noise points — it struggles here because the customer segments aren't separated by density gaps.
- **Cluster 2** (high income + high spending) is the most valuable segment for targeted marketing. **Cluster 4** (high income + low spending) is the hidden opportunity.

## Quick Start

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('Mall_Customers.csv')
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```
