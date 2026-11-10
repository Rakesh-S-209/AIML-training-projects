# Telco Customer Churn Dataset

A classic binary classification dataset from IBM that predicts whether a telecommunications customer will leave (churn) within the next month.

---

## Overview

| Property | Description |
|----------|-------------|
| **Source** | [IBM Watson Analytics](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) |
| **Total samples** | 7,043 customers |
| **Features** | 19 (excluding `customerID` and `Churn`) |
| **Target** | `Churn` — `Yes` or `No` |
| **Data type** | Mixed (categorical + numerical) |

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | String | Unique customer identifier (not used for training) |
| `gender` | Category | Customer's gender: *Female, Male* |
| `SeniorCitizen` | Integer | Whether the customer is 65+: *0 = No, 1 = Yes* |
| `Partner` | Category | Has a partner: *Yes, No* |
| `Dependents` | Category | Has dependents: *Yes, No* |
| `tenure` | Integer | Number of months the customer has stayed |
| `PhoneService` | Category | Has phone service: *Yes, No* |
| `MultipleLines` | Category | Multiple lines: *Yes, No, No phone service* |
| `InternetService` | Category | ISP: *DSL, Fiber optic, No* |
| `OnlineSecurity` | Category | Online security: *Yes, No, No internet service* |
| `OnlineBackup` | Category | Online backup: *Yes, No, No internet service* |
| `DeviceProtection` | Category | Device protection: *Yes, No, No internet service* |
| `TechSupport` | Category | Tech support: *Yes, No, No internet service* |
| `StreamingTV` | Category | Streaming TV: *Yes, No, No internet service* |
| `StreamingMovies` | Category | Streaming movies: *Yes, No, No internet service* |
| `Contract` | Category | Contract type: *Month-to-month, One year, Two year* |
| `PaperlessBilling` | Category | Paperless billing: *Yes, No* |
| `PaymentMethod` | Category | Payment method: *Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)* |
| `MonthlyCharges` | Float | Monthly charge (USD) |
| `TotalCharges` | Float | Total charges (USD) — note: contains 11 blank values |
| `Churn` | Target | Churned? *Yes, No* |

---

## Key Statistics

- **Churn rate**: ~26.6% (1,869 churned out of 7,043)
- **Tenure range**: 1–72 months
- **Monthly charges range**: $18.25–$118.75
- **High correlation**: Shorter tenure, month-to-month contracts, electronic check payments, and fiber optic service tend to correlate with higher churn.

---

## Preprocessing Notes

1. **Missing values**: `TotalCharges` has 11 empty cells (`" "`) — rows should be imputed (median strategy used in the notebook).
2. **Drop `customerID`**: This identifier has no predictive power.
3. **Encode target**: `Churn` → `Yes=1`, `No=0`.
4. **One-Hot Encode categorical columns**: All object-type columns get binary encodings (with `drop_first=True` to avoid dummy variable trap).
5. **Processed shape**: 7,043 rows × 31 columns after encoding.

---

## Models

### Logistic Regression & Random Forest (Baseline — **failed on raw data**)

These sklearn models require numeric inputs and failed on raw string columns.

```
ValueError: could not convert string to float: '7590-VHVEG'
```

### CatBoost (Native categorical support — **works without encoding**)

- **Iterations**: 100  
- **Learning rate**: 0.1  
- **Depth**: 6  
- **Accuracy**: ~81.26% on 20% test split (unprocessed data)
- ** Advantage**: Accepts `cat_features` list directly; handles string columns natively.

### Standard Preprocessing + sklearn Models (After One-Hot Encoding)

The notebook also demonstrates full preprocessing:
- Drop `customerID`
- Impute `TotalCharges` (median)
- Encode `Churn` (binary)
- One-Hot encode remaining categorical columns (`drop_first=True`)
- Train/test split (80/20, random_state=42)

After this, models like Logistic Regression, RandomForest, XGBoost, etc., can be applied.

---

## License

This dataset is provided for educational and research purposes under the IBM Community Dataset License. Use it responsibly and attribute the original source when publishing.

---

## References

- IBM Watson Analytics Community Blog, *Telco Customer Churn* (2019).
- Kaggle: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (original mirror).
