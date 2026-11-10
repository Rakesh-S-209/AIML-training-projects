# Employee Salary Prediction (Adult Income)

Predict whether an individual's annual income exceeds $50,000 based on census data.

---

## Overview

| Property | Description |
|----------|-------------|
| **Source** | [UCI Adult Income Dataset](https://archive.ics.uci.edu/dataset/2/adult) |
| **Total samples** | 48,842 |
| **Features** | 13 (after preprocessing) |
| **Target** | `income` — `<=50K` or `>50K` (binary classification) |

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `age` | Integer | Person's age |
| `workclass` | Category | Employment type (Private, Self-emp, Local-gov, etc.) |
| `fnlwgt` | Integer | Final weight (sampling weight) |
| `educational-num` | Integer | Years of education |
| `marital-status` | Category | Marital status |
| `occupation` | Category | Job type |
| `relationship` | Category | Family relationship |
| `race` | Category | Race |
| `gender` | Binary | Male/Female |
| `capital-gain` | Float | Capital gains |
| `capital-loss` | Float | Capital losses |
| `hours-per-week` | Integer | Hours worked per week |
| `native-country` | Category | Country of origin |

---

## Preprocessing

1. Replace `?` (missing values) with `Not-listed` in `workclass`, `occupation`, `native-country`
2. Drop redundant `education` column (already captured by `educational-num`)
3. Encode categorical features using `LabelEncoder`
4. Train-test split (standard 80/20)

---

## Models

The notebook compares multiple classifiers:
- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **K-Nearest Neighbors**
- **Support Vector Classifier**

---

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

---

## License

Dataset is available under the UCI Data License.
