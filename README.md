# Data Science Salary Analysis & Prediction

An end-to-end data analytics and machine learning project analyzing tech salary patterns from 3,755 real-world records and predicting annual compensation (USD) using a tuned Random Forest Regressor.

---

## Overview

- **Dataset**: 3,755 rows, 11 features (`ds_salaries.csv`, 2020–2023).
- **Goal**: Analyze key salary drivers (experience level, company location, remote ratio, year) and build an inference-ready salary prediction model.
- **Workflow**: EDA → Hierarchical Group-wise IQR Outlier Filtering → Feature Encoding → Random Forest Modeling → Serialization (`.pkl`).

---

## Dataset

| Feature | Type | Description |
|---|---|---|
| `work_year` | Numerical | Year the salary was paid (2020–2023) |
| `experience_level` | Categorical | `EN` (Entry), `MI` (Mid), `SE` (Senior), `EX` (Executive) |
| `employment_type` | Categorical | `FT` (Full-time), `PT` (Part-time), `CT` (Contract), `FL` (Freelance) |
| `job_title` | Categorical | Job title (e.g., Data Engineer, Data Scientist, Data Analyst) |
| `salary` | Numerical | Gross salary in local currency |
| `salary_currency` | Categorical | Currency code (e.g., USD, EUR, GBP) |
| `salary_in_usd` | Numerical | **Target**: Annual gross salary converted to USD |
| `employee_residence`| Categorical | Country code of employee residence |
| `remote_ratio` | Categorical | Work mode: `0` (On-site), `50` (Hybrid), `100` (Remote) |
| `company_location` | Categorical | Country code of company headquarters |
| `company_size` | Categorical | `S` (<50), `M` (50–250), `L` (>250 employees) |

---

## Key Insights

### 1. Compensation by Experience Level
| Level | Code | Share | Median (USD) | Mean (USD) |
|---|:---:|:---:|:---:|:---:|
| Entry-Level | `EN` | 8.5% | $70,000 | $78,546 |
| Mid-Level | `MI` | 21.4% | $100,000 | $104,526 |
| Senior-Level | `SE` | 67.0% | $146,000 | $153,051 |
| Executive | `EX` | 3.0% | $196,000 | $194,931 |

### 2. Yearly Growth (2020 – 2023)
- **2020**: Median $73,065
- **2021**: Median $80,000 (+9.5%)
- **2022**: Median $131,300 (+64.1%)
- **2023**: Median $143,860 (+9.6%)
- Overall median growth: **+96.9%** from 2020 to 2023.

### 3. Remote Work vs. On-site
- **On-site (0%)**: 1,923 records | Median: $139,600
- **Fully Remote (100%)**: 1,643 records | Median: $135,000
- **Hybrid (50%)**: 189 records | Median: $63,312

### 4. Top Job Roles
Data Engineer (1,040), Data Scientist (840), Data Analyst (612), Machine Learning Engineer (289), Analytics Engineer (103).

---

## Data Cleaning & Outlier Removal

A 3-stage grouped IQR filtering was applied on `salary_in_usd`:
1. By `employment_type`: 67 outliers removed (3,755 → 3,688)
2. By `company_size`: 15 outliers removed (3,688 → 3,673)
3. By `experience_level`: 15 outliers removed (3,673 → 3,658)

**Total**: 97 outliers dropped (2.58%), yielding **3,658 clean rows**.

---

## Modeling & Evaluation

- **Features Used**: `experience_level`, `work_year`, `company_location`, `employee_residence`, `salary_currency`
- **Target**: `salary_in_usd`
- **Encoding**: `LabelEncoder` per categorical column
- **Split**: 80% Train / 20% Test (Random State: 42)
- **Algorithm**: `RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=1, random_state=42)`

### Test Results

| Metric | Value |
|---|:---:|
| R² Score | **0.3948** |
| Root Mean Squared Error (RMSE) | **$44,215.78** |
| Mean Absolute Error (MAE) | **$35,006.70** |

---

## Model Artifacts

- [`random_forest_salary_model.pkl`](file:///Users/huynef/Downloads/Analytics-and-Predict-Data-Scientist-main/random_forest_salary_model.pkl): Trained Random Forest model.
- [`label_encoders.pkl`](file:///Users/huynef/Downloads/Analytics-and-Predict-Data-Scientist-main/label_encoders.pkl): Fitted encoders for categorical inputs.

---

## Quickstart

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn joblib jupyter matplotlib seaborn
```

### 2. Run Notebook
```bash
jupyter notebook data-science-salaries-analyse-and-predict.ipynb
```

---

## Inference Example

Use the trained model to predict compensation for a new profile:

```python
import joblib
import pandas as pd

# Load saved artifacts
model = joblib.load('random_forest_salary_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# Input data
new_sample = pd.DataFrame([{
    'experience_level': 'Senior',
    'work_year': 2023,
    'company_location': 'US',
    'employee_residence': 'US',
    'salary_currency': 'USD'
}])

# Encode and predict
for col in encoders:
    new_sample[col] = encoders[col].transform(new_sample[col])

pred_salary = model.predict(new_sample)[0]
print(f"Predicted Salary: ${pred_salary:,.2f} USD")
# Output: Predicted Salary: $163,890.89 USD
```

---

## Project Structure

```text
Analytics-and-Predict-Data-Scientist-main/
├── ds_salaries.csv                                 # Raw dataset
├── data-science-salaries-analyse-and-predict.ipynb # Analysis & ML notebook
├── random_forest_salary_model.pkl                  # Trained model
├── label_encoders.pkl                              # Preprocessing encoders
└── README.md                                       # Documentation
```

---

## License

MIT License.
