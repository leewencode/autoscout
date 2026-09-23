# AutoScout Vehicle Valuation & Automated Pricing Engine

## Project Overview
An end-to-end machine learning and data science pipeline built in Python to clean data, engineer features, and predict vehicle prices using the AutoScout dataset. This project systematically processes raw automotive data, handles missing values via multi-stage group imputation, handles categorical encodings, and compares multiple regression models (including XGBoost, LightGBM, and CatBoost) to achieve an $R^2$ score of ~0.96.

---

## Notebook Workflow
1. **Data Cleaning (`data_cleaning.ipynb`):** Ingests raw JSON data, parses strings, and cleans text attributes.
2. **Missing Value Analysis (`missing_value_analysis.ipynb`):** Implements multi-stage group imputation strategies.
3. **Outlier Analysis (`outlier_analysis.ipynb`):** Manages boxplot anomalies and statistical feature corrections.
4. **Data Encoding (`data_encoding.ipynb`):** Handles one-hot encoding and multi-level categorical parsing.
5. **Modeling (`modeling.ipynb`):** Tunes hyperparameters via GridSearchCV across multiple regressors, evaluates performance metrics (MSE, RMSE, MAE, $R^2$), and generates visual comparison plots.

---

## Tech Stack & Dependencies
- **Language:** Python
- **Environment Management:** `uv`
- **Core Libraries:**
  - `pandas`, `numpy` (Data manipulation & cleaning)
  - `scikit-learn`, `xgboost`, `lightgbm`, `catboost` (Regression models & machine learning)
  - `matplotlib`, `seaborn` (Data visualization & plotting)
  - `joblib` (Model serialization)

---

## Project Structure
```text
├── notebooks/          # Jupyter notebooks for cleaning, encoding, and modeling
├── pyproject.toml      # Local uv dependency configuration file
├── requirements.txt    # Cloud/Colab environment compatibility file
└── README.md           # Project documentation
