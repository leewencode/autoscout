# AutoScout24 Vehicle Valuation & Automated Pricing Engine

## Project Overview
An end-to-end machine learning and data science pipeline built in Python to clean data, engineer features, and predict vehicle prices using the AutoScout24 dataset. It compares 14 regression models (including XGBoost, LightGBM, and CatBoost) and achieves an $R^2$ of 0.96.

## Notebook Workflow
1. **Data Cleaning (`data_cleaning.ipynb`):** Ingests raw JSON data, parses strings, and cleans text attributes.
2. **Missing Value Analysis (`missing_value_analysis.ipynb`):** Implements multi-stage group imputation strategies.
3. **Outlier Analysis (`outlier_analysis.ipynb`):** Manages boxplot anomalies and statistical feature corrections.
4. **Data Encoding (`data_encoding.ipynb`):** Handles one-hot encoding and multi-level categorical parsing.
5. **Modeling (`modeling.ipynb`):** Tunes hyperparameters via `GridSearchCV` across multiple regressors, evaluates performance metrics (MSE, RMSE, MAE, $R^2$), and generates visual comparison plots.
