"""
tune_xgboost.py

Author: Your Name
Date: 2025-xx-xx
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Performs hyperparameter tuning of an XGBoost regressor using
    GridSearchCV to predict black hole masses (logBHMA_hb). Uses
    expanded feature set including both Hβ and Mg II line features.
    Reports the best hyperparameters and model performance.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - xgboost

Usage:
    python tune_xgboost.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    Console printout of best parameters and evaluation metrics
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# resolve paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_in = os.path.join(base_dir, "data", "processed", "spiders_quasar_features.csv")

# load data
df = pd.read_csv(csv_in, encoding="latin1")

# clean
df = df[
    (df["edd_ratio1"] > 0) &
    (df["l5100"] > 0) &
    (df["l3000"] > 0) &
    (df["fwhm1_hb"] > 0) &
    (df["virialfwhm_mgII"] > 0) &
    (df["logBHMA_hb"] > 0)
].copy()

# log-transform
df["log_l5100"] = np.log10(df["l5100"])
df["log_l3000"] = np.log10(df["l3000"])
df["log_fwhm1_hb"] = np.log10(df["fwhm1_hb"])
df["log_virialfwhm_mgII"] = np.log10(df["virialfwhm_mgII"])

# features
features = [
    "redshift",
    "edd_ratio1",
    "log_l5100",
    "log_l3000",
    "log_fwhm1_hb",
    "log_virialfwhm_mgII"
]
X = df[features]
y = df["logBHMA_hb"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# hyperparameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2]
}

# XGBoost with GridSearch
grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best CV score (neg MAE): {grid_search.best_score_:.3f}")
print(f"✅ Test MAE: {mae:.3f}")
print(f"✅ Test R²: {r2:.3f}")
