"""
train_xgboost.py

Author: Your Name
Date: 2025-xx-xx
Project: Black Hole Mass Estimation with Machine Learning

Description:
    This script trains an XGBoost regression model to predict
    black hole masses (logBHMA_hb) using log-scaled features
    from the SPIDERS Quasar DR16 catalog. It filters invalid
    rows, log-transforms features, and prints model performance.

Required Libraries:
    - pandas
    - numpy
    - xgboost
    - scikit-learn

Usage:
    python train_xgboost.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    Console printout of MAE and R² score
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# resolve path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_in = os.path.join(base_dir, "data", "processed", "spiders_quasar_features.csv")

# load data
df = pd.read_csv(csv_in, encoding="latin1")

# clean
df = df[
    (df["edd_ratio1"] > 0) &
    (df["l5100"] > 0) &
    (df["fwhm1_hb"] > 0) &
    (df["logBHMA_hb"] > 0)
].copy()

# log-transform
df["log_l5100"] = np.log10(df["l5100"])
df["log_fwhm1_hb"] = np.log10(df["fwhm1_hb"])

# features
features = ["redshift", "edd_ratio1", "log_l5100", "log_fwhm1_hb"]
X = df[features]
y = df["logBHMA_hb"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# XGBoost regressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ XGBoost MAE: {mae:.3f}")
print(f"✅ XGBoost R²: {r2:.3f}")
