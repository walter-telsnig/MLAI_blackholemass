"""
train_xgboost_extended.py

Author: Walter Telsnig
Date: 2025-07-06
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains an XGBoost regression model to predict black hole masses
    (logBHMA_hb) using expanded feature sets including both Hβ and Mg II
    properties. Filters invalid values and applies log-transforms.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - xgboost
    - joblib

Usage:
    python src/ml/train_xgboost_extended.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    - models/xgb_model.pkl
    - outputs/metrics/xgb_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

def train_xgb_model(data_path, model_path, metrics_path):
    # Load and clean
    df = pd.read_csv(data_path, encoding="latin1")
    df = df[
        (df["edd_ratio1"] > 0) &
        (df["l5100"] > 0) &
        (df["l3000"] > 0) &
        (df["fwhm1_hb"] > 0) &
        (df["virialfwhm_mgII"] > 0) &
        (df["logBHMA_hb"] > 0)
    ].copy()

    # Log-transform
    df["log_l5100"] = np.log10(df["l5100"])
    df["log_l3000"] = np.log10(df["l3000"])
    df["log_fwhm1_hb"] = np.log10(df["fwhm1_hb"])
    df["log_virialfwhm_mgII"] = np.log10(df["virialfwhm_mgII"])

    # Features & target
    feature_cols = [
        "redshift",
        "edd_ratio1",
        "log_l5100",
        "log_l3000",
        "log_fwhm1_hb",
        "log_virialfwhm_mgII"
    ]
    X = df[feature_cols].values
    y = df["logBHMA_hb"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"MAE": mae, "R2": r2}, f, indent=4)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"✅ XGBoost trained successfully.\nMAE: {mae:.4f} | R²: {r2:.4f}")

if __name__ == "__main__":
    train_xgb_model(
        data_path="data/processed/spiders_quasar_features.csv",
        model_path="models/xgb_model.pkl",
        metrics_path="outputs/metrics/xgb_metrics.json"
    )
