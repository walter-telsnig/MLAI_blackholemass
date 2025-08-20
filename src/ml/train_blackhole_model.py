"""
train_blackhole_model.py

Author: Walter Telsnig
Date: 2025-07-06
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains a Random Forest regression model to predict
    black hole masses (logBHMA_hb) from quasar spectral features
    including redshift, Eddington ratio, and continuum luminosity.
    Wide–scale features are log-transformed before training.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - joblib

Usage:
    python src/ml/train_blackhole_model.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    - models/rf_model.pkl
    - outputs/metrics/rf_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_rf_model(data_path, model_path, metrics_path):
    # Load and clean
    df = pd.read_csv(data_path, encoding="latin1")
    df = df[
        (df["edd_ratio1"] > 0) &
        (df["l5100"] > 0) &
        (df["fwhm1_hb"] > 0) &
        (df["logBHMA_hb"] > 0)
    ].copy()

    # Log-transform
    df["log_l5100"] = np.log10(df["l5100"])
    df["log_fwhm1_hb"] = np.log10(df["fwhm1_hb"])

    # Features & target
    feature_cols = ["redshift", "edd_ratio1", "log_l5100", "log_fwhm1_hb"]
    X = df[feature_cols].values
    y = df["logBHMA_hb"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = RandomForestRegressor(n_estimators=100, random_state=42)
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

    print(f"✅ Random Forest trained successfully.\nMAE: {mae:.4f} | R²: {r2:.4f}")

if __name__ == "__main__":
    train_rf_model(
        data_path="data/processed/spiders_quasar_features.csv",
        model_path="models/rf_model.pkl",
        metrics_path="outputs/metrics/rf_metrics.json"
    )
