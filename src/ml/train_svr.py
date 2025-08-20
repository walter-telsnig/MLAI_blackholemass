"""
train_svr.py

Author: Walter Telsnig
Date: 2025-07-06
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains a Support Vector Regression (SVR) model to predict
    black hole masses (logBHMA_hb) from quasar spectral features
    including redshift, Eddington ratio, continuum luminosities, and line widths.
    Features are standardized before training.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - joblib

Usage:
    python src/ml/train_svr.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    - models/svr_model.pkl
    - outputs/metrics/svr_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_svr_model(data_path, model_path, metrics_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Select features and target
    feature_cols = [
        'redshift',
        'edd_ratio1',
        'l5100',
        'l3000',
        'fwhm1_hb',
        'virialfwhm_mgII'
    ]
    X = data[feature_cols].values
    y = data['logBHMA_hb'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features to mean=0, var=1
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    # Instantiate SVR
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    # Train SVR
    svr.fit(X_train, y_train)

    # Predict
    y_pred = svr.predict(X_test)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"MAE": mae, "R2": r2}, f, indent=4)

    # Save trained model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": svr, "scaler": scaler}, model_path)

    print(f"✅ SVR trained successfully.\nMAE: {mae:.4f} | R²: {r2:.4f}")

if __name__ == "__main__":
    train_svr_model(
        data_path="data/processed/spiders_quasar_features.csv",
        model_path="models/svr_model.pkl",
        metrics_path="outputs/metrics/svr_metrics.json"
    )
