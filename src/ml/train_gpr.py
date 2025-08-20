"""
train_gpr.py

Author: Walter Telsnig
Date: 2025-07-06
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains a Gaussian Process Regression (GPR) model to predict
    black hole masses (logBHMA_hb) from quasar spectral features
    including redshift, Eddington ratio, continuum luminosities, and line widths.
    Features are standardized to speed up convergence, and the optimizer
    is run without restarts for quick turnaround.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - joblib

Usage:
    python src/ml/train_gpr.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    - models/gpr_model.pkl
    - outputs/metrics/gpr_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def train_gpr_model(data_path, model_path, metrics_path):
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

    # Define GPR kernel
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)

    # Train GPR (no restarts for speed)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=0,
        alpha=0.1
    )
    gpr.fit(X_train, y_train)

    # Predict with uncertainty
    y_pred, y_std = gpr.predict(X_test, return_std=True)

    # Evaluate performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"MAE": mae, "R2": r2}, f, indent=4)

    # Save trained model (and scaler for future use)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": gpr, "scaler": scaler}, model_path)

    print(f"✅ GPR trained successfully.\nMAE: {mae:.4f} | R²: {r2:.4f}")

if __name__ == "__main__":
    train_gpr_model(
        data_path="data/processed/spiders_quasar_features.csv",
        model_path="models/gpr_model.pkl",
        metrics_path="outputs/metrics/gpr_metrics.json"
    )
