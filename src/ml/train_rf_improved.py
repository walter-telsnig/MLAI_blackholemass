"""
train_rf_improved.py

Author: Walter Telsnig
Date: 2025-07-06
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains an optimized Random Forest regression model with hyperparameter tuning
    to predict black hole masses (logBHMA_hb) from quasar spectral features
    including redshift, Eddington ratio, continuum luminosities, and line widths.
    Uses RandomizedSearchCV over key hyperparameters, out-of-bag scoring, and
    a large ensemble to maximize R².

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - joblib

Usage:
    python src/ml/train_rf_improved.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    - models/rf_improved_model.pkl
    - outputs/metrics/rf_improved_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import randint, uniform

def train_rf_improved(data_path, model_path, metrics_path):
    # Load and clean data
    df = pd.read_csv(data_path)
    df = df[
        (df["edd_ratio1"] > 0) &
        (df["l5100"] > 0) &
        (df["l3000"] > 0) &
        (df["fwhm1_hb"] > 0) &
        (df["virialfwhm_mgII"] > 0) &
        (df["logBHMA_hb"] > 0)
    ].copy()

    # Log-transform skewed features
    df["log_l5100"]        = np.log10(df["l5100"])
    df["log_l3000"]        = np.log10(df["l3000"])
    df["log_fwhm1_hb"]     = np.log10(df["fwhm1_hb"])
    df["log_virial_mgII"]  = np.log10(df["virialfwhm_mgII"])

    # Feature matrix and target
    feature_cols = [
        "redshift",
        "edd_ratio1",
        "log_l5100",
        "log_l3000",
        "log_fwhm1_hb",
        "log_virial_mgII"
    ]
    X = df[feature_cols].values
    y = df["logBHMA_hb"].values

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Base model with OOB scoring
    base_rf = RandomForestRegressor(
        oob_score=True,
        n_jobs=-1,
        random_state=42
    )

    # Hyperparameter distributions
    param_dist = {
        "n_estimators": randint(200, 800),
        "max_depth":    randint(5, 30),
        "min_samples_split": randint(2, 10),
        "min_samples_leaf":  randint(1, 5),
        "max_features":     ["auto", "sqrt", 0.5, 0.8]
    }

    # Randomized search
    rnd_search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=50,
        scoring="r2",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    rnd_search.fit(X_train, y_train)

    # Best estimator
    best_rf = rnd_search.best_estimator_

    # Evaluate on test set
    y_pred = best_rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({
            "MAE": mae,
            "R2": r2,
            "OOB_R2": best_rf.oob_score_,
            "Best_Params": rnd_search.best_params_
        }, f, indent=4)

    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_rf, model_path)

    print("✅ Improved RF trained.")
    print(f"Test MAE: {mae:.4f} | Test R²: {r2:.4f}")
    print(f"OOB R²: {best_rf.oob_score_:.4f}")
    print(f"Best params: {rnd_search.best_params_}")

if __name__ == "__main__":
    train_rf_improved(
        data_path="data/processed/spiders_quasar_features.csv",
        model_path="models/rf_improved_model.pkl",
        metrics_path="outputs/metrics/rf_improved_metrics.json"
    )
