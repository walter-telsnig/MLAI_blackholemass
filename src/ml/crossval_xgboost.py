"""
crossval_xgboost.py

Author: Your Name
Date: 2025-xx-xx
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Performs k-fold cross-validation on an XGBoost regression model to predict
    black hole masses, and plots validation error across folds, plus a scatter
    plot of predicted vs. true black hole masses pooled over all CV folds.

Required Libraries:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scikit-learn
    - xgboost

Usage:
    python crossval_xgboost.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    Saves plots to outputs/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# resolve paths
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_in = os.path.join(base_dir, "data", "processed", "spiders_quasar_features.csv")
fig_dir = os.path.join(base_dir, "outputs", "figures")
os.makedirs(fig_dir, exist_ok=True)

# load
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
X = df[features].to_numpy()
y = df["logBHMA_hb"].to_numpy()

# cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mae_scores = []
all_true = []
all_pred = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

# plot boxplot of fold MAEs
plt.figure()
sns.boxplot(y=mae_scores)
plt.title("MAE across 5-fold Cross-Validation")
plt.ylabel("Mean Absolute Error (dex)")
plt.savefig(os.path.join(fig_dir, "cv_mae_boxplot.png"))
plt.close()

# plot all true vs. pred pooled
plt.figure()
sns.scatterplot(x=all_true, y=all_pred)
plt.xlabel("True log(BH mass)")
plt.ylabel("Predicted log(BH mass)")
plt.title("Cross-Validation: True vs. Predicted")
plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], "r--")
plt.savefig(os.path.join(fig_dir, "cv_true_vs_pred.png"))
plt.close()

print(f"✅ Cross-validation finished, plots saved in {fig_dir}")
