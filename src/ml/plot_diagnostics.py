"""
plot_diagnostics.py

Author: Your Name
Date: 2025-xx-xx
Project: Black Hole Mass Estimation with Machine Learning

Description:
    This script loads the trained Random Forest model's predictions
    and the test dataset, and generates diagnostic plots:
    - true vs. predicted log black hole masses
    - residual histogram
    - feature distribution histograms

Required Libraries:
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn

Usage:
    python plot_diagnostics.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    Saves diagnostic plots to outputs/figures/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# resolve path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_in = os.path.join(base_dir, "data", "processed", "spiders_quasar_features.csv")
fig_dir = os.path.join(base_dir, "outputs", "figures")
os.makedirs(fig_dir, exist_ok=True)

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

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# re-train the same random forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# --------------------
# PLOT 1: predicted vs true
# --------------------
plt.figure()
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("True log(BH mass)")
plt.ylabel("Predicted log(BH mass)")
plt.title("True vs. Predicted Black Hole Mass")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.savefig(os.path.join(fig_dir, "true_vs_pred.png"))
plt.close()

# --------------------
# PLOT 2: residuals histogram
# --------------------
plt.figure()
sns.histplot(residuals, bins=30, kde=True)
plt.xlabel("Residual (True - Predicted)")
plt.title("Residual Distribution")
plt.savefig(os.path.join(fig_dir, "residuals_hist.png"))
plt.close()

# --------------------
# PLOT 3: feature distributions
# --------------------
for feature in features:
    plt.figure()
    sns.histplot(df[feature], bins=30, kde=True)
    plt.xlabel(feature)
    plt.title(f"Distribution of {feature}")
    plt.savefig(os.path.join(fig_dir, f"{feature}_distribution.png"))
    plt.close()

print(f"✅ Diagnostics saved in {fig_dir}")
