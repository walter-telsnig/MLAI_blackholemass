"""
train_nn_pytorch.py

Author: Your Name
Date: 2025-xx-xx
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains a simple feedforward neural network using PyTorch to predict
    black hole masses (logBHMA_hb) from an expanded feature set including
    both Hβ and Mg II lines.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - torch

Usage:
    python train_nn_pytorch.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    Prints MAE and R² on test data.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# resolve path
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
X = df[features].to_numpy(dtype=np.float32)
y = df["logBHMA_hb"].to_numpy(dtype=np.float32)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert to tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train).unsqueeze(1)
X_test_t  = torch.from_numpy(X_test)
y_test_t  = torch.from_numpy(y_test).unsqueeze(1)

# define neural net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = Net()

# loss and optimizer
criterion = nn.L1Loss()  # MAE
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {loss.item():.4f}")

# evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).squeeze().numpy()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ PyTorch NN Test MAE: {mae:.3f}")
print(f"✅ PyTorch NN Test R²: {r2:.3f}")
