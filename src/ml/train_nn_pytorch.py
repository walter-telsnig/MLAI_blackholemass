#!/usr/bin/env python3
"""
train_nn_pytorch.py

Author: Walter Telsnig
Date: 2025-07-06
Project: Black Hole Mass Estimation with Machine Learning

Description:
    Trains a simple feedforward neural network using PyTorch to predict
    black hole masses (logBHMA_hb) from quasar spectral features
    including redshift, Eddington ratio, continuum luminosities, and line widths.
    Features are standardized before training.

Required Libraries:
    - pandas
    - numpy
    - scikit-learn
    - torch

Usage:
    python src/ml/train_nn_pytorch.py

Inputs:
    data/processed/spiders_quasar_features.csv

Outputs:
    - models/nn_model.pth
    - outputs/metrics/nn_metrics.json
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_nn_model(data_path, model_path, metrics_path, epochs=100, batch_size=32, lr=1e-3):
    # Load dataset
    df = pd.read_csv(data_path)

    # Select features and target
    feature_cols = [
        'redshift',
        'edd_ratio1',
        'l5100',
        'l3000',
        'fwhm1_hb',
        'virialfwhm_mgII'
    ]
    X = df[feature_cols].values
    y = df['logBHMA_hb'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = FeedforwardNN(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).squeeze().numpy()

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"MAE": mae, "R2": r2}, f, indent=4)

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler": scaler
    }, model_path)

    print(f"✅ NN trained successfully.\nMAE: {mae:.4f} | R²: {r2:.4f}")

if __name__ == "__main__":
    train_nn_model(
        data_path="data/processed/spiders_quasar_features.csv",
        model_path="models/nn_model.pth",
        metrics_path="outputs/metrics/nn_metrics.json",
        epochs=100,
        batch_size=32,
        lr=1e-3
    )

