"""
train_aml_model.py
====================================
Fraud Detection & AML — Model 3 Training Pipeline
Strictly adheres to CEO Implementation Guide Sections 4.2 & 4.3

Models:
1. Isolation Forest (Scikit-Learn) -> Spatial Anomaly Detection
2. Deep Autoencoder (PyTorch)      -> Reconstruction Error Anomaly Detection
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("1. Loading AML Transaction Ledger...")
df = pd.read_csv("fraud_detection_aml/data/aml_transactions.csv")

# EXACT CEO Feature List (13 Features)
FRAUD_FEATURES = [
    'amount_log',           # log(transaction amount)
    'hour_of_day',          # 0-23 (night = higher risk)
    'day_of_week',          # 0=Mon, 6=Sun
    'days_since_last_txn',  # activity frequency
    'amount_vs_30d_avg',    # deviation from personal average
    'amount_vs_peers',      # vs similar customers
    'txn_count_1h',         # velocity: transactions per hour
    'txn_count_24h',        # velocity: transactions per day
    'amount_sum_24h',       # total value in 24 hours
    'unique_merchants_7d',  # merchant diversity
    'channel_encoded',      # ATM=0, branch=1, mobile=2, online=3
    'location_change_flag', # 1 if location differs from usual
    'amount_round_flag',    # 1 if round number (structuring signal)
]

# ---------------------------------------------------------
# SECTION 4.2: ISOLATION FOREST TRAINING
# ---------------------------------------------------------
print("2. Preparing Data (Training on NORMAL transactions only)...")
# Isolate only normal transactions (fraud_flag == 0) for training
df_normal = df[df['fraud_flag'] == 0].copy()

X_train = df_normal[FRAUD_FEATURES].fillna(0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("3. Training Isolation Forest (CEO Spec 4.2)...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.005,   # Expect ~0.5% fraud rate
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

# Calculate dynamic threshold (bottom 1% of scores)
train_scores = iso_forest.score_samples(X_train_scaled)
if_threshold = np.percentile(train_scores, 1)
print(f"   -> Isolation Forest Threshold Score: {if_threshold:.4f}")

# ---------------------------------------------------------
# SECTION 4.3: DEEP LEARNING AUTOENCODER (PYTORCH)
# ---------------------------------------------------------
print("\n4. Training PyTorch Autoencoder (CEO Spec 4.3)...")

class TransactionAutoencoder(nn.Module):
    '''Learns to reconstruct normal transactions.
       High reconstruction error = anomaly (potential fraud).'''
    def __init__(self, input_dim=13):
        super().__init__()
        # Encoder: compress normal transaction pattern
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)   # bottleneck
        )
        # Decoder: reconstruct from compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x):
        x_hat = self.forward(x)
        return ((x - x_hat)**2).mean(dim=1)  # MSE per sample

# Prepare PyTorch Tensors and DataLoader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
dataset = TensorDataset(X_train_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

autoencoder = TransactionAutoencoder(input_dim=len(FRAUD_FEATURES))
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training Loop
epochs = 50
print(f"   -> Running {epochs} epochs...")
for epoch in range(epochs):
    epoch_loss = 0
    for batch in dataloader:
        batch_x = batch[0]
        recon = autoencoder(batch_x)
        loss = criterion(recon, batch_x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

# Set threshold at 95th percentile of training reconstruction error
with torch.no_grad():
    train_errors = autoencoder.reconstruction_error(X_train_tensor).numpy()
ae_threshold = np.percentile(train_errors, 95)
print(f"   -> Autoencoder Fraud Threshold (MSE): {ae_threshold:.6f}")

# ---------------------------------------------------------
# SAVING ARTIFACTS FOR LAYER 4.4 (REAL-TIME PIPELINE)
# ---------------------------------------------------------
os.makedirs("fraud_detection_aml/models", exist_ok=True)
joblib.dump(iso_forest, "fraud_detection_aml/models/iso_forest.pkl")
joblib.dump(scaler, "fraud_detection_aml/models/scaler.pkl")
torch.save(autoencoder.state_dict(), "fraud_detection_aml/models/autoencoder.pth")
print("\n✅ All models and scalers saved to 'fraud_detection_aml/models/'")

# ---------------------------------------------------------
# 5. THE PROOF: EVALUATING THE "RED TEAM" ATTACKS
# ---------------------------------------------------------
print("\n" + "="*60)
print("🚨 BLIND AUDIT: SCORING THE INJECTED 'RED TEAM' ATTACKS")
print("="*60)

# Extract ONLY the fraud cases to see if the AI catches them
df_fraud = df[df['fraud_flag'] == 1].copy()
X_fraud = df_fraud[FRAUD_FEATURES].fillna(0)
X_fraud_scaled = scaler.transform(X_fraud)

# Score Isolation Forest
fraud_if_scores = iso_forest.score_samples(X_fraud_scaled)
df_fraud['if_flagged'] = fraud_if_scores < if_threshold

# Score Autoencoder
X_fraud_tensor = torch.tensor(X_fraud_scaled, dtype=torch.float32)
with torch.no_grad():
    fraud_ae_errors = autoencoder.reconstruction_error(X_fraud_tensor).numpy()
df_fraud['ae_flagged'] = fraud_ae_errors > ae_threshold

# Calculate Capture Rates
if_capture_rate = df_fraud['if_flagged'].mean() * 100
ae_capture_rate = df_fraud['ae_flagged'].mean() * 100
ensemble_captured = ((df_fraud['if_flagged']) | (df_fraud['ae_flagged'])).sum()
total_fraud = len(df_fraud)

print(f"Total Hidden Attacks           : {total_fraud}")
print(f"Caught by Isolation Forest     : {df_fraud['if_flagged'].sum()} ({if_capture_rate:.1f}%)")
print(f"Caught by PyTorch Autoencoder  : {df_fraud['ae_flagged'].sum()} ({ae_capture_rate:.1f}%)")
print(f"HYBRID ENSEMBLE CAPTURE RATE   : {ensemble_captured} / {total_fraud} ({(ensemble_captured/total_fraud)*100:.1f}%)")
print("="*60)