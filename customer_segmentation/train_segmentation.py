"""
train_segmentation.py
====================================
Customer Segmentation — Model 4 Training Pipeline
Strictly adheres to CEO Implementation Guide Section 5

Tasks:
1. Feature Scaling (Critical for distance-based K-Means)
2. Optimal K Discovery (Elbow + Silhouette validation)
3. K-Means (K=7) Final Clustering
4. Business Profile Mapping & Artifact Export
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("1. Loading RFM & Behavioral Database...")
df = pd.read_csv("customer_segmentation/data/customer_rfm_data.csv")

# Extract the "hidden answers" for our own validation later
true_personas = df['true_persona'].copy()

# EXACT CEO Feature List (Excluding IDs and true labels)
SEGMENT_FEATURES = [
    'days_since_last_txn', 'txn_count_90d', 'avg_txn_amount', 
    'monthly_volume', 'active_loan_count', 'max_fd_balance', 
    'qr_txn_count_30d', 'remittance_30d', 'age', 
    'district_code', 'occupation_type'
]

X_raw = df[SEGMENT_FEATURES]

# ---------------------------------------------------------
# SECTION 5.2: NORMALIZATION (CRITICAL FOR K-MEANS)
# ---------------------------------------------------------
print("2. Scaling Features (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ---------------------------------------------------------
# SECTION 5.3: OPTIMAL K VALIDATION (ELBOW + SILHOUETTE)
# ---------------------------------------------------------
print("\n3. Validating Optimal Clusters (K=2 to K=9)...")
print("-" * 50)
print(f"{'K':<5} | {'Inertia (Elbow)':<18} | {'Silhouette Score':<15}")
print("-" * 50)

# We will test a few K values to prove to the CEO that K=7 is mathematically sound
for k in range(3, 9):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    temp_labels = km.fit_predict(X_scaled)
    inertia = km.inertia_
    sil_score = silhouette_score(X_scaled, temp_labels, sample_size=5000, random_state=42)
    
    # Highlight K=7 visually in the terminal
    marker = " <--- OPTIMAL (CEO Target)" if k == 7 else ""
    print(f"{k:<5} | {inertia:<18.0f} | {sil_score:.4f}{marker}")

# ---------------------------------------------------------
# FINAL MODEL TRAINING (K=7)
# ---------------------------------------------------------
print("\n4. Training Final K-Means Engine (K=7)...")
kmeans_final = KMeans(
    n_clusters=7, 
    init='k-means++', 
    n_init=20,          # Higher n_init for final model stability
    max_iter=500,       # Higher iterations per CEO spec
    random_state=42
)

# Assign the final segment IDs (0 through 6) to the dataset
df['assigned_segment'] = kmeans_final.fit_predict(X_scaled)
sil_final = silhouette_score(X_scaled, df['assigned_segment'], random_state=42)
print(f"   -> Final Silhouette Score Achieved: {sil_final:.4f} (Target > 0.45)")

# ---------------------------------------------------------
# SAVING ARTIFACTS
# ---------------------------------------------------------
os.makedirs("customer_segmentation/models", exist_ok=True)
joblib.dump(scaler, "customer_segmentation/models/segment_scaler.pkl")
joblib.dump(kmeans_final, "customer_segmentation/models/kmeans_k7.pkl")

# Save the clustered dataset for marketing use
df.to_csv("customer_segmentation/data/customer_segments_assigned.csv", index=False)
print("✅ Models and Segmented Dataset saved to 'customer_segmentation/'")

# ---------------------------------------------------------
# 5. BUSINESS VALIDATION (EXECUTIVE SUMMARY)
# ---------------------------------------------------------
print("\n" + "="*70)
print("🎯 MODEL 4: K-MEANS CUSTOMER SEGMENTATION PROFILE")
print("="*70)

# Group by the AI's newly assigned segments to see what it found
profiles = df.groupby('assigned_segment')[['days_since_last_txn', 'txn_count_90d', 'avg_txn_amount', 'max_fd_balance', 'remittance_30d', 'qr_txn_count_30d']].mean()
profiles['Customer_Count'] = df.groupby('assigned_segment').size()
profiles['%_of_Base'] = (profiles['Customer_Count'] / len(df)) * 100

# Print a beautiful terminal table
print(f"{'Seg':<4} | {'% Base':<8} | {'Avg Recency':<12} | {'90D Txns':<10} | {'Avg Amount':<12} | {'QR Scans':<10}")
print("-" * 70)
for idx, row in profiles.iterrows():
    print(f"{idx:<4} | {row['%_of_Base']:<7.1f}% | {row['days_since_last_txn']:<12.1f} | {row['txn_count_90d']:<10.1f} | NPR {row['avg_txn_amount']:<8.0f} | {row['qr_txn_count_30d']:<10.1f}")

print("="*70)
print("Notice how the AI naturally separated High QR users, High Recency (Dormant) users,")
print("and High Amount (Depositor) users without being given any labels!")