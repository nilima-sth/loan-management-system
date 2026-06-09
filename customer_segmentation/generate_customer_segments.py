"""
generate_customer_segments.py
====================================
Customer Segmentation — Model 4 Data Pipeline
Generates `customer_rfm_data.csv`: A synthetic representation of 
10,000 bank customers with Recency, Frequency, Monetary (RFM), 
and digital banking features.

Engineered with 7 underlying Gaussian clusters to simulate real Nepalese
banking personas (NRN Remittance, Digital Youth, Rural Microfinance, etc.).
"""

import pandas as pd
import numpy as np
import os
import random

# ─────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────
OUTPUT_PATH = "customer_segmentation/data/customer_rfm_data.csv"
NUM_CUSTOMERS = 10000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("1. Initializing Odoo RFM & Customer Feature Simulation...")

def generate_segment_data(n, segment_name, params):
    """Generates a cluster of customers using normal distributions based on profile parameters."""
    df = pd.DataFrame({
        'days_since_last_txn': np.clip(np.random.normal(params['recency_mu'], params['recency_sd'], n), 0, 365).astype(int),
        'txn_count_90d': np.clip(np.random.normal(params['freq_mu'], params['freq_sd'], n), 0, 500).astype(int),
        'avg_txn_amount': np.clip(np.random.normal(params['monetary_mu'], params['monetary_sd'], n), 100, 5000000),
        'monthly_volume': np.clip(np.random.normal(params['vol_mu'], params['vol_sd'], n), 0, 10000000),
        'active_loan_count': np.clip(np.round(np.random.normal(params['loan_mu'], params['loan_sd'], n)), 0, 3).astype(int),
        'max_fd_balance': np.clip(np.random.normal(params['fd_mu'], params['fd_sd'], n), 0, 50000000),
        'qr_txn_count_30d': np.clip(np.random.normal(params['qr_mu'], params['qr_sd'], n), 0, 150).astype(int),
        'remittance_30d': np.clip(np.random.normal(params['rem_mu'], params['rem_sd'], n), 0, 1000000),
        'age': np.clip(np.random.normal(params['age_mu'], params['age_sd'], n), 18, 80).astype(int),
        # 0: Salaried, 1: Business, 2: Agriculture, 3: Student, 4: Foreign Employment
        'occupation_type': np.random.choice(params['occ_choices'], n, p=params['occ_probs']),
        'district_code': np.random.choice(range(1, 78), n), # 1 to 77 districts in Nepal
        'true_persona': segment_name # Hidden label for our own validation
    })
    return df

print("2. Engineering the 7 Core Nepalese Banking Personas...")

# 1. High-Value Depositor (11%) - Old money, big FDs, no QR
n_1 = int(NUM_CUSTOMERS * 0.11)
seg_1 = generate_segment_data(n_1, 'High-Value Depositor', {
    'recency_mu': 15, 'recency_sd': 10, 'freq_mu': 10, 'freq_sd': 5,
    'monetary_mu': 150000, 'monetary_sd': 50000, 'vol_mu': 500000, 'vol_sd': 200000,
    'loan_mu': 0.2, 'loan_sd': 0.4, 'fd_mu': 2500000, 'fd_sd': 1000000, # High FD
    'qr_mu': 1, 'qr_sd': 2, 'rem_mu': 0, 'rem_sd': 0, 'age_mu': 55, 'age_sd': 10,
    'occ_choices': [0, 1], 'occ_probs': [0.4, 0.6]
})

# 2. NRN Remittance Active (19%) - Heavy foreign inflows
n_2 = int(NUM_CUSTOMERS * 0.19)
seg_2 = generate_segment_data(n_2, 'NRN Remittance Active', {
    'recency_mu': 5, 'recency_sd': 3, 'freq_mu': 25, 'freq_sd': 10,
    'monetary_mu': 45000, 'monetary_sd': 15000, 'vol_mu': 150000, 'vol_sd': 50000,
    'loan_mu': 0.5, 'loan_sd': 0.5, 'fd_mu': 100000, 'fd_sd': 50000,
    'qr_mu': 5, 'qr_sd': 3, 'rem_mu': 85000, 'rem_sd': 20000, # Massive Remittance
    'age_mu': 35, 'age_sd': 8, 'occ_choices': [4, 0], 'occ_probs': [0.8, 0.2]
})

# 3. Active SME Borrower (21%) - High volume, business loans
n_3 = int(NUM_CUSTOMERS * 0.21)
seg_3 = generate_segment_data(n_3, 'Active SME Borrower', {
    'recency_mu': 2, 'recency_sd': 2, 'freq_mu': 120, 'freq_sd': 40, # High frequency
    'monetary_mu': 75000, 'monetary_sd': 25000, 'vol_mu': 2500000, 'vol_sd': 800000, # High volume
    'loan_mu': 1.8, 'loan_sd': 0.4, 'fd_mu': 50000, 'fd_sd': 20000, # Definite Loan
    'qr_mu': 40, 'qr_sd': 20, 'rem_mu': 0, 'rem_sd': 0, 'age_mu': 45, 'age_sd': 10,
    'occ_choices': [1], 'occ_probs': [1.0] # 100% Business
})

# 4. Micro Finance Rural (27%) - Agrictulture, low volume, low digital
n_4 = int(NUM_CUSTOMERS * 0.27)
seg_4 = generate_segment_data(n_4, 'Micro Finance Rural', {
    'recency_mu': 25, 'recency_sd': 15, 'freq_mu': 5, 'freq_sd': 3,
    'monetary_mu': 8000, 'monetary_sd': 3000, 'vol_mu': 15000, 'vol_sd': 5000,
    'loan_mu': 0.8, 'loan_sd': 0.4, 'fd_mu': 0, 'fd_sd': 0,
    'qr_mu': 0, 'qr_sd': 0.5, 'rem_mu': 5000, 'rem_sd': 5000, 'age_mu': 40, 'age_sd': 12,
    'occ_choices': [2], 'occ_probs': [1.0] # 100% Agriculture
})

# 5. QR Digital-First Youth (13%) - Mobile heavy, low balance, high velocity
n_5 = int(NUM_CUSTOMERS * 0.13)
seg_5 = generate_segment_data(n_5, 'QR Digital-First Youth', {
    'recency_mu': 1, 'recency_sd': 1, 'freq_mu': 90, 'freq_sd': 20,
    'monetary_mu': 1500, 'monetary_sd': 800, 'vol_mu': 45000, 'vol_sd': 15000,
    'loan_mu': 0, 'loan_sd': 0, 'fd_mu': 0, 'fd_sd': 0,
    'qr_mu': 60, 'qr_sd': 15, 'rem_mu': 0, 'rem_sd': 0, 'age_mu': 24, 'age_sd': 4, # Young, High QR
    'occ_choices': [3, 0], 'occ_probs': [0.7, 0.3] # Mostly students
})

# 6. Dormant At-Risk (9%) - Inactive for 3+ months
n_6 = int(NUM_CUSTOMERS * 0.09)
seg_6 = generate_segment_data(n_6, 'Dormant At-Risk', {
    'recency_mu': 180, 'recency_sd': 60, 'freq_mu': 0, 'freq_sd': 1, # Ghost town
    'monetary_mu': 500, 'monetary_sd': 500, 'vol_mu': 0, 'vol_sd': 500,
    'loan_mu': 0, 'loan_sd': 0, 'fd_mu': 0, 'fd_sd': 0,
    'qr_mu': 0, 'qr_sd': 0, 'rem_mu': 0, 'rem_sd': 0, 'age_mu': 38, 'age_sd': 15,
    'occ_choices': [0, 1, 2, 3], 'occ_probs': [0.25, 0.25, 0.25, 0.25]
})

# 7. High-Risk Borrower (Remainder ~10%) - Maxed out loans, struggling
n_7 = NUM_CUSTOMERS - (n_1 + n_2 + n_3 + n_4 + n_5 + n_6)
seg_7 = generate_segment_data(n_7, 'High-Risk Borrower', {
    'recency_mu': 45, 'recency_sd': 20, 'freq_mu': 8, 'freq_sd': 4,
    'monetary_mu': 15000, 'monetary_sd': 8000, 'vol_mu': 35000, 'vol_sd': 15000,
    'loan_mu': 2.5, 'loan_sd': 0.5, 'fd_mu': 0, 'fd_sd': 0, # Highly leveraged
    'qr_mu': 2, 'qr_sd': 2, 'rem_mu': 0, 'rem_sd': 0, 'age_mu': 42, 'age_sd': 10,
    'occ_choices': [0, 1], 'occ_probs': [0.5, 0.5]
})

print("3. Compiling and Shuffling Customer Database...")
final_df = pd.concat([seg_1, seg_2, seg_3, seg_4, seg_5, seg_6, seg_7], ignore_index=True)

# Add Primary Key
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
final_df.insert(0, 'customer_id', final_df.index + 1)

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*60)
print("🎯 MODEL 4: CUSTOMER SEGMENTATION DATASET COMPLETE")
print("="*60)
print(f"Total Customers Generated : {len(final_df):,}")
print("-" * 60)
print("Underlying Mathematical Clusters (Hidden Labels):")
print(final_df['true_persona'].value_counts().to_string())
print("="*60)
print(f"Saved to: {OUTPUT_PATH}")