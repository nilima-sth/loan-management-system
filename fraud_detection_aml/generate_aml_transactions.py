"""
generate_aml_transactions.py
====================================
Fraud Detection & AML — Model 3 Data Pipeline
Simulates a live Odoo `account_move_line` ledger with engineered
features ready for Isolation Forest and Autoencoder ingestion.

Injects 3 specific attack vectors:
  1. Structuring (Smurfing just under 1M NPR CTR limits)
  2. Velocity Attacks (Account Takeover / rapid drain)
  3. Spatial/Temporal Outliers (3 AM massive transfers)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ─────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────
OUTPUT_PATH = "fraud_detection_aml/data/aml_transactions.csv"
NUM_NORMAL_TXNS = 10000
NUM_FRAUD_TXNS = 50   # 0.5% contamination to match CEO guide
NUM_CUSTOMERS = 500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print("1. Initializing Odoo Transaction Ledger Simulation...")

# ─────────────────────────────────────────────────
# 1. GENERATE NORMAL TRANSACTIONS
# ─────────────────────────────────────────────────
def generate_normal_transactions(n_txns):
    data = []
    start_date = datetime(2025, 1, 1)
    
    for _ in range(n_txns):
        # Normal behavior is mostly daytime, standard channels, smaller amounts
        customer_id = random.randint(1, NUM_CUSTOMERS)
        amount = round(np.random.lognormal(mean=7.5, sigma=1.2), 2) # Heavily skewed to smaller amounts (NPR 500 - 50k)
        
        # Simulate daytime hours (8 AM to 8 PM mostly)
        hour = int(np.random.normal(14, 4)) % 24
        day_of_week = random.randint(0, 6)
        
        # Normal velocity and history
        txn_count_1h = random.randint(0, 1)
        txn_count_24h = random.randint(1, 3)
        amount_sum_24h = amount + random.uniform(0, 5000)
        
        # Channel (0=ATM, 1=Branch, 2=Mobile, 3=Online)
        channel = np.random.choice([0, 1, 2, 3], p=[0.2, 0.1, 0.6, 0.1])
        
        # Round flag logic (normal people rarely deposit exactly 50,000 unless it's ATM)
        round_flag = 1 if amount % 1000 == 0 and channel in [0, 1] else 0
        
        data.append({
            'customer_id': customer_id,
            'amount': amount,
            'amount_log': np.log1p(amount),
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'days_since_last_txn': round(random.uniform(0.5, 15.0), 1),
            'amount_vs_30d_avg': round(random.uniform(0.5, 1.5), 2),
            'amount_vs_peers': round(random.uniform(0.5, 2.0), 2),
            'txn_count_1h': txn_count_1h,
            'txn_count_24h': txn_count_24h,
            'amount_sum_24h': amount_sum_24h,
            'unique_merchants_7d': random.randint(1, 5),
            'channel_encoded': channel,
            'location_change_flag': np.random.choice([0, 1], p=[0.95, 0.05]),
            'amount_round_flag': round_flag,
            'fraud_flag': 0,
            'attack_type': 'None'
        })
    return data

# ─────────────────────────────────────────────────
# 2. INJECT CYBER & AML ATTACKS
# ─────────────────────────────────────────────────
def inject_fraud_attacks(n_fraud):
    fraud_data = []
    
    for i in range(n_fraud):
        attack_choice = random.choice(['Structuring', 'Velocity', 'Midnight_Outlier'])
        
        if attack_choice == 'Structuring':
            # Smurfing just under NRB 1,000,000 limit
            amount = random.choice([990000, 995000, 999000])
            fraud_data.append({
                'amount': amount,
                'amount_log': np.log1p(amount),
                'hour_of_day': random.randint(10, 15),
                'day_of_week': random.randint(0, 4),
                'days_since_last_txn': random.uniform(0.1, 1.0),
                'amount_vs_30d_avg': random.uniform(5.0, 10.0), # 10x higher than normal
                'amount_vs_peers': random.uniform(8.0, 15.0),
                'txn_count_1h': random.randint(2, 4),
                'txn_count_24h': random.randint(3, 6),
                'amount_sum_24h': amount * random.randint(2, 4),
                'unique_merchants_7d': random.randint(1, 2),
                'channel_encoded': 1, # Usually done at Branch
                'location_change_flag': 0,
                'amount_round_flag': 1,
                'fraud_flag': 1,
                'attack_type': 'Structuring'
            })
            
        elif attack_choice == 'Velocity':
            # Account Takeover (Rapid drain via mobile)
            amount = random.uniform(50000, 200000)
            fraud_data.append({
                'amount': amount,
                'amount_log': np.log1p(amount),
                'hour_of_day': random.randint(0, 23),
                'day_of_week': random.randint(0, 6),
                'days_since_last_txn': 0.0, # Rapid fire
                'amount_vs_30d_avg': random.uniform(1.0, 3.0),
                'amount_vs_peers': random.uniform(1.0, 3.0),
                'txn_count_1h': random.randint(8, 15), # IMPOSSIBLE normally
                'txn_count_24h': random.randint(10, 20),
                'amount_sum_24h': amount * 10,
                'unique_merchants_7d': random.randint(8, 15), # Hitting multiple targets
                'channel_encoded': 2, # Mobile
                'location_change_flag': 1, # Hacker from different location
                'amount_round_flag': 0,
                'fraud_flag': 1,
                'attack_type': 'Velocity'
            })
            
        elif attack_choice == 'Midnight_Outlier':
            # 3 AM massive wire transfer
            amount = random.uniform(1500000, 5000000)
            fraud_data.append({
                'amount': amount,
                'amount_log': np.log1p(amount),
                'hour_of_day': random.randint(2, 4), # Dead of night
                'day_of_week': random.randint(0, 6),
                'days_since_last_txn': random.uniform(10.0, 30.0),
                'amount_vs_30d_avg': random.uniform(20.0, 50.0), # Massive spike
                'amount_vs_peers': random.uniform(20.0, 50.0),
                'txn_count_1h': 1,
                'txn_count_24h': 1,
                'amount_sum_24h': amount,
                'unique_merchants_7d': 1,
                'channel_encoded': 3, # Online Banking
                'location_change_flag': 1, # Different IP
                'amount_round_flag': 1,
                'fraud_flag': 1,
                'attack_type': 'Midnight_Outlier'
            })
            
        fraud_data[-1]['customer_id'] = random.randint(1, NUM_CUSTOMERS)
            
    return fraud_data

# ─────────────────────────────────────────────────
# 3. COMPILE AND EXPORT
# ─────────────────────────────────────────────────
print("2. Generating Normal Transactions...")
normal_df = pd.DataFrame(generate_normal_transactions(NUM_NORMAL_TXNS))

print("3. Injecting Structuring, Velocity, and Midnight Attacks...")
fraud_df = pd.DataFrame(inject_fraud_attacks(NUM_FRAUD_TXNS))

# Combine and shuffle so the fraud is hidden
final_df = pd.concat([normal_df, fraud_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Ensure directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
final_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*60)
print("🎯 MODEL 3: AML DATASET GENERATION COMPLETE")
print("="*60)
print(f"Total Transactions  : {len(final_df):,}")
print(f"Normal Transactions : {len(normal_df):,} (Class 0)")
print(f"Injected Attacks    : {len(fraud_df):,} (Class 1)")
print("-" * 60)
print("Attack Distribution Breakdown:")
print(fraud_df['attack_type'].value_counts().to_string())
print("="*60)
print(f"Saved to: {OUTPUT_PATH}")