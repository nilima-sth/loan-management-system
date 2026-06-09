"""
eda_analysis_model3.py
Generates the Visual and Textual Exploratory Data Analysis
for the Model 3 Fraud/AML Dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading AML Transaction Ledger for EDA...")
df = pd.read_csv("fraud_detection_aml/data/aml_transactions.csv")

# ==========================================
# 1. GENERATE VISUAL CHARTS (WHITE BACKGROUND)
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model 3: Unsupervised AML Threat Detection EDA', fontsize=20, y=1.02, fontweight='bold', color='#333333')

# Palette: Gray for Normal, Red for Fraud
palette = {0: '#bdc3c7', 1: '#e74c3c'}

# Chart A: The Structuring Attack (Amount vs 30-Day Average)
sns.scatterplot(data=df, x='amount_log', y='amount_vs_30d_avg', hue='fraud_flag', palette=palette, alpha=0.6, ax=axes[0, 0])
axes[0, 0].set_title('Detecting "Structuring" (Deviations from 30-Day Avg)', fontweight='bold')
axes[0, 0].set_xlabel('Transaction Amount (Log Scale)')
axes[0, 0].set_ylabel('Multiplier of Personal 30-Day Avg')

# Chart B: The Midnight Outlier (Time of Day vs Amount)
sns.scatterplot(data=df, x='hour_of_day', y='amount_log', hue='fraud_flag', palette=palette, alpha=0.6, ax=axes[0, 1])
axes[0, 1].set_title('Temporal Threat Matrix (Midnight Outliers)', fontweight='bold')
axes[0, 1].set_xlabel('Hour of Day (0-23)')
axes[0, 1].set_ylabel('Transaction Amount (Log Scale)')
axes[0, 1].set_xticks(range(0, 24, 2))

# Chart C: The Velocity Attack (Account Takeover)
sns.boxplot(data=df, x='attack_type', y='txn_count_1h', palette="Set2", ax=axes[1, 0])
axes[1, 0].set_title('Velocity Attacks (Transactions in 1 Hour)', fontweight='bold')
axes[1, 0].set_ylabel('Transactions per Hour')
axes[1, 0].set_xlabel('Identified Behavior')

# Chart D: Feature Distribution: Amount vs Peers
sns.kdeplot(data=df, x='amount_vs_peers', hue='fraud_flag', fill=True, palette=palette, common_norm=False, ax=axes[1, 1])
axes[1, 1].set_title('Peer Group Deviation (Amount vs Similar Customers)', fontweight='bold')
axes[1, 1].set_xlabel('Multiplier vs Peer Group')
axes[1, 1].set_xlim(0, 20)

plt.tight_layout()
plt.savefig("fraud_detection_aml/data/model3_eda_visuals.png", bbox_inches='tight', facecolor='white')
print("✅ Visual charts saved to 'fraud_detection_aml/data/model3_eda_visuals.png'")

# ==========================================
# 2. PRINT TEXT REPORT
# ==========================================
print("\n" + "="*60)
print("📊 MODEL 3: AML THREAT VECTORS TEXT REPORT")
print("="*60)

print(f"Total Transactions Scanned: {len(df):,}")
print(f"Normal Activity (Gray)    : {len(df[df['fraud_flag']==0]):,}")
print(f"Injected Anomalies (Red)  : {len(df[df['fraud_flag']==1]):,}")

print("\n--- The 3 Threat Vectors Isolated ---")
attacks = df[df['fraud_flag'] == 1]

# Structuring
struct = attacks[attacks['attack_type'] == 'Structuring']
print(f"\n1. STRUCTURING (Smurfing the 1M NPR CTR Limit):")
print(f"   Count: {len(struct)}")
print(f"   Avg Amount: NPR {struct['amount'].mean():,.2f}")
print(f"   Avg Multiplier vs Normal 30D Avg: {struct['amount_vs_30d_avg'].mean():.2f}x")

# Velocity
vel = attacks[attacks['attack_type'] == 'Velocity']
print(f"\n2. VELOCITY ATTACKS (Account Takeovers):")
print(f"   Count: {len(vel)}")
print(f"   Avg Txn per Hour: {vel['txn_count_1h'].mean():.1f} txns/hr (Normal is ~{df[df['fraud_flag']==0]['txn_count_1h'].mean():.1f})")

# Midnight
mid = attacks[attacks['attack_type'] == 'Midnight_Outlier']
print(f"\n3. MIDNIGHT OUTLIERS (Temporal Anomalies):")
print(f"   Count: {len(mid)}")
print(f"   Avg Transfer Time: Hour {mid['hour_of_day'].mean():.1f} AM")
print(f"   Avg Amount: NPR {mid['amount'].mean():,.2f}")

print("="*60)