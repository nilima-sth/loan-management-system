"""
eda_analysis_model4.py
Generates the Visual Dashboard for Model 4: Customer Segmentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading Segmented Customer Database...")
df = pd.read_csv("customer_segmentation/data/customer_segments_assigned.csv")

# Map the raw numbers to the business names for a beautiful chart
segment_map = {
    0: 'NRN/Mid-Tier',
    1: 'Dormant At-Risk',
    2: 'Active SME',
    3: 'QR Digital Youth',
    4: 'Standard Retail',
    5: 'High-Value Depositor',
    6: 'Micro Finance Rural'
}
df['Persona'] = df['assigned_segment'].map(segment_map)

# ==========================================
# GENERATE VISUAL CHARTS (WHITE BACKGROUND)
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('Model 4: K-Means Customer Behavioral Segmentation', fontsize=22, y=1.02, fontweight='bold', color='#333333')

# Standard Bank Palette
palette = sns.color_palette("Set2", 7)

# Chart A: Recency vs Frequency (The Activity Matrix)
sns.scatterplot(data=df, x='days_since_last_txn', y='txn_count_90d', hue='Persona', palette=palette, alpha=0.7, ax=axes[0, 0], s=60)
axes[0, 0].set_title('Engagement Matrix: Recency vs. Frequency', fontweight='bold', fontsize=14)
axes[0, 0].set_xlabel('Recency (Days Since Last Transaction)')
axes[0, 0].set_ylabel('Frequency (Transactions in Last 90 Days)')

# Chart B: Monetary Value vs Digital Adoption
sns.scatterplot(data=df, x='qr_txn_count_30d', y='avg_txn_amount', hue='Persona', palette=palette, alpha=0.7, ax=axes[0, 1], s=60)
axes[0, 1].set_title('Value Matrix: Digital Adoption vs. Avg Balance', fontweight='bold', fontsize=14)
axes[0, 1].set_xlabel('Digital Adoption (QR Scans per Month)')
axes[0, 1].set_ylabel('Monetary Value (Avg Txn Amount NPR)')
axes[0, 1].set_yscale('log') # Log scale because of High-Value Depositors

# Chart C: The Ghost Town (Isolating Dormant Accounts)
sns.boxplot(data=df, x='Persona', y='days_since_last_txn', palette=palette, ax=axes[1, 0])
axes[1, 0].set_title('Churn Risk (Identifying Dormant Accounts)', fontweight='bold', fontsize=14)
axes[1, 0].set_ylabel('Days Since Last Transaction')
axes[1, 0].tick_params(axis='x', rotation=45)

# Chart D: Segment Population Breakdown
counts = df['Persona'].value_counts(normalize=True) * 100
sns.barplot(x=counts.values, y=counts.index, palette=palette, ax=axes[1, 1])
axes[1, 1].set_title('Portfolio Distribution (% of Customer Base)', fontweight='bold', fontsize=14)
axes[1, 1].set_xlabel('Percentage of Total Portfolio (%)')

plt.tight_layout()
plt.savefig("customer_segmentation/data/model4_eda_visuals.png", bbox_inches='tight', facecolor='white')
print("✅ Visual dashboard saved to 'customer_segmentation/data/model4_eda_visuals.png'")