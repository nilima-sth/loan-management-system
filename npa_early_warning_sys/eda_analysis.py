import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Loading Databases for EDA...")
df_master = pd.read_csv("credit_scoring_engine/data/lms_advanced_master.csv")
df_master['loan_account_id'] = df_master.index + 1
df_schedule = pd.read_csv("npa_early_warning_sys/data/lms_repayment_schedule.csv")

# Merge target variable for analysis so we can compare Good vs Bad loans
df = df_schedule.merge(df_master[['loan_account_id', 'defaulted']], on='loan_account_id')

# ==========================================
# 1. GENERATE VISUAL CHARTS (WHITE BACKGROUND)
# ==========================================
# Set a clean, professional white grid theme for corporate reports
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model 2: Live Time-Series & Behavioral EDA', fontsize=20, y=1.02, fontweight='bold', color='#333333')

# Define standard bank colors (Safe = Green, Risk = Red)
bank_palette = ['#2ecc71', '#e74c3c']

# Chart A: Average Payment Ratio by Class
sns.histplot(data=df, x='payment_ratio', hue='defaulted', bins=50, palette=bank_palette, ax=axes[0, 0], stat='density', common_norm=False)
axes[0, 0].set_title('Overall Payment Ratio Distribution', fontweight='bold')
axes[0, 0].set_xlim(0, 1.2)

# Chart B: Missed Payments Comparison
missed_rates = df.groupby('defaulted')['missed'].mean() * 100
sns.barplot(x=missed_rates.index, y=missed_rates.values, palette=bank_palette, ax=axes[0, 1])
axes[0, 1].set_title('Percentage of Missed Months by Class', fontweight='bold')
axes[0, 1].set_xticklabels(['Stable (0)', 'Impending NPA (1)'])
axes[0, 1].set_ylabel('% Missed')

# Chart C: The "Falling Slope"
print("Calculating rolling trends for visualization (this takes a few seconds)...")
def get_trend(group):
    if len(group) < 6: return np.nan
    y = group['payment_ratio'].tail(6).values
    return np.polyfit(np.arange(len(y)), y, 1)[0]

trends = df.groupby('loan_account_id').apply(get_trend).reset_index(name='payment_trend')
trends = trends.merge(df_master[['loan_account_id', 'defaulted']], on='loan_account_id')

sns.kdeplot(data=trends, x='payment_trend', hue='defaulted', fill=True, palette=bank_palette, ax=axes[1, 0])
axes[1, 0].set_title('The "Falling Slope" (6-Month Trend Distribution)', fontweight='bold')
axes[1, 0].set_xlim(-0.3, 0.1)
# Changed axvline to black so it shows up on the white background
axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)

# Chart D: Months Since Last Full Pay
def get_months_since(group):
    n = len(group)
    perfect = group[group['missed'] == 0]
    return n - (perfect.index.max() + 1) if not perfect.empty else n

months_since = df.groupby('loan_account_id').apply(get_months_since).reset_index(name='months_since')
months_since = months_since.merge(df_master[['loan_account_id', 'defaulted']], on='loan_account_id')

sns.boxplot(data=months_since, x='defaulted', y='months_since', palette=bank_palette, ax=axes[1, 1])
axes[1, 1].set_title('Months Since Last Perfect Payment', fontweight='bold')
axes[1, 1].set_xticklabels(['Stable (0)', 'Impending NPA (1)'])

plt.tight_layout()
# Added facecolor='white' to explicitly ensure the saved image has a white background
plt.savefig("npa_early_warning_sys/data/model2_eda_visuals.png", bbox_inches='tight', facecolor='white')
print("✅ Visual charts saved to 'npa_early_warning_sys/data/model2_eda_visuals.png'")

# ==========================================
# 2. PRINT TEXT REPORT
# ==========================================
print("\n" + "="*60)
print("📊 MODEL 2: TIME-SERIES EDA TEXT REPORT")
print("="*60)

print(f"Total Relational Records: {len(df):,} monthly payments")
print(f"Unique Loan Accounts    : {df['loan_account_id'].nunique():,}")

print("\n--- System-Wide Missed Payment Rate ---")
print(f"Total Missed Months: {df['missed'].sum():,} out of {len(df):,} ({df['missed'].mean()*100:.2f}%)")

print("\n--- Behavioral Contrast: Stable vs NPA ---")
for cls, name in [(0, "Stable (Class 0)"), (1, "Impending NPA (Class 1)")]:
    sub = df[df['defaulted'] == cls]
    print(f"\n{name}:")
    print(f"  Avg Payment Ratio : {sub['payment_ratio'].mean():.4f}")
    print(f"  Missed Month Rate : {sub['missed'].mean()*100:.2f}%")

print("\n--- Rolling Feature Summary (The Falling Slope) ---")
for cls, name in [(0, "Stable (Class 0)"), (1, "Impending NPA (Class 1)")]:
    sub = trends[trends['defaulted'] == cls]
    print(f"{name} Avg 6-Month Trend: {sub['payment_trend'].mean():.5f}")

print("="*60)