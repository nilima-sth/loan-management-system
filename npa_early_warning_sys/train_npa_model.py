import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Relational Databases...")
df_master = pd.read_csv("credit_scoring_engine/data/lms_advanced_master.csv")
df_master['loan_account_id'] = df_master.index + 1

df_schedule = pd.read_csv("npa_early_warning_sys/data/lms_repayment_schedule.csv")

print("2. Engineering Rolling Time-Series Features")
def calculate_rolling_features(group):
    group = group.sort_values('month_index')
    n = len(group)
    if n < 6: return None 
        
    feats = {}
    feats['loan_account_id'] = group['loan_account_id'].iloc[0]
    
    # 1. Current DPD 
    last_record = group.iloc[-1]
    feats['current_dpd'] = 30 if last_record['missed'] == 1 else 0
    
    # 2. Missed Counts & Payment Ratios 
    feats['missed_count_6m'] = group.tail(6)['missed'].sum()
    feats['missed_count_3m'] = group.tail(3)['missed'].sum()
    feats['payment_ratio_6m'] = group.tail(6)['amount_paid'].sum() / group.tail(6)['amount_due'].sum()
    
    # 3. Average Payment Gap (CEO Spec)
    group['payment_gap'] = group['amount_due'] - group['amount_paid']
    feats['avg_payment_gap'] = group['payment_gap'].tail(6).mean()
    
    # 4. Mathematical Trend (CEO Spec)
    y_values = group['payment_ratio'].tail(6).values
    feats['payment_trend'] = np.polyfit(np.arange(len(y_values)), y_values, 1)[0]
    
    # 5. Months since last perfect payment (CEO Spec)
    perfect_months = group[group['missed'] == 0]
    feats['months_since_last_full_pay'] = n - (perfect_months.index.max() + 1) if not perfect_months.empty else n
    
    # 6. Loan Age & Percentage Remaining (CEO Spec)
    feats['loan_age_months'] = n
    feats['pct_remaining'] = 1 - (n / 60) # Assuming standard 60 month tenure for calculation
        
    return pd.Series(feats)

# Apply the feature engineering (Flattens the schedule back to 1 row per customer)
npa_features_df = df_schedule.groupby('loan_account_id').apply(calculate_rolling_features).dropna().reset_index(drop=True)

# Merge the target variable ('defaulted') from the master table
final_npa_df = npa_features_df.merge(df_master[['loan_account_id', 'defaulted']], on='loan_account_id')
print(f"   -> Extracted rolling features for {len(final_npa_df)} accounts.")

print("3. Preparing Data for Stacking Ensemble...")
X = final_npa_df.drop(['loan_account_id', 'defaulted'], axis=1)
y = final_npa_df['defaulted']

# Standard split (No SMOTE - relying on CEO native weighting)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("4. Training Level 0 & Level 1 Stacking Classifier (CEO Spec 3.3)...")
# Base learners (level 0) - EXACT CEO PARAMS
base_models = [
    ('xgb', xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=25, eval_metric='auc', random_state=42, n_jobs=-1
    )),
    ('rf', RandomForestClassifier(
        n_estimators=300, max_depth=8,
        class_weight='balanced', random_state=42, n_jobs=-1
    )),
]

# Meta-learner (level 1)
meta_learner = LogisticRegression(C=1.0, random_state=42)

stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    stack_method='predict_proba',
    passthrough=True, 
    n_jobs=-1
)

# FIXED: Now strictly using X_train and y_train
stacking.fit(X_train, y_train)

print("\n" + "="*60)
print("🚨 MODEL 2: NPA EARLY WARNING SYSTEM (90-DAY PREDICTION)")
print("="*60)

y_pred = stacking.predict(X_test)
y_pred_proba = stacking.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)

print(f"STACKING AUC SCORE : {auc:.4f} (Target > 0.93 per documentation)")
print("-" * 60)
print("Early Warning Detection Report (Test Set):")
print(classification_report(y_test, y_pred, target_names=["Stable Loan (0)", "Impending NPA (1)"]))
print("="*60)