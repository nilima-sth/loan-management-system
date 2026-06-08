import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

print("1. Loading Advanced Master LMS Data (Natively Time-Stamped)...")
df = pd.read_csv(r"D:\lms_ai_brain\data\lms_advanced_master.csv")

print("2. Preprocessing & Encoding Text Features...")
target = df["defaulted"]
features_raw = df.drop("defaulted", axis=1)

# Encode categoricals into numbers
X_encoded = pd.get_dummies(features_raw, drop_first=True)
X_encoded["defaulted"] = target 

print("3. Executing Strict Temporal Split (Time-Series Validation Rule)...")
df_train = X_encoded[X_encoded['close_year'] <= 2022]
df_test  = X_encoded[X_encoded['close_year'] >= 2023]

X_train = df_train.drop(["defaulted", "close_year"], axis=1)
y_train = df_train["defaulted"]

X_test  = df_test.drop(["defaulted", "close_year"], axis=1)
y_test  = df_test["defaulted"]

print(f"   -> Historical Training Data Pool (2018-2022) : {X_train.shape[0]} loans")
print(f"   -> Future Testing Evaluation Pool (2023-2024): {X_test.shape[0]} loans")

print("4. Applying SMOTE to Balance Training Data Imbalance...")
smote = SMOTE(sampling_strategy=0.25, random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"   -> Balanced Training Dimensions              : {X_train_bal.shape[0]} samples")

print("5. Training Temporal XGBoost Model on RTX GPU Engine...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    scale_pos_weight=4,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)

print("\n" + "="*60)
print("🎯 USE CASE 1: CREDIT SCORING ENGINE - TEMPORAL VALIDATION REPORT")
print("="*60)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"OVERALL ACCURACY : {accuracy * 100:.2f}%")
print(f"AUC SCORE        : {auc:.4f} (Target > 0.91 per documentation)")
print("-" * 60)
print("Detailed Classification Report (NRB Temporal Audit Format):")
print(classification_report(y_test, y_pred, target_names=["Repaid (0)", "Defaulted (1)"]))

# ---------------------------------------------------------
# NEW: ADVANCED BANKING & REGULATORY METRICS
# ---------------------------------------------------------
print("="*60)
print("🏦 ADVANCED BANKING & REGULATORY METRICS (NRB STANDARD)")
print("="*60)

# 1. GINI INDEX
gini = (2 * auc) - 1
print(f"GINI INDEX       : {gini:.4f} (Target: > 0.60 for strong separation)")

# 2. KS STATISTIC
proba_defaulted = y_pred_proba[y_test == 1]
proba_repaid = y_pred_proba[y_test == 0]
ks_stat, ks_pvalue = ks_2samp(proba_defaulted, proba_repaid)
print(f"KS STATISTIC     : {ks_stat:.4f} (Target: > 0.40 for scorecard approval)")

# 3. PSI (POPULATION STABILITY INDEX)
def calculate_psi(expected, actual, buckets=10):
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    expected_perc = np.percentile(expected, breakpoints)
    expected_counts = np.histogram(expected, expected_perc)[0]
    actual_counts = np.histogram(actual, expected_perc)[0]
    expected_rates = expected_counts / len(expected)
    actual_rates = actual_counts / len(actual)
    expected_rates = np.where(expected_rates == 0, 0.0001, expected_rates)
    actual_rates = np.where(actual_rates == 0, 0.0001, actual_rates)
    psi_values = (actual_rates - expected_rates) * np.log(actual_rates / expected_rates)
    return np.sum(psi_values)

train_pred_proba = model.predict_proba(X_train_bal)[:, 1]
psi_score = calculate_psi(train_pred_proba, y_pred_proba)

print(f"PSI (Drift)      : {psi_score:.4f} ", end="")
if psi_score < 0.1:
    print("(Status: GREEN - Stable Population, No significant drift)")
elif psi_score < 0.25:
    print("(Status: AMBER - Slight drift detected, monitor closely)")
else:
    print("(Status: RED - Severe drift, model retraining required)")

# ---------------------------------------------------------
# STEP 6: RISK EXPLAINABILITY
# ---------------------------------------------------------
print("="*60)
print("🔍 STEP 6: RISK EXPLAINABILITY LOG (NRB COMPLIANCE)")
print("="*60)

importance_scores = model.get_booster().get_score(importance_type='gain')
clean_impacts = sorted(
    [(col, importance_scores.get(col, 0.0)) for col in X_train.columns],
    key=lambda x: x[1], reverse=True
)

print("Top 5 Regulatory Risk Factors Governing Future Predictions:")
for rank, (feat, score) in enumerate(clean_impacts[:5], 1):
    print(f" Rank {rank} -> {feat:<28} : Gain Weight = {score:.2f}")

print("="*60)