✓ Default rate achieved  : 18.25%  (target 15–18%)

── Data Quality Report ──────────────────────────────────────────────
✓ Null values              : 0
✓ income_stability values  : [np.float64(0.4), np.float64(0.7), np.float64(1.0)]  OK
✓ collateral_score values  : [np.float64(0.4), np.float64(0.5), np.float64(0.6), np.float64(0.8), np.float64(1.0)]  OK
✓ All range checks         : PASSED
✓ insurance_flag              : values={0, 1}  OK
✓ remittance_inflow_flag      : values={0, 1}  OK
✓ defaulted                   : values={0, 1}  OK
✓ Class distribution       : Repaid=6,540  Defaulted=1,460  (18.2% default rate)
✓ Shape                    : 8,000 rows × 29 columns

── Feature Correlations with `defaulted` (all features) ────────────
  emi_income_ratio               +0.3532  ████████████
  dscr                           −0.3106  ███████████
  max_dpd_24m                    +0.2711  █████████
  overdue_count_12m              +0.2136  ███████
  active_loan_count              +0.2041  ███████
  bounce_count_12m               +0.1811  ██████
  loan_tenure_months             −0.1656  █████
  restructure_count              +0.1336  ████
  hard_enquiry_6m                +0.1177  ████
  income_stability               −0.1001  ███
  insurance_flag                 −0.0675  ██
  collateral_score               −0.0670  ██
  ltv_ratio                      +0.0594  ██
  income_cagr_2y                 −0.0525  █
  avg_bank_balance_6m            −0.0508  █
  monthly_income_npr             −0.0499  █
  avg_deposit_12m                −0.0455  █
  product_count                  −0.0419  █
  remittance_inflow_flag         −0.0374  █
  bank_tenure_months             −0.0319  █
  festival_proximity             −0.0302  █
  credit_history_months          −0.0295  █
  district_npa_rate              +0.0249  
  applicant_age                  −0.0136  
  bs_month_applied               −0.0076  
  loan_amount_npr                −0.0064  

── Default Rate by Province ─────────────────────────────────────────
  Sudurpashchim           19.90%  ███████████
  Province No. 1          19.51%  ███████████
  Madhesh                 19.28%  ███████████
  Lumbini                 19.25%  ███████████
  Karnali                 18.03%  ██████████
  Bagmati                 17.21%  ██████████
  Gandaki                 16.11%  █████████

── Default Rate by income_stability ─────────────────────────────────
  0.4 (Agriculture )  27.69%  ████████████████
  0.7 (Business    )  19.13%  ███████████
  1.0 (Salaried    )  15.36%  █████████

── Default Rate by collateral_score ─────────────────────────────────
  0.4 (Shares   )  21.52%  ████████████
  0.5 (Land     )  20.30%  ████████████
  0.6 (Building )  18.74%  ███████████
  0.8 (Gold     )  15.29%  █████████
  1.0 (FD       )  13.08%  ███████

 Dataset saved → lms_advanced_master.csv
   Rows    : 8,000
   Columns : 29  (27 numeric, 2 categorical)
   Memory  : 2682 KB

--------------------------------CREDIT SCORE ENGINE-------------------------------------------------

1. Loading Advanced Master LMS Data...
2. Preprocessing & Encoding Text Features...
3. Executing Strict Temporal Split...
Historical Training Data Pool (2018-2022) : 5642 loans
Future Testing Evaluation Pool (2023-2024): 2358 loans
4. Applying SMOTE to Balance Training Data Imbalance...
Balanced Training Dimensions              : 5850 samples
5. Training Temporal XGBoost Model...

============================================================
USE CASE 1: CREDIT SCORING ENGINE - TEMPORAL VALIDATION REPORT
============================================================
OVERALL ACCURACY : 83.46%
AUC SCORE        : 0.8047 (Target > 0.91 per documentation)
------------------------------------------------------------
Detailed Classification Report (NRB Temporal Audit Format):
               precision    recall  f1-score   support

   Repaid (0)       0.89      0.91      0.90      1908
Defaulted (1)       0.58      0.50      0.53       450

     accuracy                           0.83      2358
    macro avg       0.73      0.71      0.72      2358
 weighted avg       0.83      0.83      0.83      2358

============================================================
STEP 6: RISK EXPLAINABILITY LOG (NRB COMPLIANCE)
============================================================
Top 5 Regulatory Risk Factors Governing Future Predictions:
 Rank 1 -> emi_income_ratio             : Gain Weight = 17.41
 Rank 2 -> max_dpd_24m                  : Gain Weight = 11.49
 Rank 3 -> income_stability             : Gain Weight = 11.26
 Rank 4 -> insurance_flag               : Gain Weight = 8.43
 Rank 5 -> overdue_count_12m            : Gain Weight = 7.63
============================================================