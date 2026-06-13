PS D:\lms_ai_brain> (Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& d:\lms_ai_brain\venv\Scripts\Activate.ps1)

(venv) PS D:\lms_ai_brain> & d:\lms_ai_brain\venv\Scripts\python.exe d:/lms_ai_brain/nlp_engine/eval_engine.py



[CONFIG] Total intent classes: 12

[CONFIG] Intents: ['account_balance', 'emi_inquiry', 'loan_apply', 'branch_location', 'interest_rate', 'cic_check', 'complaint_lodge', 'qr_payment_help', 'remittance_receive', 'fd_interest', 'general_greeting', 'escalate_human']



======================================================================

STEP 1: DATA PREPARATION

======================================================================

Total raw samples: 144

Training samples  : 115

Validation samples: 29

✅ Data preparation complete. Ready for training and evaluation!



======================================================================

STEP 2: MODEL TRAINING (Phase 2 Classifier)

======================================================================

Note: Using TF-IDF + Logistic Regression as evaluation stand-in

      for mBERT. All metric code is identical to production.



[CONFIG] Target model : bert-base-multilingual-cased

[CONFIG] Num labels   : 12

[CONFIG] Learning rate: 2e-5 (design doc spec)

[CONFIG] Epochs       : 10

[CONFIG] Batch size   : 32 (train) / 64 (eval)



✅ Model trained in 0.05s



======================================================================

STEP 3: RUNNING CHATBOT EVALUATION

======================================================================



Final Model Accuracy : 75.86%

Final Model F1-Score : 74.98%



======================================================================

DETAILED INTENT CLASSIFICATION REPORT

======================================================================

                    precision    recall  f1-score   support



   account_balance     1.0000    0.5000    0.6667         2

       emi_inquiry     1.0000    0.5000    0.6667         2

        loan_apply     0.5000    1.0000    0.6667         2

   branch_location     1.0000    1.0000    1.0000         3

     interest_rate     0.6667    0.6667    0.6667         3

         cic_check     1.0000    0.5000    0.6667         2

   complaint_lodge     0.6667    0.6667    0.6667         3

   qr_payment_help     0.7500    1.0000    0.8571         3

remittance_receive     0.6667    1.0000    0.8000         2

       fd_interest     1.0000    0.5000    0.6667         2

  general_greeting     1.0000    0.5000    0.6667         2

    escalate_human     0.7500    1.0000    0.8571         3



          accuracy                         0.7586        29

         macro avg     0.8333    0.7361    0.7373        29

      weighted avg     0.8218    0.7586    0.7498        29



======================================================================

CONFUSION MATRIX (rows=True, cols=Predicted)

======================================================================

Intent                  0   1   2   3   4   5   6   7   8   9  10  11

----------------------------------------------------------------------

account_balance         1   0   0   0   0   0   0   0   1   0   0   0

emi_inquiry             0   1   0   0   0   0   0   1   0   0   0   0

loan_apply              0   0   2   0   0   0   0   0   0   0   0   0

branch_location         0   0   0   3   0   0   0   0   0   0   0   0

interest_rate           0   0   1   0   2   0   0   0   0   0   0   0

cic_check               0   0   1   0   0   1   0   0   0   0   0   0

complaint_lodge         0   0   0   0   0   0   2   0   0   0   0   1

qr_payment_help         0   0   0   0   0   0   0   3   0   0   0   0

remittance_receive      0   0   0   0   0   0   0   0   2   0   0   0

fd_interest             0   0   0   0   1   0   0   0   0   1   0   0

general_greeting        0   0   0   0   0   0   1   0   0   0   1   0

escalate_human          0   0   0   0   0   0   0   0   0   0   0   3



======================================================================

LIVE CHATBOT DEMO — Sample Predictions

======================================================================



Query                                         Predicted Intent          Confidence

-------------------------------------------------------------------------------------

मेरो balance कति छ?                           account_balance                83.8%

EMI kati garnu parne?                         emi_inquiry                    57.0%

Home loan apply garna ke chahincha?           loan_apply                     44.2%

Nearest ATM kaha xa?                          branch_location                51.6%

FD byaj kati ho?                              interest_rate                  38.7%

QR payment garna milcha?                      qr_payment_help                78.4%

I want to speak to a human                    escalate_human                 49.6%

Namaste                                       general_greeting               34.8%

Mero account ma wrong transaction bhayo       complaint_lodge                49.2%

Remittance kaise receive garne?               remittance_receive             51.1%



======================================================================

DESIGN DOCUMENT COMPLIANCE VERIFICATION

======================================================================



  Target F1-Score (doc spec) : > 93%

  Achieved F1-Score          : 74.98%  ⚠️  NEEDS MORE DATA



  Target Accuracy (doc spec) : > 93%

  Achieved Accuracy          : 75.86%  ⚠️  NEEDS MORE DATA



  Model Architecture in doc  : mBERT (bert-base-multilingual-cased)

  Phase 1 (Rule-based) target: 85% ← Lower baseline (expected)

  Phase 2 (mBERT) target     : 93% ← Current evaluation target

  Phase 3 (RAG) target       : 96% ← Future target

Traceback (most recent call last):

  File "d:\lms_ai_brain\nlp_engine\eval_engine.py", line 491, in <module>

    with open("/home/claude/eval_results.json", "w", encoding="utf-8") as f:

FileNotFoundError: [Errno 2] No such file or directory: '/home/claude/eval_results.json'

(venv) PS D:\lms_ai_brain> 