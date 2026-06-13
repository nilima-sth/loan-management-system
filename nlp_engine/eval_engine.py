import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------
# 1. THE CEO'S CHOSEN MODEL & PARAMETERS
# ---------------------------------------------------------
MODEL_NAME = 'bert-base-multilingual-cased'
NUM_INTENTS = 12

print(f"🚀 Initializing Production Architecture: {MODEL_NAME}")

# Initialize the Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_INTENTS
)

# ---------------------------------------------------------
# 2. DATASET INGESTION (The 10,000+ Rows)
# ---------------------------------------------------------
# Assumes you have a CSV named 'full_banking_intents.csv' 
# with columns: 'text' (the query) and 'label' (0-11)
data_path = os.path.join(os.path.dirname(__file__), "full_banking_intents.csv")

print("📊 Loading full dataset...")
df = pd.read_csv(data_path)

# Convert pandas dataframe to HuggingFace Dataset
hf_dataset = Dataset.from_pandas(df)

# Split 80% Train, 20% Evaluation
dataset_splits = hf_dataset.train_test_split(test_size=0.2, seed=42)

# Tokenization Function
def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True, 
        max_length=64  # Short banking queries don't need 512 tokens
    )

print("⚙️ Tokenizing data (this may take a moment)...")
tokenized_datasets = dataset_splits.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['test']

# ---------------------------------------------------------
# 3. EVALUATION METRICS (The Grader)
# ---------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

# ---------------------------------------------------------
# 4. TRAINING EXECUTION
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir='./banking_chatbot_nepali',
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=200,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='best',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    fp16=True, # Enables Mixed Precision to speed up GPU training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("🔥 Commencing Training Phase...")
trainer.train()

# ---------------------------------------------------------
# 5. FINAL EVALUATION & EXPORT
# ---------------------------------------------------------
print("✅ Training Complete. Running final evaluation on holdout set...")
eval_results = trainer.evaluate()

print(f"\n🏆 PRODUCTION MODEL RESULTS:")
print(f"Final Accuracy : {eval_results['eval_accuracy'] * 100:.2f}%")
print(f"Final F1-Score : {eval_results['eval_f1'] * 100:.2f}%")

# Save the final production-ready model
trainer.save_model('./banking_chatbot_nepali_FINAL')
tokenizer.save_pretrained('./banking_chatbot_nepali_FINAL')
print("💾 Model saved securely to disk.")