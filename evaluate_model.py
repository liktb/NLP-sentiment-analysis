import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np

# 1. LOAD THE TEST DATASET
print("Loading test reviews...")
test_df = pd.read_csv('rotten_tomatoes_train_test/test_reviews.csv')
test_dataset = Dataset.from_pandas(test_df)

# 2. LOAD YOUR SAVED MODEL AND TOKENIZER
model_path = "./my_rotten_tomatoes_model"
print(f"Loading custom model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval() # Set model to evaluation mode

# 3. TOKENIZE THE TEST TEXT
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_test = test_dataset.map(preprocess_function, batched=True)

# 4. RUN INFERENCE ON THE TEST SET
print("Running predictions on the test split...")
all_preds = []
all_labels = test_df['label'].values

# Process in small batches to prevent memory overflow
batch_size = 16
for i in range(0, len(tokenized_test), batch_size):
    batch_texts = tokenized_test[i : i + batch_size]["text"]
    
    # Tokenize batch
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class indices (highest probability logit)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    all_preds.extend(preds)

all_preds = np.array(all_preds)

# 5. CALCULATE METRICS
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted') # Handles multi-class or binary securely

print("\n==============================")
print(f" FINAL MODEL PERFORMANCE ")
print("==============================")
print(f"Accuracy : {acc * 100:.2f}%")
print(f"F1-Score : {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))
print("==============================")