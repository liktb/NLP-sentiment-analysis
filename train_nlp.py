import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import numpy as np

# 1. LOAD YOUR SEPARATE TRAIN AND TEST CSV FILES
print("Loading train and test datasets from folder...")
train_df = pd.read_csv('rotten_tomatoes_train_test/train_reviews.csv')
test_df = pd.read_csv('rotten_tomatoes_train_test/test_reviews.csv')

# Convert pandas DataFrames into Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Combine them into a DatasetDict
datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 2. LOAD A PRETRAINED TOKENIZER
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    # Tokenizer converts words into numbers the AI model can read
    return tokenizer(examples["text"], truncation=True, max_length=128)

print("Tokenizing text data...")
tokenized_datasets = datasets.map(preprocess_function, batched=True)

# Data collator handles dynamic padding for batches
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. LOAD THE PRETRAINED MODEL WITH A CLASSIFICATION HEAD
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=2, 
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1}
)

# 4. DEFINE HOW SUCCESS IS MEASURED
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# 5. CONFIGURE TRAINING ARGUMENTS (Optimized for your RTX 4070 GPU)
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none"
)

# 6. INITIALIZE TRAINER AND START TRAINING
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training loop on your local GPU...")
trainer.train()

# Evaluate final performance on the test set
eval_results = trainer.evaluate()
print(f"Final Evaluation Results: {eval_results}")

# Save your custom fine-tuned model locally!
model.save_pretrained("./my_rotten_tomatoes_model")
tokenizer.save_pretrained("./my_rotten_tomatoes_model")
print("Model trained and saved successfully to './my_rotten_tomatoes_model'!")