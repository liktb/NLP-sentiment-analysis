import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load your custom saved model from your local folder
model_path = "./my_rotten_tomatoes_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Test comment
comment = "This movie was an absolute waste of time, incredibly boring and poorly acted."

# Tokenize input text
inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128)

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_id = logits.argmax().item()
label = model.config.id2label[predicted_class_id]

print(f"Comment: '{comment}'")
print(f"AI Prediction: {label} (Class ID: {predicted_class_id})")