Yes, adding a direct link to your Hugging Face model repository inside your GitHub `README.md` is standard practice for professional machine learning engineering. It shows judges that you understand proper artifact management and reproducibility.

Here is your updated, copy-pasteable professional `README.md` containing the new **Pretrained Model & Weights** section.

Open your `README.md` in VS Code, replace everything with this block, and save:

```markdown
# Rotten Tomatoes AI Sentiment Classification Pipeline

A production-grade Natural Language Processing pipeline built using PyTorch, Hugging Face Transformers (**DistilBERT**), and accelerated locally via NVIDIA CUDA. Designed for high accuracy and rapid text inference on customer review datasets.

---

##  Model Performance & Evaluation Metrics
Evaluated on **2,133 unseen test reviews**:

- **Overall Test Accuracy:** `86.03%`
- **Class Metrics:**
  - **Negative:** Precision: `0.86` | Recall: `0.86` | F1-Score: `0.86`
  - **Positive:** Precision: `0.86` | Recall: `0.86` | F1-Score: `0.86`

---

##  Pretrained Model & Artifacts
To ensure GitHub repository size efficiency and seamless project reproducibility, the fine-tuned DistilBERT weights and tokenizer configurations are hosted directly on the Hugging Face Hub:

- **Hugging Face Model Hub:** [Rexxxxxtxx/rotten-tomatoes-distilbert-sentiment](https://huggingface.co/Rexxxxxtxx/rotten-tomatoes-distilbert-sentiment)

---

##  Project Structure
```text
NLP/
│
├── rotten_tomatoes_train_test/ # Dataset folder (train_reviews.csv / test_reviews.csv splits)
├── train_nlp.py                # Tokenization, Hugging Face Trainer setup, training loop & artifact saving
├── test_model.py               # Single-string inference script loading the saved model weights
├── evaluate_model.py           # Comprehensive evaluation script (Accuracy, F1-Score, Classification Report)
├── push_to_hf.py               # Automated script to push local model artifacts to Hugging Face Hub
├── requirements.txt            # Locked python package dependencies
├── .gitignore                  # Git exclusion rules for large model weights and checkpoints
└── README.md                   # Project documentation & metrics summary

```

---

##  How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Train the Model (Optional)

*(Fine-tunes DistilBERT locally on your GPU using the training split)*

```bash
python train_nlp.py

```

### 3. Evaluate Model Performance

Run evaluation across the test set to generate accuracy and F1 metrics:

```bash
python evaluate_model.py

```

### 4. Run Single-Comment Prediction

Test custom review text strings against your saved local or remote model weights:

```bash
python test_model.py

```

---

##  Author

* **Pipeline Architecture, Training & Execution:** [Ryan Dinto](https://github.com/liktb)

```

---

### Push the Update to GitHub:
Run these terminal commands to update your repository live:
```bash
git add README.md
git commit -m "Add Hugging Face model repository link and update project structure"
git push origin main

```
