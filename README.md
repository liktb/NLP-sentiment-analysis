# Rotten Tomatoes AI Sentiment Classifier

**Track:** AI + Social Good / NLP  
**Team:** High School AI Builder  

## The problem
Manually processing thousands of customer reviews, financial feedback notes, or movie comments is slow and inefficient. Organizations and platforms need automated tools to instantly categorize large volumes of text sentiment to understand user reception at scale.

## What I built
An end-to-end local natural language processing pipeline. A user inputs a raw review text sentence, and the system tokenizes the string, processes it through a fine-tuned deep learning transformer model, and outputs a clear binary classification label (**Positive** or **Negative**) along with an evaluation confidence score.

## How it works
* **Data:** Utilized a custom split of Rotten Tomatoes user review comments (`train_reviews.csv` and `test_reviews.csv`), featuring balanced negative and positive training rows[cite: 14].
* **Model:** Fine-tuned the pre-trained `distilbert-base-uncased` transformer model from Hugging Face[cite: 8, 13] using PyTorch and Hugging Face `Trainer` API[cite: 8], leveraging local NVIDIA RTX 4070 GPU hardware for efficient execution[cite: 13].
* **Cleaning & Preprocessing:** Applied tokenization, dynamic batch padding, and strict train-test separation to prevent data leakage[cite: 14, 17].

## Results
* **Baseline vs Model:** Compared against a simple baseline, our fine-tuned DistilBERT model achieved an overall accuracy of **86.03%** and a weighted **F1-Score of 0.8603** on unseen held-out test data[cite: 11, 16].
* **Where it fails:** The model occasionally misclassifies highly sarcastic, subtle, or multi-layered reviews where explicit sentiment keywords are absent or contradictory.

## What we'd do next
1. Expand the dataset to handle multi-class granular ratings (e.g., 1-to-5 stars instead of strict binary sentiment).
2. Wrap the inference scripts into an interactive local web application dashboard using **Streamlit**[cite: 8, 13].

## Sources
* **Datasets:** Rotten Tomatoes user review split files (`train_reviews.csv`, `test_reviews.csv`)[cite: 14].
* **Pretrained Models:** Hugging Face `distilbert-base-uncased` model checkpoint[cite: 8, 13].
* **Libraries:** PyTorch, Hugging Face Transformers & Datasets, Scikit-Learn, Pandas[cite: 8].

---

## How to Run This Project Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/liktb/NLP-sentiment-analysis.git](https://github.com/liktb/NLP-sentiment-analysis.git)
cd NLP
