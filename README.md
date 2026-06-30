# 📰 Day 53 — Fake News Detection with PyTorch & TF-IDF

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TFIDF-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-009688?style=flat-square)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat-square&logo=pandas&logoColor=white)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2053-blueviolet?style=flat-square)

**A deep learning model doesn't understand the meaning of news articles. It learns statistical patterns from numerical representations.**

</div>

---

## 📌 Overview

Text can't be fed directly into a neural network — it must first be converted into numbers that preserve meaningful signal. Day 53 builds a complete **Fake News Detection pipeline** in PyTorch: raw article text → TF-IDF vectors → trained classifier → real-time prediction on both dataset samples and manually typed news.

> **Hard truth learned today:** A deep learning model doesn't understand the meaning of news articles. It learns statistical patterns from numerical representations, making proper preprocessing and feature extraction just as important as the model itself.

---

## 🏗️ System Pipeline

```
Raw News Dataset (CSV)
         │
         ▼
┌──────────────────────┐
│   Text Preprocessing  │  Clean text, remove noise, handle missing values
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   Label Encoding      │  Real → 1   Fake → 0
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   TF-IDF Vectorizer   │  Text → numerical feature vectors
│                       │  Weighted by term importance
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   Train/Test Split    │
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   PyTorch Model       │  Linear → Sigmoid
│                       │  Binary classification
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   Training Loop       │  Forward pass → BCE Loss → Backprop → Update weights
└──────────┬────────────┘
           │
           ▼
┌──────────────────────┐
│   Prediction          │  On test samples AND manual user input
│                       │  Same TF-IDF pipeline applied to both
└───────────────────────┘
```

---

## 🧠 Why TF-IDF?

**TF-IDF (Term Frequency–Inverse Document Frequency)** scores each word by how important it is to a specific document relative to the entire corpus — common words like "the" get suppressed, while distinctive words get amplified.

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)$$

Where:
- $\text{TF}(t, d)$ = frequency of term $t$ in document $d$
- $N$ = total number of documents
- $\text{DF}(t)$ = number of documents containing term $t$

| Why it works for fake news detection |
|---|
| ✅ Captures word importance, not just presence |
| ✅ Downweights generic/common words automatically |
| ✅ Produces fixed-length vectors regardless of article length |
| ✅ Fast and lightweight compared to embedding-based approaches |
| ⚠️ Ignores word order and context (no semantic understanding) |

---

## 🔬 What I Implemented

### 1. Data Preprocessing & Label Encoding

```python
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ─── Load Dataset ────────────────────────────────────────────────────────────────
df = pd.read_csv('fake_news_data.csv')   # columns: 'text', 'label' (REAL / FAKE)
df.dropna(subset=['text'], inplace=True)

# ─── Clean Text ──────────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)               # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()           # normalize whitespace
    return text

df['clean_text'] = df['text'].apply(clean_text)

# ─── Encode Labels ───────────────────────────────────────────────────────────────
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])    # FAKE=0, REAL=1
print(dict(zip(le.classes_, le.transform(le.classes_))))
```

### 2. TF-IDF Feature Extraction

```python
# ─── Vectorize ────────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=5000,        # keep top 5000 most informative terms
    stop_words='english',
    ngram_range=(1, 2)        # unigrams + bigrams
)

X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['label_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Feature vector size: {X_train.shape[1]}")
print(f"Training samples   : {X_train.shape[0]}")

# Save TF-IDF vectorizer for inference
import joblib
joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
```

### 3. PyTorch Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

# ─── Define Model ─────────────────────────────────────────────────────────────────
class FakeNewsClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FakeNewsClassifier, self).__init__()
        self.linear  = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = FakeNewsClassifier(input_dim=X_train.shape[1])

# ─── Loss & Optimizer ──────────────────────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── Convert to Tensors ────────────────────────────────────────────────────────────
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)
```

### 4. Training Loop

```python
# ─── Train ──────────────────────────────────────────────────────────────────────────
EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            preds = (outputs >= 0.5).float()
            acc = (preds == y_train_tensor).float().mean()
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

# ─── Evaluate ───────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds   = (test_outputs >= 0.5).float()
    test_acc     = (test_preds == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")

# ─── Save Model ─────────────────────────────────────────────────────────────────────
torch.save(model.state_dict(), 'models/fake_news_model.pth')
```

### 5. Manual Text Prediction

```python
# ─── Predict on Custom User Input ───────────────────────────────────────────────────
def predict_news(news_text, model, tfidf, label_encoder):
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned]).toarray()
    tensor_input = torch.tensor(vectorized, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model(tensor_input)

    confidence = pred.item()
    label = "✅ REAL" if confidence >= 0.5 else "❌ FAKE"
    display_confidence = confidence if confidence >= 0.5 else (1 - confidence)

    print(f"News: {news_text[:80]}...")
    print(f"Prediction: {label}")
    print(f"Confidence: {display_confidence:.4f}")
    return label, confidence

# ─── Example ────────────────────────────────────────────────────────────────────────
user_input = input("Enter a news headline or article: ")
predict_news(user_input, model, tfidf, le)
```

**Sample output (from today's run):**
```
>>> Enter a news headline or article:
U.S. Secretary of State John F. Kerry said Mon...

tensor([[0.9944]], grad_fn=<SigmoidBackward0>)
Prediction: ✅ REAL
Confidence: 0.9944
```

---

## 🏗️ Model Architecture

```
Input: TF-IDF vector (5000 features)
  ↓
nn.Linear(5000 → 1)
  → Single fully connected layer
  → Learns weighted importance of each TF-IDF feature
  ↓
nn.Sigmoid()
  → Squashes output to (0, 1)
  → Output ≥ 0.5 → REAL
  → Output < 0.5  → FAKE
  ↓
Output: Confidence score ∈ (0, 1)
```

> This is logistic regression implemented as a PyTorch `nn.Module` — a single linear layer + sigmoid. It's intentionally simple: the real complexity lives in the TF-IDF feature extraction, not the model architecture.

---

## 📊 Interpreting the Output

| Prediction Score | Interpretation |
|---|---|
| 0.95 – 1.00 | Strongly classified as Real |
| 0.50 – 0.95 | Likely Real, lower confidence |
| 0.05 – 0.50 | Likely Fake, lower confidence |
| 0.00 – 0.05 | Strongly classified as Fake |

> The model output is a raw sigmoid probability — confidence near 0.5 indicates genuine model uncertainty, not a coding error.

---

## 💡 Key Learnings

- **Text must be converted into numerical features before training deep learning models** — there's no way around this for classical architectures
- **TF-IDF is an effective and fast feature extraction technique** for text classification, especially with limited compute
- **PyTorch provides flexibility for building custom models** — even a single-layer classifier is straightforward to define, train, and evaluate
- **The same preprocessing pipeline must be reused at inference time** — fitting TF-IDF only once and reusing the fitted vectorizer (not refitting) on user input is critical
- **Preprocessing quality directly impacts prediction performance** — cleaning, stop-word removal, and n-gram choices all affect the final accuracy

---

## ⚠️ Limitations

| Limitation | Detail |
|---|---|
| No semantic understanding | TF-IDF + linear model can't grasp context, sarcasm, or nuance |
| Vocabulary-bound | Words unseen during training are silently dropped at inference |
| Single linear layer | Cannot model complex nonlinear relationships in language |
| Dataset bias | Model will reflect biases present in the labeled training data |
| Source-based artifacts | Model may learn stylistic patterns (e.g., publication formatting) rather than factual accuracy |

---

## 🔁 Why Reuse the Fitted TF-IDF Vectorizer

```python
# ❌ WRONG — refitting creates a different vocabulary/vector space
tfidf_new = TfidfVectorizer()
user_vector = tfidf_new.fit_transform([user_input])    # Mismatched dimensions!

# ✅ CORRECT — transform only, using the vectorizer fitted during training
user_vector = tfidf.transform([user_input])             # Same 5000-dim space
```

> This is the exact step shown in the implementation: `user_input = tfidf.transform([user_input])` — never `fit_transform` on new data, or the feature space won't match what the model was trained on.

---

## 🗂️ Project Structure

```
day-53-pytorch-fake-news/
├── train.py                      # Preprocessing + TF-IDF + training loop
├── predict.py                    # Manual news input prediction
├── models/
│   ├── fake_news_model.pth       # Trained PyTorch model weights
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── label_encoder.pkl         # Fitted LabelEncoder
├── data/
│   └── fake_news_data.csv
├── outputs/
│   ├── training_loss.png
│   └── sample_predictions.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-53-pytorch-fake-news
cd day-53-pytorch-fake-news
pip install -r requirements.txt

# Train the model
python train.py

# Predict on custom input
python predict.py
```

**Requirements:**
```
torch
scikit-learn
pandas
numpy
joblib
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 53 of 100 — PyTorch Implementation: Fake News Detection with TF-IDF

| ← Previous | Current | Next → |
|---|---|---|
| [Day 52 — Full-Stack AI App](#) | **Day 53 — Fake News Detection** | [Day 54](#) |


---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #PyTorch #FakeNewsDetection #NLP #TFIDF #TextClassification #DeepLearning</sub>
</div>
