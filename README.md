# Spam Email Classifier

A lightweight NLP classification project that detects whether a message is **spam** or **ham** using bag-of-words features and a Multinomial Naive Bayes classifier.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![CI](https://github.com/rrstii/spam-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/rrstii/spam-detector/actions/workflows/ci.yml)

## Project Overview

The project demonstrates a compact end-to-end text-classification pipeline:

1. Create a small labeled dataset of spam and legitimate messages.
2. Convert text into numerical features with `CountVectorizer`.
3. Split the data using a reproducible stratified train/test split.
4. Train a `MultinomialNB` classifier inside a scikit-learn `Pipeline`.
5. Evaluate accuracy, precision, recall, F1-score, and the confusion matrix.
6. Classify previously unseen example messages.

## Model Pipeline

```text
Raw messages
     ↓
CountVectorizer
     ↓
Bag-of-words features
     ↓
Multinomial Naive Bayes
     ↓
Spam / Ham prediction
```

## Results

With the included dataset and fixed stratified split, the demonstration achieves **90% test accuracy** on 10 held-out messages. The confusion matrix uses `[ham, spam]` ordering:

```text
[[4, 1],
 [0, 5]]
```

The dataset contains only 31 manually created examples, so this result is **not representative of production performance**.

## Dataset

The dataset is intentionally small and embedded directly in `spam.py` so the machine-learning workflow remains easy to inspect and reproduce. It contains examples of promotional, suspicious, personal, and everyday messages.

## Installation

```bash
git clone https://github.com/rrstii/spam-detector.git
cd spam-detector
pip install -r requirements.txt
```

For development and tests:

```bash
pip install -r requirements-dev.txt
```

## Usage

```bash
python spam.py
```

Run the test suite with:

```bash
python -m pytest -q
```

The script displays the dataset preview, class distribution, evaluation metrics, and predictions for new messages.

## Project Structure

```text
spam-detector/
├── spam.py
├── tests/
│   └── test_spam.py
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
├── .gitignore
├── .github/workflows/ci.yml
└── README.md
```

## Limitations and Next Steps

This is an educational NLP project. A more robust version could:

- Use a larger real-world dataset such as SMS Spam Collection.
- Compare raw counts with TF-IDF features.
- Use cross-validation for a more stable estimate.
- Compare Naive Bayes with linear classifiers.
- Separate data loading from training and inference.
- Expose the classifier through a small web API.

## Tech Stack

**Python · Pandas · scikit-learn · NLP · CountVectorizer · Multinomial Naive Bayes**
