# Spam Email Classifier

A lightweight NLP classification project that detects whether a message is **spam** or **ham** using bag-of-words features and a Multinomial Naive Bayes classifier.

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Project Overview

The project demonstrates a simple end-to-end text-classification pipeline:

1. Create a small labeled dataset of spam and legitimate messages.
2. Convert text into numerical features with `CountVectorizer`.
3. Split the data into training and test sets.
4. Train a `MultinomialNB` classifier.
5. Evaluate predictions with accuracy and a confusion matrix.
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

With the included dataset and fixed train/test split, the demonstration achieves approximately **90% test accuracy**.

The confusion matrix is also printed by the script. Because the dataset contains only 31 manually created examples, this result is **not representative of production performance**.

## Dataset

The dataset is intentionally small and embedded directly in `spam.py` so the machine-learning workflow remains easy to inspect and reproduce. It contains examples of promotional, suspicious, personal, and everyday messages.

## Installation

```bash
git clone https://github.com/rrstii/spam-detector.git
cd spam-detector
pip install -r requirements.txt
```

## Usage

```bash
python spam.py
```

The script displays the dataset preview, class distribution, test accuracy, confusion matrix, and predictions for new messages.

## Project Structure

```text
spam-detector/
├── spam.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Limitations and Next Steps

This is an educational NLP project. A more robust version could:

- Use a larger real-world dataset such as SMS Spam Collection.
- Replace raw counts with TF-IDF features.
- Report precision, recall, and F1-score.
- Use stratified cross-validation.
- Compare Naive Bayes with linear classifiers.
- Separate data, training, and inference code into reusable modules.
- Expose the classifier through a small web API.

## Tech Stack

**Python · Pandas · scikit-learn · NLP · CountVectorizer · Multinomial Naive Bayes**
