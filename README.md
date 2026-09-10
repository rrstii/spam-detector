# Spam Email Classifier

A lightweight text-classification project that detects whether a message is **spam** or **ham** (not spam) using a bag-of-words representation and Multinomial Naive Bayes.

## Overview

The pipeline is intentionally simple so the core machine-learning workflow is easy to understand:

1. Build a small labeled dataset of spam and ham messages.
2. Convert text into numerical features with `CountVectorizer`.
3. Split the data into training and test sets.
4. Train a `MultinomialNB` classifier.
5. Evaluate predictions with accuracy and a confusion matrix.
6. Run the trained model on new messages.

## Example

The script includes unseen example messages such as:

- Promotional or suspicious messages → `spam`
- Normal meeting and personal messages → `ham`

## Results

The current demonstration reports approximately **90% test accuracy**. Because the dataset is small and manually created, this number should not be interpreted as production-level performance.

## Installation

```bash
pip install -r requirements.txt
```

Or install the dependencies directly:

```bash
pip install pandas scikit-learn
```

## Usage

```bash
python spam.py
```

The program prints the dataset preview, label distribution, test accuracy, confusion matrix, and predictions for new messages.

## Project Structure

```text
.
├── spam.py
├── requirements.txt
└── README.md
```

## Limitations & Next Steps

The current dataset is deliberately small and synthetic. A stronger version could use a public dataset such as the SMS Spam Collection and add:

- TF-IDF features
- precision, recall, and F1-score
- cross-validation
- model comparison
- a larger real-world dataset
- a small web/API interface for inference

## Tech Stack

Python · Pandas · scikit-learn · NLP · Multinomial Naive Bayes
