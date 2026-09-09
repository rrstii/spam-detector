# Spam Email Classifier

A simple machine learning project that classifies text messages as spam or ham (not spam) using a Naive Bayes classifier.

## How it works
1. A small set of example messages (spam and ham) is used as training data.
2. The text is converted into numeric features using `CountVectorizer` (bag-of-words).
3. A Multinomial Naive Bayes model is trained on the data.
4. The model is tested on unseen messages to check if it classifies them correctly.

## Run it
```
pip install pandas scikit-learn
python spam.py
```

## Results
- Accuracy: ~90% on the test set
- Correctly classifies new messages like "click here to win a free laptop" as spam and "let's meet at the library at 5" as ham

## Notes
The dataset used here is small and made up for demonstration purposes. For a more robust model, a larger real-world dataset (like the SMS Spam Collection dataset) would give better results.
