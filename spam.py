"""Simple spam-message classifier using a bag-of-words Naive Bayes model."""

from typing import List, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_SEED = 1
TEST_SIZE = 0.3

EMAILS: Sequence[Tuple[str, str]] = [
    ("win a free iphone now click here", "spam"),
    ("congratulations you won a lottery claim your prize", "spam"),
    ("limited time offer buy now and save 50%", "spam"),
    ("you have been selected for a cash reward click link", "spam"),
    ("free money click this link now", "spam"),
    ("act now to claim your free gift card", "spam"),
    ("urgent your account will be suspended click here to verify", "spam"),
    ("get rich quick with this one simple trick", "spam"),
    ("hot singles in your area waiting to chat", "spam"),
    ("you are a winner claim your free vacation now", "spam"),
    ("lowest price on viagra buy now", "spam"),
    ("earn money from home no experience needed", "spam"),
    ("your loan has been approved click to withdraw", "spam"),
    ("exclusive deal just for you click now", "spam"),
    ("free trial no credit card needed sign up now", "spam"),
    ("hey are we still meeting for lunch tomorrow", "ham"),
    ("can you send me the report before friday", "ham"),
    ("happy birthday hope you have a great day", "ham"),
    ("the meeting has been moved to 3pm", "ham"),
    ("don't forget to bring your laptop tomorrow", "ham"),
    ("thanks for helping me with the project", "ham"),
    ("let me know if you are free this weekend", "ham"),
    ("i attached the file you asked for", "ham"),
    ("see you at the gym later", "ham"),
    ("can we reschedule our call to next week", "ham"),
    ("i finished the homework can you check it", "ham"),
    ("mom said dinner is ready come home", "ham"),
    ("here is the invoice for last month", "ham"),
    ("good morning hope you slept well", "ham"),
    ("the professor moved the exam to monday", "ham"),
    ("let's grab coffee sometime this week", "ham"),
]

NEW_MESSAGES = [
    "click here to win a free laptop",
    "hey can you call me back when you're free",
    "urgent claim your prize before it expires",
    "let's meet at the library at 5",
]


def build_dataset() -> pd.DataFrame:
    """Return the example messages as a labeled DataFrame."""
    return pd.DataFrame(EMAILS, columns=["text", "label"])


def build_model() -> Pipeline:
    """Build the vectorization + Naive Bayes pipeline."""
    return Pipeline(
        steps=[
            ("vectorizer", CountVectorizer()),
            ("classifier", MultinomialNB()),
        ]
    )


def train_model(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_SEED
) -> Tuple[Pipeline, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset, train the classifier, and return the test data."""
    if df.empty:
        raise ValueError("Dataset must not be empty")

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    model = build_model()
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> None:
    """Print standard classification metrics."""
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0))

    print("Confusion matrix [ham, spam]:")
    print(confusion_matrix(y_test, predictions, labels=["ham", "spam"]))


def predict_messages(model: Pipeline, messages: List[str]) -> List[str]:
    """Classify a list of previously unseen messages."""
    if not messages:
        return []
    return list(model.predict(messages))


def main() -> None:
    """Run the complete spam-detection example."""
    df = build_dataset()
    print(df.head())
    print("\nCount per label:")
    print(df["label"].value_counts())

    model, _, x_test, _, y_test = train_model(df)
    evaluate_model(model, x_test, y_test)

    print("Predictions for new messages:")
    for message, label in zip(NEW_MESSAGES, predict_messages(model, NEW_MESSAGES)):
        print(f"'{message}' -> {label}")


if __name__ == "__main__":
    main()
