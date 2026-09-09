import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# small dataset i made myself, normally you would use a bigger csv file
# but this is enough to show how it works
emails = [
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

df = pd.DataFrame(emails, columns=["text", "label"])
print(df.head())
print("\ncount per label:")
print(df["label"].value_counts())

x = df["text"]
y = df["label"]

# turn text into numbers, basically counting words
vectorizer = CountVectorizer()
x_vec = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.3, random_state=1)

model = MultinomialNB()
model.fit(x_train, y_train)

preds = model.predict(x_test)

acc = accuracy_score(y_test, preds)
print(f"\naccuracy: {acc}")

print("confusion matrix:")
print(confusion_matrix(y_test, preds, labels=["ham", "spam"]))

# testing on new messages that were not in the training data
new_messages = [
    "click here to win a free laptop",
    "hey can you call me back when you're free",
    "urgent claim your prize before it expires",
    "let's meet at the library at 5"
]

new_vec = vectorizer.transform(new_messages)
new_preds = model.predict(new_vec)

for msg, label in zip(new_messages, new_preds):
    print(f"'{msg}' -> {label}")
