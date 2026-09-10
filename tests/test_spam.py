import pandas as pd
import pytest

from spam import build_dataset, build_model, predict_messages, train_model


def test_dataset_has_expected_schema_and_labels():
    df = build_dataset()
    assert list(df.columns) == ["text", "label"]
    assert set(df["label"]) == {"ham", "spam"}
    assert len(df) == 31


def test_model_can_fit_and_predict():
    df = build_dataset()
    model, _, x_test, _, y_test = train_model(df)

    assert len(x_test) == len(y_test)
    predictions = model.predict(x_test)
    assert len(predictions) == len(y_test)
    assert set(predictions).issubset({"ham", "spam"})


def test_predict_messages_returns_expected_count():
    model = build_model()
    df = build_dataset()
    model.fit(df["text"], df["label"])

    messages = ["free prize click now", "are we still meeting today"]
    predictions = predict_messages(model, messages)

    assert len(predictions) == len(messages)


def test_empty_dataset_is_rejected():
    empty = pd.DataFrame(columns=["text", "label"])
    with pytest.raises(ValueError):
        train_model(empty)
