import os
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/iris_data_adapted_for_feast.csv"
METRICS_PATH = "metrics/train_metrics.json"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

# Minimum acceptable performance
MIN_ACCURACY = 0.90
MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_F1 = 0.90


def load_dataset():
    assert os.path.exists(DATA_PATH), (
        "Dataset not found. "
        "Run 'dvc pull' before executing tests."
    )

    return pd.read_csv(DATA_PATH)


def load_model():
    assert os.path.exists(MODEL_PATH), (
        "Model not found. "
        "Run training or execute 'dvc pull'."
    )

    return joblib.load(MODEL_PATH)


def prepare_test_data(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


def test_model_file_exists():
    """Verify trained model exists."""

    assert os.path.exists(MODEL_PATH)


def test_model_can_be_loaded():
    """Verify joblib model loads successfully."""

    model = load_model()

    assert model is not None


def test_model_prediction_shape():
    """Prediction count should equal test sample count."""

    model = load_model()
    df = load_dataset()

    X_test, y_test = prepare_test_data(df)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_prediction_labels():
    """Predictions should contain only valid Iris classes."""

    model = load_model()
    df = load_dataset()

    X_test, _ = prepare_test_data(df)

    predictions = model.predict(X_test)

    valid = {
        "setosa",
        "versicolor",
        "virginica",
    }

    assert set(predictions).issubset(valid)


def test_model_accuracy():
    """Accuracy should exceed threshold."""

    model = load_model()
    df = load_dataset()

    X_test, y_test = prepare_test_data(df)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= MIN_ACCURACY


def test_model_precision():
    """Precision should exceed threshold."""

    model = load_model()
    df = load_dataset()

    X_test, y_test = prepare_test_data(df)

    predictions = model.predict(X_test)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
    )

    assert precision >= MIN_PRECISION


def test_model_recall():
    """Recall should exceed threshold."""

    model = load_model()
    df = load_dataset()

    X_test, y_test = prepare_test_data(df)

    predictions = model.predict(X_test)

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
    )

    assert recall >= MIN_RECALL


def test_model_f1_score():
    """F1 score should exceed threshold."""

    model = load_model()
    df = load_dataset()

    X_test, y_test = prepare_test_data(df)

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    assert f1 >= MIN_F1


def test_metrics_json_exists():
    """Metrics JSON should exist."""

    assert os.path.exists(METRICS_PATH)


def test_metrics_json_contents():
    """Metrics JSON should contain required keys."""

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    required = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    ]

    for key in required:
        assert key in metrics


def test_saved_metrics_threshold():
    """Stored metrics should meet quality thresholds."""

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert metrics["accuracy"] >= MIN_ACCURACY
    assert metrics["precision"] >= MIN_PRECISION
    assert metrics["recall"] >= MIN_RECALL
    assert metrics["f1_score"] >= MIN_F1