import json
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

DATA_PATH = "data/iris_data_adapted_for_feast.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

METRICS_DIR = "metrics"
METRICS_PATH = os.path.join(METRICS_DIR, "train_metrics.json")


# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

def load_dataset(path):
    """Load dataset."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    return df


def preprocess(df):
    """Prepare features and labels."""

    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]

    target_column = "species"

    X = df[feature_columns]
    y = df[target_column]

    return X, y


def train_model(X_train, y_train):
    """Train Random Forest."""

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


def evaluate(model, X_test, y_test):
    """Evaluate model."""

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )

    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def save_metrics(metrics):
    os.makedirs(METRICS_DIR, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {METRICS_PATH}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

def main():

    print("Loading dataset...")

    df = load_dataset(DATA_PATH)

    print(df.head())

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training model...")

    model = train_model(X_train, y_train)

    print("Evaluating model...")

    metrics = evaluate(model, X_test, y_test)

    print("\nEvaluation Metrics")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    save_model(model)

    save_metrics(metrics)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()