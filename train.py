import argparse
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
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    )

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

    return pd.read_csv(path)

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

def build_model(n_estimators, max_depth):
    """Create model instance."""

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )

def train_model(model, X_train, y_train):
    """Train model."""

    model.fit(X_train, y_train)
    return model

def calculate_metrics(y_true, predictions):
    """Calculate classification metrics."""

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train/test and CV."""

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )
    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )
    cv_scores = cross_val_score(
        model,
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=5,
        scoring="accuracy",
    )
    metrics = {
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1_score": train_metrics["f1_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "confusion_matrix": confusion_matrix(
            y_test,
            test_predictions,
        ).tolist(),
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

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth",
    )
    return parser.parse_args()

# --------------------------------------------------------------------

# Main

# --------------------------------------------------------------------

def main():

    args = parse_arguments()
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(
        f"Training model "
        f"(n_estimators={args.n_estimators}, "
        f"max_depth={args.max_depth})"
    )
    model = build_model(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    model = train_model(
        model,
        X_train,
        y_train,
    )
    metrics = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    print("\nEvaluation Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    save_model(model)
    save_metrics(metrics)
print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()
