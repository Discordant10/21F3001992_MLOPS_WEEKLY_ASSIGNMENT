import pandas as pd
import joblib
import json
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from sklearn.model_selection import train_test_split

# ------------------------
# Create folders
# ------------------------

os.makedirs(
    "metrics",
    exist_ok=True
)

# ------------------------
# Load data
# ------------------------

df = pd.read_csv(
    "data/iris.csv"
)

target = df.columns[-1]

X = df.drop(
    columns=[target]
)

y = df[target]

# ------------------------
# Same split as training
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------
# Load model
# ------------------------

model = joblib.load(
    "models/model.pkl"
)

# ------------------------
# Predict
# ------------------------

preds = model.predict(
    X_test
)

acc = accuracy_score(
    y_test,
    preds
)

report = classification_report(
    y_test,
    preds
)

# ------------------------
# Save metrics
# ------------------------

metrics = {
    "accuracy": float(acc)
}

with open(
    "metrics/inference_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

with open(
    "metrics/classification_report.txt",
    "w"
) as f:

    f.write(
        report
    )

print("Inference Complete")
print("Accuracy:", acc)