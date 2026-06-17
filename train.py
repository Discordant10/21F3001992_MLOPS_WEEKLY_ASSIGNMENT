from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import joblib
import json
import argparse

from datetime import datetime

# ------------------------
# Arguments
# ------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_version",
    required=True,
    help="raw, v1, v2 ..."
)

args = parser.parse_args()

DATA_VERSION = args.data_version

# ------------------------
# Config
# ------------------------

DATA_BUCKET = "irisdata-week1"
ARTIFACT_BUCKET = "irisdata-week1-artefacts"

TRAIN_FILE = f"train/{DATA_VERSION}_train.csv"

# ------------------------
# Download train data
# ------------------------

client = storage.Client()

bucket = client.bucket(DATA_BUCKET)

bucket.blob(
    TRAIN_FILE
).download_to_filename(
    "train.csv"
)

print(f"Downloaded {TRAIN_FILE}")

# ------------------------
# Load data
# ------------------------

df = pd.read_csv("train.csv")

target = df.columns[-1]

X = df.drop(columns=[target])

y = df[target]

# ------------------------
# Train model
# ------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

preds = model.predict(X)

acc = accuracy_score(y, preds)

# ------------------------
# Artifact folder
# ------------------------

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

artifact_folder = (
    f"{DATA_VERSION}_{timestamp}"
)

# ------------------------
# Save model
# ------------------------

joblib.dump(
    model,
    "model.pkl"
)

# ------------------------
# Save metrics
# ------------------------

metrics = {
    "accuracy": float(acc),
    "rows": len(df),
    "data_version": DATA_VERSION,
    "artifact_folder": artifact_folder
}

with open(
    "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ------------------------
# Save log
# ------------------------

with open(
    "train.log",
    "w"
) as f:

    f.write(
        f"Version: {DATA_VERSION}\n"
        f"Rows: {len(df)}\n"
        f"Accuracy: {acc}\n"
    )

# ------------------------
# Upload artifacts
# ------------------------

artifact_bucket = client.bucket(
    ARTIFACT_BUCKET
)

for file in [
    "model.pkl",
    "metrics.json",
    "train.log"
]:

    artifact_bucket.blob(
        f"{artifact_folder}/{file}"
    ).upload_from_filename(
        file
    )

print("\nTraining Complete")
print("Accuracy:", acc)
print("Artifact Folder:", artifact_folder)
