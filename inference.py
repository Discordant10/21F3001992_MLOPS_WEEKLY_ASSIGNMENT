from google.cloud import storage

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

import pandas as pd
import joblib
import json
import argparse

# ------------------------
# Arguments
# ------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_version",
    required=True
)

parser.add_argument(
    "--artifact_folder",
    required=True
)

args = parser.parse_args()

DATA_VERSION = args.data_version

ARTIFACT_FOLDER = args.artifact_folder

# ------------------------
# Config
# ------------------------

DATA_BUCKET = "irisdata-week1"

ARTIFACT_BUCKET = "irisdata-week1-artefacts"

EVAL_FILE = f"eval/{DATA_VERSION}_eval.csv"

# ------------------------
# Download files
# ------------------------

client = storage.Client()

bucket = client.bucket(
    DATA_BUCKET
)

bucket.blob(
    EVAL_FILE
).download_to_filename(
    "eval.csv"
)

artifact_bucket = client.bucket(
    ARTIFACT_BUCKET
)

artifact_bucket.blob(
    f"{ARTIFACT_FOLDER}/model.pkl"
).download_to_filename(
    "model.pkl"
)

# ------------------------
# Load model
# ------------------------

model = joblib.load(
    "model.pkl"
)

# ------------------------
# Load evaluation data
# ------------------------

df = pd.read_csv(
    "eval.csv"
)

target = df.columns[-1]

X = df.drop(
    columns=[target]
)

y = df[target]

# ------------------------
# Predict
# ------------------------

preds = model.predict(X)

acc = accuracy_score(
    y,
    preds
)

report = classification_report(
    y,
    preds
)

# ------------------------
# Save predictions
# ------------------------

pred_df = pd.DataFrame(
    {
        "actual": y,
        "prediction": preds
    }
)

pred_df.to_csv(
    "predictions.csv",
    index=False
)

# ------------------------
# Save metrics
# ------------------------

metrics = {
    "accuracy": float(acc),
    "data_version": DATA_VERSION,
    "artifact_folder": ARTIFACT_FOLDER
}

with open(
    "inference_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

with open(
    "classification_report.txt",
    "w"
) as f:

    f.write(
        report
    )

# ------------------------
# Upload outputs
# ------------------------

for file in [
    "predictions.csv",
    "inference_metrics.json",
    "classification_report.txt"
]:

    artifact_bucket.blob(
        f"{ARTIFACT_FOLDER}/{file}"
    ).upload_from_filename(
        file
    )

print("\nInference Complete")
print("Accuracy:", acc)
print("Artifact Folder:", ARTIFACT_FOLDER)
