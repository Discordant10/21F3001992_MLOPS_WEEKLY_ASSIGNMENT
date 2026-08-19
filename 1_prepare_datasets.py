import json
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from google.cloud import storage

BUCKET_NAME = os.getenv("GCP_BUCKET", "irisdata-raw")

iris = load_iris()
X, y, target_names = iris.data, iris.target, iris.target_names

# 80/20 Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def create_records(X_data, y_data):
    v1_train, v2_train = [], []
    v1_test, v2_test = [], []

    for features, target in zip(X_data, y_data):
        species_name = target_names[target]

        v1_in = f"sepal_length: {features[0]}, sepal_width: {features[1]}, petal_length: {features[2]}, petal_width: {features[3]}"
        v1_out = species_name

        v2_in = f"A flower specimen has a sepal length of {features[0]} cm, sepal width of {features[1]} cm, petal length of {features[2]} cm, and petal width of {features[3]} cm. Identify the iris species."
        v2_out = f"This is Iris {species_name}."

        # Training format (Strict Gemini contents schema)
        v1_train.append({
            "contents": [
                {"role": "user", "parts": [{"text": v1_in}]},
                {"role": "model", "parts": [{"text": v1_out}]}
            ]
        })
        v2_train.append({
            "contents": [
                {"role": "user", "parts": [{"text": v2_in}]},
                {"role": "model", "parts": [{"text": v2_out}]}
            ]
        })

        # Test format (Easy evaluation parsing)
        v1_test.append({"input_text": v1_in, "output_text": v1_out})
        v2_test.append({"input_text": v2_in, "output_text": v2_out})

    return v1_train, v2_train, v1_test, v2_test

v1_train, v2_train, v1_test, v2_test = create_records(X_train, y_train)
_, _, v1_eval, v2_eval = create_records(X_test, y_test)

files = {
    "v1_train.jsonl": v1_train,
    "v2_train.jsonl": v2_train,
    "v1_test.jsonl": v1_eval,
    "v2_test.jsonl": v2_eval
}

# Write files locally
for filename, rows in files.items():
    with open(filename, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# Upload to GCS
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
for filename in files.keys():
    bucket.blob(filename).upload_from_filename(filename)

print("Train and test datasets created and uploaded to GCS.")