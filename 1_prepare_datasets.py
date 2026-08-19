import json
import os
from sklearn.datasets import load_iris
from google.cloud import storage

BUCKET_NAME = "irisdata-raw"  

iris = load_iris()
X, y, target_names = iris.data, iris.target, iris.target_names

v1_rows = []
v2_rows = []

for features, target in zip(X, y):
    species_name = target_names[target]

    # Task 1: v1 Raw Feature Format
    v1_in = f"sepal_length: {features[0]}, sepal_width: {features[1]}, petal_length: {features[2]}, petal_width: {features[3]}"
    v1_out = species_name

    v1_rows.append({
        "contents": [
            {"role": "user", "parts": [{"text": v1_in}]},
            {"role": "model", "parts": [{"text": v1_out}]}
        ]
    })

    # Task 2: v2 Natural Language Format
    v2_in = f"A flower specimen has a sepal length of {features[0]} cm, sepal width of {features[1]} cm, petal length of {features[2]} cm, and petal width of {features[3]} cm. Identify the iris species."
    v2_out = f"This is Iris {species_name}."

    v2_rows.append({
        "contents": [
            {"role": "user", "parts": [{"text": v2_in}]},
            {"role": "model", "parts": [{"text": v2_out}]}
        ]
    })

# Save clean JSONL files locally
v1_file = "v1_train.jsonl"
v2_file = "v2_train.jsonl"

with open(v1_file, "w") as f:
    for r in v1_rows:
        f.write(json.dumps(r) + "\n")

with open(v2_file, "w") as f:
    for r in v2_rows:
        f.write(json.dumps(r) + "\n")

# Upload clean files to GCS
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
bucket.blob(v1_file).upload_from_filename(v1_file)
bucket.blob(v2_file).upload_from_filename(v2_file)

# Delete local files from VM 
if os.path.exists(v1_file):
    os.remove(v1_file)
if os.path.exists(v2_file):
    os.remove(v2_file)

print("Files generated in strict Gemini format, uploaded to GCS, and removed from VM.")