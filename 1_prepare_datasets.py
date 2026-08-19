import json

from google.cloud import storage
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

BUCKET_NAME = "irisdata-raw"

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

v1_train, v2_train, v1_test, v2_test = [], [], [], []

for X, y, v1_list, v2_list in [
    (X_train, y_train, v1_train, v2_train),
    (X_test, y_test, v1_test, v2_test),
]:
    for features, label in zip(X, y):
        species = iris.target_names[label]

        # v1: Raw Feature Format
        v1_list.append(
            {
                "input_text": f"sepal_length: {features[0]}, sepal_width: {features[1]}, petal_length: {features[2]}, petal_width: {features[3]}",
                "output_text": species,
            }
        )

        # v2: Natural Language Format
        v2_list.append(
            {
                "input_text": f"A flower specimen has a sepal length of {features[0]} cm, sepal width of {features[1]} cm, petal length of {features[2]} cm, and petal width of {features[3]} cm. Identify the iris species.",
                "output_text": f"This is Iris {species}.",
            }
        )

# Save files locally
files = {
    "v1_train.jsonl": v1_train,
    "v2_train.jsonl": v2_train,
    "v1_test.jsonl": v1_test,
    "v2_test.jsonl": v2_test,
}

for filename, data in files.items():
    with open(filename, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

# Upload training data to GCS
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
for filename in ["v1_train.jsonl", "v2_train.jsonl"]:
    bucket.blob(filename).upload_from_filename(filename)
    print(f"Uploaded {filename} to gs://{BUCKET_NAME}/")
