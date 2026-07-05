import os
import json
import joblib
import pandas as pd

from feast import FeatureStore

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create Output folders
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Connect to feast
store = FeatureStore(
    repo_path="feature_repo"
)

# Reading only Entity info from csv and building entity df. Features come from feast store
entity_df = pd.read_csv(
    "data/iris_data_adapted_for_feast.csv"
)
entity_df["event_timestamp"] = pd.to_datetime(
    entity_df["event_timestamp"],
    utc=True
)

entity_df["created_timestamp"] = pd.to_datetime(
    entity_df["created_timestamp"],
    utc=True
)
entity_rows = entity_df[
    [
        "iris_id",
        "event_timestamp"
    ]
]

# pulling historical features from feast
training_df = store.get_historical_features(

    entity_df=entity_rows,

    features=[
        "iris_features:sepal_length",
        "iris_features:sepal_width",
        "iris_features:petal_length",
        "iris_features:petal_width"
    ]

).to_df()

# merge lables. The four feature columns came from Feast. The label came from the CSV.
training_df = training_df.merge(
    entity_df[
        [
            "iris_id",
            "event_timestamp",
            "species"
        ]
    ],
    on=[
        "iris_id",
        "event_timestamp"
    ]
)
print(training_df.head())


# Create train test splits
X = training_df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]
y = training_df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Train
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

preds = model.predict(X_test)
accuracy = accuracy_score(
    y_test,
    preds,
)

# save model and metrics
joblib.dump(
    model,
    "models/model.pkl",
)

metrics = {
    "accuracy": float(accuracy),
    "rows": len(training_df),
}

with open(
    "metrics/train_metrics.json",
    "w",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )

print()
print("Training Complete")
print("Accuracy:", accuracy)
