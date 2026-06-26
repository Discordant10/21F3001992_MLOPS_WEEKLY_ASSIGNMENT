import pandas as pd
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ------------------------
# Create folders
# ------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# ------------------------
# Load data
# ------------------------

df = pd.read_csv(
    "data/iris.csv"
)

target = df.columns[-1]

X = df.drop(columns=[target])

y = df[target]

# ------------------------
# Train split
# ------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------
# Train model
# ------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ------------------------
# Evaluate
# ------------------------

preds = model.predict(
    X_test
)

acc = accuracy_score(
    y_test,
    preds
)

# ------------------------
# Save model
# ------------------------

joblib.dump(
    model,
    "models/model.pkl"
)

# ------------------------
# Save metrics
# ------------------------

metrics = {
    "accuracy": float(acc),
    "rows": len(df)
}

with open(
    "metrics/train_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Training Complete")
print("Accuracy:", acc)