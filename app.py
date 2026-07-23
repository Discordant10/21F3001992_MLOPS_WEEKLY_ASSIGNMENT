from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from feast import FeatureStore
from mlflow_utils import load_registered_model


app = FastAPI(
    title="IRIS Prediction API",
    version="1.0.0",
)

print("Loading model...")
model = load_registered_model()
print("Model loaded")

project_root = Path(__file__).resolve().parent

import subprocess

feature_repo = (
    project_root
    / "feature_repo"
)

print("Applying Feast repository...")

subprocess.run(
    ["feast", "apply"],
    cwd=str(feature_repo),
    check=True,
)

store = FeatureStore(
    repo_path=str(feature_repo)
)

print("Feast repository loaded")

class PredictionRequest(BaseModel):
    iris_id: int


@app.get("/")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
    request: PredictionRequest,
):
    features = (
        store.get_online_features(
            features=[
                "iris_features:sepal_length",
                "iris_features:sepal_width",
                "iris_features:petal_length",
                "iris_features:petal_width",
            ],
            entity_rows=[
                {
                    "iris_id": request.iris_id
                }
            ],
        )
        .to_dict()
    )

    if (
        len(features["sepal_length"]) == 0
        or features["sepal_length"][0]
        is None
    ):
        return {
            "error":
            f"No features found "
            f"for iris_id="
            f"{request.iris_id}"
        }

    X = pd.DataFrame(
        {
            "sepal_length": [
                features["sepal_length"][0]
            ],
            "sepal_width": [
                features["sepal_width"][0]
            ],
            "petal_length": [
                features["petal_length"][0]
            ],
            "petal_width": [
                features["petal_width"][0]
            ],
        }
    )

    prediction = model.predict(X)

    return {
        "iris_id": request.iris_id,
        "prediction": str(
            prediction[0]
        ),
    }
