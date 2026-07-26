from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from feast import FeatureStore
from mlflow_utils import load_registered_model

app = FastAPI(
    title="IRIS Prediction API",
    version="1.0.0",
)

project_root = Path(__file__).resolve().parent
feature_repo = project_root / "feature_repo"

model = None
store = None


class PredictionRequest(BaseModel):
    iris_id: int


@app.on_event("startup")
def startup_event():
    global model, store

    print("========== API STARTUP ==========")

    print("Loading MLflow model...")
    model = load_registered_model()
    print("Model loaded successfully.")

    print("Initializing Feast FeatureStore...")
    store = FeatureStore(repo_path=str(feature_repo))
    print("FeatureStore initialized.")

    print("========== STARTUP COMPLETE ==========")


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "iris-api"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None or store is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet."
        )

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
        ).to_dict()
    )

    if (
        len(features["sepal_length"]) == 0
        or features["sepal_length"][0] is None
    ):
        raise HTTPException(
            status_code=404,
            detail=f"No features found for iris_id={request.iris_id}"
        )

    X = pd.DataFrame(
        {
            "sepal_length": [features["sepal_length"][0]],
            "sepal_width": [features["sepal_width"][0]],
            "petal_length": [features["petal_length"][0]],
            "petal_width": [features["petal_width"][0]],
        }
    )

    prediction = model.predict(X)

    return {
        "iris_id": request.iris_id,
        "prediction": str(prediction[0]),
    }
