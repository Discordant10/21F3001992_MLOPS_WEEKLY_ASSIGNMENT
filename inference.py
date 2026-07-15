import argparse
from pathlib import Path

import pandas as pd

from mlflow_utils import load_registered_model
from feast import FeatureStore

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--id",
        type=int,
        default=1,
    )

    args = parser.parse_args()
    iris_id = args.id
    model = load_registered_model()
    print("Model loaded")

    project_root = Path(__file__).resolve().parent
    feature_repo = project_root / "feature_repo"
    store = FeatureStore(
        repo_path=str(feature_repo)
    )
    features = store.get_online_features(
        features=[
            "iris_features:sepal_length",
            "iris_features:sepal_width",
            "iris_features:petal_length",
            "iris_features:petal_width",
        ],
        entity_rows=[
            {"iris_id": iris_id}
        ],
    ).to_dict()

    print()
    print("Retrieved Features:")
    print(features)

    X = pd.DataFrame({
        "sepal_length": [features["sepal_length"][0]],
        "sepal_width": [features["sepal_width"][0]],
        "petal_length": [features["petal_length"][0]],
        "petal_width": [features["petal_width"][0]],
    })

    prediction = model.predict(X)

    print()
    print("Prediction:")
    print(prediction[0])


if __name__ == "__main__":
    main()
