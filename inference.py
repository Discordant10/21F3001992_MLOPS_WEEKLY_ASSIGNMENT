import joblib
from feast import FeatureStore
from pathlib import Path

# Locate feature repository
PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_REPO = PROJECT_ROOT / "feature_repo"
store = FeatureStore(
    repo_path=str(FEATURE_REPO)
)

#Load model
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)

#Choose Dataset ID (hardcoded for now)
iris_id = 1001

#Retrieve features
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
print(features)

#Convert to 2D array which the model expects
X = [[
    features["sepal_length"][0],
    features["sepal_width"][0],
    features["petal_length"][0],
    features["petal_width"][0],
]]

#Predict
prediction = model.predict(X)
print()
print("Prediction:")
print(prediction[0])

