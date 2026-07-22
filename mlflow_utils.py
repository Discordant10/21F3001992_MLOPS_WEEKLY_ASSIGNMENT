import mlflow.pyfunc
import yaml

# MLflow settings are stored in params.yaml

PARAMS_PATH = "params.yaml"
MODEL_CACHE = None


def load_params():
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def load_registered_model():
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    params = load_params()
    tracking_uri = params["mlflow"]["tracking_uri"]
    model_name = params["mlflow"]["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}/latest"
    print(f"Loading model from MLflow: " f"{model_uri}")
    MODEL_CACHE = mlflow.pyfunc.load_model(model_uri)
    return MODEL_CACHE
