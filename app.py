from fastapi import FastAPI

from inference import predict_iris

app = FastAPI()


@app.get("/")
def health():
    return {"status": "running"}


@app.get("/predict/{iris_id}")
def predict(iris_id: int):

    prediction = predict_iris(iris_id)

    return {
        "iris_id": iris_id,
        "prediction": prediction,
    }
