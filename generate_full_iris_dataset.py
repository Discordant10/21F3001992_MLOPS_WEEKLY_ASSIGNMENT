import os
from datetime import datetime, timedelta

import pandas as pd
from sklearn.datasets import load_iris

OUTPUT_CSV = "data/iris_data_adapted_for_feast.csv"
OUTPUT_PARQUET = "data/iris_data_adapted_for_feast.parquet"

def main():

    iris = load_iris(as_frame=True)

    df = iris.frame.copy()

    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    species_mapping = {
        0: "setosa",
        1: "versicolor",
        2: "virginica",
    }

    df["species"] = df["species"].map(species_mapping)

    df["iris_id"] = range(1, len(df) + 1)

    base_time = datetime(2025, 1, 1, 0, 0, 0)

    df["event_timestamp"] = [
        base_time + timedelta(minutes=i)
        for i in range(len(df))
    ]

    df["created_timestamp"] = df["event_timestamp"]

    df = df[
        [
            "iris_id",
            "event_timestamp",
            "created_timestamp",
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "species",
        ]
    ]

    os.makedirs("data", exist_ok=True)

    df.to_csv(OUTPUT_CSV, index=False)

    df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        engine="pyarrow",
    )

    print("Dataset generated successfully")
    print(f"Rows: {len(df)}")
    print()
    print(df["species"].value_counts())
    print()
    print(f"CSV     : {OUTPUT_CSV}")
    print(f"PARQUET : {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
