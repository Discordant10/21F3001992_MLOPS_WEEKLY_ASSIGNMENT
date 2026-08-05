import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from generate_poisoned_data import generate_datasets

def run_pipeline():
    data_dir = "data"
    generate_datasets(data_dir)

    # Load clean dataset to build clean validation test set
    clean_df = pd.read_csv(os.path.join(data_dir, "iris_clean.csv"))
    X_clean = clean_df.drop(columns=["target"])
    y_clean = clean_df["target"]

    # Fixed clean validation set to evaluate true model performance across all runs
    _, X_val, _, y_val = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    poison_configs = [
        ("0% Clean", os.path.join(data_dir, "iris_clean.csv"), 0.00),
        ("5% Poisoned", os.path.join(data_dir, "iris_poisoned_5.csv"), 0.05),
        ("10% Poisoned", os.path.join(data_dir, "iris_poisoned_10.csv"), 0.10),
        ("50% Poisoned", os.path.join(data_dir, "iris_poisoned_50.csv"), 0.50),
    ]

    mlflow.set_experiment("IRIS_MLSecOps_Poisoning_Analysis")

    for name, filepath, level in poison_configs:
        df = pd.read_csv(filepath)
        X_train = df.drop(columns=["target"])
        y_train = df["target"]

        with mlflow.start_run(run_name=f"Run_{int(level*100)}pct_poison"):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average="weighted")
            rec = recall_score(y_val, y_pred, average="weighted")
            f1 = f1_score(y_val, y_pred, average="weighted")

            # Log Parameters
            mlflow.log_param("poisoning_level_pct", int(level * 100))
            mlflow.log_param("poisoning_ratio", level)
            mlflow.log_param("total_training_samples", len(X_train))
            mlflow.log_param("corrupted_samples_count", int(len(X_train) * level))

            # Log Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)

            # Log Model Artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    run_pipeline()