import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from evidently import Report
from evidently.presets import DataDriftPreset
from fairlearn.metrics import MetricFrame
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    print("--- Task 1: Training Model with Location Excluded ---")
    # Load Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    # Add random location attribute (0 or 1)
    np.random.seed(42)
    X["location"] = np.random.choice([0, 1], size=len(X))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train classifier excluding the 'location' attribute
    features = iris.feature_names
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train[features], y_train)
    y_pred = clf.predict(X_test[features])

    print("Model trained successfully.\n")

    print("--- Task 2: Fairness Audit with Fairlearn ---")
    # Define metrics for Fairlearn MetricFrame
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda y_true, y_pred: precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall": lambda y_true, y_pred: recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
    }

    # Compute metrics disaggregated by location
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=X_test["location"],
    )

    print("Metrics by Location Group (0 vs 1):")
    print(mf.by_group)
    print("\n")

    print("--- Task 3: Generating SHAP Summary Plots ---")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X[features])

    # For Iris, class 0: setosa, class 1: versicolor, class 2: virginica
    # Generate and save plot for virginica (Class 2)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values[2], X[features], show=False)
    plt.title("SHAP Summary Plot - Virginica")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/shap_virginica_summary.png")
    plt.close()
    print("SHAP plot for Virginica saved to outputs/shap_virginica_summary.png\n")

    print("--- Task 4: Detecting Data Drift with Evidently ---")
    # Simulate a production dataset by adding a constant offset to petal length
    production_data = X[features].copy()
    production_data["petal length (cm)"] += 2.0

    # Generate Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=X[features], current_data=production_data)

    drift_report.save_html("outputs/data_drift_report.html")
    print("Data Drift report generated and saved to outputs/data_drift_report.html\n")


if __name__ == "__main__":
    main()
