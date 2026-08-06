import os
from pyexpat import features

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
    features_list = iris.feature_names
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train[features_list], y_train)
    y_pred = clf.predict(X_test[features_list])

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
    shap_values = explainer.shap_values(X[features_list])

    # Handle both old (list) and new (3D array) SHAP output formats
    if isinstance(shap_values, list):
        shap_virginica = shap_values[2]
    else:
        # For newer SHAP versions: slice all samples, all features, 3rd class
        shap_virginica = shap_values[:, :, 2]

    # Generate and save plot for virginica (Class 2)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_virginica, X[features_list], show=False)
    plt.title("SHAP Summary Plot - Virginica")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/shap_virginica_summary.png")
    plt.close()
    print("SHAP plot for Virginica saved to outputs/shap_virginica_summary.png\n")

    print("--- Task 4: Detecting Data Drift with Evidently ---")
    # Simulate a production dataset by adding a constant offset to petal length
    production_data = X[features_list].copy()
    production_data["petal length (cm)"] += 2.0

    # Generate Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])

    # Capture the output of the run as an evaluation snapshot
    my_eval = drift_report.run(
        reference_data=X[features_list], current_data=production_data
    )

    # Call save_html on the returned snapshot object
    my_eval.save_html("outputs/data_drift_report.html")
    print("Data Drift report generated and saved to outputs/data_drift_report.html\n")

    print("--- Task 5: Generating Markdown Report ---")
    group_str = mf.by_group.to_string()

    # Create the content for the report.md file cleanly
    report_content = (
        "# IRIS Model Evaluation Report\n\n"
        "## Fairness Audit\n"
        "The model was evaluated for fairness across the synthetic `location` attribute using Fairlearn.\n\n"
        "**Metrics by Location Group (0 vs 1):**\n"
        "```text\n"
        f"{group_str}\n"
        "```\n\n"
        "## Explainability\n"
        "A SHAP summary plot has been generated to evaluate feature importance for the 'Virginica' class.\n"
        "* **Artifact:** `shap_virginica_summary.png`\n\n"
        "## Data Drift\n"
        "A baseline data drift report has been generated comparing the training data against simulated production data.\n"
        "* **Artifact:** `data_drift_report.html`\n"
    )

    # Save the report
    with open("outputs/report.md", "w") as f:
        f.write(report_content)

    print("Markdown report generated and saved to outputs/report.md\n")


if __name__ == "__main__":
    main()
