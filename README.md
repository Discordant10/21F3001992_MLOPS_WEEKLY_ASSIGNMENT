# IRIS Classification Pipeline with MLOps

## IIT Madras BS Degree Programme

**Course:** MLOps Weekly Assignment – Week 9

**Student:** Parag Seth

**Roll Number:** 21F3001992

---

## Project Overview

This repository implements a robust, end-to-end Machine Learning Operations (MLOps) pipeline for the IRIS classification dataset. Building upon continuous integration, data versioning, and feature stores, this iteration (Week 9) introduces critical production-grade guardrails: **Model Fairness, Explainability, and Data Drift Detection**.

### Key Pipeline Capabilities:

* **Model Training & Tracking:** Decision Tree Classifier tracked via MLflow.
* **Feature Management:** Feast Feature Store for standardized feature retrieval.
* **Data Versioning:** DVC combined with Google Cloud Storage (GCS).
* **Continuous Integration (CI):** GitHub Actions with automated PyTest suites and CML reporting.
* **Model Fairness:** Fairlearn auditing to ensure demographic parity across synthetic location attributes.
* **Explainability:** SHAP integration for visual feature contribution analysis (focusing on the 'Virginica' class).
* **Data Drift Monitoring:** Evidently HTML reports to track feature distribution shifts between training and production-simulated data.

---

## Technologies Used

| Category | Tools / Libraries |
| --- | --- |
| **Core ML & Data** | Python 3.11, Scikit-Learn, Pandas, NumPy |
| **Feature Store** | Feast |
| **Tracking & Registry** | MLflow |
| **Data Versioning** | DVC, Google Cloud Storage (GCS) |
| **CI/CD & Reporting** | GitHub Actions, PyTest, CML (Continuous Machine Learning) |
| **Trust & Monitoring** | Fairlearn (Fairness), SHAP (Explainability), Evidently (Drift) |

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dvc/
├── data/
│   ├── iris_data_adapted_for_feast.csv
│   └── iris_data_adapted_for_feast.parquet
├── feature_repo/
│   ├── data/
│   ├── feature_store.yaml
│   └── features.py
├── metrics/
│   ├── train_metrics.json
├── outputs/
│   ├── report.md                      # NEW: Consolidated report
│   ├── evidently_drift_report.html    # NEW: Data drift baseline report
│   └── shap_virginica_summary.png     # NEW: SHAP feature importance plot
├── tests/
│   ├── test_data_validation.py
│   └── test_model.py
├── train.py                           # UPDATED: Includes Fairness, SHAP, and Evidently logic
├── inference.py
├── mlflow_utils.py
├── params.yaml
├── dvc.yaml
├── requirements.txt
└── README.md

```

---

## Dataset

The project uses the Feast-adapted IRIS dataset supplied as part of the course, augmented with a synthetic attribute for fairness testing.

* **Training Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`
* **Fairness Attribute:** `location` (Values: 0 or 1). *Note: This feature is explicitly excluded from model training to prevent bias but is retained in the pipeline for Fairlearn demographic parity auditing.*
* **Target:** `species`
* **Metadata:** `iris_id`, `event_timestamp`, `created_timestamp`

---

## Trust, Explainability, and Monitoring (Week 9 Features)

### 1. Fairness Auditing (Fairlearn)

The pipeline utilizes `fairlearn.metrics` to calculate the demographic parity difference across the synthetic `location` attribute. The system asserts that the model performs equally well (tracking accuracy, precision, and recall) regardless of the data's location origin.

### 2. Model Interpretability (SHAP)

To ensure the Decision Tree's predictions are transparent, the pipeline generates a SHAP summary plot (`shap_virginica_summary.png`). This visualization explicitly details which features (e.g., petal length/width) most heavily influence the model when predicting the 'Virginica' class.

### 3. Data Drift Detection (Evidently)

An `evidently_drift_report.html` is generated during the training phase. It compares the baseline training dataset against a simulated production dataset to detect statistical distribution shifts. This artifact serves as a vital monitoring tool for model degradation over time.

---

## Core MLOps Workflows

### DVC Pipeline (Data Versioning)

DVC tracks datasets and metrics, pushing remote states to GCS. Models are versioned through the MLflow Registry.

```bash
dvc pull     # Pull versioned data
dvc repro    # Run pipeline
dvc push     # Push data updates

```

### Feast Feature Store

```bash
cd feature_repo
feast apply
feast materialize 2025-09-01T00:00:00 2100-01-01T00:00:00

```

### MLflow Experiment Tracking

Running the training script logs hyperparameters, evaluation metrics, fairness metrics, confusion matrices, and the model artifact to MLflow.

```bash
python train.py

```

*Configuration is managed in `params.yaml` (e.g., `n_estimators`, `max_depth`).*

### Automated Unit Tests (PyTest)

The testing suite validates dataset integrity, schema expectations, numerical thresholds, missing values, and model performance metrics (Accuracy, Precision, Recall, F1 > 0.90).

```bash
pytest tests -v

```

---

## Continuous Integration (GitHub Actions)

The `.github/workflows/ci.yml` file automates the entire pipeline on every Pull Request.

**Automated Steps:**

1. Setup Python & Install Dependencies.
2. Authenticate to Google Cloud (via `GCP_SA_KEY` secret).
3. Pull data via DVC.
4. Train the model (generates MLflow logs, SHAP plots, and Evidently reports).
5. Execute PyTest suite.
6. Publish Markdown reports, test metrics, and artifacts via Continuous Machine Learning (CML) as a PR comment and GitHub Actions Summary.

---

## Week-wise Progress

* **Week 1:** Vertex AI Pipeline, Basic Model Training.
* **Week 2:** Data Versioning (DVC), Google Cloud Storage Remote.
* **Week 3:** Feast Feature Store Integration.
* **Week 4:** CI/CD via GitHub Actions, PyTest Data/Model Validation, CML PR Reporting.
* **Week 5:** Hyperparameter Tuning, MLflow Tracking & Registry, Registry-Based Inference.
* **...**
* **Week 9:** Integrated Model Fairness (Fairlearn), Explainability (SHAP), and Data Drift Monitoring (Evidently). Excluded sensitive attributes from training while maintaining strict audit trails.