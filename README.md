# IRIS Classification Pipeline with MLOps

## IIT Madras BS Degree Programme

**Course:** MLOps Weekly Assignment – Week 5
**Roll Number:** 21F3001992

---

# Project Overview

This repository implements a complete Machine Learning Operations (MLOps) pipeline for the IRIS classification dataset.

The project demonstrates an end-to-end ML workflow including:

* Model Training
* Hyperparameter Tuning
* Feature Management using Feast
* Experiment Tracking using MLflow
* Model Registry using MLflow
* Data Versioning using DVC
* Google Cloud Storage (GCS) Remote Storage
* Continuous Integration using GitHub Actions
* Automated Data Validation and Model Evaluation using PyTest
* Continuous Machine Learning (CML) Reporting

---

# Repository Structure

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
│   ├── .gitignore
│   └── train_metrics.json
├── reports/
│   ├── .gitignore
│   └── report.md
├── tests/
│   ├── test_data_validation.py
│   └── test_model.py
├── train.py
├── inference.py
├── mlflow_utils.py
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Technologies Used

* Python 3.11
* Scikit-Learn
* Pandas
* NumPy
* Feast Feature Store
* MLflow
* DVC
* Google Cloud Storage
* GitHub Actions
* PyTest
* CML

---

# Project Workflow

```text
Developer
        │
        ▼
Pull Request
        │
        ▼
GitHub Actions
        │
        ▼
Install Dependencies
        │
        ▼
Authenticate to Google Cloud
        │
        ▼
DVC Pull
        │
        ▼
Train Model
        │
        ▼
Log Experiment to MLflow
        │
        ▼
Register Model in MLflow Registry
        │
        ▼
Run PyTest Suite
        │
        ▼
Generate Markdown Report
        │
        ▼
Publish GitHub Actions Summary
        │
        ▼
Upload Report Artifact
        │
        ▼
Publish Pull Request Comment using CML
```

---

# Dataset

The project uses the Feast-adapted IRIS dataset supplied as part of the course.

Features:

* sepal_length
* sepal_width
* petal_length
* petal_width

Additional Metadata:

* iris_id
* event_timestamp
* created_timestamp

Target:

* species

---

# DVC Pipeline

Initialize DVC

```bash
dvc init
```

Pull versioned data

```bash
dvc pull
```

Run pipeline

```bash
dvc repro
```

Push data updates

```bash
dvc push
```

DVC now tracks datasets and metrics only.

Models are stored and versioned through the MLflow Model Registry.

---

# Feast Feature Store

Apply feature definitions

```bash
cd feature_repo

feast apply
```

Materialize features

```bash
feast materialize 2025-09-01T00:00:00 2100-01-01T00:00:00
```

---

# MLflow Configuration

Configuration is stored in:

```text
params.yaml
```

Example:

```yaml
model:
  n_estimators: 50
  max_depth: 5

mlflow:
  tracking_uri: http://<server-ip>:5000
  experiment_name: iris-random-forest
  registered_model_name: IrisClassifier
```

---

# MLflow Experiment Tracking

Train the model

```bash
python train.py
```

Each training run logs:

* Hyperparameters
* Evaluation Metrics
* Confusion Matrix
* Trained Model Artifact

The trained model is automatically registered in the MLflow Model Registry.

---

# Hyperparameter Tuning

The training pipeline supports hyperparameter tuning through:

```yaml
model:
  n_estimators:
  max_depth:
```

Different parameter combinations create separate MLflow experiment runs, allowing side-by-side comparison through the MLflow UI.

---

# Model Registry

Registered Model:

```text
IrisClassifier
```

The latest registered model version is retrieved automatically during inference and testing.

---

# Run Inference

```bash
python inference.py --id 1
```

Inference:

1. Loads the latest model from the MLflow Model Registry
2. Fetches features from the Feast Online Store
3. Generates predictions

---

# Automated Unit Tests

Run all tests locally

```bash
pytest tests -v
```

## Data Validation Tests

The test suite validates:

* Dataset availability
* Expected schema
* Required timestamp columns
* iris_id column
* Numeric feature types
* Missing values
* Valid species labels
* Finite numerical feature values
* Reasonable feature value ranges

## Model Evaluation Tests

The test suite verifies:

* MLflow registered model loads successfully
* Prediction generation
* Prediction validity
* Accuracy threshold
* Precision threshold
* Recall threshold
* F1-score threshold
* Metrics file generation

---

# GitHub Actions Continuous Integration

Workflow file:

```text
.github/workflows/ci.yml
```

The workflow automatically runs on:

* Every Pull Request

Workflow Steps:

1. Checkout Repository
2. Set up Python
3. Install Dependencies
4. Authenticate with Google Cloud
5. Pull versioned data using DVC
6. Train the model
7. Log experiment to MLflow
8. Execute the PyTest suite
9. Generate a Markdown report
10. Publish the report to the GitHub Actions Summary
11. Upload the report as a GitHub Actions artifact
12. Publish the report as a Pull Request comment using CML

---

# Google Cloud Configuration

Create a Service Account with appropriate Storage permissions.

Generate a JSON key.

Add the JSON key as a GitHub Actions Secret.

Required secret:

```text
GCP_SA_KEY
```

---

# GitHub Secrets

Repository

Settings

→ Secrets and Variables

→ Actions

Required Secret:

```text
GCP_SA_KEY
```

---

# Continuous Machine Learning (CML)

The workflow publishes test results in three different ways.

## 1. GitHub Actions Summary

A formatted Markdown report is displayed directly on the GitHub Actions run page.

## 2. GitHub Actions Artifact

The generated Markdown report is uploaded as a downloadable workflow artifact.

## 3. Pull Request Comment

When the workflow is triggered by a Pull Request, CML automatically publishes the report as a comment on the Pull Request.

The report includes:

* Test execution status
* Test Accuracy
* Test Precision
* Test Recall
* Test F1 Score

---

# Expected Model Performance

Typical performance:

* Test Accuracy ≈ 0.90 – 0.97
* Test Precision ≈ 0.90 – 0.97
* Test Recall ≈ 0.90 – 0.97
* Test F1 Score ≈ 0.90 – 0.97

---

# Week-wise Progress

## Week 1

* Vertex AI Pipeline
* Model Training

## Week 2

* Data Versioning using DVC
* Google Cloud Storage Remote

## Week 3

* Feast Feature Store Integration

## Week 4

* GitHub Actions Continuous Integration
* Automated Data Validation
* Automated Model Evaluation
* DVC Integration within CI
* GitHub Actions Summary Report
* GitHub Actions Artifact Upload
* CML Pull Request Reporting

## Week 5

* Hyperparameter Tuning
* MLflow Experiment Tracking
* MLflow Model Registry
* Registry-Based Inference
* Registry-Based Testing
* Removal of Model Tracking from DVC

---

# Author

**Name:** Parag Seth
**Roll Number:** 21F3001992
