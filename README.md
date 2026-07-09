# IRIS Classification Pipeline with CI/CD

## IIT Madras BS Degree Programme

**Course:** MLOps Weekly Assignment – Week 4
**Roll Number:** 21F3001992
---

# Project Overview

This repository implements a complete Machine Learning Operations (MLOps) pipeline for the IRIS classification dataset.

The project demonstrates an end-to-end ML workflow including:

* Model Training
* Feature Management using Feast
* Data and Model Versioning using DVC
* Google Cloud Storage (GCS) Remote Storage
* Continuous Integration using GitHub Actions
* Automated Data Validation and Model Evaluation using PyTest
* Continuous Machine Learning (CML) Reporting
* Automated CI execution on every Push and Pull Request

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
│   └── train_metrics.json (generated)
├── models/
│   ├── .gitignore
│   └── model.pkl (generated)
├── reports/
│   ├── .gitignore
│   └── report.md (generated)
├── tests/
│   ├── test_data_validation.py
│   └── test_model.py
├── train.py
├── inference.py
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
Push / Pull Request
        │
        ▼
GitHub Actions
        │
        ▼
Checkout Repository
        │
        ▼
Install Python & Dependencies
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
Run PyTest Suite
        │
        ▼
Generate Markdown Test Report
        │
        ▼
Publish GitHub Actions Summary
        │
        ▼
Upload Report Artifact
        │
        ▼
Publish Pull Request Comment using CML
(Pull Requests only)
```

---

# Dataset

The project uses the **Feast-adapted IRIS dataset** supplied as part of the course.

Features:
* sepal_length
* sepal_width
* petal_length
* petal_width

Additional metadata:
* event_timestamp
* created_timestamp
* iris_id

Target:
* species

---

# DVC Pipeline

Initialize DVC
```bash
dvc init
```

Pull versioned artifacts
```bash
dvc pull
```

Run the complete pipeline
```bash
dvc repro
```

Push updated artifacts
```bash
dvc push
```
---

# Feast Feature Store

Apply Feature Repository
```bash
cd feature_repo

feast apply
```

Materialize Features
```bash
feast materialize 2025-09-01T00:00:00 2100-01-01T00:00:00
```

---

# Model Training
Train the model
```bash
python train.py
```

Generated outputs
```text
models/model.pkl
metrics/train_metrics.json
```

---

# Run Inference
```bash
python inference.py
```

---

# Automated Unit Tests
Run all tests locally
```bash
pytest tests -v
```

## Data Validation Tests

The CI pipeline validates:
* Dataset availability
* Expected schema
* Required timestamp columns
* iris_id column
* Numeric feature types
* Missing values
* Valid species labels
* Presence of valid classes
* Finite numerical feature values
* Reasonable feature value ranges

## Model Evaluation Tests
The CI pipeline verifies:
* Trained model exists
* Model loads successfully
* Prediction generation
* Prediction validity
* Accuracy threshold
* Precision threshold
* Recall threshold
* F1-score threshold
* Metrics file generation

---

# GitHub Actions Continuous Integration
Workflow file
```text
.github/workflows/ci.yml
```

The workflow automatically runs on:
* Every Push
* Every Pull Request

Workflow Steps
1. Checkout Repository
2. Set up Python
3. Install Dependencies
4. Authenticate with Google Cloud
5. Pull versioned artifacts using DVC
6. Train the model
7. Execute the PyTest suite
8. Generate a Markdown test report
9. Publish the report to the GitHub Actions Summary
10. Upload the report as a GitHub Actions artifact
11. Publish the report as a Pull Request comment using CML (Pull Requests only)
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
Required Secret
```text
GCP_SA_KEY
```
---

# Continuous Machine Learning (CML)
The workflow publishes test results in three different ways.

## 1. GitHub Actions Summary
A formatted Markdown report is displayed directly on the GitHub Actions run page for every workflow execution.

## 2. GitHub Actions Artifact
The generated Markdown report is uploaded as a downloadable workflow artifact.

## 3. Pull Request Comment
When the workflow is triggered by a Pull Request, CML automatically publishes the report as a comment on the Pull Request conversation.
This report includes:
* Test execution status
* Accuracy
* Precision
* Recall
* F1-score
---

# Expected Model Performance
The trained Random Forest model is expected to satisfy the minimum evaluation thresholds defined in the automated test suite.
Typical performance:
* Accuracy ≈ 0.96
* Precision ≈ 0.96
* Recall ≈ 0.96
* F1-score ≈ 0.96
---

# Week-wise Progress
## Week 1
* Vertex AI Pipeline
* Model Training

## Week 2
* Data & Model Versioning using DVC
* Google Cloud Storage Remote

## Week 3
* Feast Feature Store Integration

## Week 4
* GitHub Actions Continuous Integration
* Automated Data Validation
* Automated Model Evaluation
* DVC Integration within CI
* GitHub Actions Summary Report
* GitHub Actions Report Artifact
* Continuous Machine Learning (CML) Pull Request Reporting
---

# Author
**Name:** Parag Seth
**Roll Number:** 21F3001992
