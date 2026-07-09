# IRIS Classification Pipeline with CI/CD

## IIT Madras BS Degree Programme

**Course:** MLOps Weekly Assignment - Week 4

**Roll Number:** 21F3001992

---

# Project Overview

This project implements a complete Machine Learning Operations (MLOps) pipeline for the Iris classification dataset.

The pipeline demonstrates:

- Model Training
- Feature Management using Feast
- Data & Model Versioning using DVC
- Google Cloud Storage Remote
- Continuous Integration using GitHub Actions
- Automated Testing using PyTest
- Continuous Machine Learning (CML) Reports

---

# Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── iris_data_adapted_for_feast.csv
│
├── feature_repo/
│
├── metrics/
│   └── metrics.json
│
├── models/
│   └── iris_model.pkl
│
├── reports/
│
├── tests/
│   ├── test_data_validation.py
│   └── test_model.py
│
├── train.py
├── inference.py
├── dvc.yaml
├── requirements.txt
└── README.md
```

---

# Technologies Used

- Python 3.11
- Scikit-Learn
- Pandas
- NumPy
- Feast
- DVC
- Google Cloud Storage
- GitHub Actions
- PyTest
- CML

---

# Project Workflow

```
Developer

↓

Push Code

↓

GitHub Actions

↓

Install Dependencies

↓

Authenticate to Google Cloud

↓

DVC Pull

↓

Load Dataset + Model

↓

Run Unit Tests

↓

Generate Metrics

↓

Generate Report

↓

Publish PR Comment using CML
```

---

# Dataset

Dataset:

```
IRIS Dataset
```

Features

- sepal_length
- sepal_width
- petal_length
- petal_width

Target

```
species
```

Classes

- setosa
- versicolor
- virginica

---

# DVC Pipeline

Initialize DVC

```bash
dvc init
```

Pull versioned files

```bash
dvc pull
```

Run pipeline

```bash
dvc repro
```

Push changes

```bash
dvc push
```

---

# Feast

Apply feature definitions

```bash
cd feature_repo

feast apply
```

Materialize features

```bash
feast materialize-incremental
```

---

# Train Model

Run

```bash
python train.py
```

Outputs

```
models/iris_model.pkl

metrics/metrics.json
```

---

# Run Inference

```bash
python inference.py
```

---

# Unit Tests

Run

```bash
pytest tests -v
```

Tests Included

## Data Validation

- Dataset exists
- Correct schema
- No missing values
- Numeric feature types
- Valid feature ranges
- Duplicate check

## Model Evaluation

- Model loads successfully
- Predictions generated
- Accuracy threshold
- Precision threshold
- Recall threshold
- F1 threshold

---

# GitHub Actions

Workflow file

```
.github/workflows/ci.yml
```

Runs automatically on

- Push
- Pull Request

Workflow Steps

1. Checkout Repository
2. Install Python
3. Install Dependencies
4. Authenticate Google Cloud
5. DVC Pull
6. Train Model
7. Execute Tests
8. Generate Report
9. Publish CML Comment

---

# Google Cloud Configuration

Create a Service Account with

- Storage Object Viewer
- Storage Object Admin

Create JSON Key

Store JSON as GitHub Secret

```
GCP_SA_KEY
```

---

# GitHub Secrets

Repository

Settings

Secrets and Variables

Actions

Required Secret

```
GCP_SA_KEY
```

---

# Continuous Machine Learning (CML)

After every Pull Request

GitHub Actions

↓

Runs tests

↓

Creates report

↓

Posts report directly on Pull Request

Example

| Metric | Value |
|---------|------:|
| Accuracy | 0.9667 |
| Precision | 0.9673 |
| Recall | 0.9667 |
| F1 Score | 0.9666 |

---

# Expected Accuracy

The trained Random Forest classifier consistently achieves

```
Accuracy > 96%
```

---

# Week-wise Progress

## Week 1

- Vertex AI Pipeline
- Model Training

## Week 2

- DVC
- Google Cloud Storage

## Week 3

- Feast Feature Store

## Week 4

- GitHub Actions
- CI Pipeline
- PyTest
- CML Reports

---

# Author

**Name:** Parag Seth

**Roll Number:** 21F3001992