# IRIS Classification Pipeline with MLOps

## IIT Madras BS Degree Programme

**Course:** MLOps Weekly Assignment – Week 6  
**Roll Number:** 21F3001992  
**Name:** Parag Seth

---

# Project Overview

This repository implements a complete end-to-end MLOps pipeline for the IRIS classification dataset.

The project demonstrates:

- Data Versioning using DVC
- Feature Management using Feast
- Experiment Tracking using MLflow
- Model Registry using MLflow
- Automated Testing using PyTest
- Continuous Integration using GitHub Actions
- Docker Containerization
- Google Artifact Registry
- Kubernetes Deployment using GKE
- Continuous Deployment using GitHub Actions

---

# System Architecture

```text
                    ┌─────────────────┐
                    │   Developer     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ GitHub Repo     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ GitHub Actions  │
                    │ CD Pipeline     │
                    └────────┬────────┘
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼

    ┌─────────────────┐          ┌─────────────────┐
    │ Docker Build    │          │ Automated Tests │
    └────────┬────────┘          └─────────────────┘
             │
             ▼

    ┌─────────────────────────────────────┐
    │ Google Artifact Registry            │
    └────────┬────────────────────────────┘
             │
             ▼

    ┌─────────────────────────────────────┐
    │ Google Kubernetes Engine (GKE)      │
    └────────┬────────────────────────────┘
             │
             ▼

    ┌─────────────────────────────────────┐
    │ FastAPI IRIS Prediction Service     │
    └────────┬────────────────────────────┘
             │
             ▼

    ┌─────────────────────────────────────┐
    │ Feast Feature Store                 │
    └────────┬────────────────────────────┘
             │
             ▼

    ┌─────────────────────────────────────┐
    │ MLflow Model Registry               │
    └─────────────────────────────────────┘
```

---

# Repository Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── data/
│   ├── iris_data_adapted_for_feast.csv
│   └── iris_data_adapted_for_feast.parquet
│
├── feature_repo/
│   ├── feature_store.yaml
│   ├── features.py
│   └── data/
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── tests/
│   ├── test_data_validation.py
│   └── test_model.py
│
├── Dockerfile
├── .dockerignore
├── app.py
├── train.py
├── inference.py
├── mlflow_utils.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

---

# Technologies Used

## Machine Learning

- Scikit-Learn
- Pandas
- NumPy

## Feature Store

- Feast

## Experiment Tracking

- MLflow

## Data Versioning

- DVC
- Google Cloud Storage

## Testing

- PyTest

## CI/CD

- GitHub Actions

## Deployment

- Docker
- Google Artifact Registry
- Google Kubernetes Engine (GKE)
- FastAPI
- Uvicorn

---

# Dataset

The project uses the Feast-adapted IRIS dataset.

Features:

- sepal_length
- sepal_width
- petal_length
- petal_width

Metadata:

- iris_id
- event_timestamp
- created_timestamp

Target:

- species

---

# DVC Data Versioning

Initialize DVC:

```bash
dvc init
```

Pull data:

```bash
dvc pull
```

Run pipeline:

```bash
dvc repro
```

Push updates:

```bash
dvc push
```

---

# Feast Feature Store

Apply definitions:

```bash
cd feature_repo
feast apply
```

Materialize:

```bash
feast materialize 2024-09-01T00:00:00 2100-01-01T00:00:00
```

---

# MLflow Model Registry

The model is tracked and registered in MLflow.

Registered model:

```text
IrisClassifier
```

The deployed API automatically loads the latest registered version from MLflow.

---

# Training

Run training:

```bash
python train.py
```

The training pipeline:

1. Loads versioned data
2. Trains RandomForestClassifier
3. Evaluates performance
4. Logs metrics to MLflow
5. Registers model in MLflow

---

# Local Inference

Run:

```bash
python inference.py --id 1
```

Steps:

1. Load latest MLflow model
2. Retrieve features from Feast
3. Generate prediction

---

# Docker Containerization

Build image:

```bash
docker build -t iris-api .
```

Run locally:

```bash
docker run -p 8080:8080 iris-api
```

Health check:

```bash
curl http://localhost:8080/
```

Prediction:

```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d '{"iris_id":1}'
```

---

# Google Artifact Registry

Docker images are pushed to:

```text
us-central1-docker.pkg.dev/<PROJECT_ID>/iris-api/iris-api
```

Artifact Registry acts as the container image repository for Kubernetes deployments.

---

# Kubernetes Deployment

Deployment:

```text
k8s/deployment.yaml
```

Service:

```text
k8s/service.yaml
```

Deploy:

```bash
kubectl apply -f k8s/
```

Verify:

```bash
kubectl get deployment
kubectl get pods
kubectl get svc
```

---

# GitHub Actions CI

Workflow:

```text
.github/workflows/ci.yml
```

Functions:

- Install dependencies
- Authenticate to GCP
- Pull DVC data
- Train model
- Run tests
- Generate reports

---

# GitHub Actions Continuous Deployment

Workflow:

```text
.github/workflows/cd.yml
```

Triggered on:

```text
Push to week_6 branch
```

Pipeline Steps:

1. Checkout repository
2. Authenticate with GCP
3. Configure Docker
4. Build Docker image
5. Push image to Artifact Registry
6. Retrieve GKE credentials
7. Deploy to Kubernetes
8. Restart deployment
9. Verify rollout

---

# API Endpoints

## Health Check

```http
GET /
```

Response:

```json
{
  "status": "healthy"
}
```

---

## Prediction

```http
POST /predict
```

Request:

```json
{
  "iris_id": 1
}
```

Response:

```json
{
  "iris_id": 1,
  "prediction": "setosa"
}
```

---

# Deployment Validation

Verify service:

```bash
kubectl get svc
```

Verify pods:

```bash
kubectl get pods
```

Health check:

```bash
curl http://<LOAD_BALANCER_IP>/
```

Prediction:

```bash
curl -X POST http://<LOAD_BALANCER_IP>/predict \
-H "Content-Type: application/json" \
-d '{"iris_id":1}'
```

---

# Week-wise Progress

## Week 1

- Vertex AI Pipeline
- Model Training

## Week 2

- DVC
- GCS Remote Storage

## Week 3

- Feast Feature Store

## Week 4

- GitHub Actions CI
- Automated Testing
- CML Reporting

## Week 5

- MLflow Tracking
- MLflow Model Registry
- Registry-based Inference

## Week 6

- Docker Containerization
- Artifact Registry
- Google Kubernetes Engine
- GitHub Actions Continuous Deployment
- Public Prediction API

---

# Author

**Parag Seth**  
**Roll Number:** 21F3001992