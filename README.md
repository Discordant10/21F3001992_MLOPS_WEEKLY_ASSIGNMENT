# IRIS Classification MLOps Pipeline (Week 7)

## Overview

This repository implements a complete end-to-end MLOps pipeline for the IRIS classification problem using modern MLOps tools and Google Cloud Platform.

The project covers:

- Data Version Control (DVC)
- Feast Feature Store
- MLflow Model Registry
- FastAPI inference service
- Docker containerization
- Google Artifact Registry
- Google Kubernetes Engine (GKE)
- GitHub Actions CI/CD

The deployed API loads the latest registered model from MLflow, materializes features into Feast's online store, and serves predictions through a REST API running on Kubernetes.

---

# Project Structure

```
.
├── app.py
├── train.py
├── inference.py
├── mlflow_utils.py
├── Dockerfile
├── requirements.txt
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── feature_repo/
├── data/
├── models/
├── metrics/
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── tests/
└── .github/
    └── workflows/
        ├── ci.yml
        └── cd.yml
```

---

# Technology Stack

- Python 3.11
- FastAPI
- Scikit-learn
- Feast
- MLflow
- DVC
- Docker
- GitHub Actions
- Google Cloud Platform
- Artifact Registry
- Google Kubernetes Engine (GKE)

---

# Pipeline Overview

## Training Pipeline

Dataset
↓
Feature Engineering
↓
Feast Apply
↓
Feast Materialize
↓
Model Training
↓
Metrics Generation
↓
MLflow Model Registry
↓
Registered Model

---

## Deployment Pipeline

Git Push

↓

GitHub Actions

↓

Docker Build

↓

Artifact Registry

↓

GKE Deployment

↓

FastAPI Service

↓

Prediction API

---

# Docker

The inference application is containerized using Docker.

Build locally:

```bash
docker build -t iris-api .
```

Run locally:

```bash
docker run -p 8080:8080 iris-api
```

Test:

```bash
curl http://localhost:8080/
```

Prediction:

```bash
curl -X POST http://localhost:8080/predict \
-H "Content-Type: application/json" \
-d '{"iris_id":1}'
```

Expected output:

```json
{
  "iris_id":1,
  "prediction":"setosa"
}
```

---

# Kubernetes

Deployment files are available under:

```
k8s/
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

# MLflow

The inference API automatically loads the latest registered model from MLflow.

```
models:/IrisClassifier/latest
```

---

# Feast

The application automatically:

- Applies the Feast repository
- Materializes the online feature store
- Loads features during inference

Feature View:

```
iris_features
```

Entity:

```
iris_id
```

---

# GitHub Actions

## Continuous Integration

Runs automatically on every push.

Performs:

- Install dependencies
- Run tests
- Execute training pipeline
- Validate artifacts

---

## Continuous Deployment

Runs on pushes to:

```
week_6
```

Pipeline:

1. Checkout repository
2. Authenticate with Google Cloud
3. Configure Docker
4. Build Docker image
5. Push image to Artifact Registry
6. Fetch GKE credentials
7. Deploy updated image to Kubernetes
8. Wait for rollout completion

---

# Google Cloud Services Used

- Compute Engine
- Artifact Registry
- Google Kubernetes Engine
- Cloud IAM

---

# REST API

## Health Check

```
GET /
```

Response

```json
{
  "status":"healthy"
}
```

---

## Prediction

```
POST /predict
```

Input

```json
{
  "iris_id":1
}
```

Response

```json
{
  "iris_id":1,
  "prediction":"setosa"
}
```

---

# Model

Algorithm

```
Random Forest Classifier
```

Problem

```
Multi-class Classification
```

Dataset

```
IRIS Dataset
```

---

# How to Run

Clone repository

```bash
git clone <repository_url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
dvc repro
```

Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Open

```
http://localhost:8080
```

---

# Week-wise Progress

## Week 1

- Project setup
- Dataset preparation

## Week 2

- DVC pipeline
- Data versioning

## Week 3

- Feature engineering

## Week 4

- Feast integration

## Week 5

- MLflow model registry
- FastAPI inference service

## Week 6

- Docker containerization
- Artifact Registry
- Kubernetes deployment
- GitHub Actions CD pipeline

## Week 7

- Fully automated deployment to Google Kubernetes Engine
- Docker image published to Artifact Registry through GitHub Actions
- Automatic Feast repository initialization and feature materialization at application startup
- End-to-end deployment validation
- REST API successfully serving predictions from the deployed Kubernetes service

---

# Author

Parag Seth 
21F3001992
IIT Madras - Online BS Degree
Course: MLOps