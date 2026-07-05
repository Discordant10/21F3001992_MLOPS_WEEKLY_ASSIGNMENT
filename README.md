# MLOps Weekly Assignment – Week 3

## Integrating Feast Feature Store into the IRIS Pipeline

### Student Details

* **Roll Number:** 21F3001992
* **Course:** MLOps
* **Assignment:** Week 3 – Feast Feature Store Integration

---

# Project Overview

This project extends the Week 2 MLOps pipeline by integrating the **Feast Feature Store** into the IRIS machine learning workflow.

The objective is to eliminate training-serving skew by ensuring that both model training and inference retrieve engineered features from a single feature store.

The project retains **DVC** for reproducibility while introducing **Feast** for feature management.

---

# Objectives

* Build a local Feast Feature Repository.
* Define Entity, Data Source and Feature View.
* Register feature definitions using Feast.
* Materialize features into the online store.
* Train the model using Feast Offline Store.
* Perform inference using Feast Online Store.
* Maintain reproducibility using DVC.

---

# Project Structure

```
21F3001992_MLOPS_WEEKLY_ASSIGNMENT/

│
├── data/
│   ├── iris_data_adapted_for_feast.parquet
│   └── iris_data_adapted_for_feast.csv
│
├── feature_repo/
│   ├── feature_store.yaml
│   ├── features.py
│   └── data/
│       ├── registry.db
│       └── online_store.db
│
├── models/
│   └── model.pkl
│
├── metrics/
│   ├── train_metrics.json
│   └── inference_metrics.json
│
├── train.py
├── inference.py
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md
```

---

# Technologies Used

* Python 3.11
* Feast 0.45
* DVC
* Scikit-Learn
* Pandas
* Joblib
* SQLite
* PyArrow

---

# Feature Store Design

## Entity

```
iris_id
```

Each Iris plant is uniquely identified using `iris_id`.

---

## Data Source

```
iris_data_adapted_for_feast.parquet
```

The dataset contains:

* iris_id
* event_timestamp
* created_timestamp
* sepal_length
* sepal_width
* petal_length
* petal_width
* species

---

## Feature View

The feature view exposes the following features:

* sepal_length
* sepal_width
* petal_length
* petal_width

The **species** column is intentionally excluded from the Feature View because it is the prediction target (label) rather than an input feature.

---

# Pipeline Architecture

```
                DVC
                 │
                 ▼
        Feast Apply
                 │
                 ▼
       Materialize Features
                 │
        ┌────────┴────────┐
        ▼                 ▼
Offline Store      Online Store
        │                 │
        ▼                 ▼
     Training        Inference
        │                 │
        └────────┬────────┘
                 ▼
          Random Forest Model
```

---

# Training Workflow

1. Read entity IDs and timestamps.
2. Retrieve historical features using Feast Offline Store.
3. Merge retrieved features with labels.
4. Split into training and testing datasets.
5. Train Random Forest classifier.
6. Save trained model.
7. Save training metrics.

---

# Inference Workflow

1. Accept an Iris Entity ID.
2. Retrieve latest feature values from Feast Online Store.
3. Construct inference feature vector.
4. Load trained model.
5. Predict Iris species.

---

# DVC Pipeline

The project uses DVC to orchestrate the complete workflow.

```
dvc repro
↓
feast apply
↓
feast materialize
↓
python train.py
↓
python inference.py
```

---

# Running the Project

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Apply Feast definitions

```bash
cd feature_repo

feast apply
```

---

## 3. Materialize features

```bash
feast materialize 2025-09-01T00:00:00 2100-01-01T00:00:00
```

---

## 4. Train the model

```bash
python train.py
```

---

## 5. Perform inference

```bash
python inference.py
```

---

## 6. Execute the complete pipeline

```bash
dvc repro
```

---

# Outputs

Training generates:

```
models/model.pkl

metrics/train_metrics.json
```

Inference generates:

```
metrics/inference_metrics.json
```

---

# Assignment Tasks Completed

| Task                        | Status |
| --------------------------- | ------ |
| Initialize Feast Repository | ✅      |
| Define Entity               | ✅      |
| Define Data Source          | ✅      |
| Define Feature View         | ✅      |
| Apply Feature Definitions   | ✅      |
| Materialize Features        | ✅      |
| Offline Feature Retrieval   | ✅      |
| Model Training Using Feast  | ✅      |
| Online Feature Retrieval    | ✅      |
| Real-time Inference         | ✅      |
| DVC Integration             | ✅      |

---

# Notes

* Feast is used as the single source of truth for feature retrieval.
* DVC is retained for pipeline reproducibility.
* Training uses Feast Offline Store (`get_historical_features()`).
* Inference uses Feast Online Store (`get_online_features()`).
* The `species` column is used only as the prediction label and is not included in the Feature View to avoid target leakage.

---

# Future Work

The next enhancement is to replace the local file-based offline store with a Google BigQuery backend while retaining the same training and inference code, demonstrating Feast's storage abstraction capabilities.
