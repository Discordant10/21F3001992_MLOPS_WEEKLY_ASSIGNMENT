# Week 8: MLSecOps - Data Poisoning Attack & Mitigation Analysis

This repository contains the code, configuration files, and experiment tracking logs for **Week 8: MLSecOps**. The primary objective of this project is to simulate, evaluate, and mitigate **Data Poisoning** attacks on machine learning pipelines using the standard Iris dataset, MLflow, and Google Cloud Platform (GCP) infrastructure.

---

## 📋 Table of Contents

1. [Project Overview](https://www.google.com/search?q=%23project-overview)
2. [Repository Structure](https://www.google.com/search?q=%23repository-structure)
3. [Task 1: Threat Vectors in ML Systems](https://www.google.com/search?q=%23task-1-threat-vectors-in-ml-systems)
4. [Task 2: Data Poisoning Dataset Generation](https://www.google.com/search?q=%23task-2-data-poisoning-dataset-generation)
5. [Task 3: GCP Deployment & MLflow Architecture](https://www.google.com/search?q=%23task-3-gcp-deployment--mlflow-architecture)
6. [Task 4: Performance Evaluation & Results](https://www.google.com/search?q=%23task-4-performance-evaluation--results)
7. [Task 5: Mitigation & Data Quality Analysis](https://www.google.com/search?q=%23task-5-mitigation--data-quality-analysis)
8. [Setup & Execution Guide](https://www.google.com/search?q=%23setup--execution-guide)

---

## 🎯 Project Overview

In this module, we explore how data corruption affects model training, performance, and generalizability. We inject varying degrees of feature and label noise (0%, 5%, 10%, and 50%) into the Iris dataset, train Random Forest Classifiers on each variant, and track metric degradation against a clean validation set using an MLflow tracking server deployed on GCP.

---

## 📁 Repository Structure

```text
.
├── data/                           # Generated datasets (local/ignored in git)
│   ├── iris_clean.csv              # Baseline clean dataset (0% poisoning)
│   ├── iris_poisoned_5.csv         # 5% corrupted samples
│   ├── iris_poisoned_10.csv        # 10% corrupted samples
│   └── iris_poisoned_50.csv        # 50% corrupted samples
├── generate_poisoned_data.py       # Script to corrupt features and targets
├── train.py                        # Model training, evaluation, & MLflow logging
├── params.yaml                     # Configuration parameters for dataset & model
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project documentation

```

---

## 🛡️ Task 1: Threat Vectors in ML Systems

| Threat Vector | Target Pipeline Stage | Mechanism & Real-World Example |
| --- | --- | --- |
| **Data Poisoning** | Data Collection & Preprocessing | Injecting corrupted samples or mislabeled data to degrade model performance or create backdoor triggers.<br>

<br>*Example*: Submitting deceptive training labels to crowdsourced dataset repositories. |
| **Adversarial Examples** | Inference & Prediction | Applying imperceptible noise to input features during runtime to force high-confidence misclassifications.<br>

<br>*Example*: Placing specific tape patterns on speed limit signs to trick self-driving car vision systems. |
| **Model Extraction** | Deployment & Serving API | Querying a deployed model endpoint systematically with crafted inputs to reconstruct a duplicate local surrogate model.<br>

<br>*Example*: Replicating proprietary credit scoring APIs via black-box input-output harvesting. |
| **Prompt Injection** | Input Processing (LLMs) | Embedding untrusted instructions inside user inputs or documents to override system prompts and execute unauthorized tasks.<br>

<br>*Example*: Hiding commands in uploaded PDFs that compel an AI assistant to leak sensitive user information. |

---

## 🧪 Task 2: Data Poisoning Dataset Generation

The `generate_poisoned_data.py` script loads the baseline Iris dataset and creates 4 dataset variants:

* **0% (Clean Baseline)**: Original Iris dataset (150 rows).
* **5% Poisoned**: 7 samples corrupted.
* **10% Poisoned**: 15 samples corrupted.
* **50% Poisoned**: 75 samples corrupted.

### Corruption Methodology

1. **Target Selection**: Random indices are selected without replacement according to the target percentage.
2. **Feature Noise**: All four features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) for target rows are replaced with uniform random values bounded within original feature minimums and maximums.
3. **Label Noise**: Corresponding targets are assigned a random label chosen from available class targets $\{0, 1, 2\}$.

---

## ☁️ Task 3: GCP Deployment & MLflow Architecture

### Network & Infrastructure Specifications

* **GCP Project ID**: `project-ccb283b3-613f-40a9-a23`
* **MLflow Tracking Server VM**: `mlflow-server` (`10.128.0.4`)
* **Model Training VM**: `iris-training-vm` (`10.128.0.7`)
* **Firewall Rule**: `allow-mlflow-5000` opening TCP port 5000 on network tag `mlflow-server`.

```text
┌─────────────────────────┐         HTTP / Port 5000         ┌─────────────────────────┐
│    iris-training-vm     │ ───────────────────────────────> │      mlflow-server      │
│     (10.128.0.7)        │   Logs Params, Metrics, Models   │      (10.128.0.4)       │
└─────────────────────────┘                                  └─────────────────────────┘

```

---

## 📊 Task 4: Performance Evaluation & Results

All models were evaluated on a **fixed 20% clean validation holdout set** (30 samples) stratified from the original clean dataset to measure absolute generalizability.

### Metric Summary Table

| Experiment Run Name | Poisoning Level | Training Samples Corrupted | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| --- | --- | --- | --- | --- | --- | --- |
| `Run_0pct_poison` | **0%** | 0 / 150 | **0.9667** | **0.9697** | **0.9667** | **0.9665** |
| `Run_5pct_poison` | **5%** | 7 / 150 | **0.9667** | **0.9697** | **0.9667** | **0.9665** |
| `Run_10pct_poison` | **10%** | 15 / 150 | **0.9000** | **0.9111** | **0.9000** | **0.8980** |
| `Run_50pct_poison` | **50%** | 75 / 150 | **0.3333** | **0.1111** | **0.3333** | **0.1667** |

### Key Analytical Findings

1. **Degradation Point**:
* **0% to 5%**: The model demonstrates robustness against minor noise (0.0% accuracy loss), as ensemble decision trees effectively isolate sparse outlier branches.
* **10% Noise**: Measurable degradation occurs; accuracy drops from **96.67% to 90.00%**.
* **50% Noise**: Complete model collapse occurs. Accuracy falls to **33.33%**, which is mathematically equivalent to random guessing on a 3-class balanced dataset.


2. **Metrics Affected First**: Precision and F1 scores degrade before overall accuracy because corrupted boundary points cause class-specific false positives before altering broad majority classifications.

---

## 🔒 Task 5: Mitigation & Data Quality Analysis

### Production Mitigation Strategies

1. **Automated Schema & Range Enforcement**: Utilize data validation frameworks (e.g., Great Expectations) prior to model ingestion to flag feature values exceeding statistical expected boundaries.
2. **Outlier & Anomaly Detection**: Implement Isolation Forests, Mahalanobis Distance, or Local Outlier Factor (LOF) algorithms during dataset preprocessing to detect unnatural feature combinations.
3. **Data Lineage & Cryptographic Verification**: Track data provenance using version control systems (e.g., DVC) and verify data checksums before pipeline ingestion to guarantee source integrity.

### Data Quantity vs. Data Quality Tradeoff

* **Is Collecting More Data Enough?** No. Increasing data volume does not offset poisoning if the ingestion pipeline continues to ingest data with a constant noise ratio. If 50% of ingested data is corrupted, scaling from 1,000 to 1,000,000 samples results in 500,000 bad samples, maintaining degraded decision boundaries.
* **Impact of Clean Ratio**: Collecting additional data only helps if the net *clean data ratio* increases. As noise density grows, the dataset size required for model convergence increases non-linearly.

---

## 🚀 Setup & Execution Guide

### Prerequisites

* Python 3.10+
* GCP VM instances (`mlflow-server` and `iris-training-vm`) with SSH access

### 1. Local Environment Setup

```bash
git checkout -b week_8
pip install -r requirements.txt

```

### 2. Configure MLflow Tracking Server (on `mlflow-server` VM)

```bash
# Start MLflow Server listening on all interfaces
nohup mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns > mlflow.log 2>&1 &

```

### 3. Run Pipeline (on `iris-training-vm` VM)

```bash
# Set remote tracking server address
export MLFLOW_TRACKING_URI=http://10.128.0.4:5000

# Execute training and logging script
python train.py

```