# Week 2 Assignment – Integrating DVC into the IRIS ML Pipeline

## Student Details

* **Student ID:** 21F3001992
* **Course:** MLOps
* **Week:** 2

---

# Project Overview

This project demonstrates the integration of **Data Version Control (DVC)** into an IRIS machine learning pipeline.

The objective is to version control datasets and trained model artifacts while storing the actual data in a **Google Cloud Storage (GCS)** remote. Git is used for versioning the source code and DVC metadata, while DVC manages large files such as datasets and trained models.

The project includes:

* DVC initialization
* Google Cloud Storage configured as the DVC remote
* Versioning of datasets and trained models
* Training and inference scripts
* Dataset augmentation across multiple iterations
* Switching between dataset/model versions using Git and DVC checkout

---

# Repository Structure

```text
21F3001992_MLOPS_WEEKLY_ASSIGNMENT

├── data/
│   └── iris.csv.dvc
│
├── models/
│   └── model.pkl.dvc
│
├── metrics/
│   ├── train_metrics.json.dvc
│   ├── inference_metrics.json.dvc
│   └── classification_report.txt.dvc
│
├── train.py
├── inference.py
├── main.ipynb
├── requirements.txt
├── dvc.yaml
├── dvc.lock
├── README.md
├── .gitignore
└── .dvc/
```

---

# Technology Stack

* Python
* Google Cloud Platform (Vertex AI Workbench)
* Google Cloud Storage
* Git
* DVC
* Scikit-learn
* Pandas

---

# DVC Remote Storage

The project uses **Google Cloud Storage** as the default DVC remote.

The remote stores:

* Dataset versions
* Trained model versions
* Metrics
* Other tracked artifacts

Git stores only lightweight `.dvc` pointer files.

---

# Project Workflow

1. Load the IRIS dataset.
2. Store the dataset under the `data/` directory.
3. Track the dataset using DVC.
4. Train a Random Forest classifier.
5. Save the trained model under `models/`.
6. Save training metrics under `metrics/`.
7. Track model and metrics using DVC.
8. Run inference.
9. Save inference metrics and classification report.
10. Track updated artifacts using DVC.

---

# Running the Project

## Train the Model

```bash
python train.py
```

The script:

* Loads `data/iris.csv`
* Splits the dataset into training and testing sets
* Trains a Random Forest classifier
* Saves the trained model
* Saves training metrics

---

## Run Inference

```bash
python inference.py
```

The script:

* Loads the trained model
* Evaluates it on the test split
* Generates predictions
* Saves inference metrics
* Saves the classification report

---

# DVC Commands Used

## Initialize DVC

```bash
dvc init
```

---

## Configure GCS Remote

```bash
dvc remote add -d gcsremote gs://<YOUR_DVC_BUCKET>
```

---

## Track Dataset

```bash
dvc add data/iris.csv
```

---

## Track Model

```bash
dvc add models/model.pkl
```

---

## Track Metrics

```bash
dvc add metrics/train_metrics.json
dvc add metrics/inference_metrics.json
dvc add metrics/classification_report.txt
```

---

## Push Data to Remote

```bash
dvc push
```

---

## Pull Data from Remote

```bash
dvc pull
```

---

## Restore Files

```bash
dvc checkout
```

---

# Multiple Dataset Versions

To simulate multiple data iterations:

1. Modify or augment `data/iris.csv`.
2. Run:

```bash
dvc add data/iris.csv
git add .
git commit -m "Dataset Version X"
dvc push
```

3. Retrain the model.

Each Git commit references the corresponding DVC-tracked dataset and model versions, enabling complete reproducibility.

---

# Restoring Previous Versions

To restore an earlier dataset/model combination:

```bash
git checkout <commit_hash>
dvc checkout
```

To return to the latest version:

```bash
git checkout week_2
dvc checkout
```

---

# Files Tracked by DVC

* `data/iris.csv`
* `models/model.pkl`
* `metrics/train_metrics.json`
* `metrics/inference_metrics.json`
* `metrics/classification_report.txt`

---

# Reproducibility

The project is fully reproducible.

A reviewer can:

1. Clone the repository.
2. Install dependencies.
3. Configure the DVC remote.
4. Run:

```bash
dvc pull
```

5. Execute:

```bash
python train.py
python inference.py
```

to reproduce the pipeline.

---

# Assignment Objectives Covered

* DVC initialized
* Google Cloud Storage configured as DVC remote
* Dataset versioned using DVC
* Model versioned using DVC
* Multiple dataset iterations created
* Git + DVC checkout demonstrated
* Fully reproducible repository
