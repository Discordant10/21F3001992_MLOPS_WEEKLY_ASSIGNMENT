## Model Card: IRIS Classifier
### 1. Model Details

Architecture: Decision Tree Classifier

Dataset: IRIS Dataset (augmented with a synthetic location attribute)

Objective: Predict the species of an iris flower based on its sepal and petal measurements.

### 2. Data Preprocessing & Infrastructure
The underlying data pipeline runs on Google Cloud Platform, leveraging PySpark for distributed data transformations and utilizing Google Cloud Storage as the primary data repository. Preprocessing protocols include a rigorous data cleaning step to standardize inconsistent city names into a unified format, utilizing a strict mapping list that converts abbreviations like "cmbt" to "Coimbatore" and "trivandrm" to "Trivandrum".

Data quality guardrails are implemented to handle null values during transformation, preventing downstream errors such as blank output tables. The pipeline actively monitors for these nulls, gracefully filtering out corrupted records—such as isolating 306 invalid transactions caused by null transaction_date fields. Prior to model training, the synthetic location attribute (Location 0 vs. Location 1) is intentionally excluded from the feature set to facilitate fairness auditing.

### 3. Fairness and Bias

Audit Tool: Fairlearn

Demographic Parity: The model was audited across the synthetic location groups (0 and 1).

Metrics: The model achieved a 1.0 (100%) score across accuracy, precision, and recall for both location groups, demonstrating no disparate impact or bias across these segments during the evaluation.

### 4. Explainability / Interpretability

Method: SHAP (SHapley Additive exPlanations)

Summary Plot: (Insert shap_virginica_summary.png here)

Feature Contributions: The SHAP analysis specifically evaluates the model's predictions for the 'Virginica' class, visually breaking down which feature measurements most heavily influence the model's decision-making process.

### 5. Out-of-Scope Use & Monitoring

Drift Detection: Evidently

Monitoring Protocol: An initial HTML data drift report has been generated to establish a baseline distribution of the features. This artifact will be used in continuous monitoring to detect any distribution shifts between the training data and live production data, ensuring model performance does not degrade over time.