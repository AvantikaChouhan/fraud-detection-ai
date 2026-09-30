# FraudStream AI — Complete Project Documentation

## 1. Project Overview

**FraudStream AI** is an end-to-end **Data Engineering-focused fraud detection and AI analytics platform**.

The project demonstrates how a modern data engineering pipeline can process a highly imbalanced credit card transaction dataset, transform it through a Databricks Medallion Architecture, perform data quality validation and analytics, apply a machine learning fraud detection model, maintain fraud detection business rules using SCD Type 2, and integrate a local Large Language Model for supporting fraud explanations.

The project is intentionally designed with **Data Engineering as the primary focus**, while Machine Learning and Generative AI are integrated as supporting components.

### Project Focus

```text
Data Engineering     → Primary Layer
Machine Learning     → Fraud Detection Layer
Generative AI        → Explanation Layer
Dashboard            → Analytics / Investigation Layer
```

Approximate project focus:

- Data Engineering: ~70%
- Machine Learning: ~20%
- Generative AI: ~10%

---

# 2. Problem Statement

Financial transaction fraud detection is a challenging data engineering and machine learning problem because fraudulent transactions are usually a very small percentage of total transactions.

A fraud analytics platform needs to:

1. Ingest transaction data reliably.
2. Preserve raw source data.
3. Validate and clean incoming records.
4. Handle duplicate and invalid records.
5. Build analytics-ready data models.
6. Detect potentially fraudulent transactions.
7. Store model predictions and probabilities.
8. Provide SQL-based analytical insights.
9. Monitor data quality.
10. Maintain changing fraud detection business rules.
11. Provide understandable explanations for model predictions.
12. Present results through an interactive dashboard.

FraudStream AI demonstrates these capabilities using a historical credit card transaction dataset.

---

# 3. Project Objectives

The main objectives of the project are:

- Build an end-to-end Data Engineering pipeline using Databricks.
- Implement a Bronze → Silver → Gold Medallion Architecture.
- Perform data validation and duplicate handling.
- Create analytics-ready Delta tables.
- Build a fraud detection model using Spark ML.
- Handle severe class imbalance using class weighting.
- Store and analyze fraud predictions.
- Implement SQL-based fraud analytics.
- Implement reusable data quality checks.
- Demonstrate SCD Type 2 using fraud detection business rules.
- Integrate a local LLM using Ollama.
- Generate grounded explanations for selected fraud predictions.
- Build an interactive Streamlit investigation dashboard.
- Maintain a clean, modular, Git-ready project structure.

---

# 4. Architecture

The current implemented architecture is:

```text
                    ┌─────────────────────────┐
                    │   Kaggle Fraud Dataset  │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │  Databricks Source      │
                    │  creditcard_source      │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │         BRONZE          │
                    │ Raw Ingestion Layer     │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │         SILVER          │
                    │ Validation + Cleaning   │
                    │ Duplicate Removal       │
                    │ Quarantine              │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │          GOLD           │
                    │ Analytics-Ready Data    │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │     Fraud ML Model      │
                    │ Weighted Logistic Reg.  │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │ Fraud Predictions       │
                    │ Probability + Decision  │
                    └────────────┬────────────┘
                                 ↓
              ┌──────────────────┼──────────────────┐
              ↓                  ↓                  ↓
       SQL Analytics       Data Quality         SCD Type 2
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │       Local GenAI       │
                    │   Ollama + Llama 3.2    │
                    └────────────┬────────────┘
                                 ↓
                    ┌─────────────────────────┐
                    │   Streamlit Dashboard   │
                    └─────────────────────────┘
```

---

# 5. Technology Stack

## Data Engineering

- Python
- PySpark
- Databricks
- Delta Lake
- Spark SQL
- Medallion Architecture

## Machine Learning

- Spark ML
- Logistic Regression
- VectorAssembler
- Class-weighted learning
- Fraud probability scoring

## Generative AI

- Ollama
- Llama 3.2 3B
- Python Ollama Client

## Dashboard

- Streamlit
- Plotly
- Pandas

## Development

- Git
- Git-ready project structure

---

# 6. Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

Dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dataset Profile

Original dataset:

```text
Rows    : 284,807
Columns : 31
```

Columns:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
Class
```

## Feature Types

The dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The `V1–V28` features are anonymized.

Therefore, the project does not assign business meanings to individual V1–V28 features.

## Target

```text
Class = 0 → Normal
Class = 1 → Fraud
```

## Class Distribution

```text
Normal Transactions : 284,315
Fraud Transactions  : 492
```

Fraud represents approximately:

```text
0.172749%
```

of the original dataset.

This makes the problem highly imbalanced.

---

# 7. Initial Data Profiling

Initial profiling identified:

```text
Rows                 : 284,807
Columns              : 31
Missing Values       : 0
Duplicate Rows       : 1,081
Normal Transactions  : 284,315
Fraud Transactions   : 492
```

The dataset contains no missing values.

However, duplicate rows are present and are handled in the Silver layer.

The dataset does not contain:

- Customer ID
- Merchant
- City
- Location
- Payment method
- Customer behavioral attributes

These attributes are therefore not invented anywhere in the pipeline.

---

# 8. Databricks Environment

The project is implemented using **Databricks Free Edition** with the serverless/default environment.

The Databricks workspace contains the project notebooks and Delta tables.

Workspace project folder:

```text
FraudStream_AI
```

---

# 9. Source Layer

The Kaggle dataset is first loaded into a Databricks source table:

```text
workspace.default.creditcard_source
```

This source table acts as the starting point for the Databricks data pipeline.

The source layer keeps the original dataset available for downstream processing.

---

# 10. Bronze Layer

The Bronze layer is responsible for raw data ingestion.

Table:

```text
workspace.default.bronze_transactions
```

The Bronze layer preserves the source data and adds ingestion metadata.

Additional fields:

```text
_ingestion_time
_source
_batch_id
```

The current ingestion metadata contains:

```text
_source   = kaggle_creditcard
_batch_id = initial_load
```

## Bronze Implementation

The source table is read using Spark:

```python
source_df = spark.table("workspace.default.creditcard_source")
```

Metadata is added using Spark functions.

The resulting DataFrame is written as a Delta table.

## Bronze Result

```text
Bronze Rows = 284,807
```

The Bronze layer intentionally preserves the incoming records before business-level cleaning.

---

# 11. Silver Layer

The Silver layer is responsible for validation and cleaning.

Table:

```text
workspace.default.silver_transactions
```

Invalid records are written to:

```text
workspace.default.quarantine_transactions
```

## Validation Rules

The following validation rules are applied:

### Rule 1 — Amount

Transaction amount must be greater than or equal to zero.

```text
Amount >= 0
```

### Rule 2 — Class

The fraud class must be either:

```text
0
1
```

### Rule 3 — Duplicate Removal

Duplicate rows are removed using:

```python
dropDuplicates()
```

## Silver Processing Flow

```text
Bronze
   ↓
Validate Amount
   ↓
Validate Class
   ↓
Invalid Records → Quarantine
   ↓
Remove Duplicates
   ↓
Add Silver Processing Timestamp
   ↓
Silver Delta Table
```

## Silver Results

```text
Bronze Rows        : 284,807
Silver Rows        : 283,726
Duplicates Removed : 1,081
Invalid Rows       : 0
```

The difference between Bronze and Silver is entirely explained by duplicate removal.

---

# 12. Quarantine Layer

Invalid records are written to:

```text
workspace.default.quarantine_transactions
```

The current dataset produced:

```text
Invalid Rows = 0
```

The quarantine table is still part of the architecture because it provides a controlled location for invalid data rather than silently discarding it.

This pattern can be extended in a production pipeline for:

- Schema violations
- Invalid business values
- Missing mandatory fields
- Negative amounts
- Invalid transaction classes

---

# 13. Gold Layer

The Gold layer contains analytics-ready tables.

The primary Gold tables are:

```text
workspace.default.fact_transactions
workspace.default.fraud_summary
workspace.default.amount_summary
```

---

## 13.1 Fact Transactions

Table:

```text
workspace.default.fact_transactions
```

This table contains the cleaned transaction records required for downstream analytics and machine learning.

The table includes:

- Time
- V1–V28
- Amount
- Class
- `_ingestion_time`

---

## 13.2 Fraud Summary

Table:

```text
workspace.default.fraud_summary
```

This table contains aggregated fraud statistics.

Metrics include:

- Total transactions
- Fraud transactions
- Normal transactions
- Fraud rate
- Fraud transaction amount
- Total transaction amount

---

## 13.3 Amount Summary

Table:

```text
workspace.default.amount_summary
```

This table summarizes transaction amounts by fraud class.

Metrics include:

- Transaction count
- Total amount
- Average amount
- Maximum amount

---

# 14. Machine Learning

The fraud detection layer is implemented using **Spark ML Logistic Regression**.

The model reads from:

```text
workspace.default.fact_transactions
```

---

# 15. Feature Engineering

The model uses the following features:

```text
V1
V2
V3
...
V28
Amount
```

There are 29 model features in total.

A Spark `VectorAssembler` combines the feature columns into a single feature vector.

```python
feature_columns = [
    "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]
```

The target is:

```text
Class
```

---

# 16. Train/Test Split

The Gold dataset is divided into training and testing data using:

```python
train_df, test_df = ml_df.randomSplit(
    [0.8, 0.2],
    seed=42
)
```

Results:

```text
Training Transactions : 227,179
Testing Transactions  : 56,547
```

Fraud distribution:

```text
Training Fraud : 387
Testing Fraud  : 86
```

---

# 17. Class Imbalance

Fraud detection is highly imbalanced.

Most transactions belong to the normal class, while only a very small percentage are fraud.

A standard model can therefore become biased toward the majority class.

To address this, class weights are calculated.

The weighting strategy gives fraud transactions greater importance during model training.

Conceptually:

```text
Fraud Weight  > Normal Weight
```

This is implemented using Spark ML's `weightCol`.

---

# 18. Logistic Regression Model

The model uses:

```python
LogisticRegression(
    featuresCol="features",
    labelCol="Class",
    weightCol="class_weight",
    maxIter=50
)
```

Model version:

```text
logistic_regression_v1
```

The model outputs:

- Prediction
- Probability vector

The fraud probability is extracted from the probability vector.

---

# 19. Model Evaluation

The current test results are:

```text
True Positive  : 82
True Negative  : 55,123
False Positive : 1,338
False Negative : 4
```

Metrics:

```text
Precision : 0.0577
Recall    : 0.9535
F1 Score  : 0.1089
PR-AUC    : 0.6808
```

Equivalent percentages:

```text
Precision : 5.77%
Recall    : 95.35%
F1 Score  : 10.89%
PR-AUC    : 0.6808
```

## Interpretation

The current class-weighted configuration prioritizes identifying fraud transactions.

The model catches:

```text
82 / 86
```

fraud transactions in the test set.

It misses:

```text
4
```

fraud transactions.

However, it also produces:

```text
1,338
```

false positives.

This explains the relatively low precision.

The current model is therefore a baseline that can be improved through:

- Threshold tuning
- Feature engineering
- Alternative algorithms
- Hyperparameter tuning
- Model comparison

These improvements are future enhancements rather than part of the current implementation.

---

# 20. Fraud Prediction Table

Model predictions are persisted in:

```text
workspace.default.fact_fraud_predictions
```

The table contains:

```text
prediction_id
Time
Amount
Class
prediction
fraud_probability
model_version
prediction_time
```

## Prediction Generation

The Spark ML prediction output contains a probability vector.

The fraud probability is extracted using:

```python
vector_to_array(col("probability"))[1]
```

The prediction table also stores:

```text
prediction_time
model_version
prediction_id
```

---

# 21. Prediction Results

The test dataset contains:

```text
Total Predictions : 56,547
Predicted Fraud   : 1,420
Predicted Normal  : 55,127
```

Predicted fraud rate:

```text
2.5112%
```

The predicted fraud rate is higher than the actual fraud prevalence because the current class-weighted model prioritizes recall.

---

# 22. SQL Analytics

SQL analytics are implemented in:

```text
05_FraudStream_SQL
```

The SQL layer analyzes the prediction table.

Key analysis areas include:

- Prediction distribution
- Fraud probability bands
- Confusion matrix
- Transaction amount analysis

---

# 23. Probability Band Analysis

Fraud probabilities are grouped into:

```text
0.00 - 0.24
0.25 - 0.49
0.50 - 0.74
0.75 - 0.89
0.90 - 1.00
```

Current results:

| Probability Band | Count | Average Probability | Total Amount |
|---|---:|---:|---:|
| 0.90–1.00 | 318 | 0.9620 | 124,275.87 |
| 0.75–0.89 | 287 | 0.8323 | 93,304.82 |
| 0.50–0.74 | 815 | 0.6086 | 165,981.34 |
| 0.25–0.49 | 2,830 | 0.3437 | 452,911.57 |
| 0.00–0.24 | 52,297 | 0.0498 | 4,209,302.61 |

This analysis helps understand how model probabilities are distributed across the test data.

---

# 24. Confusion Matrix Analysis

The current confusion matrix is:

| Actual | Prediction | Count |
|---|---|---:|
| Normal | Normal | 55,123 |
| Normal | Fraud | 1,338 |
| Fraud | Normal | 4 |
| Fraud | Fraud | 82 |

This corresponds to:

```text
TN = 55,123
FP = 1,338
FN = 4
TP = 82
```

---

# 25. Transaction Amount Analysis

Predicted fraud transactions:

```text
Count       : 1,420
Total Amount: 383,562.03
Average     : 270.11
Maximum     : 8,360.00
```

Predicted normal transactions:

```text
Count       : 55,127
Total Amount: 4,662,214.18
Average     : 84.57
Maximum     : 5,356.42
```

This analysis provides an additional business view of the model's predictions.

---

# 26. Data Quality

Data quality validation is implemented in:

```text
06_FraudStream_DataQuality
```

The checks are designed to validate the Silver/Gold transaction data.

## Checks

### Total Row Count

Confirms the expected number of cleaned transactions.

### Null Amount

Checks whether transaction amounts contain null values.

### Invalid Class

Checks whether values outside:

```text
0, 1
```

exist.

### Negative Amount

Checks whether:

```text
Amount < 0
```

exists.

### Duplicate Groups

Checks for duplicate transaction groups after cleaning.

---

# 27. Data Quality Results

Current results:

```text
Total Transactions : 283,726
Null Amount        : 0
Invalid Class      : 0
Negative Amount    : 0
Duplicate Groups   : 0
```

All current data quality checks pass.

Results are persisted in:

```text
workspace.default.data_quality_results
```

This allows data quality results to become part of the data platform instead of existing only as temporary notebook output.

---

# 28. SCD Type 2

SCD Type 2 is implemented in:

```text
07_FraudStream_SCD2
```

The table is:

```text
workspace.default.dim_fraud_rules
```

---

# 29. Why SCD Type 2 Uses Fraud Rules

The source dataset does not contain:

- Customer ID
- Natural customer key
- Customer attributes
- Changing customer information

Therefore, creating a fake customer dimension would not accurately represent the source data.

Instead, SCD Type 2 is implemented on a realistic business-rule dimension.

This demonstrates historical tracking without inventing source attributes.

---

# 30. Fraud Rule Dimension

Initial rules:

| Rule ID | Rule | Threshold | Severity |
|---|---|---:|---|
| 1 | High Fraud Probability | 0.90 | HIGH |
| 2 | Medium Fraud Probability | 0.50 | MEDIUM |
| 3 | Low Fraud Probability | 0.25 | LOW |

---

# 31. Simulated Rule Change

A business rule change is simulated for the High Fraud Probability rule.

Old rule:

```text
Threshold = 0.90
```

New rule:

```text
Threshold = 0.85
```

The previous version is retained with its historical end date.

The new version is marked as current.

Conceptually:

```text
Old Rule
0.90
   ↓
Historical Version

New Rule
0.85
   ↓
Current Version
```

This demonstrates the core purpose of SCD Type 2:

> Preserve historical versions while maintaining the current version.

---

# 32. Generative AI Layer

Generative AI is implemented as a supporting layer after the ML prediction stage.

The architecture is:

```text
Fraud ML Model
      ↓
Fraud Prediction
      ↓
Fraud Probability
      ↓
Risk Classification
      ↓
Local LLM
      ↓
Supporting Explanation
```

The LLM is **not responsible for determining whether a transaction is fraudulent**.

---

# 33. Local LLM Setup

The project uses:

```text
Ollama
```

with:

```text
llama3.2:3b
```

Ollama version used:

```text
0.34.4
```

Python package:

```text
ollama
```

Python version used:

```text
3.13.5
```

The local model approach avoids requiring an external cloud LLM service or Azure subscription.

---

# 34. GenAI Input

The explanation function receives:

```text
amount
prediction
fraud_probability
model_version
```

A deterministic risk level is calculated in Python.

Risk levels:

```text
prediction = 1 and probability >= 0.90
    → HIGH

prediction = 1 and probability >= 0.50
    → MEDIUM

prediction = 1
    → LOW

prediction = 0
    → NORMAL
```

---

# 35. GenAI Prompt Grounding

The LLM is instructed to analyze only the supplied facts.

The prompt explicitly prevents the model from:

- Claiming certainty
- Claiming the transaction is definitely fraudulent
- Describing probability as model accuracy
- Inventing customer information
- Inventing merchant information
- Inventing location
- Inventing behavioral information
- Interpreting V1–V28
- Mentioning unavailable merchant information

The recommendation is controlled by Python:

```text
Review transaction
```

This creates a separation between deterministic application logic and LLM-generated supporting text.

---

# 36. GenAI Output

The expected JSON structure is:

```json
{
  "risk_level": "...",
  "summary": "...",
  "recommendation": "..."
}
```

However, Python validates and controls the final deterministic fields.

The LLM is mainly responsible for:

```text
summary
```

Python controls:

```text
risk_level
recommendation
```

This reduces the risk of the LLM changing important business logic.

---

# 37. GenAI Output Dataset

Selected fraud predictions are processed locally.

Output:

```text
data/fraud_explanations.csv
```

The file is uploaded to Databricks as:

```text
workspace.default.fraud_explanations_source
```

The final table is:

```text
workspace.default.fraud_explanations
```

Current explanation count:

```text
20
```

The current implementation generates explanations for the top selected predicted-fraud transactions rather than all 56,547 predictions.

---

# 38. Full Prediction Export

The complete prediction table is exported locally as:

```text
data/fraud_predictions.csv
```

Current size:

```text
56,547 rows
```

Columns:

```text
prediction_id
Time
Amount
Class
prediction
fraud_probability
model_version
prediction_time
```

This full prediction dataset is used by the Streamlit dashboard.

---

# 39. Streamlit Dashboard

The project contains an interactive dashboard built using:

```text
Streamlit
Plotly
Pandas
```

Main files:

```text
dashboard/app.py
dashboard/data.py
```

---

# 40. Dashboard Features

## KPI Section

The dashboard dynamically calculates:

- Total predictions
- Predicted fraud
- Predicted normal
- Fraud rate

The KPIs update based on the active filters.

---

## Risk Filter

Users can filter transactions by:

```text
ALL
HIGH
MEDIUM
LOW
NORMAL
```

Risk level is derived from fraud probability.

---

## Fraud Probability Filter

The dashboard provides a probability range filter:

```text
0.00 → 1.00
```

This allows users to investigate transactions based on model probability.

---

## Transaction Search

Users can search for a specific:

```text
prediction_id
```

---

# 41. Risk Classification in Dashboard

The dashboard derives risk levels using probability bands:

```text
0.00 - 0.24 → NORMAL
0.25 - 0.49 → LOW
0.50 - 0.89 → MEDIUM
0.90 - 1.00 → HIGH
```

The risk classification is a presentation/investigation layer and does not replace the ML model's prediction.

---

# 42. Dashboard Analytics

The dashboard contains:

### Risk Distribution

Shows the number of transactions in each risk category.

### Prediction Performance

Displays model prediction behavior including:

- True positives
- False positives
- True negatives
- False negatives

### Amount Analysis

Provides transaction amount analysis across predictions.

### Fraud Investigation

Displays selected predicted-fraud transactions for investigation.

---

# 43. AI Investigation Assistant

The dashboard integrates the locally generated GenAI explanations.

For a selected transaction, the dashboard can display:

```text
Transaction Amount
Fraud Probability
Risk Level
AI Explanation
Recommendation
```

Example:

```text
Transaction Amount : $766.36
Fraud Probability  : 100.00%
Risk Level         : HIGH

AI Explanation:
Transaction amount $766.36 predicted to be fraudulent
with fraud probability of 1.00

Recommendation:
Review transaction
```

The explanation is intended as analyst support rather than an autonomous fraud decision.

---

# 44. Local Transaction Simulator

A Python transaction simulator exists in:

```text
src/streaming_simulator.py
```

The simulator reads:

```text
data/creditcard.csv
```

and writes individual JSON transaction files into:

```text
data/stream_input/
```

Each generated transaction contains:

```text
transaction_id
time
amount
class
```

The simulator introduces a delay between transactions.

---

# 45. Current Streaming Status

The local simulator is implemented as a transaction generation component.

However, **Kafka and Spark Structured Streaming are not part of the current implemented production pipeline**.

A local PySpark Structured Streaming attempt encountered Windows Hadoop/native checkpoint environment issues.

The current Spark processing pipeline therefore runs in Databricks.

Kafka and Structured Streaming remain future enhancements.

---

# 46. Project Structure

```text
fraud-detection-ai/
│
├── README.md
│
├── docs/
│   └── 01_project_overview.md
│
├── src/
│   ├── profile_data.py
│   ├── streaming_simulator.py
│   └── stream_processor.py
│
├── ai/
│   └── fraud_explanation.py
│
├── dashboard/
│   ├── app.py
│   └── data.py
│
├── notebooks/
│   ├── 01_FraudStream_Bronze.ipynb
│   ├── 02_FraudStream_Silver.ipynb
│   ├── 03_FraudStream_Gold.ipynb
│   ├── 04_FraudStream_ML.ipynb
│   ├── 05_FraudStream_SQL.ipynb
│   ├── 06_FraudStream_DataQuality.ipynb
│   ├── 07_FraudStream_SCD2.ipynb
│   └── 08_FraudStream_GenAI.ipynb
│
├── data/
│   ├── creditcard.csv
│   ├── fraud_predictions.csv
│   ├── fraud_explanations.csv
│   └── stream_input/
│
└── .gitignore
```

---

# 47. Databricks Notebook Organization

## 01_FraudStream_Bronze

Responsible for:

- Reading source table
- Adding ingestion metadata
- Creating Bronze Delta table

Table:

```text
workspace.default.bronze_transactions
```

---

## 02_FraudStream_Silver

Responsible for:

- Validation
- Quarantine
- Duplicate removal
- Silver processing timestamp

Tables:

```text
workspace.default.quarantine_transactions
workspace.default.silver_transactions
```

---

## 03_FraudStream_Gold

Responsible for:

- Fact transaction table
- Fraud summary
- Amount summary

Tables:

```text
workspace.default.fact_transactions
workspace.default.fraud_summary
workspace.default.amount_summary
```

---

## 04_FraudStream_ML

Responsible for:

- Feature assembly
- Train/test split
- Class weighting
- Logistic Regression
- Model evaluation
- Prediction generation
- Prediction persistence

Table:

```text
workspace.default.fact_fraud_predictions
```

---

## 05_FraudStream_SQL

Responsible for:

- Prediction analysis
- Probability bands
- Confusion matrix
- Amount analysis

---

## 06_FraudStream_DataQuality

Responsible for:

- Null checks
- Invalid class checks
- Negative amount checks
- Duplicate checks
- Persisting DQ results

Table:

```text
workspace.default.data_quality_results
```

---

## 07_FraudStream_SCD2

Responsible for:

- Fraud rule dimension
- Rule versioning
- Historical tracking
- Current rule management

Table:

```text
workspace.default.dim_fraud_rules
```

---

## 08_FraudStream_GenAI

Responsible for:

- Loading GenAI explanation source data
- Writing explanation results to Databricks

Table:

```text
workspace.default.fraud_explanations
```

---

# 48. Databricks Table Inventory

| Table | Layer | Purpose |
|---|---|---|
| `creditcard_source` | Source | Original dataset |
| `bronze_transactions` | Bronze | Raw ingested data |
| `quarantine_transactions` | Silver | Invalid records |
| `silver_transactions` | Silver | Cleaned transactions |
| `fact_transactions` | Gold | Analytics-ready transactions |
| `fraud_summary` | Gold | Fraud summary metrics |
| `amount_summary` | Gold | Amount analytics |
| `fact_fraud_predictions` | ML | Model predictions |
| `data_quality_results` | DQ | Quality check results |
| `dim_fraud_rules` | SCD2 | Fraud rule history |
| `fraud_explanations_source` | GenAI | Uploaded AI explanations |
| `fraud_explanations` | GenAI | Final explanation table |

---

# 49. End-to-End Data Flow

The complete implemented workflow is:

```text
Kaggle Dataset
      ↓
Databricks Source Table
      ↓
Bronze Ingestion
      ↓
Silver Validation
      ↓
Quarantine Invalid Records
      ↓
Remove Duplicates
      ↓
Gold Data Modeling
      ↓
Fraud ML Model
      ↓
Fraud Predictions
      ↓
SQL Analytics
      ↓
Data Quality
      ↓
SCD Type 2 Fraud Rules
      ↓
Local GenAI Explanations
      ↓
Streamlit Dashboard
```

---

# 50. Key Design Decisions

## 50.1 Data Engineering First

The project is intentionally built as a Data Engineering project.

The foundation is:

```text
Ingestion
   ↓
Processing
   ↓
Validation
   ↓
Data Modeling
   ↓
Analytics
```

Machine Learning and GenAI are added on top of this foundation.

---

## 50.2 ML Makes the Fraud Decision

The ML model is responsible for:

```text
Fraud Prediction
Fraud Probability
```

The LLM does not replace the ML model.

---

## 50.3 GenAI Explains Rather Than Decides

The LLM receives structured prediction information and generates a supporting explanation.

This separation makes the architecture:

```text
ML → Decision
LLM → Explanation
```

---

## 50.4 No Artificial Customer Dimension

The dataset does not contain a customer identifier.

Therefore, the project does not create a fake customer dimension just to demonstrate SCD Type 2.

Instead, business rules are used.

---

## 50.5 Local LLM Instead of Cloud LLM

Ollama + Llama 3.2 3B was selected so the GenAI layer can run locally without requiring an Azure subscription or external API.

---

## 50.6 Bronze Preserves Raw Data

Duplicate records are not removed at ingestion time.

They are retained in Bronze and handled in Silver.

This maintains a cleaner separation between:

```text
Raw Data
     ↓
Validated Data
```

---

# 51. Limitations

The current implementation has the following limitations:

1. The dataset is historical rather than production transaction data.
2. The transaction features `V1–V28` are anonymized.
3. There is no customer or merchant information.
4. The current ML model is a baseline Logistic Regression model.
5. Precision is relatively low because the current configuration prioritizes recall.
6. Threshold tuning has not yet been implemented.
7. Only selected predicted-fraud transactions receive GenAI explanations.
8. The local transaction simulator does not currently feed Kafka.
9. Kafka is not part of the current implemented architecture.
10. Spark Structured Streaming is not part of the current Databricks pipeline.
11. The dashboard currently works from an exported local prediction CSV.
12. Production deployment and monitoring are not implemented.

---

# 52. Future Enhancements

Potential future improvements include:

## Real-Time Processing

```text
Python Transaction Simulator
          ↓
        Kafka
          ↓
Spark Structured Streaming
          ↓
Bronze → Silver → Gold
```

This would extend the current batch-oriented pipeline toward real-time processing.

---

## ML Improvements

Possible improvements include:

- Threshold tuning
- Hyperparameter tuning
- Feature engineering
- Model comparison
- Random Forest
- Gradient Boosted Trees
- Advanced anomaly detection
- Model monitoring

---

## Data Engineering Improvements

Potential improvements include:

- Automated orchestration
- Incremental processing
- Streaming ingestion
- Schema evolution handling
- Production data quality monitoring
- Data lineage
- Automated alerts

---

## GenAI Improvements

Potential improvements include:

- Better explanation templates
- More structured explanation output
- Retrieval-based grounding
- Explanation auditing
- Batch explanation generation
- Analyst feedback integration

---

## Dashboard Improvements

Potential improvements include:

- Real-time dashboard refresh
- More drill-down capabilities
- Historical model comparison
- Model monitoring views
- Data quality monitoring page
- Rule history visualization

---

# 53. Security and Data Handling

The project uses a public anonymized dataset.

No customer-identifying information is intentionally introduced into the project.

The GenAI layer also follows a grounding approach that prevents the LLM from inventing unavailable customer, merchant, location, or behavioral information.

Generated local data files are excluded from Git where appropriate using `.gitignore`.

---

# 54. Git and Repository Structure

The project uses Git for version control.

The repository excludes local environment files and generated CSV data.

Current `.gitignore` includes:

```text
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
data/*.csv
data/stream_input/
```

This prevents large local datasets and generated files from being committed unnecessarily.

---

# 55. Final Project Results

## Data Engineering

```text
Original Rows        : 284,807
Silver Rows          : 283,726
Duplicates Removed   : 1,081
Invalid Rows          : 0
```

## Machine Learning

```text
Test Transactions : 56,547
True Positives     : 82
True Negatives     : 55,123
False Positives    : 1,338
False Negatives    : 4

Precision          : 5.77%
Recall             : 95.35%
F1 Score           : 10.89%
PR-AUC             : 0.6808
```

## Predictions

```text
Total Predictions : 56,547
Predicted Fraud   : 1,420
Predicted Normal  : 55,127
```

## Data Quality

```text
Null Amounts      : 0
Invalid Classes   : 0
Negative Amounts  : 0
Duplicate Groups  : 0
```

## GenAI

```text
Local Model       : Llama 3.2 3B
Runtime           : Ollama
Selected Outputs  : 20
```

---

# 56. Complete Project Summary

FraudStream AI demonstrates an end-to-end Data Engineering workflow for fraud analytics.

The project starts with a public credit card transaction dataset and builds a structured Databricks pipeline:

```text
Source
  ↓
Bronze
  ↓
Silver
  ↓
Gold
```

The Gold layer feeds a class-weighted Logistic Regression model that produces fraud predictions and probabilities.

The predictions are then analyzed using SQL and validated through data quality checks.

SCD Type 2 is used to maintain historical versions of fraud detection business rules.

A local Llama 3.2 3B model running through Ollama provides supporting explanations for selected fraud predictions.

Finally, a Streamlit dashboard brings together:

```text
Fraud KPIs
    +
Risk Analysis
    +
Prediction Performance
    +
Transaction Analysis
    +
AI Explanations
```

The overall architecture maintains a clear separation of responsibilities:

```text
Data Engineering
        ↓
Builds and prepares the data

Machine Learning
        ↓
Makes the fraud prediction

Generative AI
        ↓
Explains the prediction

Streamlit
        ↓
Presents the results
```

This design keeps the core fraud detection logic deterministic and model-driven while using GenAI as an additional analytical support layer.

---

# 57. Project Completion Checklist

```text
[✓] Dataset profiling
[✓] Databricks source table
[✓] Bronze layer
[✓] Silver layer
[✓] Quarantine layer
[✓] Duplicate handling
[✓] Gold layer
[✓] Fraud summary
[✓] Amount summary
[✓] Machine Learning
[✓] Class imbalance handling
[✓] Fraud predictions
[✓] SQL analytics
[✓] Data quality checks
[✓] SCD Type 2
[✓] Local GenAI
[✓] GenAI grounding
[✓] Streamlit dashboard
[✓] Local transaction simulator
[✓] Git project structure
```

Future items:

```text
[ ] Kafka
[ ] Spark Structured Streaming
[ ] Real-time processing
[ ] ML threshold tuning
[ ] Advanced model experimentation
[ ] Production deployment
```

---

# 58. Conclusion

FraudStream AI combines modern Data Engineering practices with Machine Learning and Generative AI to create a complete fraud analytics platform.

The project demonstrates:

- Data ingestion
- Data validation
- Data cleaning
- Medallion Architecture
- Delta Lake
- Data modeling
- SQL analytics
- Machine Learning
- Class imbalance handling
- Data quality
- SCD Type 2
- Local LLM integration
- AI grounding
- Interactive analytics

The core design principle is:

```text
Build a reliable data platform first.
Then integrate ML.
Then add GenAI as a supporting layer.
```

This keeps the project centered on Data Engineering while demonstrating how ML and GenAI can be integrated into a practical modern data platform.