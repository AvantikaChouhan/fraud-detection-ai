# FraudStream AI

### Fraud Detection & AI Analytics Platform

FraudStream AI is an end-to-end **Data Engineering project** that builds a fraud detection and analytics platform using **Databricks, PySpark, Delta Lake, SQL, Machine Learning, Generative AI, and Streamlit**.

The platform processes credit card transaction data through a **Medallion Architecture (Bronze → Silver → Gold)**, applies a fraud detection model, performs analytics and data quality validation, manages fraud detection rules using **SCD Type 2**, and provides AI-assisted explanations for selected fraud predictions.

---

## Architecture

```text
Kaggle Credit Card Fraud Dataset
              ↓
     Databricks Source Table
              ↓
           Bronze
              ↓
     Silver + Quarantine
              ↓
             Gold
              ↓
      Fraud ML Model
              ↓
     Fraud Predictions
              ↓
   SQL + Data Quality + SCD2
              ↓
       Local GenAI
    Ollama + Llama 3.2
              ↓
     Streamlit Dashboard
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| Data Engineering | Python, PySpark, Databricks, Delta Lake |
| Data Processing | Spark SQL, PySpark |
| Data Modeling | Medallion Architecture, SCD Type 2 |
| Machine Learning | Spark ML, Logistic Regression |
| Generative AI | Ollama, Llama 3.2 3B |
| Dashboard | Streamlit, Plotly, Pandas |
| Version Control | Git |

---

## Key Features

- Medallion Architecture: **Bronze → Silver → Gold**
- Data validation and quarantine
- Duplicate detection and removal
- Analytics-ready Gold tables
- Class-weighted fraud detection model
- Fraud probability scoring
- SQL-based fraud analytics
- Data quality validation
- SCD Type 2 for fraud detection rules
- Local GenAI-powered fraud explanations
- Interactive Streamlit dashboard
- Modular Databricks notebook pipeline

---

## Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

**Source:**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Summary

- 284,807 transactions
- 31 columns
- 492 fraud transactions
- Highly imbalanced classification dataset
- `V1–V28`: anonymized transaction features
- `Amount`: transaction amount
- `Class`: fraud indicator

```text
Class = 0 → Normal Transaction
Class = 1 → Fraud Transaction
```

---

## Data Engineering Pipeline

### Bronze

Raw transaction data is ingested into:

```text
workspace.default.bronze_transactions
```

Additional ingestion metadata is added:

- `_ingestion_time`
- `_source`
- `_batch_id`

### Silver

The Silver layer performs:

- Amount validation
- Class validation
- Duplicate removal
- Data cleaning

Invalid records are written to:

```text
workspace.default.quarantine_transactions
```

Clean records are stored in:

```text
workspace.default.silver_transactions
```

### Gold

Analytics-ready tables include:

```text
workspace.default.fact_transactions
workspace.default.fraud_summary
workspace.default.amount_summary
```

---

## Machine Learning

A **class-weighted Logistic Regression** model is implemented using Spark ML.

### Features

```text
V1, V2, ... V28
Amount
```

### Target

```text
Class
```

The model generates:

- Fraud prediction
- Fraud probability
- Model version
- Prediction timestamp

Predictions are stored in:

```text
workspace.default.fact_fraud_predictions
```

### Current Model Results

| Metric | Result |
|---|---:|
| Precision | 5.77% |
| Recall | 95.35% |
| F1 Score | 10.89% |
| PR-AUC | 0.6808 |

The current model configuration prioritizes fraud recall, resulting in a higher number of false positives.

---

## Data Quality

Data quality checks are implemented in Databricks for:

- Null values
- Invalid class values
- Negative transaction amounts
- Duplicate records

Results are persisted in:

```text
workspace.default.data_quality_results
```

Current validation results:

```text
Null Amounts      : 0
Invalid Classes   : 0
Negative Amounts  : 0
Duplicate Groups  : 0
```

---

## SCD Type 2

SCD Type 2 is implemented for fraud detection business rules.

Table:

```text
workspace.default.dim_fraud_rules
```

Example fraud probability thresholds:

```text
High Risk   → 0.90
Medium Risk → 0.50
Low Risk    → 0.25
```

Historical rule changes are maintained using SCD Type 2.

---

## Generative AI

Generative AI is implemented as a **supporting explanation layer**.

The ML model remains responsible for the fraud prediction, while the LLM provides a concise explanation based only on the supplied prediction information.

### Local AI Stack

```text
Ollama
   ↓
Llama 3.2 3B
   ↓
Python
```

The GenAI layer is designed to:

- Avoid making the fraud decision
- Avoid unsupported transaction details
- Avoid interpreting anonymized `V1–V28` features
- Provide concise prediction explanations
- Provide a transaction review recommendation

Generated explanations are stored in:

```text
workspace.default.fraud_explanations
```

---

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for fraud investigation and analytics.

### Dashboard Capabilities

- Fraud KPIs
- Risk-level filtering
- Fraud probability filtering
- Transaction search
- Risk distribution
- Prediction performance
- Transaction amount analysis
- Fraud investigation table
- AI Investigation Assistant

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Project Structure

```text
fraud-detection-ai/
│
├── README.md
│
├── docs/
│   └── project_overview.md
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

## Databricks Notebooks

| Notebook | Purpose |
|---|---|
| `01_FraudStream_Bronze` | Raw data ingestion |
| `02_FraudStream_Silver` | Data validation and cleaning |
| `03_FraudStream_Gold` | Gold tables and analytics |
| `04_FraudStream_ML` | Fraud detection model |
| `05_FraudStream_SQL` | SQL analytics |
| `06_FraudStream_DataQuality` | Data quality checks |
| `07_FraudStream_SCD2` | SCD Type 2 implementation |
| `08_FraudStream_GenAI` | GenAI explanation layer |

---

## End-to-End Workflow

```text
Source Data
    ↓
Bronze Ingestion
    ↓
Silver Validation & Cleaning
    ↓
Gold Data Modeling
    ↓
Machine Learning
    ↓
Fraud Predictions
    ↓
SQL Analytics
    ↓
Data Quality
    ↓
SCD Type 2
    ↓
GenAI Explanations
    ↓
Streamlit Dashboard
```

---

## Future Enhancements

- Kafka-based transaction ingestion
- Spark Structured Streaming
- Real-time fraud processing
- ML threshold optimization
- Additional fraud detection models
- Model monitoring
- Automated data quality monitoring
- Improved GenAI grounding
- Production deployment

---

## Documentation

Detailed project documentation is available in:

```text
docs/project_overview.md
```

The documentation covers the complete implementation, architecture, Databricks pipeline, ML workflow, SQL analytics, data quality, SCD Type 2, GenAI integration, dashboard, design decisions, limitations, and future enhancements.

---

## License

This project is intended for educational and portfolio purposes.