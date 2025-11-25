# Fully Working MLOps Repo Template (Toy Churn)

This is a fully working starter MLOps project using:
- Airflow
- AutoML (FLAML)
- MLflow
- DVC
- Evidently
- Docker

It implements a **toy churn prediction** pipeline using a synthetic `events.csv`
in `data/raw/`, and demonstrates:
- ingestion
- label creation (churn = inactivity)
- feature engineering
- AutoML training + MLflow logging
- a weekly training DAG in Airflow
