# Dockerfile
FROM python:3.11-slim

# Airflow + project environment
ENV AIRFLOW_HOME=/opt/airflow \
    PYTHONPATH=/opt/airflow

WORKDIR /opt/airflow

# System deps needed for Airflow + Postgres
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create basic folder structure inside the image
RUN mkdir -p dags src config data /opt/airflow/logs /mlflow/artifacts
