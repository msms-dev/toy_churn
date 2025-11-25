from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.utils.config_loader import load_config
from src.ingestion.api_client import load_raw_events
from src.labelling.build_labels import build_churn_labels
from src.features.build_features import build_features
from src.models.automl_train import run_automl_training

def prepare_training_dataset(**context):
    cfg = load_config()
    history_days = cfg["label_definition"]["history_days"]
    churn_window = cfg["label_definition"]["churn_window_days"]

    df_events = load_raw_events()
    labels_df = build_churn_labels(df_events, history_days, churn_window)
    dataset = build_features(df_events, labels_df, history_days)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / "training_dataset.parquet"
    dataset.to_parquet(path)
    context["ti"].xcom_push(key="training_dataset_path", value=str(path))

def train_model(**context):
    path = context["ti"].xcom_pull(key="training_dataset_path", task_ids="prepare_training_dataset")
    run_automl_training(path)

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="training_dag",
    default_args=default_args,
    schedule_interval="@once",
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    t_prepare = PythonOperator(
        task_id="prepare_training_dataset",
        python_callable=prepare_training_dataset,
        provide_context=True,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        provide_context=True,
    )

    t_prepare >> t_train
