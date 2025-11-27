from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


def prepare_training_dataset(**context):
    from src.utils.config_loader import load_config
    from src.ingestion.api_client import load_raw_events
    from src.labelling.build_labels import build_churn_labels
    from src.features.build_features import build_features

    cfg = load_config()
    history_days = cfg["label_definition"]["history_days"]
    churn_window = cfg["label_definition"]["churn_window_days"]

    df_events = load_raw_events()
    labels_df = build_churn_labels(df_events, history_days, churn_window)
    dataset = build_features(df_events, labels_df, history_days)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / "training_dataset.parquet"
    dataset.to_parquet(dataset_path)

    ti = context["ti"]
    ti.xcom_push(key="training_dataset_path", value=str(dataset_path))


def train_model(**context):
    from src.models.automl_train import run_automl_training

    ti = context["ti"]
    dataset_path = ti.xcom_pull(
        key="training_dataset_path",
        task_ids="prepare_training_dataset",
    )
    run_automl_training(dataset_path)


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="toy_churn_training_dag",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["toy_churn", "training"],
) as dag:

    t_prepare = PythonOperator(
        task_id="prepare_training_dataset",
        python_callable=prepare_training_dataset,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_prepare >> t_train

