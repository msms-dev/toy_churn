from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator


def prepare_scoring_features(**context):
    from src.utils.config_loader import load_config
    from src.ingestion.api_client import load_raw_events
    from src.features.build_features import build_scoring_features

    cfg = load_config()
    history_days = cfg["label_definition"]["history_days"]

    df_events = load_raw_events()
    features_df = build_scoring_features(df_events, history_days)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    scoring_path = processed_dir / "scoring_features.parquet"
    features_df.to_parquet(scoring_path)

    ti = context["ti"]
    ti.xcom_push(
        key="scoring_features_path",
        value=str(scoring_path),
    )


def score_users(**context):
    from pathlib import Path as _Path
    from src.models.predict_batch import predict_batch

    ti = context["ti"]
    scoring_path = ti.xcom_pull(
        key="scoring_features_path",
        task_ids="prepare_scoring_features",
    )
    scoring_path = _Path(scoring_path)

    predict_batch(scoring_path, None)


default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="toy_churn_scoring_dag",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["toy_churn", "scoring"],
) as dag:

    t_prepare_scoring = PythonOperator(
        task_id="prepare_scoring_features",
        python_callable=prepare_scoring_features,
    )

    t_score = PythonOperator(
        task_id="score_users",
        python_callable=score_users,
    )

    t_prepare_scoring >> t_score

