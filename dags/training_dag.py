from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.ingestion.bigquery_loader import load_raw_events_from_bigquery
from src.features.ga_features import prepare_training_dataset
from src.models.automl_train import run_automl_training
from src.utils.config_loader import load_config
from src.utils.simulation import compute_simulated_ga_date
from airflow.models import Variable


def load_bigquery_events(ds: str, **context):
    cfg = load_config()
    sim = cfg["simulation"]

    historical_days = int(cfg["bigquery"]["historical_days"])  # keep if you still want it elsewhere

    # Map "today" (Airflow ds) -> simulated GA day (moves 7 days per training run)
    ga_day = compute_simulated_ga_date(
        logical_ds=ds,
        sim_start_date=sim["sim_start_date"],
        ga_start_date=sim["ga_start_date"],
        ga_end_date=sim["ga_end_date"],
        step_days=int(sim["training_step_days"]),
    )

    warmup_days = int(sim["warmup_days"])
    train_start = (datetime.strptime(sim["ga_start_date"], "%Y-%m-%d").date()
                   - timedelta(days=warmup_days))
    train_end = ga_day  # expanding end

    print(f"[training] ds={ds} -> ga_day={ga_day} train_start={train_start} train_end={train_end}")

    raw_df = load_raw_events_from_bigquery(
        start_date=str(train_start),
        end_date=str(train_end),
        project_id=None,
        table_pattern=None,
    )

    if raw_df.empty:
        raise ValueError(f"No raw events returned for training window {train_start}..{train_end}")

    processed_dir = Path(cfg["paths"]["data_dir"]) / "training_raw"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = processed_dir / f"raw_{ds}.parquet"
    raw_df.to_parquet(raw_path)

    context["ti"].xcom_push(key="raw_path", value=str(raw_path))
    context["ti"].xcom_push(key="ga_day", value=str(ga_day))
    context["ti"].xcom_push(key="train_start", value=str(train_start))
    context["ti"].xcom_push(key="train_end", value=str(train_end))

def build_training_dataset(**context):
    cfg = load_config()
    ti = context["ti"]
    raw_path = ti.xcom_pull(task_ids="load_bigquery_events", key="raw_path")

    import pandas as pd
    raw_df = pd.read_parquet(raw_path)

    X, y = prepare_training_dataset(raw_df)

    out_dir = Path(cfg["paths"]["data_dir"]) / "training_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "train_dataset.parquet"
    df = X.copy()
    df["label"] = y.values
    df.to_parquet(dataset_path)

    ti.xcom_push(key="dataset_path", value=str(dataset_path))


def train_model(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="build_training_dataset", key="dataset_path")
    run_automl_training(dataset_path)


default_args = {"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=3)}

with DAG(
    dag_id="ga_training_dag",
    schedule="@weekly",
    start_date=datetime(2025, 12, 15),  # real-world sim start
    catchup=False,
    default_args=default_args,
    tags=["training", "bigquery"],
) as dag:

    t_load = PythonOperator(
        task_id="load_bigquery_events",
        python_callable=load_bigquery_events,
        op_kwargs={"ds": "{{ ds }}"},
    )

    t_build = PythonOperator(
        task_id="build_training_dataset",
        python_callable=build_training_dataset,
    )

    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    t_load >> t_build >> t_train
