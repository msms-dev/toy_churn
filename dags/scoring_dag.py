# dags/scoring_dag.py
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from src.ingestion.bigquery_loader import load_raw_events_from_bigquery
from src.features.ga_features import prepare_scoring_features
from src.scoring.score import score_users
from src.utils.config_loader import load_config
from src.utils.simulation import compute_simulated_ga_date


def load_scoring_events(ds: str, **context):
    cfg = load_config()
    sim = cfg["simulation"]

    ga_day = compute_simulated_ga_date(
        logical_ds=ds,
        sim_start_date=sim["sim_start_date"],
        ga_start_date=sim["ga_start_date"],
        ga_end_date=sim["ga_end_date"],
        step_days=int(sim["scoring_step_days"]),  # usually 1
    )

    df = load_raw_events_from_bigquery(
        start_date=str(ga_day),
        end_date=str(ga_day),
        project_id=None,
        table_pattern=None,
    )

    print(f"[load_scoring_events] ds={ds} -> ga_day={ga_day} rows={len(df)}")

    # persist ga_day for downstream tasks
    context["ti"].xcom_push(key="ga_day", value=str(ga_day))

    if df.empty:
        # Option A: fail (forces you to pick a day with data)
        raise ValueError(f"No GA rows returned for simulated day {ga_day}")

        # Option B (alternative): return empty JSON and let downstream handle
        # return "[]"

    return df.to_json(orient="records")


def build_scoring_features(raw_json: str, **context):
    import pandas as pd
    from io import StringIO
    
    raw_df = pd.read_json(StringIO(raw_json), orient="records")

    feats = prepare_scoring_features(raw_df)

    if feats.empty:
        raise ValueError("prepare_scoring_features produced empty features")

    print(f"[build_scoring_features] features shape={feats.shape}")
    return feats.to_json(orient="records")


def run_scoring(features_json: str, **context):
    import pandas as pd

    cfg = load_config()
    ti = context["ti"]
    ga_day = ti.xcom_pull(task_ids="load_scoring_events", key="ga_day")

    feats_df = pd.read_json(features_json, orient="records")

    scores_dir = cfg["paths"]["scores_dir"]
    Path(scores_dir).mkdir(parents=True, exist_ok=True)

    registered_name = cfg["mlflow"]["registered_model_name"]
    model_uri = f"models:/{registered_name}/Production"

    output_path = score_users(
        features_df=feats_df,
        model_uri=model_uri,
        scores_dir=scores_dir,
        score_date=ga_day,  # save using GA date (2017-..)
    )

    print(f"[run_scoring] wrote: {output_path}")
    return output_path


default_args = {"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="ga_scoring_dag",
    schedule="@daily",
    start_date=datetime(2025, 12, 15),  # real-world start of your simulation
    catchup=False,
    default_args=default_args,
    tags=["ga_demo", "scoring"],
) as dag:

    t_load = PythonOperator(
        task_id="load_scoring_events",
        python_callable=load_scoring_events,
        op_kwargs={"ds": "{{ ds }}"},
    )

    t_features = PythonOperator(
        task_id="build_scoring_features",
        python_callable=build_scoring_features,
        op_kwargs={"raw_json": "{{ ti.xcom_pull(task_ids='load_scoring_events') }}"},
    )

    t_score = PythonOperator(
        task_id="run_scoring",
        python_callable=run_scoring,
        op_kwargs={"features_json": "{{ ti.xcom_pull(task_ids='build_scoring_features') }}"},
    )

    t_load >> t_features >> t_score

