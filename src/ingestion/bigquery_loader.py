# src/ingestion/bigquery_loader.py
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd
from google.cloud import bigquery


def get_bigquery_client(project_id: Optional[str] = None) -> bigquery.Client:
    """
    Create a BigQuery client using the service-account key pointed to by
    GOOGLE_APPLICATION_CREDENTIALS inside the container.

    If project_id is not given, it uses the GCP_PROJECT_ID environment variable.
    """
    project_id = project_id or os.getenv("GCP_PROJECT_ID")
    if project_id is None:
        raise RuntimeError(
            "GCP_PROJECT_ID env var is not set. "
            "Set it in docker-compose.yml under environment."
        )

    return bigquery.Client(project=project_id)


def load_raw_events_from_bigquery(
    start_date: str,
    end_date: str,
    project_id: Optional[str] = None,
    table_pattern: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load Google Analytics sessions data for the Google Merchandise Store
    between start_date and end_date (inclusive).

    Parameters
    ----------
    start_date : str
        Date string in 'YYYY-MM-DD' format.
    end_date : str
        Date string in 'YYYY-MM-DD' format.
    project_id : str, optional
        GCP project id. If omitted, uses GCP_PROJECT_ID env var.
    table_pattern : str, optional
        Fully qualified BigQuery table pattern. If omitted, uses BQ_GA_TABLE env var,
        defaulting to the public GA sample:
        'bigquery-public-data.google_analytics_sample.ga_sessions_*'

    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row per GA session, with selected columns.
    """
    client = get_bigquery_client(project_id)

    table_pattern = table_pattern or os.getenv(
        "BQ_GA_TABLE",
        "bigquery-public-data.google_analytics_sample.ga_sessions_*",
    )

    # Parse strings to Python date objects for query parameters
    start_dt = datetime.fromisoformat(start_date).date()
    end_dt = datetime.fromisoformat(end_date).date()

    query = f"""
    SELECT
      fullVisitorId,
      visitId,
      visitStartTime,
      date,
      channelGrouping,
      device.browser AS browser,
      device.operatingSystem AS operating_system,
      geoNetwork.country AS country,
      trafficSource.source AS source,
      trafficSource.medium AS medium,
      totals.pageviews,
      totals.timeOnSite,
      totals.transactions,
      totals.totalTransactionRevenue
    FROM `{table_pattern}`
    WHERE
      PARSE_DATE('%Y%m%d', date) BETWEEN @start_date AND @end_date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", start_dt),
            bigquery.ScalarQueryParameter("end_date", "DATE", end_dt),
        ]
    )

    query_job = client.query(query, job_config=job_config)
    df = query_job.to_dataframe()

    # Convert date/time fields to proper types
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["visitStartTime"] = pd.to_datetime(df["visitStartTime"], unit="s")

    return df
