# src/features/ga_features.py
from __future__ import annotations

from datetime import timedelta
from typing import Tuple

import pandas as pd


def _clean_ga_raw_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning to ensure numeric columns are numeric and NAs are handled.
    """
    df = raw_df.copy()

    # Convert known numeric cols to numeric (coerce errors to NaN)
    numeric_cols = [
        "pageviews",
        "timeOnSite",
        "transactions",
        "totalTransactionRevenue",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_training_dataset(
    raw_df: pd.DataFrame,
    churn_gap_days: int = 30,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert raw GA sessions into user-level training features and binary churn labels.

    Expected columns in raw_df (from BigQuery query):
      - fullVisitorId
      - visitId
      - date (as datetime)
      - channelGrouping
      - browser
      - operatingSystem
      - country
      - source
      - medium
      - pageviews
      - timeOnSite
      - transactions
      - totalTransactionRevenue

    Label definition (simple example):
      - max_date = max(date in dataset)
      - For each user, last_visit_date = max(date)
      - churn_label = 1 if last_visit_date <= max_date - churn_gap_days else 0
    """
    if raw_df.empty:
        raise ValueError("No raw events returned from BigQuery for training.")

    df = _clean_ga_raw_df(raw_df)

    # Aggregate user-level features
    agg = (
        df.groupby("fullVisitorId")
        .agg(
            sessions=("visitId", "nunique"),
            total_pageviews=("pageviews", "sum"),
            total_time_on_site=("timeOnSite", "sum"),
            total_transactions=("transactions", "sum"),
            total_revenue=("totalTransactionRevenue", "sum"),
            first_visit=("date", "min"),
            last_visit=("date", "max"),
            n_countries=("country", "nunique"),
            n_sources=("source", "nunique"),
            n_mediums=("medium", "nunique"),
        )
        .reset_index()
    )

    # Fill NaNs for numeric aggregates
    for col in [
        "total_pageviews",
        "total_time_on_site",
        "total_transactions",
        "total_revenue",
    ]:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0)

    max_date = agg["last_visit"].max()
    churn_threshold_date = max_date - timedelta(days=churn_gap_days)

    agg["churn_label"] = (agg["last_visit"] <= churn_threshold_date).astype(int)

    # Features: everything except id + dates + label
    feature_cols = [
        c
        for c in agg.columns
        if c not in ("fullVisitorId", "first_visit", "last_visit", "churn_label")
    ]

    X = agg[feature_cols]
    y = agg["churn_label"]

    # Insert identifier column at the front so we can keep track of users
    X.insert(0, "fullVisitorId", agg["fullVisitorId"])

    return X, y


def prepare_scoring_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare user-level features from raw GA sessions for a single scoring day.
    Aggregation mirrors the training features (minus labels and date fields).

    Expected columns in raw_df:
      - fullVisitorId
      - visitId
      - date
      - country, source, medium
      - pageviews, timeOnSite, transactions, totalTransactionRevenue
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = _clean_ga_raw_df(raw_df)

    agg = (
        df.groupby("fullVisitorId")
        .agg(
            sessions=("visitId", "nunique"),
            total_pageviews=("pageviews", "sum"),
            total_time_on_site=("timeOnSite", "sum"),
            total_transactions=("transactions", "sum"),
            total_revenue=("totalTransactionRevenue", "sum"),
            n_countries=("country", "nunique"),
            n_sources=("source", "nunique"),
            n_mediums=("medium", "nunique"),
        )
        .reset_index()
    )

    for col in [
        "total_pageviews",
        "total_time_on_site",
        "total_transactions",
        "total_revenue",
    ]:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0)

    return agg
