from datetime import timedelta
import pandas as pd


def build_features(
    df_events: pd.DataFrame,
    labels_df: pd.DataFrame,
    history_days: int,
) -> pd.DataFrame:
    # existing function – leave as is
    df = df_events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    labels_df["reference_date"] = pd.to_datetime(labels_df["reference_date"])

    rows = []
    for _, row in labels_df.iterrows():
        user = row["user_id"]
        ref_date = row["reference_date"]
        hist_start = ref_date - timedelta(days=history_days)

        hist = df[
            (df["user_id"] == user)
            & (df["event_date"] > hist_start)
            & (df["event_date"] <= ref_date)
        ]

        if hist.empty:
            continue

        last_event_date = hist["event_date"].max()
        rows.append(
            {
                "user_id": user,
                "reference_date": ref_date,
                "events_7d": hist[
                    hist["event_date"] > ref_date - timedelta(days=7)
                ]["event_count"].sum(),
                "events_30d": hist["event_count"].sum(),
                "days_active_30d": hist[hist["event_count"] > 0][
                    "event_date"
                ].nunique(),
                "days_since_last_event": (ref_date - last_event_date).days,
            }
        )

    features_df = pd.DataFrame(rows)
    dataset = features_df.merge(
        labels_df, on=["user_id", "reference_date"], how="inner"
    )
    return dataset


def build_scoring_features(
    df_events: pd.DataFrame,
    history_days: int,
) -> pd.DataFrame:
    """
    Build features for all users as of the latest event_date.

    This is a simplified "daily snapshot":
    - use max(event_date) as the reference_date
    - compute the same feature set as in training
    - no labels, just features + user_id + reference_date
    """
    df = df_events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])

    if df.empty:
        raise ValueError("No events available to build scoring features.")

    ref_date = df["event_date"].max()
    hist_start = ref_date - timedelta(days=history_days)

    hist = df[(df["event_date"] > hist_start) & (df["event_date"] <= ref_date)]

    rows = []
    for user, group in hist.groupby("user_id"):
        last_event_date = group["event_date"].max()
        rows.append(
            {
                "user_id": user,
                "reference_date": ref_date,
                "events_7d": group[
                    group["event_date"] > ref_date - timedelta(days=7)
                ]["event_count"].sum(),
                "events_30d": group["event_count"].sum(),
                "days_active_30d": group[group["event_count"] > 0][
                    "event_date"
                ].nunique(),
                "days_since_last_event": (ref_date - last_event_date).days,
            }
        )

    return pd.DataFrame(rows)
