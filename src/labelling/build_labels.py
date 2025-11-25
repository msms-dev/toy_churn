from datetime import timedelta
import pandas as pd

def build_churn_labels(
    df_events: pd.DataFrame,
    history_days: int,
    churn_window_days: int,
) -> pd.DataFrame:
    """
    df_events: columns ['user_id', 'event_date', 'event_count']
    """
    df = df_events.copy().sort_values(["user_id", "event_date"])
    min_date = df["event_date"].min()
    max_date = df["event_date"].max()

    labels = []
    latest_ref = max_date - timedelta(days=churn_window_days)
    ref_date = min_date + timedelta(days=history_days)

    while ref_date <= latest_ref:
        hist_start = ref_date - timedelta(days=history_days)
        churn_end = ref_date + timedelta(days=churn_window_days)

        hist = df[(df["event_date"] > hist_start) & (df["event_date"] <= ref_date)]
        fut = df[(df["event_date"] > ref_date) & (df["event_date"] <= churn_end)]

        hist_users = (
            hist.groupby("user_id")["event_count"]
            .sum()
            .reset_index(name="hist_events")
        )
        fut_users = (
            fut.groupby("user_id")["event_count"]
            .sum()
            .reset_index(name="future_events")
        )

        merged = hist_users.merge(fut_users, on="user_id", how="left")
        merged["future_events"] = merged["future_events"].fillna(0)
        merged["churn"] = (merged["future_events"] == 0).astype(int)
        merged["reference_date"] = ref_date

        labels.append(merged[["user_id", "reference_date", "churn"]])
        ref_date += timedelta(days=1)

    return pd.concat(labels, ignore_index=True)
