from pathlib import Path
import pandas as pd
from src.utils.config_loader import load_config

def load_raw_events() -> pd.DataFrame:
    """
    Example ingestion: load events from a local CSV.
    """
    cfg = load_config()
    csv_path = Path(cfg["data"]["raw_events_csv"])
    df = pd.read_csv(csv_path)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df
