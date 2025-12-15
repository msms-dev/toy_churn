from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Tuple


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def compute_simulated_ga_date(
    logical_ds: str,
    sim_start_date: str,
    ga_start_date: str,
    ga_end_date: str,
    step_days: int,
) -> date:
    """
    Map Airflow logical date (ds) -> a simulated GA date.

    Example:
      sim_start_date = 2025-12-14
      ga_start_date  = 2017-04-03
      step_days      = 1 (scoring) or 7 (training)
      ds=2025-12-14 -> 2017-04-03
      ds=2025-12-15 -> 2017-04-04
    """
    ds_date = datetime.strptime(logical_ds, "%Y-%m-%d").date()
    sim0 = datetime.strptime(sim_start_date, "%Y-%m-%d").date()
    ga0 = datetime.strptime(ga_start_date, "%Y-%m-%d").date()
    ga_end = datetime.strptime(ga_end_date, "%Y-%m-%d").date()

    delta_days = (ds_date - sim0).days
    if delta_days < 0:
        # if user manually triggers earlier than sim_start_date
        delta_days = 0

    steps = delta_days // step_days
    ga_day = ga0 + timedelta(days=steps * step_days)

    if ga_day > ga_end:
        raise ValueError(f"Simulated GA date {ga_day} exceeds ga_end_date {ga_end}")

    return ga_day