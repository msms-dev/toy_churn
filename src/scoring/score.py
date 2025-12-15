# src/scoring/score.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import mlflow
import mlflow.sklearn

from src.utils.config_loader import load_config

def score_users(
    features_df: pd.DataFrame,
    model_uri: str,
    scores_dir: str,
    score_date: str,
    id_col: str = "fullVisitorId",
) -> str:
    """
    Score users using an MLflow model URI and save date-stamped parquet.

    model_uri example: "models:/toy_churn_model/Production"
    """
    if features_df.empty:
        raise ValueError("No features to score.")

    if id_col not in features_df.columns:
        raise KeyError(f"Missing id column '{id_col}' in features_df.")

    # Load the Production model from MLflow Registry
    model = mlflow.pyfunc.load_model(model_uri)

    # Predict probabilities if available, else raw predictions
    X = features_df.drop(columns=[id_col])
    preds = model.predict(X)

    # If preds are 2D probabilities, take column 1; otherwise assume already 1D
    if hasattr(preds, "shape") and len(getattr(preds, "shape", [])) == 2 and preds.shape[1] >= 2:
        churn_score = preds[:, 1]
    else:
        churn_score = preds

    out = pd.DataFrame(
        {
            id_col: features_df[id_col].astype(str),
            "churn_score": churn_score,
            "score_date": score_date,
        }
    )

    Path(scores_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(scores_dir) / f"churn_scores_{score_date}.parquet")
    out.to_parquet(out_path, index=False)

    return out_path

