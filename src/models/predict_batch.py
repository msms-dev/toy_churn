from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.utils.config_loader import load_config

# Must match training
FEATURE_COLS = ["events_7d", "events_30d", "days_active_30d", "days_since_last_event"]


def load_production_model(cfg: dict):
    """Load the latest Production model from MLflow Model Registry."""
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    model_name = cfg["mlflow"]["registered_model_name"]

    model_uri = f"models:/{model_name}/Production"
    print(f"🔹 Loading model from registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    return model


def load_scaler(cfg: dict):
    """Load the scaler saved during training."""
    models_dir = Path(cfg["paths"]["models_dir"])
    scaler_path = models_dir / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. "
            "Have you run training and logged the scaler?"
        )
    print(f"🔹 Loading scaler from: {scaler_path}")
    return joblib.load(scaler_path)


def predict_batch(
    input_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Load a batch of feature rows, score churn probabilities, and save results.

    Expects `input_path` to be a Parquet file with at least FEATURE_COLS,
    and ideally a `user_id` column so we can attach scores per user.
    """
    cfg = load_config()
    model = load_production_model(cfg)
    scaler = load_scaler(cfg)

    print(f"🔹 Reading features from: {input_path}")
    df = pd.read_parquet(input_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input data is missing required feature columns: {missing}. "
            f"Expected: {FEATURE_COLS}"
        )

    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # For sklearn classifiers logged by MLflow, predict_proba is available
    print("🔹 Generating churn scores...")
    proba = model.predict_proba(X_scaled)[:, 1]

    # Build output frame
    if "user_id" in df.columns:
        out = df[["user_id"]].copy()
    else:
        out = df.index.to_series().rename("row_id").to_frame()

    out["churn_score"] = proba

    # Add reference to model version if you want (optional)
    out["model_name"] = cfg["mlflow"]["registered_model_name"]

    if output_path is None:
        output_path = Path("data/processed/churn_scores.parquet")
    else:
        output_path = Path(output_path)

    #  NEW: more robust parquet write to avoid "Resource deadlock avoided"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If the file already exists, delete it first (helps with overlayfs weirdness)
    if output_path.exists():
        output_path.unlink()

    # Explicitly open the file handle and let pandas write to it
    with open(output_path, "wb") as f:
        out.to_parquet(f)

    print(f"✅ Saved churn scores to: {output_path}")
    print(out.head())

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch churn scoring using MLflow registry model.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/processed/scoring_features.parquet",
        help="Path to Parquet file with features to score.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/churn_scores.parquet",
        help="Where to write scores Parquet file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_batch(Path(args.input_path), Path(args.output_path))
