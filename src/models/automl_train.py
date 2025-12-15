from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import mlflow
import pandas as pd
from flaml import AutoML
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config_loader import load_config


def run_automl_training(dataset_path: Union[str, Path]) -> float:
    """
    Train a churn model on the GA training dataset stored at dataset_path.

    Expects a parquet file with at least:
      - 'fullVisitorId' column (user id)
      - 'label' column (target: 0/1 churn)
      - all other columns treated as numeric features.
    """
    dataset_path = Path(dataset_path)

    cfg = load_config()
    ml_cfg = cfg["mlflow"]

    # ---- MLflow configuration (matches config.yaml) ----
    mlflow.set_tracking_uri(ml_cfg["tracking_uri"])
    mlflow.set_experiment(ml_cfg["experiment_name"])

    # ---- Load dataset ----
    df = pd.read_parquet(dataset_path)

    target_col = "label"
    id_col = "fullVisitorId"

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset")

    # Use all non-id, non-target columns as features
    feature_cols = [c for c in df.columns if c not in (target_col, id_col)]

    if not feature_cols:
        raise ValueError("No feature columns found in dataset for training.")

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    # ---- Preprocessing ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    automl = AutoML()

    with mlflow.start_run(run_name="ga_automl_churn") as run:
        automl.fit(
            X_train_s,
            y_train,
            task="classification",
            time_budget=60,
            metric="roc_auc",
        )

        y_val_proba = automl.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)

        # ---- Log metrics & params ----
        mlflow.log_metric("val_roc_auc", auc)
        mlflow.log_param("best_estimator", automl.best_estimator)
        mlflow.log_param("feature_cols", ",".join(feature_cols))

        # ---- Save scaler + model as artifacts ----
        paths_cfg = cfg["paths"]
        models_dir = Path(paths_cfg["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = models_dir / "ga_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        best_model = automl.model

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=ml_cfg["registered_model_name"],
        )

        print(f"Trained model with ROC-AUC={auc:.3f}")

    return auc
