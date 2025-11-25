from pathlib import Path
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import mlflow

from src.utils.config_loader import load_config

FEATURE_COLS = ["events_7d", "events_30d", "days_active_30d", "days_since_last_event"]

def run_automl_training(dataset_path: str | Path) -> float:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = pd.read_parquet(dataset_path)
    X = df[FEATURE_COLS].values
    y = df["churn"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    automl = AutoML()

    with mlflow.start_run(run_name="automl_churn") as run:
        automl.fit(
            X_train_s, y_train,
            task="classification",
            time_budget=60,
            metric="roc_auc",
        )

        y_val_proba = automl.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)

        mlflow.log_metric("val_roc_auc", auc)
        mlflow.log_param("best_estimator", automl.best_estimator)

        models_dir = Path(cfg["paths"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        mlflow.sklearn.log_model(
            sk_model=automl,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )

        print(f"Trained model with ROC-AUC={auc:.3f}")

    return auc
