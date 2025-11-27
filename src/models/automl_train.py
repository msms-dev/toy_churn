from pathlib import Path
import pandas as pd
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import mlflow

from src.utils.config_loader import load_config
from src.ingestion.api_client import load_raw_events
from src.labelling.build_labels import build_churn_labels
from src.features.build_features import build_features

FEATURE_COLS = ["events_7d", "events_30d", "days_active_30d", "days_since_last_event"]


def run_automl_training(dataset_path: str | Path) -> float:
    cfg = load_config()
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    df = pd.read_parquet(dataset_path)
    X = df[FEATURE_COLS].values
    y = df["churn"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
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
            X_train_s,
            y_train,
            task="classification",
            time_budget=60,
            metric="roc_auc",
        )

        y_val_proba = automl.predict_proba(X_val_s)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)

        mlflow.log_metric("val_roc_auc", auc)
        mlflow.log_param("best_estimator", automl.best_estimator)

        cfg_paths = cfg["paths"]
        models_dir = Path(cfg_paths["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        best_model = automl.model

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )
        
        print(f"Trained model with ROC-AUC={auc:.3f}")

    return auc


if __name__ == "__main__":
    """
    End-to-end runner:
    - Load raw events
    - Build labels
    - Build features
    - Save training dataset
    - Run AutoML training + log to MLflow
    """
    cfg = load_config()
    history_days = cfg["label_definition"]["history_days"]
    churn_window = cfg["label_definition"]["churn_window_days"]

    print("🔹 Loading raw events...")
    df_events = load_raw_events()

    print("🔹 Building churn labels...")
    labels_df = build_churn_labels(df_events, history_days, churn_window)

    print("🔹 Building features & training dataset...")
    dataset = build_features(df_events, labels_df, history_days)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = processed_dir / "training_dataset.parquet"
    dataset.to_parquet(dataset_path)
    print(f"✅ Saved training dataset to {dataset_path}")

    print("🔹 Starting AutoML training...")
    auc = run_automl_training(dataset_path)
    print(f"🎉 Done. Validation ROC-AUC: {auc:.3f}")

