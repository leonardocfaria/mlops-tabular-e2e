"""Train, track (MLflow file store), register, promote via alias, export for serving."""
import os
from pathlib import Path

# SPEC asks for a local file store; MLflow 3 keeps it behind an opt-in. A sqlite:/// URI also works.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL, NUMERIC, TARGET, validate

MODEL_NAME = "adult-gbm"
ALIAS = "production"
AUC_FLOOR = 0.75
SEED = 42
PARAMS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": SEED}


def build_pipeline() -> Pipeline:
    features = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",  # numeric columns go straight to the trees
    )
    return Pipeline([("features", features), ("model", HistGradientBoostingClassifier(**PARAMS))])


def train_and_register(df, tracking_uri: str, experiment: str = "adult") -> tuple[str, str, float]:
    validate(df)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    pipe = build_pipeline().fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_val)[:, 1]
    auc = float(roc_auc_score(y_val, proba))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({**PARAMS, "n_train": len(X_tr), "n_val": len(X_val)})
        mlflow.log_metrics({"val_auc": auc, "val_accuracy": float(accuracy_score(y_val, proba > 0.5))})
        # skops (safe format); we trained this model ourselves, so trust only the tree node type it flags
        info = mlflow.sklearn.log_model(
            pipe, name="model", registered_model_name=MODEL_NAME, input_example=X_val.head(3),
            skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])
    version = str(info.registered_model_version)
    MlflowClient(tracking_uri, registry_uri=tracking_uri).set_model_version_tag(MODEL_NAME, version, "val_auc", str(auc))
    return run.info.run_id, version, auc


def promote(tracking_uri: str, version: str, auc: float) -> bool:
    """Point the `production` alias at `version` if it clears the floor and is no worse than production."""
    client = MlflowClient(tracking_uri, registry_uri=tracking_uri)
    if auc < AUC_FLOOR:
        return False
    try:
        current = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        if auc < float(current.tags.get("val_auc", 0)):
            return False
    except mlflow.exceptions.MlflowException:
        pass  # no production model yet
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    return True


def export_production(tracking_uri: str, dst: str | Path) -> Path:
    """Copy the production model's MLflow artifacts to a plain folder (portable into Docker)."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    Path(dst).mkdir(parents=True, exist_ok=True)
    return Path(mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(dst)))
