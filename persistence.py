import os
import joblib
import warnings

try:
    import mlflow  # type: ignore
    import mlflow.sklearn  # type: ignore
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

def save_pipeline(path: str, pipeline) -> str:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    joblib.dump(pipeline, path)
    return path

def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def mlflow_log_model(pipeline, artifact_path: str = "model", run_name: str = "autox_run", params: dict = None):
    if not _HAS_MLFLOW:
        raise ImportError("mlflow not installed. Install with `pip install mlflow` to enable MLflow logging.")
    params = params or {}
    mlflow.set_experiment("autox_experiments")
    with mlflow.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.sklearn.log_model(pipeline, artifact_path)
        run = mlflow.active_run().info
    return {"run_id": run.run_id, "artifact_uri": run.artifact_uri}
