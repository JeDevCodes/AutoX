"""
AutoX - Automated Machine Learning Pipeline
"""

from .ingestion import load_data, detect_schema
from .preprocessing import build_preprocessor, fit_preprocessor, save_preprocessor, load_preprocessor
from .feature_engineering import (
    add_polynomial_interactions,
    add_polynomial_features,
    add_lag_features
)
from .model_zoo import available_models, build_model, default_param_grid, model_metadata
from .trainer import Trainer
from .evaluation import scoring_name, build_leaderboard, save_leaderboard_html
from .persistence import save_pipeline, load_pipeline, mlflow_log_model
from .orchestrator import AutoX

__version__ = "0.1.0"

__all__ = [
    "load_data",
    "detect_schema",
    "build_preprocessor",
    "fit_preprocessor",
    "save_preprocessor",
    "load_preprocessor",
    "add_polynomial_interactions",
    "add_polynomial_features",
    "add_lag_features",
    "available_models",
    "build_model",
    "default_param_grid",
    "model_metadata",
    "Trainer",
    "scoring_name",
    "build_leaderboard",
    "save_leaderboard_html",
    "save_pipeline",
    "load_pipeline",
    "mlflow_log_model",
    "AutoX",
]