"""
Model evaluation utilities.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)


def scoring_name(task: str, metric: Optional[str] = None) -> str:
    """
    Get sklearn scoring name for a task.
    
    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    metric : str, optional
        Override metric name.
    
    Returns
    -------
    str
        Sklearn scoring string.
    """
    if metric:
        return metric
    if task == 'classification':
        return 'f1_macro'
    return 'r2'


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Returns
    -------
    Dict with accuracy, precision, recall, f1, and optionally roc_auc.
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    # ROC AUC (only for binary or if proba available for multiclass)
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                if y_proba.ndim == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
                else:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
            else:
                metrics['roc_auc_ovr'] = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                ))
        except Exception:
            pass
    
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Returns
    -------
    Dict with mse, rmse, mae, r2, explained_variance.
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'explained_variance': float(explained_variance_score(y_true, y_pred))
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate model predictions.
    
    Parameters
    ----------
    y_true : array-like
        True labels/values.
    y_pred : array-like
        Predicted labels/values.
    task : str
        'classification' or 'regression'.
    y_proba : array-like, optional
        Predicted probabilities (classification only).
    
    Returns
    -------
    Dict with evaluation metrics.
    """
    if task == 'classification':
        return evaluate_classification(y_true, y_pred, y_proba)
    else:
        return evaluate_regression(y_true, y_pred)


def build_leaderboard(
    candidate_summary: Dict[str, Dict[str, Any]],
    sort_by: str = 'cv_score',
    ascending: bool = False
) -> pd.DataFrame:
    """
    Build a leaderboard DataFrame from candidate results.
    
    Parameters
    ----------
    candidate_summary : dict
        Dict mapping model names to their results.
    sort_by : str
        Column to sort by.
    ascending : bool
        Sort order.
    
    Returns
    -------
    pd.DataFrame
        Leaderboard with model rankings.
    """
    rows = []
    for name, result in candidate_summary.items():
        row = {
            'model': name,
            'cv_score': float(result.get('cv_score', np.nan)),
            'best_params': str(result.get('best_params', {})),
            'fit_time': result.get('fit_time'),
        }
        # Add test metrics if available
        if 'test_metrics' in result:
            for k, v in result['test_metrics'].items():
                row[f'test_{k}'] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    return df.reset_index(drop=True)


def save_leaderboard_html(
    df: pd.DataFrame,
    out_path: str,
    title: str = "AutoX Leaderboard"
) -> str:
    """
    Save leaderboard as HTML file.
    
    Returns path to saved file.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #ddd; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {df.to_html(index=False, escape=False, classes='leaderboard')}
    </body>
    </html>
    """
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return out_path


def save_leaderboard_csv(df: pd.DataFrame, out_path: str) -> str:
    """Save leaderboard as CSV file."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path