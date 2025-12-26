"""
Model zoo with registry of available models and factory functions.
"""

from typing import Dict, Any, Optional, List
import warnings

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Optional imports with flags
_HAS_XGB = False
_HAS_LGB = False
_HAS_CAT = False
_HAS_STATS = False
_HAS_PROPHET = False
_HAS_TORCH = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LGB = True
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    pass

try:
    import statsmodels.api as sm
    _HAS_STATS = True
except ImportError:
    pass

try:
    try:
        from prophet import Prophet
    except ImportError:
        from fbprophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass


# Model registry
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Linear models
    'linear_regression': {
        'tasks': ['regression'],
        'can_proba': False,
        'display': 'LinearRegression'
    },
    'ridge': {
        'tasks': ['regression'],
        'can_proba': False,
        'display': 'Ridge'
    },
    'lasso': {
        'tasks': ['regression'],
        'can_proba': False,
        'display': 'Lasso'
    },
    'elasticnet': {
        'tasks': ['regression'],
        'can_proba': False,
        'display': 'ElasticNet'
    },
    'logistic_regression': {
        'tasks': ['classification'],
        'can_proba': True,
        'display': 'LogisticRegression'
    },
    
    # Trees
    'decision_tree': {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'DecisionTree'
    },
    'random_forest': {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'RandomForest'
    },
    'extra_trees': {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'ExtraTrees'
    },
    
    # Boosting
    'gradient_boosting': {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'GradientBoosting'
    },
    'adaboost': {
        'tasks': ['classification', 'regression'],
        'can_proba': False,  # Fixed: AdaBoostRegressor doesn't have predict_proba
        'display': 'AdaBoost'
    },
    
    # Other
    'knn': {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'KNeighbors'
    },
    'svm': {
        'tasks': ['classification', 'regression'],
        'can_proba': False,
        'display': 'SVM'
    },
    'naive_bayes': {
        'tasks': ['classification'],
        'can_proba': True,
        'display': 'GaussianNB'
    },
    'mlp': {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'MLP'
    },
    
    # Time-series
    'arima': {
        'tasks': ['time_series'],
        'can_proba': False,
        'display': 'ARIMA/SARIMAX'
    },
    'prophet': {
        'tasks': ['time_series'],
        'can_proba': False,
        'display': 'Prophet'
    },
}

# Extend registry for optional libraries
if _HAS_XGB:
    _MODEL_REGISTRY['xgboost'] = {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'XGBoost'
    }

if _HAS_LGB:
    _MODEL_REGISTRY['lightgbm'] = {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'LightGBM'
    }

if _HAS_CAT:
    _MODEL_REGISTRY['catboost'] = {
        'tasks': ['classification', 'regression'],
        'can_proba': True,
        'display': 'CatBoost'
    }

if _HAS_TORCH:
    _MODEL_REGISTRY['pytorch_mlp'] = {
        'tasks': ['classification', 'regression'],
        'can_proba': False,
        'display': 'PyTorch MLP'
    }


def available_models(task: Optional[str] = None) -> List[str]:
    """
    Get list of available model names.
    
    Parameters
    ----------
    task : str, optional
        Filter by task ('classification', 'regression', 'time_series').
    
    Returns
    -------
    List[str]
        Available model keys.
    """
    if task:
        return [k for k, v in _MODEL_REGISTRY.items() if task in v['tasks']]
    return list(_MODEL_REGISTRY.keys())


def model_metadata(name: Optional[str] = None) -> Dict[str, Any]:
    """Get metadata for a model or all models."""
    if name:
        key = name.lower()
        if key not in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' not in registry. Available: {list(_MODEL_REGISTRY.keys())}"
            )
        return _MODEL_REGISTRY[key]
    return _MODEL_REGISTRY.copy()


def build_model(
    name: str,
    task: str = 'classification',
    random_state: int = 42,
    **overrides
) -> Any:
    """
    Build and return a model instance.
    
    Parameters
    ----------
    name : str
        Model name (e.g., 'random_forest', 'xgboost').
    task : str
        Task type ('classification', 'regression', 'time_series').
    random_state : int
        Random seed.
    **overrides
        Additional parameters to pass to the model constructor.
    
    Returns
    -------
    Model instance (unfitted).
    
    Raises
    ------
    ValueError
        If model name is unknown or task is incompatible.
    ImportError
        If required library is not installed.
    """
    key = name.lower()
    
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {available_models()}"
        )
    
    # Validate task
    valid_tasks = ('classification', 'regression', 'time_series')
    if task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}")
    
    meta = _MODEL_REGISTRY[key]
    if task not in meta['tasks']:
        raise ValueError(
            f"Model '{name}' does not support task '{task}'. "
            f"Supported tasks: {meta['tasks']}"
        )
    
    # Remove incompatible parameters
    overrides = _filter_params(key, overrides)
    
    # Build model based on key
    if key == 'linear_regression':
        return LinearRegression(**overrides)
    
    if key == 'ridge':
        # Ridge doesn't have random_state
        return Ridge(**overrides)
    
    if key == 'lasso':
        return Lasso(random_state=random_state, **overrides)
    
    if key == 'elasticnet':
        return ElasticNet(random_state=random_state, **overrides)
    
    if key == 'logistic_regression':
        defaults = {'max_iter': 1000, 'solver': 'lbfgs'}
        params = {**defaults, **overrides}
        return LogisticRegression(random_state=random_state, **params)
    
    if key == 'decision_tree':
        if task == 'classification':
            return DecisionTreeClassifier(random_state=random_state, **overrides)
        return DecisionTreeRegressor(random_state=random_state, **overrides)
    
    if key == 'random_forest':
        defaults = {'n_jobs': -1}
        params = {**defaults, **overrides}
        if task == 'classification':
            return RandomForestClassifier(random_state=random_state, **params)
        return RandomForestRegressor(random_state=random_state, **params)
    
    if key == 'extra_trees':
        defaults = {'n_jobs': -1}
        params = {**defaults, **overrides}
        if task == 'classification':
            return ExtraTreesClassifier(random_state=random_state, **params)
        return ExtraTreesRegressor(random_state=random_state, **params)
    
    if key == 'gradient_boosting':
        if task == 'classification':
            return GradientBoostingClassifier(random_state=random_state, **overrides)
        return GradientBoostingRegressor(random_state=random_state, **overrides)
    
    if key == 'adaboost':
        if task == 'classification':
            return AdaBoostClassifier(random_state=random_state, **overrides)
        return AdaBoostRegressor(random_state=random_state, **overrides)
    
    if key == 'knn':
        if task == 'classification':
            return KNeighborsClassifier(**overrides)
        return KNeighborsRegressor(**overrides)
    
    if key == 'svm':
        if task == 'classification':
            return SVC(random_state=random_state, **overrides)
        return SVR(**overrides)
    
    if key == 'naive_bayes':
        return GaussianNB(**overrides)
    
    if key == 'mlp':
        defaults = {'max_iter': 500}
        params = {**defaults, **overrides}
        if task == 'classification':
            return MLPClassifier(random_state=random_state, **params)
        return MLPRegressor(random_state=random_state, **params)
    
    # XGBoost
    if key == 'xgboost':
        if not _HAS_XGB:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
        defaults = {'n_jobs': -1, 'verbosity': 0}
        params = {**defaults, **overrides}
        if task == 'classification':
            return XGBClassifier(random_state=random_state, **params)
        return XGBRegressor(random_state=random_state, **params)
    
    # LightGBM
    if key == 'lightgbm':
        if not _HAS_LGB:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")
        defaults = {'n_jobs': -1, 'verbosity': -1}
        params = {**defaults, **overrides}
        if task == 'classification':
            return LGBMClassifier(random_state=random_state, **params)
        return LGBMRegressor(random_state=random_state, **params)
    
    # CatBoost
    if key == 'catboost':
        if not _HAS_CAT:
            raise ImportError("catboost not installed. Run: pip install catboost")
        defaults = {'verbose': 0}
        params = {**defaults, **overrides}
        if task == 'classification':
            return CatBoostClassifier(random_state=random_state, **params)
        return CatBoostRegressor(random_state=random_state, **params)
    
    # Time series models
    if key == 'arima':
        if not _HAS_STATS:
            raise ImportError("statsmodels not installed. Run: pip install statsmodels")
        return ARIMAWrapper(**overrides)
    
    if key == 'prophet':
        if not _HAS_PROPHET:
            raise ImportError("prophet not installed. Run: pip install prophet")
        return ProphetWrapper(**overrides)
    
    # PyTorch MLP
    if key == 'pytorch_mlp':
        if not _HAS_TORCH:
            raise ImportError("torch not installed. Run: pip install torch")
        return TorchMLPWrapper(task=task, **overrides)
    
    raise ValueError(f"Model '{name}' registered but not implemented.")


def _filter_params(model_key: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out incompatible parameters for specific models."""
    filtered = params.copy()
    
    # Remove hidden_units if present (convert to proper param name)
    if 'hidden_units' in filtered:
        if model_key == 'mlp':
            filtered['hidden_layer_sizes'] = filtered.pop('hidden_units')
        elif model_key == 'pytorch_mlp':
            filtered['hidden_dims'] = filtered.pop('hidden_units')
        else:
            del filtered['hidden_units']
    
    return filtered


def default_param_grid(name: str, task: str = 'classification') -> Optional[Dict[str, List]]:
    """
    Get default hyperparameter grid for a model.
    
    Parameters
    ----------
    name : str
        Model name.
    task : str
        Task type.
    
    Returns
    -------
    Dict or None
        Parameter grid for GridSearchCV/RandomizedSearchCV.
    """
    key = name.lower()
    
    grids = {
        'linear_regression': None,
        'ridge': {'alpha': [0.1, 1.0, 10.0]},
        'lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
        'elasticnet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]},
        'logistic_regression': {'C': [0.1, 1.0, 10.0]},
        'decision_tree': {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        },
        'random_forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'extra_trees': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'gradient_boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'adaboost': {'n_estimators': [50, 100, 200]},
        'knn': {'n_neighbors': [3, 5, 7, 11]},
        'svm': {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']},
        'naive_bayes': None,
        'mlp': {
            'hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
            'alpha': [0.0001, 0.001]
        },
        'xgboost': {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1]
        },
        'lightgbm': {
            'n_estimators': [100, 200],
            'num_leaves': [31, 63],
            'learning_rate': [0.01, 0.1]
        },
        'catboost': {
            'iterations': [100, 200],
            'learning_rate': [0.01, 0.1],
            'depth': [4, 6, 8]
        },
        'arima': {'order': [(1, 0, 0), (1, 1, 0), (1, 1, 1)]},
        'prophet': None,
        'pytorch_mlp': {'epochs': [10, 20], 'lr': [0.001, 0.01]},
    }
    
    return grids.get(key)


# === Wrapper Classes ===

class ARIMAWrapper:
    """Wrapper for statsmodels SARIMAX."""
    
    def __init__(
        self,
        order: tuple = (1, 0, 0),
        seasonal_order: tuple = (0, 0, 0, 0),
        **kwargs
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.kwargs = kwargs
        self.model_ = None
        self.results_ = None
    
    def fit(self, y, exog=None):
        self.model_ = sm.tsa.statespace.SARIMAX(
            endog=y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            **self.kwargs
        )
        self.results_ = self.model_.fit(disp=False)
        return self
    
    def predict(self, start=None, end=None, exog=None, dynamic=False):
        if self.results_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.results_.predict(start=start, end=end, exog=exog, dynamic=dynamic)
    
    def summary(self):
        return self.results_.summary() if self.results_ else None


class ProphetWrapper:
    """Wrapper for Facebook Prophet."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.fitted = False
    
    def fit(self, df):
        """Fit model. df must have 'ds' and 'y' columns."""
        self.model = Prophet(**self.kwargs)
        self.model.fit(df)
        self.fitted = True
        return self
    
    def predict(self, future_df):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(future_df)
    
    def make_future_dataframe(self, periods: int, freq: str = 'D'):
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.make_future_dataframe(periods=periods, freq=freq)


class TorchMLPWrapper:
    """Simple PyTorch MLP wrapper for tabular data."""
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: tuple = (64, 32),
        output_dim: int = 1,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        task: str = 'classification'
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.task = task
        self.model = None
        self.classes_ = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build(self, output_dim: int):
        import torch.nn as nn
        
        layers = []
        dims = [self.input_dim] + list(self.hidden_dims)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
    
    def fit(self, X, y):
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        X = np.asarray(X).astype('float32')
        y = np.asarray(y)
        
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        # Determine output dimension
        if self.task == 'classification':
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            if n_classes > 2:
                output_dim = n_classes
            else:
                output_dim = 1
        else:
            output_dim = 1
        
        if self.model is None:
            self._build(output_dim)
        
        X_t = torch.tensor(X).to(self.device)
        
        if self.task == 'classification' and len(self.classes_) > 2:
            y_t = torch.tensor(y).long().to(self.device)
            loss_fn = nn.CrossEntropyLoss()
        else:
            y_t = torch.tensor(y).float().unsqueeze(1).to(self.device)
            loss_fn = nn.MSELoss() if self.task == 'regression' else nn.BCEWithLogitsLoss()
        
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        X = np.asarray(X).astype('float32')
        X_t = torch.tensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_t).cpu().numpy()
        
        if self.task == 'regression':
            return preds.ravel()
        elif len(self.classes_) > 2:
            return np.argmax(preds, axis=1)
        else:
            return (preds.ravel() > 0.5).astype(int)
    
    def predict_proba(self, X):
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification.")
        
        X = np.asarray(X).astype('float32')
        X_t = torch.tensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t).cpu().numpy()
        
        if len(self.classes_) > 2:
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        else:
            # Sigmoid
            proba = 1 / (1 + np.exp(-logits.ravel()))
            return np.column_stack([1 - proba, proba])


# Need numpy for TorchMLPWrapper
import numpy as np