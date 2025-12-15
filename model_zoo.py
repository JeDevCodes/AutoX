from typing import Dict, Any, Optional, List
import warnings

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

#initial flags
_HAS_XGB = True
_HAS_LGB = True
_HAS_CAT = True
_HAS_STATS = True
_HAS_PROPHET = True
_HAS_TORCH = True

# Try optional imports and set flags.
try:
    import xgboost as xgb  # noqa: F401
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb  # noqa: F401
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    _HAS_LGB = False

try:
    import catboost  # noqa: F401
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    _HAS_CAT = False

try:
    import statsmodels.api as sm  # for ARIMA/SARIMAX wrappers
except Exception:
    _HAS_STATS = False

try:
    try:
        from prophet import Prophet  # type: ignore
    except Exception:
        # legacy fbprophet
        from fbprophet import Prophet  # type: ignore
except Exception:
    _HAS_PROPHET = False

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
except Exception:
    _HAS_TORCH = False

# Helper metadata registry. Each key maps to a dict with 'tasks' and 'can_proba' flags.
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Linear models
    'linear_regression': {'tasks': ['regression'], 'can_proba': False, 'display': 'LinearRegression'},
    'ridge': {'tasks': ['regression'], 'can_proba': False, 'display': 'Ridge'},
    'lasso': {'tasks': ['regression'], 'can_proba': False, 'display': 'Lasso'},
    'elasticnet': {'tasks': ['regression'], 'can_proba': False, 'display': 'ElasticNet'},

    'logistic_regression': {'tasks': ['classification'], 'can_proba': True, 'display': 'LogisticRegression'},

    # Trees
    'decision_tree': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'DecisionTree'},
    'random_forest': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'RandomForest'},
    'extra_trees': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'ExtraTrees'},

    # Boosting (sklearn gradient boosting included)
    'gradient_boosting': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'GradientBoosting'},
    'adaboost': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'AdaBoost'},

    # Neighbors / SVM / NB
    'knn': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'KNeighbors'},
    'svm': {'tasks': ['classification', 'regression'], 'can_proba': False, 'display': 'SVM'},
    'naive_bayes': {'tasks': ['classification'], 'can_proba': True, 'display': 'GaussianNB'},

    # Neural nets (sklearn)
    'mlp': {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'MLP'},

    # Time-series placeholders
    'arima': {'tasks': ['time_series'], 'can_proba': False, 'display': 'ARIMA/SARIMAX'},
    'prophet': {'tasks': ['time_series'], 'can_proba': False, 'display': 'Prophet'},
}

# Extend registry if optional boosters are present
if _HAS_XGB:
    _MODEL_REGISTRY['xgboost'] = {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'XGBoost'}
if _HAS_LGB:
    _MODEL_REGISTRY['lightgbm'] = {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'LightGBM'}
if _HAS_CAT:
    _MODEL_REGISTRY['catboost'] = {'tasks': ['classification', 'regression'], 'can_proba': True, 'display': 'CatBoost'}
if _HAS_TORCH:
    _MODEL_REGISTRY['pytorch_mlp'] = {'tasks': ['classification', 'regression'], 'can_proba': False, 'display': 'PyTorch MLP (custom wrapper)'}


def available_models() -> List[str]:
    return list(_MODEL_REGISTRY.keys())

def model_metadata(name: Optional[str] = None) -> Dict[str, Any]:
    if name:
        key = name.lower()
        if key not in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' not present in registry. Available: {list(_MODEL_REGISTRY.keys())}")
        return _MODEL_REGISTRY[key]
    return _MODEL_REGISTRY.copy()


def build_model(name: str, task: str = 'classification', random_state: int = 42, **overrides):

    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {available_models()}")

    # Quick validation of requested task
    if task not in ('classification', 'regression', 'time_series'):
        raise ValueError("task must be one of 'classification', 'regression', 'time_series'")

    # Map keys to constructors
    if key == 'linear_regression':
        if task != 'regression':
            raise ValueError("LinearRegression supports regression only.")
        return LinearRegression(**overrides)

    if key == 'ridge':
        return Ridge(random_state=random_state, **overrides)

    if key == 'lasso':
        return Lasso(**overrides)

    if key == 'elasticnet':
        return ElasticNet(random_state=random_state, **overrides)

    if key == 'logistic_regression':
        # multiclass: 'auto' solver selection via liblinear or lbfgs
        # set max_iter higher for convergence
        default = {'max_iter': 1000, 'solver': 'lbfgs'}
        final = {**default, **overrides}
        return LogisticRegression(random_state=random_state, **final)

    if key == 'decision_tree':
        if task == 'classification':
            return DecisionTreeClassifier(random_state=random_state, **overrides)
        return DecisionTreeRegressor(random_state=random_state, **overrides)

    if key == 'random_forest':
        if task == 'classification':
            return RandomForestClassifier(random_state=random_state, n_jobs=-1, **overrides)
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **overrides)

    if key == 'extra_trees':
        if task == 'classification':
            return ExtraTreesClassifier(random_state=random_state, n_jobs=-1, **overrides)
        return ExtraTreesRegressor(random_state=random_state, n_jobs=-1, **overrides)

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
            return SVC(probability=False, random_state=random_state, **overrides)
        return SVR(**overrides)

    if key == 'naive_bayes':
        if task != 'classification':
            raise ValueError("GaussianNB supports classification only.")
        return GaussianNB(**overrides)

    if key == 'mlp':
        # choose classifier/regressor variant
        if task == 'classification':
            return MLPClassifier(random_state=random_state, max_iter=500, **overrides)
        return MLPRegressor(random_state=random_state, max_iter=500, **overrides)

    # Optional boosters
    if key == 'xgboost':
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed. Install via `pip install xgboost` to use this model.")
        if task == 'classification':
            return XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, **overrides)
        return XGBRegressor(random_state=random_state, n_jobs=-1, **overrides)

    if key == 'lightgbm':
        if not _HAS_LGB:
            raise ImportError("lightgbm is not installed. Install via `pip install lightgbm` to use this model.")
        if task == 'classification':
            return LGBMClassifier(random_state=random_state, n_jobs=-1, **overrides)
        return LGBMRegressor(random_state=random_state, n_jobs=-1, **overrides)

    if key == 'catboost':
        if not _HAS_CAT:
            raise ImportError("catboost is not installed. Install via `pip install catboost` to use this model.")
        # CatBoost accepts verbose / iterations etc
        if task == 'classification':
            return CatBoostClassifier(random_state=random_state, verbose=0, **overrides)
        return CatBoostRegressor(random_state=random_state, verbose=0, **overrides)

    # Time-series models
    if key == 'arima':
        if not _HAS_STATS:
            raise ImportError("statsmodels is required for ARIMA/SARIMAX. Install via `pip install statsmodels`.")
        # Do not instantiate ARIMA here because statsmodels expects endog array at construction.
        # return a callable / small wrapper class that will create the model at fit-time.
        class ARIMAWrapped:
            def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), enforce_stationarity=False, enforce_invertibility=False):
                self.order = order
                self.seasonal_order = seasonal_order
                self.enforce_stationarity = enforce_stationarity
                self.enforce_invertibility = enforce_invertibility
                self.model_ = None
                self.results_ = None

            def fit(self, y, exog=None):
                # uses SARIMAX for more flexibility
                self.model_ = sm.tsa.statespace.SARIMAX(endog=y, exog=exog, order=self.order, seasonal_order=self.seasonal_order,
                                                       enforce_stationarity=self.enforce_stationarity,
                                                       enforce_invertibility=self.enforce_invertibility)
                self.results_ = self.model_.fit(disp=False)
                return self

            def predict(self, start=None, end=None, exog=None, dynamic=False):
                if self.results_ is None:
                    raise RuntimeError("Model not fitted.")
                return self.results_.predict(start=start, end=end, exog=exog, dynamic=dynamic)

            def summary(self):
                return self.results_.summary()

        # allow overrides to be passed as constructor args
        return ARIMAWrapped(**overrides)

    if key == 'prophet':
        if not _HAS_PROPHET:
            raise ImportError("Prophet is not installed. Install via `pip install prophet` to use Prophet.")
        # Prophet requires dataframe with columns 'ds' and 'y' - we'll return a small wrapper
        class ProphetWrapped:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.model = Prophet(**kwargs)
                self.fitted = False

            def fit(self, df):
                # expects df to have 'ds' and 'y'
                self.model.fit(df)
                self.fitted = True
                return self

            def predict(self, future_df):
                if not self.fitted:
                    raise RuntimeError("Prophet model not fitted.")
                return self.model.predict(future_df)

        return ProphetWrapped(**overrides)

    if key == 'pytorch_mlp':
        if not _HAS_TORCH:
            raise ImportError("torch is not installed. Install via `pip install torch` to use pytorch_mlp.")
        # simple dense network factory wrapper for small tabular tasks
        import torch
        import torch.nn as nn
        import torch.optim as optim

        class TorchMLPWrapper:
            def __init__(self, input_dim=None, hidden_dims=(64, 32), lr=1e-3, epochs=10, batch_size=64, task='classification'):
                # if input_dim is None, user must set later
                self.input_dim = input_dim
                self.hidden_dims = hidden_dims
                self.lr = lr
                self.epochs = epochs
                self.batch_size = batch_size
                self.task = task
                self.model = None
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            def _build(self):
                dims = [self.input_dim] + list(self.hidden_dims)
                layers = []
                for i in range(len(dims)-1):
                    layers.append(nn.Linear(dims[i], dims[i+1]))
                    layers.append(nn.ReLU())
                # output
                out_dim = 1 if self.task == 'regression' else 1  # for classifier we'll apply sigmoid or BCE
                layers.append(nn.Linear(dims[-1], out_dim))
                self.model = nn.Sequential(*layers).to(self.device)

            def fit(self, X, y):
                import numpy as np
                from torch.utils.data import TensorDataset, DataLoader
                X = X.astype('float32')
                if self.input_dim is None:
                    self.input_dim = X.shape[1]
                if self.model is None:
                    self._build()
                X_t = torch.tensor(X).to(self.device)
                y_t = torch.tensor(np.array(y)).to(self.device).float().unsqueeze(1)
                ds = TensorDataset(X_t, y_t)
                loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
                opt = optim.Adam(self.model.parameters(), lr=self.lr)
                loss_fn = nn.MSELoss() if self.task == 'regression' else nn.BCEWithLogitsLoss()
                self.model.train()
                for ep in range(self.epochs):
                    for xb, yb in loader:
                        opt.zero_grad()
                        preds = self.model(xb)
                        loss = loss_fn(preds, yb)
                        loss.backward()
                        opt.step()
                return self

            def predict(self, X):
                import numpy as np
                X_t = torch.tensor(X.astype('float32')).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    preds = self.model(X_t).cpu().numpy()
                if self.task == 'regression':
                    return preds.ravel()
                else:
                    # return class labels by threshold 0.5
                    return (preds.ravel() > 0.5).astype(int)

        # user must pass input_dim in overrides or call fit with numpy/pandas X to infer input_dim
        return TorchMLPWrapper(task=task, **overrides)

    # fallback safety
    raise ValueError(f"Model key '{name}' is registered but not handled. Available keys: {available_models()}")


def wdefault_param_grid(name: str, task: str = 'classification') -> Optional[Dict[str, Any]]:
    
    key = name.lower()
    if key == 'linear_regression':
        return None
    if key == 'ridge':
        return {'alpha': [1.0, 10.0]}
    if key == 'lasso':
        return {'alpha': [0.001, 0.01, 0.1]}
    if key == 'elasticnet':
        return {'alpha': [0.01, 0.1], 'l1_ratio': [0.2, 0.5]}

    if key == 'logistic_regression':
        return {'C': [0.1, 1.0, 10.0], 'penalty': ['l2']}

    if key == 'decision_tree':
        return {'max_depth': [None, 5, 15], 'min_samples_split': [2, 5]}
    if key == 'random_forest' or key == 'extra_trees':
        return {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10]}
    if key == 'gradient_boosting':
        return {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1]}
    if key == 'adaboost':
        return {'model__n_estimators': [50, 100]}

    if key == 'knn':
        return {'model__n_neighbors': [3, 5, 7]}
    if key == 'svm':
        if task == 'classification':
            return {'model__C': [0.1, 1.0], 'model__kernel': ['rbf']}
        return {'model__C': [0.1, 1.0], 'model__kernel': ['rbf']}
    if key == 'naive_bayes':
        return None

    if key == 'mlp':
        return {'model__hidden_layer_sizes': [(64,), (64, 32)], 'model__alpha': [0.0001, 0.001]}

    if key == 'xgboost' and _HAS_XGB:
        return {'model__n_estimators': [100, 200], 'model__max_depth': [3, 6], 'model__learning_rate': [0.01, 0.1]}
    if key == 'lightgbm' and _HAS_LGB:
        return {'model__n_estimators': [100, 200], 'model__num_leaves': [31, 63], 'model__learning_rate': [0.01, 0.1]}
    if key == 'catboost' and _HAS_CAT:
        return {'model__iterations': [100, 200], 'model__learning_rate': [0.01, 0.1]}

    if key == 'arima':
        return {'order': [(1, 0, 0), (1, 1, 0)]}
    if key == 'prophet':
        return None

    if key == 'pytorch_mlp' and _HAS_TORCH:
        return {'epochs': [10], 'batch_size': [32]}

    return None