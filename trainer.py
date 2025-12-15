from typing import Optional, List, Dict, Any, Tuple
import os
import numpy as np
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, r2_score, mean_squared_error
from .model_zoo import available_models, build_model, default_param_grid, model_metadata
from .preprocessing import build_preprocessor, fit_preprocessor, load_preprocessor  # load_preprocessor if defined, otherwise joblib
from .tuning.optuna_tuner import OptunaTuner
from .persistence import save_pipeline, mlflow_log_model, load_pipeline
from .evaluation import scoring_name
import traceback

try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

DEFAULT_ARTIFACT_DIR = "artifacts"

class Trainer:
    def __init__(self,
                 task: str = 'classification',
                 preprocessor: Optional[Any] = None,
                 preprocessor_path: Optional[str] = None,
                 random_state: int = 42,
                 artifact_dir: str = DEFAULT_ARTIFACT_DIR):
        assert task in ('classification', 'regression')
        self.task = task
        self.random_state = random_state
        self.artifact_dir = os.path.abspath(artifact_dir)
        os.makedirs(self.artifact_dir, exist_ok=True)

        if preprocessor_path:
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(preprocessor_path)
            self.preprocessor = joblib.load(preprocessor_path)
        else:
            self.preprocessor = preprocessor

        self.candidate_summary: Dict[str, Dict[str, Any]] = {}
        self.best_pipeline = None
        self.best_name = None
        self.best_score = -np.inf

    def _cv_splitter(self, n_splits: int = 5):
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state) if self.task == 'classification' else KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def _score_metric(self):
        return 'f1_macro' if self.task == 'classification' else 'r2'

    def _final_metrics(self, y_true, y_pred) -> Dict[str, float]:
        if self.task == 'classification':
            return {'accuracy': float(accuracy_score(y_true, y_pred)), 'f1_macro': float(f1_score(y_true, y_pred, average='macro'))}
        else:
            mse = float(mean_squared_error(y_true, y_pred))
            return {'mse': mse, 'rmse': float(np.sqrt(mse)), 'r2': float(r2_score(y_true, y_pred))}

    def _optuna_objective_builder(self, X_train, y_train, model_key: str, n_splits: int):
        def objective(trial: 'optuna.trial.Trial'):
            params = {}
            mk = model_key.lower()

            if mk in ('random_forest', 'extra_trees'):
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)

            elif mk in ('xgboost', 'lightgbm', 'catboost', 'gradient_boosting'):
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 12)
                params['learning_rate'] = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)

            elif mk in ('decision_tree',):
                params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)

            elif mk in ('mlp', 'pytorch_mlp'):
                params['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-1)
                params['hidden_units'] = trial.suggest_categorical('hidden_units', [(64,), (64, 32), (128, 64)])

            elif mk == 'svm':
                params['C'] = trial.suggest_loguniform('C', 1e-2, 10.0)

            else:
                # default small forest
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
                params['max_depth'] = trial.suggest_int('max_depth', 3, 20)

            try:
                model = build_model(model_key, task=self.task, **params, random_state=self.random_state)
            except Exception as e:
                raise

            pipeline = Pipeline([('preproc', self.preprocessor), ('model', model)])
            cv = self._cv_splitter(n_splits=n_splits)
            scoring = self._score_metric()
            try:
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                score = float(np.mean(scores))
            except Exception as e:
                raise

            return score

        return objective

    def _run_optuna_for_model(self, model_key: str, X_train, y_train, n_trials: int = 30, n_splits: int = 3):
        if not _HAS_OPTUNA:
            raise ImportError("Optuna not installed - cannot run optuna tuning.")
        
        tuner = OptunaTuner(direction='maximize')
        objective = self._optuna_objective_builder(X_train, y_train, model_key, n_splits=n_splits)
        res = tuner.optimize(objective, n_trials=n_trials, show_progress=False)
        best_params = res['best_params']
        
        return res

    def run(self,
            X, y,
            candidates: Optional[List[str]] = None,
            use_optuna: bool = True,
            optuna_trials: int = 30,
            optuna_cv_splits: int = 3,
            test_size: float = 0.2,
            cv_splits: int = 5,
            grid_when_no_optuna: bool = True):
        
        if candidates is None:
            candidates = available_models()

        stratify_vals = y if (self.task == 'classification') else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=stratify_vals)

        for model_key in candidates:
            print(f"[Trainer] Candidate: {model_key}")
            try:
                if use_optuna and _HAS_OPTUNA:
                    try:
                        opt_res = self._run_optuna_for_model(model_key, X_train, y_train, n_trials=optuna_trials, n_splits=optuna_cv_splits)
                        best_params = opt_res['best_params']
                        cv_score = opt_res['best_value']
                    except Exception as e:
                        warnings.warn(f"Optuna tuning failed for {model_key}: {e}\n{traceback.format_exc()}")
                        best_params = None
                        cv_score = -np.inf
                else:
                    # fallback to gridsearch if available
                    grid = default_param_grid(model_key, task=self.task)

                    if grid is not None and grid_when_no_optuna:
                        # GridSearchCV expects pipeline param names if pipeline used; build pipeline with model via 'model' step

                        model_inst = build_model(model_key, task=self.task, random_state=self.random_state)
                        pipeline = Pipeline([('preproc', self.preprocessor), ('model', model_inst)])

                        if self.task == 'classification':
                            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
                        else:
                            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

                        gs = GridSearchCV(pipeline, {k if k.startswith('model__') else f"model__{k}": v for k, v in grid.items()} if any(not k.startswith('model__') for k in grid.keys()) else grid,
                                          cv=cv, scoring=self._score_metric(), n_jobs=-1, refit=True)
                        gs.fit(X_train, y_train)
                        best_pipeline = gs.best_estimator_
                        cv_score = gs.best_score_
                        best_params = gs.best_params_

                        # Save pipeline to candidate results
                        self.candidate_summary[model_key] = {'cv_score': float(cv_score), 'best_params': best_params, 'pipeline': best_pipeline}
                        # choose best
                        if cv_score > self.best_score:
                            self.best_score = cv_score
                            self.best_pipeline = best_pipeline
                            self.best_name = model_key
                        continue
                    else:
                        # neither optuna nor grid available; train with default params
                        model_inst = build_model(model_key, task=self.task, random_state=self.random_state)
                        pipeline = Pipeline([('preproc', self.preprocessor), ('model', model_inst)])
                        cv = self._cv_splitter(n_splits=`cv_splits`)

                        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=self._score_metric(), n_jobs=-1)
                        cv_score = float(np.mean(scores))
                        pipeline.fit(X_train, y_train)
                        
                        best_params = None
                        self.candidate_summary[model_key] = {'cv_score': float(cv_score), 'best_params': best_params, 'pipeline': pipeline}
                        if cv_score > self.best_score:
                            self.best_score = cv_score
                            self.best_pipeline = pipeline
                            self.best_name = model_key
                        continue

                # If we reach here and have best_params from optuna, instantiate final pipeline with those params and fit on full train
                if best_params:
                    # optuna trial params may be named like 'n_estimators' etc - pass to build_model directly
                    try:
                        model_inst = build_model(model_key, task=self.task, random_state=self.random_state, **best_params)
                        pipeline = Pipeline([('preproc', self.preprocessor), ('model', model_inst)])
                        pipeline.fit(X_train, y_train)
                        # store
                        self.candidate_summary[model_key] = {'cv_score': float(cv_score), 'best_params': best_params, 'pipeline': pipeline}
                        if cv_score > self.best_score:
                            self.best_score = cv_score
                            self.best_pipeline = pipeline
                            self.best_name = model_key
                    except Exception as e:
                        warnings.warn(f"Failed to instantiate best model for {model_key} with params {best_params}: {e}\n{traceback.format_exc()}")
                        continue

            except Exception as e:
                warnings.warn(f"Candidate {model_key} failed entirely: {e}\n{traceback.format_exc()}")
                continue

        if self.best_pipeline is None:
            raise RuntimeError("No successful model training for any candidate.")

        # final evaluation on test
        y_pred = self.best_pipeline.predict(X_test)
        metrics = self._final_metrics(y_test, y_pred)
        print(f"[Trainer] Best: {self.best_name} | test metrics: {metrics}")

        # persist best pipeline
        best_path = os.path.join(self.artifact_dir, "best_pipeline.joblib")
        save_pipeline(best_path, self.best_pipeline)
        # summary
        summary_path = os.path.join(self.artifact_dir, "candidates_summary.joblib")
        joblib.dump(self.candidate_summary, summary_path)
        print(f"[Trainer] Saved best pipeline -> {best_path} | candidate summary -> {summary_path}")

        # optional MLflow logging - catch errors and continue
        try:
            mlflow_info = None
            try:
                mlflow_info = mlflow_log_model(self.best_pipeline, artifact_path="model", run_name=f"autox_{self.best_name}", params={'task': self.task})
                print(f"[Trainer] MLflow logged: {mlflow_info}")
            except Exception as e:
                # mlflow missing or failing - just warn
                warnings.warn(f"MLflow logging skipped/failed: {e}")
        except Exception:
            pass

        return self.best_name, metrics
