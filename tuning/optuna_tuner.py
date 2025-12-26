"""
Optuna-based hyperparameter tuning.
"""

from typing import Callable, Dict, Any, Optional
import warnings

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False
    optuna = None


class OptunaTuner:
    """
    Wrapper for Optuna hyperparameter optimization.
    """
    
    def __init__(
        self,
        direction: str = "maximize",
        sampler: Optional[Any] = None,  # Fixed: Don't assign class as default
        pruner: Optional[Any] = None,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False
    ):
        """
        Initialize OptunaTuner.
        
        Parameters
        ----------
        direction : str
            'maximize' or 'minimize'.
        sampler : optuna.samplers.BaseSampler, optional
            Optuna sampler. Default uses TPESampler.
        pruner : optuna.pruners.BasePruner, optional
            Optuna pruner.
        study_name : str, optional
            Name for the study.
        storage : str, optional
            Database URL for persistent storage.
        load_if_exists : bool
            Load existing study if available.
        """
        if not _HAS_OPTUNA:
            raise ImportError(
                "Optuna is not installed. Install with: pip install optuna"
            )
        
        # Create sampler if not provided
        if sampler is None:
            sampler = TPESampler()
        
        self.study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists
        )
        self.direction = direction
    
    def optimize(
        self,
        objective: Callable[[optuna.trial.Trial], float],
        n_trials: int = 50,
        timeout: Optional[int] = None,
        show_progress: bool = True,
        n_jobs: int = 1,
        catch: tuple = (Exception,)
    ) -> Dict[str, Any]:
        """
        Run optimization.
        
        Parameters
        ----------
        objective : callable
            Objective function that takes an Optuna trial and returns a score.
        n_trials : int
            Number of trials.
        timeout : int, optional
            Timeout in seconds.
        show_progress : bool
            Show progress bar.
        n_jobs : int
            Number of parallel jobs.
        catch : tuple
            Exceptions to catch and treat as failed trials.
        
        Returns
        -------
        Dict with best_value, best_params, and study reference.
        """
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            n_jobs=n_jobs,
            catch=catch
        )
        
        return {
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_trial': self.study.best_trial,
            'study': self.study,
            'n_trials_completed': len(self.study.trials)
        }
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        return self.study.best_params
    
    def get_trials_dataframe(self):
        """Get DataFrame of all trials."""
        return self.study.trials_dataframe()


def create_optuna_objective(
    model_builder: Callable,
    X_train,
    y_train,
    cv,
    scoring: str,
    task: str,
    random_state: int = 42
) -> Callable:
    """
    Factory to create an Optuna objective function.
    
    Parameters
    ----------
    model_builder : callable
        Function that takes trial and returns (model_key, params).
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    cv : cross-validation splitter
        sklearn CV splitter.
    scoring : str
        Scoring metric.
    task : str
        Task type.
    random_state : int
        Random seed.
    
    Returns
    -------
    Callable objective function for Optuna.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from .model_zoo import build_model
    
    def objective(trial: optuna.trial.Trial) -> float:
        model_key, params = model_builder(trial)
        
        try:
            model = build_model(
                model_key,
                task=task,
                random_state=random_state,
                **params
            )
            
            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1
            )
            return float(scores.mean())
        
        except Exception as e:
            warnings.warn(f"Trial failed: {e}")
            return float('-inf') if 'maximize' else float('inf')
    
    return objective