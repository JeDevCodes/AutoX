from typing import Callable, Dict, Any, Optional

try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

if _HAS_OPTUNA:
    class OptunaTuner:
        def __init__(self, direction: str = "maximize", sampler: Optional[optuna.samplers.BaseSampler] = optuna.samplers.TPESampler,
                     pruner: Optional[optuna.pruners.BasePruner] = None, study_name: Optional[str] = None):
            self.study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name)

        def optimize(self, objective: Callable[[optuna.trial.Trial], float], n_trials: int = 50, show_progress: bool = False) -> Dict[str, Any]:
            self.study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress)
            return {'best_value': self.study.best_value, 'best_params': self.study.best_params, 'study': self.study}
else:
    class OptunaTuner:
        def __init__(self, *args, **kwargs):
            raise ImportError("Optuna is not installed. Install with `pip install optuna` to use OptunaTuner.")
