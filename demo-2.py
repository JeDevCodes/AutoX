import argparse
import os
from sklearn.datasets import load_iris, fetch_california_housing
from .profiling import profile_dataframe
from .preprocessing import build_preprocessor, fit_preprocessor, save_preprocessor
from .trainer import Trainer
from .evaluation import build_leaderboard, save_leaderboard_html

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    p.add_argument("--use-optuna", action="store_true", help="Use Optuna tuning if available")
    p.add_argument("--optuna-trials", type=int, default=20)
    p.add_argument("--`artifact`-dir", type=str, default="artifacts")
    return p.parse_args()

def main():
    args = parse_args()
    task = args.task
    if task == 'classification':
        data = load_iris(as_frame=True)
        X = data.frame.drop(columns=['target'])
        y = data.frame['target']
    else:
        data = fetch_california_housing(as_frame=True)
        X = data.frame.drop(columns=['MedHouseVal'])
        y = data.frame['MedHouseVal']

    out = os.path.abspath(args.artifact_dir)
    os.makedirs(out, exist_ok=True)

    print("[CLI] Profiling dataset...")
    schema, artifacts = profile_dataframe(X, out_dir=os.path.join(out, "profile"), sample_frac=0.5)
    print("[CLI] Building preprocessor based on schema...")
    preproc, meta = build_preprocessor(X)
    preproc = fit_preprocessor(preproc, X)
    preproc_path = os.path.join(out, "preprocessor.joblib")
    save_preprocessor(preproc, preproc_path)

    print("[CLI] Running Trainer...")
    trainer = Trainer(task=task, preprocessor_path=preproc_path, artifact_dir=out)
    best_name, metrics = trainer.run(X, y, use_optuna=args.use_optuna, optuna_trials=args.optuna_trials)
    print("[CLI] Best:", best_name, metrics)

    # leaderboard
    summary = {k: {'cv_score': v.get('cv_score'), 'best_params': v.get('best_params')} for k, v in trainer.candidate_summary.items()}
    df = build_leaderboard(summary)
    save_leaderboard_html(df, os.path.join(out, "leaderboard.html"))
    print("[CLI] Leaderboard saved to", os.path.join(out, "leaderboard.html"))

if __name__ == "__main__":
    main()
