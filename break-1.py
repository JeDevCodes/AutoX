# main.py (demo snippet)
import argparse
from .ingestion import load_data
from .preprocessing import build_preprocessor, fit_preprocessor, save_preprocessor
import pandas as pd
import os
from sklearn.datasets import load_iris, fetch_california_housing

def demo_with_local_dataset(task='classification'):
    # demo uses sklearn dataset to avoid external files
    if task == 'classification':
        data = load_iris(as_frame=True)
        X = data.frame.drop(columns=['target'])
        y = data.frame['target']
    else:
        data = fetch_california_housing(as_frame=True)
        X = data.frame.drop(columns=['MedHouseVal'])
        y = data.frame['MedHouseVal']

    print("Detecting schema...")
    from .ingestion import detect_schema
    _, schema = load_data  # no-op to avoid linter; we use schema from detect_schema
    schema = detect_schema(X)
    for col, info in schema.items():
        print(f"{col}: {info}")

    print("Building preprocessor...")
    preproc, meta = build_preprocessor(X)
    print("Preprocessor metadata:", meta)

    print("Fitting preprocessor (on demo data)...")
    preproc = fit_preprocessor(preproc, X)

    out_path = os.path.abspath("artifacts/preprocessor.joblib")
    print(f"Saving preprocessor to {out_path}")
    save_preprocessor(preproc, out_path)

    print("Done. Next: training module will pick up preprocessor.joblib and model training will proceed.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "regression"], default="classification")
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    demo_with_local_dataset(task=args.task)
