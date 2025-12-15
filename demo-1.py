from .trainer import Trainer
from .preprocessing import build_preprocessor
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Build preprocessor
preprocessor = build_preprocessor(X)

# Initialize trainer
trainer = Trainer(
    task='classification',
    preprocessor=preprocessor,
    random_state=42,
    artifact_dir='./my_artifacts'
)

# Run training
best_model, metrics = trainer.run(
    X, y,
    candidates=['random_forest', 'xgboost', 'lightgbm', 'logistic_regression'],
    use_optuna=True,
    optuna_trials=50,      # More trials = better tuning (but slower)
    optuna_cv_splits=3,    # CV during Optuna (faster with 3)
    test_size=0.2,
    cv_splits=5            # CV for final evaluation
)

print(f"Best model: {best_model}")
print(f"Test metrics: {metrics}")

# Access detailed results
print("\nAll candidates:")
for name, info in trainer.candidate_summary.items():
    print(f"  {name}: CV={info['cv_score']:.4f}, params={info['best_params']}")
