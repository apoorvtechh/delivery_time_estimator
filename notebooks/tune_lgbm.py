import optuna
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

TARGET = "time_taken"

def load_data(path):
    return pd.read_csv(path)

def make_X_y(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def objective(trial, X, y):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42
    }

    model = LGBMRegressor(**params)

    # CV score (Optuna tries to MINIMIZE)
    mae = -cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    ).mean()

    return mae


if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent
    train_path = root / "data" / "processed" / "train_trans.csv"

    df = load_data(train_path)
    X, y = make_X_y(df)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, X, y), n_trials=50, n_jobs=1)

    print("\n🎯 BEST LIGHTGBM PARAMS FOUND")
    print(study.best_params)
    print(f"Best MAE → {study.best_value:.5f}")
