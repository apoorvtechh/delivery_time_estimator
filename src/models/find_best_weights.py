import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error

TARGET = "time_taken"

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def make_X_y(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent

    # Load processed test data
    test_path = root / "data" / "processed" / "test_trans.csv"
    df_test = load_data(test_path)

    X_test, y_test = make_X_y(df_test, TARGET)

    # Load models
    model_dir = root / "models"
    cat = joblib.load(model_dir / "catboost_model.joblib")
    lgb = joblib.load(model_dir / "lgbm_model.joblib")

    # Get predictions
    pred_cat = cat.predict(X_test)
    pred_lgb = lgb.predict(X_test)

    print("\n🔎 Searching for best weights...")
    print("------------------------------------")

    best_w_cat = None
    best_w_lgb = None
    best_mae = float("inf")

    # Try weights from 0.0 to 1.0 (step = 0.05)
    for w_cat in np.arange(0, 1.05, 0.05):
        w_lgb = 1 - w_cat

        blended_pred = (w_cat * pred_cat) + (w_lgb * pred_lgb)
        mae = mean_absolute_error(y_test, blended_pred)

        print(f"w_cat = {w_cat:.2f}, w_lgb = {w_lgb:.2f} → MAE = {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_w_cat = w_cat
            best_w_lgb = w_lgb

    print("\n🎯 BEST WEIGHTS FOUND")
    print("------------------------------------")
    print(f"Best w_cat  = {best_w_cat:.3f}")
    print(f"Best w_lgb  = {best_w_lgb:.3f}")
    print(f"Best MAE    = {best_mae:.4f}")
