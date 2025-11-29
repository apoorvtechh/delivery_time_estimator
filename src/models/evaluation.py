import pandas as pd
import logging
import joblib
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt

# ================================================================
# LOGGER
# ================================================================
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

TARGET = "time_taken"

# ================================================================
# FUNCTION TO COMPUTE METRICS
# ================================================================
def compute_metrics(y, pred):
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)
    return mae, rmse, r2

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent

    # Paths
    train_path = root / "data" / "processed" / "train_trans.csv"
    test_path = root / "data" / "processed" / "test_trans.csv"
    params_path = root / "params.yaml"
    cat_path = root / "models" / "catboost_model.joblib"
    lgb_path = root / "models" / "lgbm_model.joblib"

    # Plot directory
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Loaded TRAIN → {train_df.shape}")
    logger.info(f"Loaded TEST  → {test_df.shape}")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]

    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]

    # Load params
    params = yaml.safe_load(open(params_path))
    weights = params["Train"]["weights"]
    w_cat = weights["cat"]
    w_lgb = weights["lgbm"]

    # Load models
    cat = joblib.load(cat_path)
    lgb = joblib.load(lgb_path)

    # Predictions (train)
    pred_train_cat = cat.predict(X_train)
    pred_train_lgb = lgb.predict(X_train)
    pred_train = (w_cat * pred_train_cat) + (w_lgb * pred_train_lgb)

    # Predictions (test)
    pred_test_cat = cat.predict(X_test)
    pred_test_lgb = lgb.predict(X_test)
    pred_test = (w_cat * pred_test_cat) + (w_lgb * pred_test_lgb)

    # Compute metrics
    train_mae, train_rmse, train_r2 = compute_metrics(y_train, pred_train)
    test_mae, test_rmse, test_r2 = compute_metrics(y_test, pred_test)

    # PRINT RESULTS
    print("\n🔥 FINAL WEIGHTED MODEL PERFORMANCE 🔥")

    print("\n📌 TRAIN PERFORMANCE:")
    print(f"MAE  : {train_mae:.4f}")
    print(f"RMSE : {train_rmse:.4f}")
    print(f"R²   : {train_r2:.4f}")

    print("\n📌 TEST PERFORMANCE:")
    print(f"MAE  : {test_mae:.4f}")
    print(f"RMSE : {test_rmse:.4f}")
    print(f"R²   : {test_r2:.4f}\n")

    # =========================================================
    # OVERFITTING CHECK
    # =========================================================
    print("=====================================")
    print("🔍 OVERFITTING CHECK")
    print("=====================================")

    print(f"MAE gap  : {abs(train_mae - test_mae):.4f}")
    print(f"RMSE gap : {abs(train_rmse - test_rmse):.4f}")
    print(f"R² gap   : {abs(train_r2 - test_r2):.4f}")

    # =========================================================
    # DIAGNOSTIC PLOTS (Regression)
    # =========================================================

    # 1. Prediction vs Actual
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, pred_test, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--')
    plt.xlabel("Actual Time")
    plt.ylabel("Predicted Time")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.savefig(plot_dir / "actual_vs_predicted.png")
    plt.close()

    # 2. Residuals
    residuals = y_test - pred_test
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=40)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(plot_dir / "residual_distribution.png")
    plt.close()

    # 3. Residuals vs Fitted
    plt.figure(figsize=(6,4))
    plt.scatter(pred_test, residuals, alpha=0.4)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.grid(True)
    plt.savefig(plot_dir / "residuals_vs_fitted.png")
    plt.close()

    # 4. Error histogram
    abs_error = np.abs(y_test - pred_test)
    plt.figure(figsize=(6,4))
    plt.hist(abs_error, bins=40)
    plt.title("Absolute Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(plot_dir / "error_histogram.png")
    plt.close()

    logger.info("Evaluation completed. Plots saved to /plots/")
