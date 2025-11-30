import pytest
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import mlflow
from mlflow import MlflowClient
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error

# =====================================================================
# 1. Load MLflow Tracking URI from .env
# =====================================================================
load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Set MLflow tracking server
mlflow.set_tracking_uri(TRACKING_URI)

# MLflow client
client = MlflowClient()

# =====================================================================
# 2. Load Latest Model from MLflow Registry
# =====================================================================
MODEL_NAME = "Swiggy-Ensemble-Model"   # your model name

def load_latest_model():
    """
    Fetch latest version of the registered model and load it.
    Returns the model bundle: {preprocessor, catboost, lgbm, weights}
    """
    latest = client.get_latest_versions(MODEL_NAME, stages=None)[0].version
    model_uri = f"models:/{MODEL_NAME}/{latest}"
    model_bundle = mlflow.sklearn.load_model(model_uri)
    return model_bundle, latest


# =====================================================================
# 3. Path to cleaned test data
# =====================================================================
ROOT = Path(__file__).parent.parent
TEST_DATA_PATH = ROOT / "data" / "interim" / "test.csv"
# Make sure this file exists — created during DVC pipeline


# =====================================================================
# 4. TEST : Model Performance using MAE Threshold ≤ 4
# =====================================================================
@pytest.mark.parametrize(
    "threshold_mae",
    [4]    # pass only if MAE <= 4 mins
)
def test_model_performance(threshold_mae):

    # Load test data
    df = pd.read_csv(TEST_DATA_PATH)

    # Remove missing rows
    df = df.dropna()

    # Separate features & target
    y = df["time_taken"]
    X = df.drop(columns=["time_taken"])

    # Load latest model bundle
    model_bundle, version = load_latest_model()

    preprocessor = model_bundle["preprocessor"]
    cat = model_bundle["catboost"]
    lgb = model_bundle["lightgbm"]
    w_cat = model_bundle["weights"]["cat"]
    w_lgb = model_bundle["weights"]["lgbm"]

    # Run preprocessing
    X_transformed = preprocessor.transform(X)

    # Predict using ensemble
    pred_cat = cat.predict(X_transformed)
    pred_lgb = lgb.predict(X_transformed)

    y_pred = (w_cat * pred_cat) + (w_lgb * pred_lgb)

    # Compute MAE
    mae = mean_absolute_error(y, y_pred)

    print("\n-------------------------------------------------------")
    print(f"Model Version Tested  : {version}")
    print(f"Mean Absolute Error   : {mae:.4f}")
    print(f"Allowed Threshold     : {threshold_mae}")
    print("-------------------------------------------------------")

    # PASS only if MAE <= threshold
    assert mae <= threshold_mae, f"FAILED ❌ MAE={mae:.4f} > threshold={threshold_mae}"

    print(f"PASSED ✅ Model MAE={mae:.4f} ≤ {threshold_mae}")


