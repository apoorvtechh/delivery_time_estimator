import os
import mlflow
import joblib
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv  # NEW

# ================================================================
# LOGGER
# ================================================================
logger = logging.getLogger("model_registry")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent

    # -----------------------------
    # 1. Load .env and tracking URI
    # -----------------------------
    env_path = root / ".env"
    load_dotenv(env_path)  # this reads MLFLOW_TRACKING_URI

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        raise ValueError("MLFLOW_TRACKING_URI not found in .env")

    mlflow.set_tracking_uri(mlflow_uri)
    logger.info(f"🚀 Using MLflow Tracking URI: {mlflow_uri}")

    # -----------------------------
    # 2. Load params + models
    # -----------------------------
    params = yaml.safe_load(open(root / "params.yaml"))
    weights = params["Train"]["weights"]

    cat_path = root / "models" / "catboost_model.joblib"
    lgb_path = root / "models" / "lgbm_model.joblib"

    logger.info("📥 Loading saved models...")
    cat_model = joblib.load(cat_path)
    lgb_model = joblib.load(lgb_path)

    # -----------------------------
    # 3. Set experiment
    # -----------------------------
    experiment_name = "Model Registration FOR TIME ESTIMATION"
    mlflow.set_experiment(experiment_name)

    # -----------------------------
    # 4. Log models inside a run
    # -----------------------------
    with mlflow.start_run(run_name="Register Saved Models") as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "weight_cat": weights["cat"],
            "weight_lgbm": weights["lgbm"],
        })

        # Log raw joblib artifacts too (optional)
        mlflow.log_artifact(str(cat_path))
        mlflow.log_artifact(str(lgb_path))

        # Log models as MLflow model artifacts
        mlflow.sklearn.log_model(
            sk_model=cat_model,
            artifact_path="catboost_model"
        )#added
        mlflow.sklearn.log_model(
            sk_model=lgb_model,
            artifact_path="lgbm_model"
        )

    # -----------------------------
    # 5. Register both models
    # -----------------------------
    cat_uri = f"runs:/{run_id}/catboost_model"
    lgb_uri = f"runs:/{run_id}/lgbm_model"

    logger.info("🔄 Registering CatBoost model...")
    cat_version = mlflow.register_model(
        model_uri=cat_uri,
        name="Swiggy-CatBoost-Model"
    )

    logger.info("🔄 Registering LightGBM model...")
    lgb_version = mlflow.register_model(
        model_uri=lgb_uri,
        name="Swiggy-LightGBM-Model"
    )

    logger.info(f"🏆 CatBoost Registered as version: {cat_version.version}")
    logger.info(f"🏆 LightGBM Registered as version: {lgb_version.version}")
    logger.info("✅ Model registration completed successfully.")
