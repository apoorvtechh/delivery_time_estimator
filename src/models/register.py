import mlflow
import mlflow.sklearn
import os
import joblib
import logging
import yaml
from pathlib import Path
from dotenv import load_dotenv

# ============================================================
# LOGGER
# ============================================================
logger = logging.getLogger("model_registry")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

# ============================================================
# LOAD ENV (.env)
# ============================================================
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

if tracking_uri is None:
    raise ValueError("❌ MLFLOW_TRACKING_URI not found in .env")

logger.info(f"🚀 Using MLflow Tracking URI: {tracking_uri}")
mlflow.set_tracking_uri(tracking_uri)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent
    model_dir = root / "models"

    # Paths
    cat_path = model_dir / "catboost_model.joblib"
    lgb_path = model_dir / "lgbm_model.joblib"
    preprocess_path = model_dir / "preprocessor.joblib"
    params_path = root / "params.yaml"

    # Load params (weights included)
    params = yaml.safe_load(open(params_path))
    w_cat = params["Train"]["weights"]["cat"]
    w_lgb = params["Train"]["weights"]["lgbm"]

    # Experiment
    experiment_name = "Model Registration FOR TIME ESTIMATION"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:

        run_id = run.info.run_id
        logger.info(f"📌 Started MLflow Run: {run_id}")

        # ---------------------------------------------
        # Load artifacts
        # ---------------------------------------------
        cat_model = joblib.load(cat_path)
        lgbm_model = joblib.load(lgb_path)
        preprocessor = joblib.load(preprocess_path)

        logger.info("Loaded CatBoost, LightGBM & Preprocessor successfully.")

        # ---------------------------------------------
        # Log raw artifacts
        # ---------------------------------------------
        mlflow.log_artifact(cat_path, artifact_path="models")
        mlflow.log_artifact(lgb_path, artifact_path="models")
        mlflow.log_artifact(preprocess_path, artifact_path="preprocessor")

        # ---------------------------------------------
        # Log ensemble weights
        # ---------------------------------------------
        mlflow.log_param("weight_catboost", w_cat)
        mlflow.log_param("weight_lightgbm", w_lgb)

        # ---------------------------------------------
        # Save combined ensemble pipeline
        # ---------------------------------------------
        combined_package = {
            "preprocessor": preprocessor,
            "catboost": cat_model,
            "lightgbm": lgbm_model,
            "weights": {
                "cat": w_cat,
                "lgbm": w_lgb,
            }
        }

        logger.info("Packaging preprocessor + models + weights...")

        mlflow.sklearn.log_model(
            sk_model=combined_package,
            artifact_path="full_pipeline",
            registered_model_name="Swiggy-Ensemble-Model"
        )

        logger.info("🎉 Swiggy-Ensemble-Model registered successfully!")

    logger.info("✅ Model registration completed successfully.")
