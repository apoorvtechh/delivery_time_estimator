"""
Test Script: test_model_loading.py
Purpose:
    - Connect to MLflow Tracking Server
    - Verify the registered model exists
    - Load latest model version from MLflow Registry
"""

import pytest
import mlflow
from mlflow import MlflowClient
import os
from dotenv import load_dotenv

# ============================================================
# Load MLflow Tracking URI from .env
# ============================================================
load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(TRACKING_URI)

# Your registered model name (hardcoded because no JSON file exists)
MODEL_NAME = "Swiggy-Ensemble-Model"


# ============================================================
# TEST: Load latest model version from registry
# ============================================================
@pytest.mark.parametrize("model_name", [MODEL_NAME])
def test_load_latest_model(model_name):
    """
    Ensures the latest model version exists and loads successfully.
    """

    client = MlflowClient()

    # Fetch all latest versions
    versions = client.get_latest_versions(name=model_name)

    assert versions, f"❌ No registered versions found for model → {model_name}"

    latest_version = versions[0].version
    print(f"\n➡ Latest version detected: {latest_version}")

    model_uri = f"models:/{model_name}/{latest_version}"
    print(f"📦 Loading model from: {model_uri}")

    # Attempt to load model
    model = mlflow.sklearn.load_model(model_uri)

    assert model is not None, "❌ Failed to load model from MLflow registry"

    print(f"✅ SUCCESS — Model loaded: {model_name} (version: {latest_version})")
