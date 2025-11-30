from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import mlflow
from mlflow import MlflowClient
import joblib
import os

from dotenv import load_dotenv


# ============================================================
# IMPORT YOUR CLEANING FUNCTION
# ============================================================
from scripts.data_clean_utils import perform_data_cleaning

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(title="Swiggy ETA Prediction API", version="1.0")

# ============================================================
# MLflow Tracking Setup
# ============================================================
load_dotenv()
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

MODEL_NAME = "Swiggy-Ensemble-Model"

print("🔍 Fetching latest model version...")
latest_ver = client.get_latest_versions(MODEL_NAME, stages=None)[0].version
print(f"🎯 Latest Model Version Found → {latest_ver}")

model_uri = f"models:/{MODEL_NAME}/{latest_ver}"

print("📦 Loading model bundle (preprocessor + models + weights)...")
model_bundle = mlflow.sklearn.load_model(model_uri)

preprocessor = model_bundle["preprocessor"]
cat_model = model_bundle["catboost"]
lgb_model = model_bundle["lightgbm"]
w_cat = model_bundle["weights"]["cat"]
w_lgb = model_bundle["weights"]["lgbm"]

print("✅ Model bundle loaded successfully!")

# ============================================================
# REQUEST BODY MODEL
# ============================================================
class InputData(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float
    Festival: str
    City: str


# ============================================================
# HOME ENDPOINT
# ============================================================
@app.get("/")
def home():
    return {
        "message": "Swiggy ETA Prediction API is running 🚀",
        "latest_model_version": latest_ver
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/predict")
def predict(data: InputData):

    # Convert input → DataFrame
    raw_df = pd.DataFrame([data.dict()])

    # Clean using DVC equivalent cleaning logic
    cleaned_df = perform_data_cleaning(raw_df)

    if cleaned_df.empty:
        return {"error": "Input cleaning removed the row (invalid input values)."}

    # Preprocess features
    X = preprocessor.transform(cleaned_df)

    # Predict from both models
    pred_cat = cat_model.predict(X)
    pred_lgb = lgb_model.predict(X)

    # Weighted average output
    final_pred = float((w_cat * pred_cat) + (w_lgb * pred_lgb))

    return {
        "predicted_time_minutes": final_pred,
        "model_version_used": latest_ver,
        "weights": {"catboost": w_cat, "lightgbm": w_lgb}
    }


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
