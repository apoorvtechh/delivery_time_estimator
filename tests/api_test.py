import pandas as pd
import requests
from pathlib import Path
import pytest

# -----------------------------------------------------------
# Path to raw dataset
# -----------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "swiggy.csv"

# -----------------------------------------------------------
# Load one random row from CSV
# -----------------------------------------------------------
df = pd.read_csv(DATA_PATH).dropna()
sample_row = df.sample(1).iloc[0]

# Extract actual target value "(min) 23" → 23
actual_time = int(sample_row["Time_taken(min)"].replace("(min) ", ""))
print(f"\n🎯 Actual Delivery Time (from dataset): {actual_time} minutes\n")

# -----------------------------------------------------------
# Build JSON payload EXACTLY matching API input schema
# -----------------------------------------------------------
payload = {
    "ID": sample_row["ID"],
    "Delivery_person_ID": sample_row["Delivery_person_ID"],
    "Delivery_person_Age": str(sample_row["Delivery_person_Age"]),
    "Delivery_person_Ratings": str(sample_row["Delivery_person_Ratings"]),
    "Restaurant_latitude": float(sample_row["Restaurant_latitude"]),
    "Restaurant_longitude": float(sample_row["Restaurant_longitude"]),
    "Delivery_location_latitude": float(sample_row["Delivery_location_latitude"]),
    "Delivery_location_longitude": float(sample_row["Delivery_location_longitude"]),
    "Order_Date": sample_row["Order_Date"],
    "Time_Orderd": sample_row["Time_Orderd"],
    "Time_Order_picked": sample_row["Time_Order_picked"],
    "Weatherconditions": sample_row["Weatherconditions"],
    "Road_traffic_density": sample_row["Road_traffic_density"],
    "Vehicle_condition": int(sample_row["Vehicle_condition"]),
    "Type_of_order": sample_row["Type_of_order"],
    "Type_of_vehicle": sample_row["Type_of_vehicle"],
    "multiple_deliveries": str(sample_row["multiple_deliveries"]),
    "Festival": sample_row["Festival"],
    "City": sample_row["City"]
}

# -----------------------------------------------------------
# Pytest test case for API
# -----------------------------------------------------------
@pytest.mark.parametrize(
    "url, data",
    [("http://127.0.0.1:8000/predict", payload)]
)
def test_predict_endpoint(url, data):

    print("\n🚀 Sending request to API...")
    response = requests.post(url=url, json=data)

    # Check for correct status
    assert response.status_code == 200, f"❌ API returned {response.status_code}"

    result = response.json()
    print("✅ API Response:", result)

    # Must contain prediction value
    assert "predicted_time_minutes" in result, "❌ Missing prediction output"

    predicted = float(result["predicted_time_minutes"])

    print(f"\n🎯 Actual time   : {actual_time} minutes")
    print(f"🤖 Predicted time: {predicted:.2f} minutes\n")

    # Optional sanity check: Prediction should be positive
    assert predicted > 0, "❌ Prediction must be greater than 0"
