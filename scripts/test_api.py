import pandas as pd
import requests
import random
import json
import pytest
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================
API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = Path(r"D:\datascience\campusx\PROJECTS\delivery_time_estimator\data\raw\swiggy.csv")

# ======================================================
# Load CSV once for all tests
# ======================================================
df = pd.read_csv(CSV_PATH)


def build_payload(sample):
    """
    Convert one dataframe row into API JSON payload.
    Ensures types match FastAPI InputData schema.
    """
    return {
        "ID": sample["ID"],
        "Delivery_person_ID": sample["Delivery_person_ID"],
        "Delivery_person_Age": str(sample["Delivery_person_Age"]),
        "Delivery_person_Ratings": str(sample["Delivery_person_Ratings"]),
        "Restaurant_latitude": float(sample["Restaurant_latitude"]),
        "Restaurant_longitude": float(sample["Restaurant_longitude"]),
        "Delivery_location_latitude": float(sample["Delivery_location_latitude"]),
        "Delivery_location_longitude": float(sample["Delivery_location_longitude"]),
        "Order_Date": sample["Order_Date"],
        "Time_Orderd": sample["Time_Orderd"],
        "Time_Order_picked": sample["Time_Order_picked"],
        "Weatherconditions": sample["Weatherconditions"],
        "Road_traffic_density": sample["Road_traffic_density"],
        "Vehicle_condition": int(sample["Vehicle_condition"]),
        "Type_of_order": sample["Type_of_order"],
        "Type_of_vehicle": sample["Type_of_vehicle"],
        "multiple_deliveries": str(sample["multiple_deliveries"]),
        "Festival": sample["Festival"],
        "City": sample["City"],
    }


def extract_true_value(sample):
    """Extract '(min) 23' → 23 from Time_taken(min) column."""
    raw = sample["Time_taken(min)"]
    cleaned = str(raw).replace("(min)", "").replace("(min) ", "").strip()
    try:
        return int(cleaned)
    except:
        return None


# ======================================================
# PYTEST TEST: API working + prediction returned
# ======================================================
@pytest.mark.parametrize("api_url", [API_URL])
def test_predict_endpoint(api_url):

    # Pick a random row
    row_idx = random.randint(0, len(df) - 1)
    sample = df.iloc[row_idx]

    print(f"\n🎯 Selected Row Index: {row_idx}")
    print(sample)

    # Build input payload
    payload = build_payload(sample)

    # Extract true value
    actual_value = extract_true_value(sample)
    print(f"\n🎯 Actual Time Taken (True Value): {actual_value}\n")

    # Send request
    print("\n🚀 Sending request to API...\n")
    response = requests.post(api_url, json=payload)

    # Validate response status
    assert response.status_code == 200, f"❌ API returned {response.status_code}: {response.text}"

    result = response.json()
    print("✅ API Response:")
    print(json.dumps(result, indent=4))

    # Validate structure
    assert "predicted_time_minutes" in result, "❌ API did not return prediction!"

    predicted_value = result["predicted_time_minutes"]

    print(f"\n📌 Predicted Time: {predicted_value}")
    print(f"🎯 True Time      : {actual_value}\n")

    # Optional: Ensure prediction is positive
    assert predicted_value > 0, "❌ Prediction must be positive"
