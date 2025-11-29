import pandas as pd
import requests
import random
import json

# ===============================
# CONFIG
# ===============================
API_URL = "http://127.0.0.1:8000/predict"
CSV_PATH = r"D:\datascience\campusx\PROJECTS\delivery_time_estimator\data\raw\swiggy.csv"

# ===============================
# LOAD CSV
# ===============================
df = pd.read_csv(CSV_PATH)

# Pick random row
row_index = random.randint(0, len(df) - 1)
sample = df.iloc[row_index]

print(f"\n🎯 Selected Row Index: {row_index}\n")
print(sample)

# ===============================
# PREP INPUT FOR API
# (matches app.py InputData model)
# ===============================
payload = {
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

# ===============================
# EXTRACT TRUE TARGET VALUE
# ===============================
target_raw = sample["Time_taken(min)"]

# Convert "(min) 23" → 23
try:
    actual_value = int(str(target_raw).replace("(min)", "").replace("(min) ", "").strip())
except:
    actual_value = None

print(f"\n🎯 Actual Time Taken (True Value): {actual_value}\n")

# ===============================
# SEND REQUEST
# ===============================
print("🚀 Sending request to API...\n")

response = requests.post(API_URL, json=payload)

if response.status_code == 200:
    result = response.json()
    print("✅ API Response:")
    print(json.dumps(result, indent=4))

    predicted = result["predicted_time_minutes"]

    print(f"\n📌 Predicted Time: {predicted}")
    print(f"🎯 True Time      : {actual_value}")

else:
    print(f"❌ API Error: {response.text}")
