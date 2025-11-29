import numpy as np
import pandas as pd
from pathlib import Path
import logging

# ================================================================
# LOGGER INITIALIZATION
# ================================================================
logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# Columns to drop
columns_to_drop = [
    "rider_id",
    "restaurant_latitude",
    "restaurant_longitude",
    "delivery_latitude",
    "delivery_longitude",
    "order_date",
    "order_time_hour",
    "order_day",
    "city_name",
    "order_day_of_week",
    "order_month",
]

# ================================================================
# TIME OF DAY
# ================================================================
def time_of_day(series: pd.Series):
    return pd.cut(
        series,
        bins=[0, 6, 12, 17, 20, 24],
        labels=["after_midnight", "morning", "afternoon", "evening", "night"],
        right=True,
    )

# ================================================================
# LOAD RAW DATA
# ================================================================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded RAW CSV → shape: {df.shape}")
    return df

# ================================================================
# RENAME COLUMNS (EXPERIMENT EXACT)
# ================================================================
def change_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.rename(str.lower, axis=1)
        .rename({
            "delivery_person_id": "rider_id",
            "delivery_person_age": "age",
            "delivery_person_ratings": "ratings",
            "delivery_location_latitude": "delivery_latitude",
            "delivery_location_longitude": "delivery_longitude",
            "time_orderd": "order_time",
            "time_order_picked": "order_picked_time",
            "weatherconditions": "weather",
            "road_traffic_density": "traffic",
            "city": "city_type",
            "time_taken(min)": "time_taken",
        }, axis=1)
    )

# ================================================================
# MAIN CLEANING LOGIC
# ================================================================
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    minor_index = df.loc[df["age"].astype(float) < 18].index
    six_star_index = df.loc[df["ratings"] == "6"].index

    logger.info(f"Dropping minors: {len(minor_index)} rows")
    logger.info(f"Dropping invalid ratings (6 stars): {len(six_star_index)} rows")

    return (
        df
        .drop(columns="id")
        .drop(index=minor_index)
        .drop(index=six_star_index)
        .replace("NaN ", np.nan)  # FIXED HERE
        .assign(
            city_name=lambda x: x["rider_id"].str.split("RES").str.get(0),

            age=lambda x: x["age"].astype(float),
            ratings=lambda x: x["ratings"].astype(float),

            restaurant_latitude=lambda x: x["restaurant_latitude"].abs(),
            restaurant_longitude=lambda x: x["restaurant_longitude"].abs(),
            delivery_latitude=lambda x: x["delivery_latitude"].abs(),
            delivery_longitude=lambda x: x["delivery_longitude"].abs(),

            order_date=lambda x: pd.to_datetime(x["order_date"], dayfirst=True),
            order_day=lambda x: x["order_date"].dt.day,
            order_month=lambda x: x["order_date"].dt.month,
            order_day_of_week=lambda x: x["order_date"].dt.day_name().str.lower(),
            is_weekend=lambda x: x["order_date"].dt.day_name().isin(["Saturday", "Sunday"]).astype(int),

            order_time=lambda x: pd.to_datetime(x["order_time"], format="mixed"),
            order_picked_time=lambda x: pd.to_datetime(x["order_picked_time"], format="mixed"),

            pickup_time_minutes=lambda x: (x["order_picked_time"] - x["order_time"]).dt.seconds / 60,

            order_time_hour=lambda x: x["order_time"].dt.hour,
            order_time_of_day=lambda x: x["order_time_hour"].pipe(time_of_day),

            weather=lambda x: (
                x["weather"]
                .str.replace("conditions ", "")
                .str.lower()
                .replace("nan", np.nan)   # FIXED HERE
            ),
            traffic=lambda x: x["traffic"].str.rstrip().str.lower(),
            type_of_order=lambda x: x["type_of_order"].str.rstrip().str.lower(),
            type_of_vehicle=lambda x: x["type_of_vehicle"].str.rstrip().str.lower(),
            festival=lambda x: x["festival"].str.rstrip().str.lower(),
            city_type=lambda x: x["city_type"].str.rstrip().str.lower(),

            multiple_deliveries=lambda x: x["multiple_deliveries"].astype(float),

            time_taken=lambda x: (
                x["time_taken"].str.replace("(min) ", "").astype(int)
            ),
        )
        .drop(columns=["order_time", "order_picked_time"])
    )

# ================================================================
# HAVERSINE DISTANCE
# ================================================================
def calculate_haversine_distance(df):
    logger.info("Calculating Haversine distance...")

    lat1 = np.radians(df["restaurant_latitude"])
    lon1 = np.radians(df["restaurant_longitude"])
    lat2 = np.radians(df["delivery_latitude"])
    lon2 = np.radians(df["delivery_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return df.assign(distance=distance)

# ================================================================
# DISTANCE TYPE
# ================================================================
def add_distance_type(df):
    df["distance_type"] = pd.cut(
        df["distance"],
        bins=[0, 5, 10, 15, 25],
        right=False,
        labels=["short", "medium", "long", "very_long"],
    )
    return df

# ================================================================
# FULL PIPELINE
# ================================================================
def perform_data_cleaning(df, save_path):

    logger.info("Running FULL DATA CLEANING PIPELINE...")

    cleaned = (
        df
        .pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(calculate_haversine_distance)
        .pipe(add_distance_type)
        .drop(columns=columns_to_drop)
    )

    cleaned.to_csv(save_path, index=False)

    logger.info(f"Final cleaned shape → {cleaned.shape}")
    logger.info(f"Saved cleaned file → {save_path}")

    return cleaned

# ================================================================
# EXECUTION ENTRY POINT
# ================================================================
if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent
    raw_path = root / "data" / "raw" / "swiggy.csv"

    save_dir = root / "data" / "cleaned"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "swiggy_cleaned.csv"

    df = load_data(raw_path)
    cleaned_df = perform_data_cleaning(df, save_path)

    print("\n==============================")
    print("✅ DATA CLEANING COMPLETE")
    print(f"➡️ Raw Data Shape     : {df.shape}")
    print(f"➡️ Cleaned Data Shape : {cleaned_df.shape}")
    print(f"➡️ Saved to           : {save_path}")
    print("==============================\n")
