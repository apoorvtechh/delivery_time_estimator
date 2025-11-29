import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn import set_config

# ================================================================
# LOGGER SETUP
# ================================================================
logger = logging.getLogger("preprocess_and_clean")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)

# ================================================================
# PREPROCESSOR CONFIG
# ================================================================
set_config(transform_output="pandas")

traffic_order = ["low", "medium", "high", "jam"]
distance_type_order = ["short", "medium", "long", "very_long"]

num_cols = ["age", "ratings", "pickup_time_minutes", "distance"]

nominal_cat_cols = [
    "weather", "type_of_order", "type_of_vehicle",
    "festival", "city_type", "is_weekend", "order_time_of_day"
]

ordinal_cat_cols = ["traffic", "distance_type"]

TARGET = "time_taken"

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", MinMaxScaler(), num_cols),
        ("nominal_encode",
         OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
         nominal_cat_cols),
        ("ordinal_encode",
         OrdinalEncoder(categories=[traffic_order, distance_type_order]),
         ordinal_cat_cols)
    ],
    remainder="passthrough",
    n_jobs=-1,
    verbose_feature_names_out=False
)

preprocessor.set_output(transform="pandas")


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    root = Path(__file__).parent.parent.parent

    train_path = root / "data" / "interim" / "train.csv"
    test_path = root / "data" / "interim" / "test.csv"

    out_dir = root / "data" / "processed"
    out_dir.mkdir(exist_ok=True, parents=True)

    train_out = out_dir / "train_trans.csv"
    test_out = out_dir / "test_trans.csv"

    preproc_path = root / "models" / "preprocessor.joblib"

    # ============================================================
    # 1️⃣ LOAD RAW TRAIN & TEST
    # ============================================================
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    logger.info(f"Loaded TRAIN → {train.shape}")
    logger.info(f"Loaded TEST  → {test.shape}")

    # ============================================================
    # 2️⃣ CONVERT STRING 'NaN' TO REAL np.nan
    # ============================================================
    nan_values = ["NaN", "NaN ", "nan", "NA", "na", "NAN"]

    train = train.replace(nan_values, np.nan)
    test = test.replace(nan_values, np.nan)

    # ============================================================
    # 3️⃣ DROP ALL ROWS WITH ANY NaN
    # ============================================================
    before_train = train.shape[0]
    before_test = test.shape[0]

    train = train.dropna()
    test = test.dropna()

    logger.info(f"Dropped TRAIN rows → {before_train - train.shape[0]}")
    logger.info(f"Dropped TEST  rows → {before_test - test.shape[0]}")

    logger.info(f"TRAIN after dropna() → {train.shape}")
    logger.info(f"TEST  after dropna() → {test.shape}")

    # ============================================================
    # 4️⃣ SPLIT INTO X AND y
    # ============================================================
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]

    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    logger.info(f"Final TRAIN (clean) → {X_train.shape}")
    logger.info(f"Final TEST  (clean) → {X_test.shape}")

    # ============================================================
    # 5️⃣ FIT PREPROCESSOR ON CLEANED TRAINING DATA
    # ============================================================
    preprocessor.fit(X_train)
    logger.info("Preprocessor fitted on cleaned train dataset.")

    # ============================================================
    # 6️⃣ TRANSFORM TRAIN & TEST
    # ============================================================
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    train_final = X_train_t.join(y_train)
    test_final = X_test_t.join(y_test)

    logger.info(f"TRAIN transformed → {train_final.shape}")
    logger.info(f"TEST  transformed → {test_final.shape}")

    train_final.to_csv(train_out, index=False)
    test_final.to_csv(test_out, index=False)

    joblib.dump(preprocessor, preproc_path)
    logger.info(f"Saved preprocessor → {preproc_path}")

    logger.info("✅ Finished: NaN cleaning + preprocessing applied.")
