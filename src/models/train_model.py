import pandas as pd
import yaml
import joblib
import logging
from pathlib import Path
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ================================================================
# LOGGER SETUP
# ================================================================
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

TARGET = "time_taken"

# ================================================================
# HELPERS
# ================================================================
def load_data(path: Path):
    df = pd.read_csv(path)
    logger.info(f"Loaded training data → shape {df.shape}")
    return df

def read_params(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def make_X_y(df: pd.DataFrame, target: str):
    return df.drop(columns=[target]), df[target]

def save_model(model, directory: Path, filename: str):
    directory.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, directory / filename)
    logger.info(f"Saved model → {directory / filename}")

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent
    train_path = root / "data" / "processed" / "train_trans.csv"
    params_path = root / "params.yaml"
    model_dir = root / "models"

    df = load_data(train_path)
    X_train, y_train = make_X_y(df, TARGET)

    params = read_params(params_path)["Train"]

    # Load CatBoost + LGBM params
    cat_params = params["CatBoost"]
    lgbm_params = params["LightGBM"]

    # Build models
    cat = CatBoostRegressor(**cat_params)
    lgb = LGBMRegressor(**lgbm_params)

    logger.info("Training CatBoost…")
    cat.fit(X_train, y_train, verbose=False)
    logger.info("Training LightGBM…")
    lgb.fit(X_train, y_train)

    # Save models
    save_model(cat, model_dir, "catboost_model.joblib")
    save_model(lgb, model_dir, "lgbm_model.joblib")

    logger.info("Training completed successfully!")
