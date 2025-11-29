import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from pathlib import Path

# ==========================================================
# CONSTANT
# ==========================================================
TARGET = "time_taken"

# ==========================================================
# LOGGER INITIALIZATION
# ==========================================================
logger = logging.getLogger("data_preparation")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ==========================================================
# LOAD DATA
# ==========================================================
def load_data(data_path: Path) -> pd.DataFrame:
    """Load cleaned dataset."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded cleaned dataset → {df.shape}")
    except FileNotFoundError:
        logger.error(f"File not found at: {data_path}")
        raise
    return df

# ==========================================================
# SPLIT DATA
# ==========================================================
def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )
    return train_data, test_data

# ==========================================================
# READ PARAMETERS
# ==========================================================
def read_params(file_path: Path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

# ==========================================================
# SAVE DATA
# ==========================================================
def save_data(data: pd.DataFrame, save_path: Path):
    data.to_csv(save_path, index=False)

# ==========================================================
# MAIN EXECUTION (DVC STAGE)
# ==========================================================
if __name__ == "__main__":

    root_path = Path(__file__).parent.parent.parent

    data_path = root_path / "data" / "cleaned" / "swiggy_cleaned.csv"

    save_data_dir = root_path / "data" / "interim"
    save_data_dir.mkdir(exist_ok=True, parents=True)

    train_path = save_data_dir / "train.csv"
    test_path = save_data_dir / "test.csv"

    df = load_data(data_path)

    params = read_params(root_path / "params.yaml")["Data_Preparation"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    train_data, test_data = split_data(df, test_size, random_state)

    logger.info(f"Train shape: {train_data.shape}")
    logger.info(f"Test shape: {test_data.shape}")
    logger.info(f"Train NA count: {train_data.isna().sum().sum()}")
    logger.info(f"Test NA count: {test_data.isna().sum().sum()}")

    save_data(train_data, train_path)
    save_data(test_data, test_path)

    logger.info(f"Train saved → {train_path}")
    logger.info(f"Test saved → {test_path}")
