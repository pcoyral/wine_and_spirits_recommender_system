"""
Loads and lightly validates the Wine Enthusiast dataset for SensoryRec.
"""

from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_COLUMNS = [
    "country", "description", "designation", "points", "price", "province",
    "region_1", "region_2", "taster_name", "taster_twitter_handle", "title",
    "variety", "winery",
]


def load_catalog(filename: str = "winemag-data.csv") -> pd.DataFrame:
    """
    Load the raw wine catalog CSV from data/raw/.

    TODO: once the full ~130k-review dataset is finalized, confirm exact
    filename/format here and update accordingly (matches the schema
    already validated against 03_Sample_Data.csv).
    """
    path = RAW_DATA_PATH / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {path}. Download the Wine Enthusiast Reviews "
            "dataset from Kaggle and place it there (see README.md)."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df
