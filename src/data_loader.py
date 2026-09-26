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

# Moët Hennessy (LVMH Wines & Spirits) wine estate brand names.
# TODO: cross-check against Moët Hennessy's official maison list before
# finalizing for the methodology write-up — the portfolio can change
# (acquisitions/divestitures), and this list was assembled from general
# knowledge, not verified against a current official source.
LVMH_WINERIES = [
    "Moët & Chandon", "Moet & Chandon", "Dom Pérignon", "Dom Perignon",
    "Veuve Clicquot", "Krug", "Ruinart", "Mercier", "Chandon",
    "Cheval Blanc", "Yquem", "Newton", "Cloudy Bay", "Cape Mentelle",
    "Terrazas de los Andes", "Numanthia", "Ao Yun", "Colgin",
]


def label_lvmh(df: pd.DataFrame, winery_col: str = "winery") -> pd.DataFrame:
    """
    Add an `is_lvmh` boolean column to the catalog, labeling rows whose
    winery matches a known Moët Hennessy (LVMH) brand.

    This labels a SUBSET within the full catalog rather than filtering it
    down — the full dataset should stay intact for model training and
    evaluation; the label is for a business-relevance case study
    (e.g. "how does SensoryRec perform on the LVMH portfolio specifically").

    Returns a copy of df with the new column added.
    """
    df = df.copy()
    df["is_lvmh"] = df[winery_col].str.contains(
        "|".join(LVMH_WINERIES), case=False, na=False
    )
    return df


def get_lvmh_subset(df: pd.DataFrame, winery_col: str = "winery") -> pd.DataFrame:
    """Return only the rows labeled as LVMH-owned wineries (see label_lvmh)."""
    labeled = label_lvmh(df, winery_col=winery_col)
    return labeled[labeled["is_lvmh"]].drop(columns="is_lvmh")


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
