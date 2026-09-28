"""Feature engineering shared by train.py and predict.py.

EDA (see eda.ipynb) showed that target = feature_6² + feature_7,
all other features are noise.
"""

import pandas as pd

FEATURE_NAMES = ["6_squared", "7"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model inputs from raw data: feature 6 squared and feature 7 as is."""
    return pd.DataFrame({"6_squared": df["6"] ** 2, "7": df["7"]}, index=df.index)
