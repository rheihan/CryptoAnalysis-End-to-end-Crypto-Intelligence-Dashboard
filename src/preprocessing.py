import numpy as np
import pandas as pd
from .config import ROLLING_WINDOW_VOL, MA_SHORT, MA_LONG


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah feature dasar: return, log_return, volatility rolling, MA, dan market_regime.
    """
    df = df.copy()
    df["return"] = df["price"].pct_change()
    df["log_return"] = np.log(df["price"]).diff()
    df["volatility"] = df["log_return"].rolling(ROLLING_WINDOW_VOL).std()

    # Moving averages
    df[f"MA_{MA_SHORT}"] = df["price"].rolling(MA_SHORT).mean()
    df[f"MA_{MA_LONG}"] = df["price"].rolling(MA_LONG).mean()

    # Market regime (simple rule)
    def _regime(row):
        if np.isnan(row[f"MA_{MA_SHORT}"]):
            return "Unknown"
        return "Bull" if row["price"] > row[f"MA_{MA_SHORT}"] else "Bear"

    df["market_regime"] = df.apply(_regime, axis=1)
    return df


def drop_na_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop baris yang ada NaN (buat modelling & visual supaya rapi).
    """
    return df.dropna()
