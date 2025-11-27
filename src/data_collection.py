import requests
import pandas as pd
from typing import Optional
from .config import DEFAULT_VS_CURRENCY, DEFAULT_DAYS

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"


def fetch_market_chart(
    coin_id: str,
    vs_currency: str = DEFAULT_VS_CURRENCY,
    days: int = DEFAULT_DAYS
) -> pd.DataFrame:
    """
    Fetch historical market data (price & volume) from CoinGecko.

    :param coin_id: e.g. 'bitcoin', 'ethereum'
    :param vs_currency: e.g. 'usd'
    :param days: number of days back from now (1, 7, 30, 365, 'max')
    :return: DataFrame with index=timestamp, columns: price, volume
    """
    url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])

    df = prices.merge(volumes, on="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    return df


def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=True)


def load_from_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
    return df
