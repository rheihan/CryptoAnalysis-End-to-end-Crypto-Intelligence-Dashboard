import numpy as np
import pandas as pd
from .config import TRADING_DAYS_PER_YEAR


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Hitung Sharpe Ratio sederhana.
    'returns' di sini return harian (hasil pct_change).
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = excess_returns.mean()
    std_excess = excess_returns.std()

    if std_excess == 0 or np.isnan(std_excess):
        return np.nan

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return sharpe


def compute_max_drawdown(prices: pd.Series) -> float:
    """
    Hitung Max Drawdown dari series harga.
    """
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    max_dd = drawdown.min()
    return max_dd


def compute_basic_risk_metrics(df: pd.DataFrame) -> dict:
    """
    Hitung metrik dasar: Sharpe & Max Drawdown.
    """
    metrics = {}
    if "return" in df.columns:
        metrics["sharpe_ratio"] = compute_sharpe_ratio(df["return"].dropna())
    else:
        metrics["sharpe_ratio"] = np.nan

    metrics["max_drawdown"] = compute_max_drawdown(df["price"].dropna())
    return metrics
