import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .config import TRADING_DAYS_PER_YEAR


def prepare_returns_for_portfolio(
    market_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Ambil kolom 'return' dari masing-masing aset dan gabungkan jadi satu dataframe.
    index: timestamp (daily)
    columns: nama aset
    """
    series_list = []
    for name, df in market_data.items():
        s = df["return"].copy()
        s.name = name
        series_list.append(s)

    returns = pd.concat(series_list, axis=1).dropna()
    # optional: resample ke daily (kalau ternyata lebih granular)
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)
    returns = returns.resample("D").last().dropna()
    return returns


def random_portfolios(
    returns: pd.DataFrame,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random portfolios (tanpa shorting) dan hitung:
    - expected annual return
    - annualized volatility
    - Sharpe ratio
    Return: (weights_matrix, returns_arr, vol_arr, sharpe_arr)
    """
    n_assets = returns.shape[1]
    daily_mean = returns.mean()
    cov = returns.cov()

    weights_all = []
    rets = []
    vols = []
    sharpes = []

    for _ in range(n_portfolios):
        # random weights pakai dirichlet (sum = 1)
        w = np.random.dirichlet(np.ones(n_assets))
        weights_all.append(w)

        port_daily_ret = (returns @ w)
        mean_daily = port_daily_ret.mean()
        std_daily = port_daily_ret.std()

        ann_ret = (1 + mean_daily) ** periods_per_year - 1
        ann_vol = std_daily * np.sqrt(periods_per_year)

        if ann_vol == 0 or np.isnan(ann_vol):
            sharpe = np.nan
        else:
            sharpe = (ann_ret - risk_free_rate) / ann_vol

        rets.append(ann_ret)
        vols.append(ann_vol)
        sharpes.append(sharpe)

    return (
        np.array(weights_all),
        np.array(rets),
        np.array(vols),
        np.array(sharpes),
    )


def find_max_sharpe_portfolio(
    returns: pd.DataFrame,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.0,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Cari portfolio dengan Sharpe ratio maksimum via random search.
    Return: (best_weights, best_ret, best_vol, best_sharpe)
    """
    weights_all, rets, vols, sharpes = random_portfolios(
        returns,
        n_portfolios=n_portfolios,
        risk_free_rate=risk_free_rate,
    )
    idx = np.nanargmax(sharpes)
    return weights_all[idx], rets[idx], vols[idx], sharpes[idx]


def equal_weight_portfolio_metrics(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Tuple[pd.Series, float, float, float]:
    """
    Hitung metrik untuk equal-weight portfolio:
    - return harian
    - annual return
    - annual volatility
    - Sharpe ratio
    """
    n_assets = returns.shape[1]
    w = np.ones(n_assets) / n_assets

    port_daily = returns @ w
    mean_daily = port_daily.mean()
    std_daily = port_daily.std()

    ann_ret = (1 + mean_daily) ** periods_per_year - 1
    ann_vol = std_daily * np.sqrt(periods_per_year)
    sharpe = (
        np.nan
        if ann_vol == 0 or np.isnan(ann_vol)
        else (ann_ret - risk_free_rate) / ann_vol
    )

    return port_daily, ann_ret, ann_vol, sharpe


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Hitung cumulative return dari return harian.
    """
    return (1 + returns).cumprod()
