import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Hitung RSI klasik 14 hari."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah indikator teknikal dan sinyal trading.

    - EMA fast / slow (trend filter)
    - MACD + MACD signal (momentum)
    - RSI (overbought / oversold)
    - trade_signal: 1 = BUY, -1 = SELL, 0 = HOLD
    - position: 1 = ON (long), 0 = OFF (cash)
    """
    out = df.copy()

    price = out["price"]

    # Trend filter: EMA 20 & EMA 50 (bisa diganti sesuai selera)
    out["ema_fast"] = price.ewm(span=20, adjust=False).mean()
    out["ema_slow"] = price.ewm(span=50, adjust=False).mean()

    # MACD 12/26 + signal 9
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI
    out["rsi"] = _rsi(price, period=14)

    # ==== RULE SINYAL ====
    # Uptrend kalau ema_fast > ema_slow
    uptrend = out["ema_fast"] > out["ema_slow"]

    # MACD cross up / down
    macd_cross_up = (out["macd"] > out["macd_signal"]) & (
        out["macd"].shift(1) <= out["macd_signal"].shift(1)
    )
    macd_cross_down = (out["macd"] < out["macd_signal"]) & (
        out["macd"].shift(1) >= out["macd_signal"].shift(1)
    )

    # RSI level
    rsi_oversold = out["rsi"] < 35
    rsi_overbought = out["rsi"] > 70

    # BUY:
    # - trend naik
    # - MACD cross up
    # - RSI baru keluar dari oversold / masih di bawah 70
    buy_signal = uptrend & macd_cross_up & ~rsi_overbought

    # SELL:
    # - MACD cross down   ATAU
    # - RSI overbought    ATAU
    # - harga tembus turun di bawah ema_slow (cut trend)
    price_breakdown = price < out["ema_slow"]
    sell_signal = macd_cross_down | rsi_overbought | price_breakdown

    out["trade_signal"] = 0
    out.loc[buy_signal, "trade_signal"] = 1
    out.loc[sell_signal, "trade_signal"] = -1

    # Position: long only (1 / 0)
    position = []
    current_pos = 0
    for sig in out["trade_signal"]:
        if sig == 1:
            current_pos = 1
        elif sig == -1:
            current_pos = 0
        position.append(current_pos)

    out["position"] = position

    return out


def backtest_strategy(
    df_sig: pd.DataFrame,
    fee_per_trade: float = 0.001,  # 0.1% per transaksi (sekali masuk / keluar)
) -> dict:
    """
    Backtest sederhana strategi long-only:

    - return harian dari df['return']
    - posisi diambil dari df['position'] (0/1)
    - fee dikenakan tiap kali terjadi perubahan posisi

    Return dict berisi:
    - equity_strategy, equity_buyhold (Series)
    - total_ret_strategy, total_ret_buyhold
    - maxdd_strategy
    - num_trades, win_rate, avg_trade_ret
    """
    df = df_sig.copy()

    if "return" not in df.columns:
        raise ValueError("Kolom 'return' wajib ada di dataframe untuk backtest.")

    # Buy & hold equity
    bh_ret = df["return"].fillna(0)
    equity_bh = (1 + bh_ret).cumprod()

    # Strategy equity
    pos = df["position"].fillna(0)

    # perpindahan posisi (entry/exit)
    pos_change = pos.diff().fillna(0).abs()
    fee_series = pos_change * fee_per_trade

    strat_daily_ret = df["return"] * pos.shift(1).fillna(0) - fee_series
    equity_strat = (1 + strat_daily_ret).cumprod()

    # Trade-level stats
    trades = df[df["trade_signal"] != 0].copy()
    num_trades = int((trades["trade_signal"] != 0).sum())

    # Kalkulasi winrate kasar: lihat return kumulatif antar sinyal SELL
    # (simple, tapi cukup buat indikasi)
    trade_returns = []
    in_trade = False
    entry_equity = None

    for i in range(len(df)):
        sig = df["trade_signal"].iloc[i]
        eq = equity_strat.iloc[i]

        if sig == 1 and not in_trade:
            in_trade = True
            entry_equity = eq
        elif sig == -1 and in_trade:
            in_trade = False
            if entry_equity is not None and entry_equity != 0:
                trade_returns.append(eq / entry_equity - 1)

    # Kalau posisi masih kebuka di akhir, tutup paksa
    if in_trade and entry_equity is not None:
        last_eq = equity_strat.iloc[-1]
        trade_returns.append(last_eq / entry_equity - 1)

    trade_returns = np.array(trade_returns) if trade_returns else np.array([])

    if len(trade_returns) > 0:
        win_rate = float((trade_returns > 0).mean())
        avg_trade_ret = float(trade_returns.mean())
    else:
        win_rate = np.nan
        avg_trade_ret = np.nan

    # Total return & max drawdown
    total_ret_strat = float(equity_strat.iloc[-1] - 1)
    total_ret_bh = float(equity_bh.iloc[-1] - 1)

    running_max = equity_strat.cummax()
    drawdown = equity_strat / running_max - 1
    maxdd = float(drawdown.min())

    return {
        "equity_strategy": equity_strat,
        "equity_buyhold": equity_bh,
        "total_ret_strategy": total_ret_strat,
        "total_ret_buyhold": total_ret_bh,
        "maxdd_strategy": maxdd,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_ret": avg_trade_ret,
    }
