import numpy as np
import pandas as pd

# =========================
# 1. HITUNG INDIKATOR + SIGNAL PER CANDLE
# =========================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pastikan kolom yang dipakai ada
    if 'Close' not in df.columns or 'Volume' not in df.columns:
        raise ValueError("DataFrame harus punya kolom 'Close' dan 'Volume'.")

    # ===== RSI 14 =====
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ===== EMA 20 & 50 =====
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # ===== MACD (12,26,9) =====
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ===== Bollinger Bands (20, 2σ) =====
    df['MA20'] = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * std20
    df['BB_Lower'] = df['MA20'] - 2 * std20

    # ===== Volume Spike (vs rata-rata 20 hari) =====
    df['Volume_Score'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # ===== Scoring + Action (BUY / SELL / HOLD) =====
    scores = []
    actions = []

    for i, row in df.iterrows():
        score = 0

        # RSI
        if not np.isnan(row.get('RSI', np.nan)):
            if row['RSI'] < 30:
                score += 2   # oversold → bullish
            elif row['RSI'] > 70:
                score -= 2   # overbought → bearish

        # MACD
        if not np.isnan(row.get('MACD', np.nan)) and not np.isnan(row.get('Signal_Line', np.nan)):
            if row['MACD'] > row['Signal_Line']:
                score += 1   # bullish
            else:
                score -= 1   # bearish

        # EMA Cross
        if not np.isnan(row.get('EMA20', np.nan)) and not np.isnan(row.get('EMA50', np.nan)):
            if row['EMA20'] > row['EMA50']:
                score += 2   # uptrend
            else:
                score -= 2   # downtrend

        # Bollinger Band
        if not np.isnan(row.get('BB_Lower', np.nan)) and not np.isnan(row.get('BB_Upper', np.nan)):
            if row['Close'] < row['BB_Lower']:
                score += 2   # dip / oversold
            elif row['Close'] > row['BB_Upper']:
                score -= 2   # bubble / overextended

        # Volume Spike
        if not np.isnan(row.get('Volume_Score', np.nan)) and row['Volume_Score'] > 1.3:
            score += 1       # buyer pressure

        scores.append(score)

        # Threshold action (bisa lu tweak)
        if score >= 4:
            actions.append("BUY")
        elif score <= -4:
            actions.append("SELL")
        else:
            actions.append("HOLD")

    df['Score'] = scores
    df['Action'] = actions

    return df


# =========================
# 2. BACKTEST STRATEGY (LONG-ONLY)
# =========================
def backtest_strategy(
    df: pd.DataFrame,
    initial_capital: float = 1000.0,
    fee_pct: float = 0.001,  # 0.1% per transaksi
):
    """
    Simple backtest:
    - Long-only (nggak ada short)
    - BUY: all-in kalau lagi tidak pegang posisi
    - SELL: close full kalau lagi pegang posisi
    - Sinyal diambil dari kolom 'Action' (BUY / SELL / HOLD)
    """

    df = df.copy()

    # Kalau belum ada Action, hitung dulu indicator & signal
    if 'Action' not in df.columns:
        df = compute_indicators(df)

    cash = float(initial_capital)
    position = 0.0      # jumlah coin
    entry_price = None  # harga entry aktif (kalau ada)
    equity_curve = []
    dates = []
    trades = []

    for idx, row in df.iterrows():
        price = float(row['Close'])
        action = row['Action']

        # ===== Eksekusi sinyal di harga close =====
        if action == "BUY" and position == 0:
            # Buy all-in
            buy_amount = cash * (1 - fee_pct)
            position = buy_amount / price
            cash = 0.0
            entry_price = price

        elif action == "SELL" and position > 0:
            # Sell full position
            sell_value = position * price * (1 - fee_pct)
            cash += sell_value
            ret_pct = (price - entry_price) / entry_price * 100

            trades.append({
                "entry_date": idx,
                "exit_date": idx,
                "entry_price": entry_price,
                "exit_price": price,
                "return_pct": ret_pct,
            })

            position = 0.0
            entry_price = None

        # ===== Equity setelah eksekusi =====
        equity = cash + position * price
        dates.append(idx)
        equity_curve.append(equity)

    # ===== Paksa close di candle terakhir kalau masih ada posisi =====
    if position > 0 and entry_price is not None:
        price = float(df.iloc[-1]['Close'])
        sell_value = position * price * (1 - fee_pct)
        cash += sell_value
        ret_pct = (price - entry_price) / entry_price * 100

        trades.append({
            "entry_date": df.index[-1],
            "exit_date": df.index[-1],
            "entry_price": entry_price,
            "exit_price": price,
            "return_pct": ret_pct,
        })

        position = 0.0
        entry_price = None
        equity_curve[-1] = cash  # update equity terakhir

    final_equity = cash

    # ===== Equity curve & max drawdown =====
    eq_series = pd.Series(equity_curve, index=dates)

    if len(eq_series) > 0:
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak
        max_dd_pct = float(drawdown.min() * 100)
    else:
        max_dd_pct = 0.0

    trades_df = pd.DataFrame(trades)

    if not trades_df.empty:
        wins = trades_df[trades_df['return_pct'] > 0]
        losses = trades_df[trades_df['return_pct'] <= 0]

        win_rate = float(len(wins) / len(trades_df) * 100)
        total_return_pct = float((final_equity / initial_capital - 1) * 100)

        if not losses.empty:
            profit_factor = float(wins['return_pct'].sum() / abs(losses['return_pct'].sum()))
        else:
            profit_factor = float('inf')
    else:
        win_rate = 0.0
        total_return_pct = float((final_equity / initial_capital - 1) * 100)
        profit_factor = 0.0

    stats = {
        "initial_capital": float(initial_capital),
        "final_equity": float(final_equity),
        "total_return_pct": total_return_pct,
        "num_trades": int(len(trades_df)),
        "win_rate": win_rate,
        "max_drawdown_pct": max_dd_pct,
        "profit_factor": profit_factor,
    }

    return eq_series, trades_df, stats
