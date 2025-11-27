import sys
import os

# Allow src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.config import SUPPORTED_COINS, DEFAULT_DAYS
from src.data_collection import fetch_market_chart
from src.preprocessing import add_basic_features, drop_na_for_model
from src.analysis import compute_correlation
from src.forecasting import train_prophet
from src.risk_metrics import compute_basic_risk_metrics
from src.sentiment import (
    add_sentiment_scores,
    aggregate_sentiment_by_time,
    align_sentiment_with_market,
)
from src.portfolio import (
    prepare_returns_for_portfolio,
    equal_weight_portfolio_metrics,
    find_max_sharpe_portfolio,
    cumulative_returns,
)

# ===================== THEME / UTILS =====================


def load_css():
    css_path = os.path.join("dashboard", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=True)
def load_data(coin_id: str, days: int) -> pd.DataFrame:
    """
    Load data dari API (live) + preprocess.
    Juga menyimpan snapshot ke data/raw sebagai parquet.
    """
    df = fetch_market_chart(coin_id=coin_id, days=days)
    df = add_basic_features(df)
    df = drop_na_for_model(df)

    # Simpan snapshot real-time ke data/raw
    try:
        raw_dir = os.path.join("data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"{coin_id}_{days}d_{ts}.parquet"
        fpath = os.path.join(raw_dir, fname)
        df.to_parquet(fpath)
    except Exception:
        # Jangan bikin app crash cuma gara-gara gagal save
        pass

    return df


def list_snapshots() -> list[str]:
    """List file parquet di data/raw (kalau folder ada)."""
    raw_dir = os.path.join("data", "raw")
    if not os.path.isdir(raw_dir):
        return []
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".parquet")]
    files.sort(reverse=True)
    return files


def chart_block(fig):
    """Wrapper card buat semua chart biar konsisten sama tema."""
    fig.update_layout(
        paper_bgcolor="#F8F8F8",
        plot_bgcolor="#FFFFFF",
        font_color="#0F172A",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#D1D5DB"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(148,163,184,0.4)",
            borderwidth=1,
        ),
    )
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def tab_header(title: str, subtitle: str, emoji: str):
    st.markdown(
        f"""
        <div class='tab-header'>
          <h2>{emoji} {title}</h2>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===================== SIGNAL & BACKTEST =====================


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Hitung RSI klasik."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean().replace(0, np.nan)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah indikator teknikal & sinyal dengan skoring:

    - EMA 20 / 50 (trend)
    - MACD 12/26 + signal 9
    - RSI 14
    - Bollinger Bands 20, 2σ
    - Volume spike (vs MA20)
    """
    out = df.copy()

    if "price" not in out.columns or "volume" not in out.columns:
        raise ValueError("DataFrame harus punya kolom 'price' dan 'volume'.")

    price = out["price"]
    volume = out["volume"]

    # EMA trend filter
    out["ema_fast"] = price.ewm(span=20, adjust=False).mean()
    out["ema_slow"] = price.ewm(span=50, adjust=False).mean()

    # MACD
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # RSI
    out["rsi"] = _rsi(price, period=14)

    # Bollinger Bands
    out["ma20"] = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    out["bb_upper"] = out["ma20"] + 2 * std20
    out["bb_lower"] = out["ma20"] - 2 * std20

    # Volume spike
    vol_ma20 = volume.rolling(20).mean()
    out["volume_score"] = volume / vol_ma20

    scores = []
    actions = []

    for _, row in out.iterrows():
        score = 0.0

        # RSI
        rsi = row.get("rsi", np.nan)
        if not np.isnan(rsi):
            if rsi < 30:
                score += 2
            elif rsi > 70:
                score -= 2

        # MACD vs signal
        macd = row.get("macd", np.nan)
        macd_sig = row.get("macd_signal", np.nan)
        if not np.isnan(macd) and not np.isnan(macd_sig):
            if macd > macd_sig:
                score += 1
            else:
                score -= 1

        # EMA trend
        ema_f = row.get("ema_fast", np.nan)
        ema_s = row.get("ema_slow", np.nan)
        if not np.isnan(ema_f) and not np.isnan(ema_s):
            if ema_f > ema_s:
                score += 2
            else:
                score -= 2

        # Bollinger
        bb_up = row.get("bb_upper", np.nan)
        bb_lo = row.get("bb_lower", np.nan)
        p = row.get("price", np.nan)
        if not np.isnan(bb_up) and not np.isnan(bb_lo) and not np.isnan(p):
            if p < bb_lo:
                score += 2
            elif p > bb_up:
                score -= 2

        # Volume
        vol_score = row.get("volume_score", np.nan)
        if not np.isnan(vol_score) and vol_score > 1.3:
            score += 1

        scores.append(score)

        if score >= 4:
            actions.append("BUY")
        elif score <= -4:
            actions.append("SELL")
        else:
            actions.append("HOLD")

    out["score"] = scores
    out["action"] = actions

    # numeric trade_signal (hanya saat ada perubahan BUY/SELL)
    trade_signals = []
    prev_action = "HOLD"
    for a in actions:
        if a == "BUY" and prev_action != "BUY":
            trade_signals.append(1)
        elif a == "SELL" and prev_action != "SELL":
            trade_signals.append(-1)
        else:
            trade_signals.append(0)
        prev_action = a
    out["trade_signal"] = trade_signals

    # position long-only
    position = []
    holding = 0
    for a in actions:
        if a == "BUY":
            holding = 1
        elif a == "SELL":
            holding = 0
        position.append(holding)
    out["position"] = position

    return out


def backtest_strategy(df_sig: pd.DataFrame, fee_per_trade: float = 0.001) -> dict:
    """
    Backtest sederhana strategi long-only pakai kolom:
    - 'price'
    - 'return'
    - 'action' (BUY / SELL / HOLD)
    """
    df = df_sig.copy()

    if "return" not in df.columns:
        raise ValueError("Kolom 'return' wajib ada di dataframe untuk backtest.")

    initial_capital = 1.0

    # Buy & hold
    bh_ret = df["return"].fillna(0)
    equity_bh = initial_capital * (1 + bh_ret).cumprod()

    cash = initial_capital
    position = 0.0
    entry_equity = None

    equity_curve = []
    trade_returns = []

    for _, row in df.iterrows():
        price = float(row["price"])
        action = row.get("action", "HOLD")

        equity_before = cash + position * price

        if action == "BUY" and position == 0:
            buy_amount = cash * (1 - fee_per_trade)
            position = buy_amount / price
            cash = 0.0
            entry_equity = equity_before

        elif action == "SELL" and position > 0:
            sell_value = position * price * (1 - fee_per_trade)
            cash = sell_value
            position = 0.0

            if entry_equity is not None and entry_equity > 0:
                exit_equity = cash
                trade_ret = exit_equity / entry_equity - 1
                trade_returns.append(trade_ret)
                entry_equity = None

        equity = cash + position * price
        equity_curve.append(equity)

    # tutup paksa di akhir kalau masih pegang posisi
    if position > 0 and entry_equity is not None:
        last_price = float(df["price"].iloc[-1])
        sell_value = position * last_price * (1 - fee_per_trade)
        cash = sell_value
        position = 0.0
        exit_equity = cash
        trade_ret = exit_equity / entry_equity - 1
        trade_returns.append(trade_ret)
        equity_curve[-1] = cash

    equity_strategy = pd.Series(equity_curve, index=df.index)
    final_equity = float(equity_strategy.iloc[-1])

    running_max = equity_strategy.cummax()
    drawdown = equity_strategy / running_max - 1
    maxdd = float(drawdown.min())

    trade_returns = np.array(trade_returns) if trade_returns else np.array([])
    if trade_returns.size > 0:
        win_rate = float((trade_returns > 0).mean())
        avg_trade_ret = float(trade_returns.mean())
        num_trades = int(trade_returns.size)
    else:
        win_rate = np.nan
        avg_trade_ret = np.nan
        num_trades = 0

    total_ret_strategy = final_equity / initial_capital - 1
    total_ret_buyhold = float(equity_bh.iloc[-1] / initial_capital - 1)

    return {
        "equity_strategy": equity_strategy,
        "equity_buyhold": equity_bh,
        "total_ret_strategy": total_ret_strategy,
        "total_ret_buyhold": total_ret_buyhold,
        "maxdd_strategy": maxdd,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_trade_ret": avg_trade_ret,
    }


# ===================== MAIN APP =====================


def main():
    st.set_page_config(page_title="Crypto Intelligence Hub", layout="wide")
    load_css()

    # ---------- HEADER ----------
    st.markdown(
        """
        <h1 class='title'>Crypto Intelligence Hub</h1>
        <p class='subtitle'>
          Real-time Analytics • Forecasting • Signals • Portfolio Optimization • Sentiment AI
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---------- SIDEBAR ----------
    st.sidebar.header("⚙ Configuration")
    coin_name = st.sidebar.selectbox("Select Asset", list(SUPPORTED_COINS.keys()), index=0)
    days = st.sidebar.slider("Data Range (Days)", 30, 365, DEFAULT_DAYS, step=30)
    coin_id = SUPPORTED_COINS[coin_name]

    # ====== DATA SOURCE (clean + adaptif) ======
    st.sidebar.markdown(
        """
        <div class="sidebar-block">
          <div class="sb-title">📂 Data Source</div>
          <div class="sb-sub">Choose how you want to feed market data into the dashboard.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_mode = st.sidebar.radio(
        label="",
        options=["Live API (real-time)", "Snapshot (from /data/raw)"],
        index=0,
    )

    snapshot_file = None
    if data_mode == "Snapshot (from /data/raw)":
        files = list_snapshots()
        if files:
            st.sidebar.markdown(
                "<div class='sb-label'>Snapshot file</div>",
                unsafe_allow_html=True,
            )
            snapshot_file = st.sidebar.selectbox("", files)
        else:
            st.sidebar.info("No snapshot found in data/raw. Falling back to Live API.")
            data_mode = "Live API (real-time)"

    st.sidebar.subheader("📥 Sentiment Upload (Optional)")
    uploaded_file = st.sidebar.file_uploader("Upload Tweet/News CSV", type=["csv"])

    # ---------- LOAD MARKET DATA ----------
    with st.spinner("Fetching and processing market data..."):
        if data_mode.startswith("Live API"):
            df = load_data(coin_id, days)
            data_source_label = f"Live API · {coin_name} ({days}d)"
        else:
            raw_dir = os.path.join("data", "raw")
            path = os.path.join(raw_dir, snapshot_file)
            df = pd.read_parquet(path)
            data_source_label = f"Snapshot · {snapshot_file}"

    # tampilkan sumber data aktif di bawah header
    st.markdown(
        f"<p class='subtitle'>Active data source: <strong>{data_source_label}</strong></p>",
        unsafe_allow_html=True,
    )

    # ---------- TABS ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📍 Overview", "🔮 Forecast", "📡 Signals", "💼 Portfolio", "🧠 Sentiment"]
    )

    # ===== TAB 1: OVERVIEW =====
    with tab1:
        tab_header(
            "Market Overview",
            "Snapshot harga, volume, dan struktur market regime untuk aset yang dipilih.",
            "📍",
        )

        col1, col2 = st.columns(2)
        with col1:
            fig_price = px.line(
                df.reset_index(), x="timestamp", y="price", title=f"{coin_name} Price"
            )
            chart_block(fig_price)

        with col2:
            fig_vol = px.bar(
                df.reset_index(), x="timestamp", y="volume", title=f"{coin_name} Volume"
            )
            chart_block(fig_vol)

        tab_header(
            "Market Regime & Correlation",
            "Identifikasi fase bull / bear dan korelasi antar fitur harga.",
            "📊",
        )

        fig_regime = px.scatter(
            df.reset_index(),
            x="timestamp",
            y="price",
            color="market_regime",
            opacity=0.9,
            title="Bull / Bear Regime",
        )
        chart_block(fig_regime)

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Return Correlation")
            corr = compute_correlation(df)
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None))
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Risk Metrics")
            metrics = compute_basic_risk_metrics(df)
            colA, colB = st.columns(2)
            colA.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            colB.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB 2: FORECAST =====
    with tab2:
        tab_header(
            "Price Forecast",
            "Forecast jangka pendek menggunakan time-series model (Prophet / fallback).",
            "🔮",
        )

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col_left, col_right = st.columns([3, 1])
        with col_left:
            horizon = st.slider(
                "Forecast Horizon (days)",
                7,
                60,
                value=30,
                step=5,
            )
        with col_right:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            run_forecast = st.button("Generate Forecast", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_forecast:
            with st.spinner("Running forecasting model..."):
                model, forecast = train_prophet(df, periods=horizon)

            actual_df = df.reset_index().rename(columns={"timestamp": "ds"})
            fig_fc = px.line(forecast, x="ds", y="yhat", title=f"{coin_name} Forecast")
            fig_fc.add_scatter(
                x=actual_df["ds"], y=actual_df["price"], mode="lines", name="Actual"
            )
            chart_block(fig_fc)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                **Forecast summary**

                - Asset : **{coin_name}**  
                - Horizon : **{horizon} days**  
                - Model : Prophet (additive trend & seasonality)  
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB 3: SIGNALS =====
    with tab3:
        tab_header(
            "Trading Signals",
            "Strategi rule-based yang menggabungkan trend, momentum (MACD), RSI, Bollinger Bands & volume dengan backtest.",
            "📡",
        )

        df_sig = build_signals(df)
        dfr = df_sig.reset_index()
        last = df_sig.iloc[-1]
        last_time = df_sig.index[-1]

        score = last["score"]
        if score >= 4:
            regime_text = "🚀 Strong Buy"
            badge_bg = "rgba(34,197,94,0.18)"
            badge_border = "#4ade80"
        elif score >= 2:
            regime_text = "📈 Buy Bias"
            badge_bg = "rgba(34,197,94,0.12)"
            badge_border = "#22c55e"
        elif score <= -4:
            regime_text = "🔻 Strong Sell"
            badge_bg = "rgba(248,113,113,0.18)"
            badge_border = "#f87171"
        elif score <= -2:
            regime_text = "📉 Sell Bias"
            badge_bg = "rgba(248,113,113,0.12)"
            badge_border = "#f97373"
        else:
            regime_text = "⚖ Neutral / Wait"
            badge_bg = "rgba(148,163,184,0.20)"
            badge_border = "#cbd5f5"

        badge_html = f"""
        <span style="
            display:inline-flex;
            align-items:center;
            gap:6px;
            padding:4px 10px;
            border-radius:999px;
            background:{badge_bg};
            border:1px solid {badge_border};
            font-size:12px;
            font-weight:600;
            color:#E5F0FF;
        ">
            {regime_text}
        </span>
        """

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1.8, 1.4, 1.6])

        with c1:
            st.markdown("### 📌 Latest Signal", unsafe_allow_html=True)
            st.markdown(badge_html, unsafe_allow_html=True)
            st.markdown(
                f"<p style='font-size:12px;color:#C4D0DD;margin-top:6px;'>Last update: {last_time}</p>",
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div style="font-size:13px;line-height:1.5;">
                    <div><span style="color:#C4D0DD;">Action</span><br><strong>{last['action']}</strong></div>
                    <div style="margin-top:6px;"><span style="color:#C4D0DD;">Score</span><br><strong>{last['score']:.1f}</strong></div>
                    <div style="margin-top:6px;"><span style="color:#C4D0DD;">Price</span><br><strong>{last['price']:.4f}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div style="font-size:13px;line-height:1.5;">
                    <div><span style="color:#C4D0DD;">RSI (14)</span><br><strong>{last['rsi']:.1f}</strong></div>
                    <div style="margin-top:6px;">
                        <span style="color:#C4D0DD;">EMA 20 / EMA 50</span><br>
                        <strong>{last['ema_fast']:.2f}</strong> / <strong>{last['ema_slow']:.2f}</strong>
                    </div>
                    <div style="margin-top:6px;">
                        <span style="color:#C4D0DD;">Volume Spike (vs 20d)</span><br>
                        <strong>{last['volume_score']:.2f}×</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        fig_sig = px.line(
            dfr,
            x="timestamp",
            y="price",
            title="Price with Signals & Trend Filter",
        )
        fig_sig.add_scatter(
            x=dfr["timestamp"],
            y=dfr["ema_fast"],
            mode="lines",
            name="EMA 20",
        )
        fig_sig.add_scatter(
            x=dfr["timestamp"],
            y=dfr["ema_slow"],
            mode="lines",
            name="EMA 50",
        )

        buys = dfr[dfr["trade_signal"] == 1]
        sells = dfr[dfr["trade_signal"] == -1]

        if not buys.empty:
            fig_sig.add_scatter(
                x=buys["timestamp"],
                y=buys["price"],
                mode="markers",
                name="BUY",
            )
        if not sells.empty:
            fig_sig.add_scatter(
                x=sells["timestamp"],
                y=sells["price"],
                mode="markers",
                name="SELL",
            )

        chart_block(fig_sig)

        bt = backtest_strategy(df_sig)
        eq_df = pd.DataFrame(
            {
                "Strategy": bt["equity_strategy"],
                "Buy & Hold": bt["equity_buyhold"],
            }
        )

        tab_header(
            "Backtest Performance",
            "Bandingkan kurva ekuitas strategi vs buy & hold dengan metrik ringkas.",
            "📈",
        )

        fig_eq = px.line(
            eq_df.reset_index(),
            x="timestamp",
            y=eq_df.columns,
            title="Equity Curve (base = 1.0)",
        )
        chart_block(fig_eq)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Return (Strategy)", f"{bt['total_ret_strategy']:.2%}")
        col2.metric("Total Return (Buy & Hold)", f"{bt['total_ret_buyhold']:.2%}")
        col3.metric("Max Drawdown (Strategy)", f"{bt['maxdd_strategy']:.2%}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Number of Trades", bt["num_trades"])
        col5.metric(
            "Win Rate",
            f"{bt['win_rate']*100:.1f}%"
            if not np.isnan(bt["win_rate"])
            else "N/A",
        )
        col6.metric(
            "Avg Return / Trade",
            f"{bt['avg_trade_ret']:.2%}"
            if not np.isnan(bt["avg_trade_ret"])
            else "N/A",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ===== TAB 4: PORTFOLIO =====
    with tab4:
        tab_header(
            "Portfolio Optimization",
            "Optimasi alokasi aset menggunakan pendekatan Equal Weight vs Max Sharpe.",
            "💼",
        )

        assets = st.multiselect(
            "Select Assets for Portfolio",
            list(SUPPORTED_COINS.keys()),
            default=list(SUPPORTED_COINS.keys())[:3],
        )

        if len(assets) >= 2:
            if st.button("Optimize Portfolio"):
                with st.spinner("Computing optimal weights..."):
                    market_data = {
                        name: load_data(SUPPORTED_COINS[name], days)
                        for name in assets
                    }
                    returns = prepare_returns_for_portfolio(market_data)

                    ew_daily, ew_ret, ew_vol, ew_sharpe = equal_weight_portfolio_metrics(
                        returns
                    )
                    best_w, best_ret, best_vol, best_sharpe = find_max_sharpe_portfolio(
                        returns
                    )

                    cum_df = pd.DataFrame(
                        {
                            "Equal Weight": cumulative_returns(ew_daily),
                            "Max Sharpe": cumulative_returns(returns @ best_w),
                        }
                    )

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Max Sharpe", f"{best_sharpe:.2f}")
                col2.metric("EW Sharpe", f"{ew_sharpe:.2f}")
                col3.metric("Assets", len(assets))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("📌 **Max Sharpe Allocation**")
                alloc_df = pd.DataFrame(
                    {"Asset": assets, "Weight": [f"{w:.2%}" for w in best_w]}
                )
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.table(alloc_df)
                st.markdown("</div>", unsafe_allow_html=True)

                fig_port = px.line(cum_df, title="Portfolio Cumulative Returns")
                chart_block(fig_port)
        else:
            st.info("Select at least 2 assets to optimize portfolio.")

    # ===== TAB 5: SENTIMENT =====
    with tab5:
        tab_header(
            "Sentiment Analysis",
            "Hubungkan sentiment tweet/news dengan pergerakan harga, return, dan volatilitas.",
            "🧠",
        )

        if not uploaded_file:
            st.info("Upload sentiment CSV in sidebar to start analysis.")
        else:
            raw_df = pd.read_csv(uploaded_file)
            text_col = st.selectbox("Text Column", raw_df.columns)
            time_col = st.selectbox("Timestamp Column", raw_df.columns)

            if st.button("Analyze Sentiment"):
                with st.spinner("Calculating sentiment scores..."):
                    scored = add_sentiment_scores(raw_df, text_col=text_col)
                    sent_daily = aggregate_sentiment_by_time(scored, time_col=time_col)
                    combined = align_sentiment_with_market(df, sent_daily)

                fig_sent = px.line(
                    combined.reset_index(),
                    x="timestamp",
                    y=["price", "sentiment"],
                    title="Price vs Sentiment (Daily)",
                )
                chart_block(fig_sent)

                with st.container():
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Correlation (Sentiment vs Return & Volatility)")
                    cols = ["sentiment", "return", "volatility"]
                    st.dataframe(combined[cols].corr())
                    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- FOOTER ----------
    st.markdown(
        "<p class='footer-text'>© 2025 | Built by Rheihandra Sang Fullstack 🚀</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
