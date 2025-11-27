# 🚀 CryptoAnalysis — End-to-End Crypto Intelligence Dashboard

![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-FF4B4B?style=flat-square)
![Model](https://img.shields.io/badge/Forecasting-Prophet-22c55e?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

**CryptoAnalysis** is a comprehensive **Python + Streamlit**-based dashboard for **real-time market monitoring**, **forecasting**, **portfolio optimization**, and **automated trading signals** using live cryptocurrency data.

Designed to deliver a premium fintech-like analytical experience — modern, responsive, futuristic, and fully integrated with real market data.

---

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [🧱 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📊 Trading Signals](#-trading-signals)
- [📈 Backtesting](#-backtesting)
- [🧠 Sentiment Analysis](#-sentiment-analysis)
- [🧪 Roadmap](#-roadmap)
- [🧑‍💻 Author](#-author)
- [📄 License](#-license)

---

## ✨ Key Features

### 🔌 Live Data & Snapshot System
- Fetch real-time data from APIs and automatically save snapshots to `/data/raw` for offline analysis.
- Supports multiple cryptocurrencies: Bitcoin, Ethereum, Solana.

### 📍 Market Intelligence
- Interactive candlestick and line charts.
- Volume analytics, bull/bear regime detection.
- Correlation matrix, volatility analysis, and Sharpe-based risk metrics.

### 🔮 Forecasting Engine (Prophet)
- Short-term price forecasting with confidence intervals.
- Compare forecasts against actual prices using time-series models.

### 📡 Hybrid Trading Signal System
Combines multiple indicators for automated signals:
- **EMA 20/50** (trend direction)
- **MACD (12/26/9)** (momentum confirmation)
- **RSI (14)** (overbought/oversold filter)
- **Bollinger Bands** (volatility & breakout detection)
- **Volume Spike** (market strength confirmation)

Generates scores and actions: **BUY / STRONG BUY / HOLD / SELL / STRONG SELL**

### 🎯 Backtesting Engine
Compare strategy performance vs. Buy & Hold:
- ✅ Equity curves
- ✅ Maximum drawdown
- ✅ Win rate
- ✅ Trade count
- ✅ Average return per trade

### 💼 Portfolio Optimization
- Equal-weight vs. Max-Sharpe allocation.
- Cumulative portfolio return charts.

### 🧠 Sentiment Analysis Module
- Upload CSV files with tweets/news.
- Compute daily sentiment scores and correlate with price changes, returns, and volatility.

---

## 🧱 Project Structure

```
CryptoAnalysis/
├── dashboard/
│   ├── streamlit_app.py    # Main Streamlit application
│   └── style.css           # Custom CSS for UI styling
├── data/
│   ├── raw/                # Real-time data snapshots (Parquet files)
│   ├── processed/          # Cleaned datasets ready for modeling
│   └── exports/            # Reports, signals, and forecasting outputs
├── models/
│   ├── forecasting/        # Saved Prophet models and scalers
│   └── signals/            # Signal configuration
├── src/
│   ├── data_collection.py  # API data fetching
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── forecasting.py      # Prophet forecasting logic
│   ├── portfolio.py        # Portfolio optimization
│   ├── sentiment.py        # Sentiment analysis
│   ├── analysis.py         # Market analysis utilities
│   ├── risk_metrics.py     # Risk calculation
│   ├── signals.py          # Trading signal generation
│   └── config.py           # Configuration settings
├── seed_raw_data.py        # Script to seed initial data
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/rheihan/CryptoAnalysis-End-to-end-Crypto-Intelligence-Dashboard.git
   cd CryptoAnalysis-End-to-end-Crypto-Intelligence-Dashboard
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Seed initial data**:
   ```bash
   python seed_raw_data.py
   ```

---

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run dashboard/streamlit_app.py
```

Navigate through the tabs:
- **Overview**: Market data visualization and risk metrics.
- **Forecast**: Generate price predictions.
- **Signals**: View trading signals and backtest results.
- **Portfolio**: Optimize asset allocations.
- **Sentiment**: Analyze market sentiment from uploaded data.

---

## 📊 Trading Signals

| Indicator       | Contribution                      |
|-----------------|-----------------------------------|
| EMA Trend       | Directional bias                  |
| MACD Cross      | Momentum confirmation             |
| RSI Level       | Market exhaustion filter          |
| Bollinger Bands | Volatility + breakout detection   |
| Volume Spike    | Market strength confirmation      |

**Scoring System**:
- 0 → HOLD
- 2 → BUY BIAS
- 4 → BUY
- 6+ → STRONG BUY
- Negative scores indicate SELL signals.

---

## 📈 Backtesting

**Example Output**:
- Strategy Return: +48.22%
- Buy & Hold: +19.03%
- Max Drawdown: -14.2%
- Trades Executed: 23
- Win Rate: 61.5%
- Avg Return Per Trade: +2.78%

---

## 🧠 Sentiment Analysis

**Example CSV Input**:
```csv
timestamp,text
2025-01-12,"Solana breaking out!"
2025-01-12,"Bitcoin looks weak"
```

**Outputs**:
- Sentiment score per entry
- Daily aggregation
- Correlation with volatility & returns
- Combined price + sentiment visualization

---

## 🧪 Roadmap

- 🔔 Telegram live alerts
- 🤖 Reinforcement learning bot
- 📦 Docker container deployment
- 📄 Automatic PDF report export
- ☁️ Hosted public version (Streamlit Cloud / Vercel / AWS)

---

## 🧑‍💻 Author

**Rheihandra**  
Data Analyst | Junior Frontend Developer | 7th Semester Information Systems Student

---

## 📄 License

This project is licensed under the MIT License — feel free to use, learn from, and develop upon it.

---

*Built with ❤️ for the crypto community*
